#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from umpeek.eval2.runner import run_full_setting  # noqa: E402


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "runs"
    / "exp2_full_comparison"
    / "A200_real_agent_qwen3_full_current"
    / "manifest"
    / "full_matrix_manifest.json"
)
DEFAULT_DEFENSES = ("undefended", "privacy_checker", "theory_of_mind")
DEFENSE_CHOICES = (*DEFAULT_DEFENSES, "stateful_counterfactual")
BACKENDS = ("Mem0", "Graphiti", "LangMem+LangGraph")
BENCHMARKS = (
    "PersonaMem-v2",
    "PersonaLens",
    "ETAPP_150x32",
    "LoCoMo_10conv_1523QA_20speakers",
)
FULL_BUDGETS = (0, 1, 2, 4, 8, 16)
SMOKE_BUDGETS = (0, 1)
FULL_BENCHMARK_LIMITS = {"PersonaLens": 5000}
ENV_DEFAULTS = {
    "UMPEEK_EVAL2_REAL_AGENT_MODE": "1",
    "UMPEEK_REAL_AGENT_MODEL": "Qwen/Qwen3-14B",
    "UMPEEK_REAL_AGENT_VLLM_BASE_URL": "http://127.0.0.1:8010/v1",
    "UMPEEK_REAL_AGENT_REQUIRE_LIVE_ENDPOINT": "1",
    "UMPEEK_REAL_AGENT_ENABLE_THINKING": "0",
    "UMPEEK_REAL_AGENT_STRICT_MODEL_CHECK": "1",
    "UMPEEK_EVAL2_GENERATE_MISSING_VISIBLE": "0",
    "UMPEEK_EVAL2_DISABLE_GENERATION_CACHE": "1",
    "UMPEEK_EVAL2_PAPER_FACING_ONLY": "1",
    "UMPEEK_EVAL2_LATENT_GOLD_MODE": "profile",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run matched UMPeek evaluations for the adaptive-defense paper artifacts."
    )
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--defense", action="append", choices=DEFENSE_CHOICES)
    parser.add_argument("--backend", action="append", choices=BACKENDS)
    parser.add_argument("--benchmark", action="append", choices=BENCHMARKS)
    parser.add_argument("--budgets", type=str, help="Comma-separated additional follow-up budgets.")
    parser.add_argument("--limit", type=int, help="Samples per backend x benchmark x budget.")
    parser.add_argument(
        "--vllm-endpoint",
        action="append",
        help="OpenAI-compatible vLLM base URL. Repeat to distribute settings across endpoints.",
    )
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _parse_budgets(raw: str | None, *, mode: str) -> tuple[int, ...]:
    values = SMOKE_BUDGETS if raw is None and mode == "smoke" else FULL_BUDGETS if raw is None else tuple(
        int(item.strip()) for item in raw.split(",") if item.strip()
    )
    if not values or tuple(sorted(set(values))) != tuple(values) or any(value < 0 for value in values):
        raise ValueError("Budgets must be unique, non-negative, and increasing.")
    return tuple(values)


def _slug(value: str) -> str:
    return "_".join(part for part in "".join(char.lower() if char.isalnum() else "_" for char in value).split("_") if part)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _tasks(
    defenses: Sequence[str],
    backends: Sequence[str],
    benchmarks: Sequence[str],
    *,
    mode: str,
    explicit_limit: int | None,
) -> list[dict[str, Any]]:
    return [
        {
            "defense": defense,
            "backend": backend,
            "benchmark": benchmark,
            "limit": explicit_limit
            if explicit_limit is not None
            else 1
            if mode == "smoke"
            else FULL_BENCHMARK_LIMITS.get(benchmark),
        }
        for defense in defenses
        for benchmark in benchmarks
        for backend in backends
    ]


def _planned_trajectory_count(manifest_path: Path, tasks: Sequence[Mapping[str, Any]]) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts = {
        (str(job.get("backend")), str(job.get("benchmark"))): int(
            job.get("input_split", {}).get("sample_count", 0) or 0
        )
        for job in manifest.get("setting_jobs", [])
        if str(job.get("method")) == "UMPeek_final"
    }
    total = 0
    for task in tasks:
        count = counts.get((str(task["backend"]), str(task["benchmark"])), 0)
        limit = int(task["limit"]) if task.get("limit") is not None else None
        total += min(limit, count) if limit is not None else count
    return total


def _setting_path(run_root: Path, task: dict[str, Any]) -> Path:
    setting = f"{_slug(str(task['backend']))}__{_slug(str(task['benchmark']))}"
    return run_root / "settings" / str(task["defense"]) / setting


def _run_task(
    task: dict[str, Any],
    *,
    manifest: Path,
    run_root: Path,
    resume: bool,
    budgets: Sequence[int],
    dry_run: bool,
) -> dict[str, Any]:
    started = time.time()
    defense = str(task["defense"])
    limit = int(task["limit"]) if task.get("limit") is not None else None
    os.environ["UMPEEK_EVAL2_DEFENSE"] = "none" if defense == "undefended" else defense
    os.environ.pop("UMPEEK_ADAPTIVE_DEFENSE_SKIP_HELDOUT", None)
    for key, value in ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)
    os.environ["UMPEEK_REAL_AGENT_VLLM_BASE_URL"] = str(task["vllm_endpoint"])

    out_dir = _setting_path(run_root, task)
    log_path = run_root / "logs" / defense / f"{out_dir.name}.log"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_handle:
            with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
                result = run_full_setting(
                    project_root=PROJECT_ROOT,
                    manifest_path=manifest,
                    backend=str(task["backend"]),
                    benchmark=str(task["benchmark"]),
                    methods=("UMPeek_final",),
                    out_dir=out_dir,
                    dry_run=dry_run,
                    limit=limit,
                    resume=resume,
                    force_smoke=False,
                    max_retries=0,
                    budget_grid_override=tuple(int(item) for item in budgets),
                    shared_budget_prefix=True,
                )
        return {
            **task,
            "status": "planned" if dry_run else "completed",
            "elapsed_s": round(time.time() - started, 3),
            "record_count": result.get("record_count", result.get("planned_total_records", 0)),
            "status_counts": result.get("status_counts", {}),
            "out_dir": str(out_dir),
            "log_path": str(log_path),
            "vllm_endpoint": str(task["vllm_endpoint"]),
        }
    except Exception as exc:
        return {
            **task,
            "status": "failed",
            "elapsed_s": round(time.time() - started, 3),
            "record_count": 0,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "out_dir": str(out_dir),
            "log_path": str(log_path),
            "vllm_endpoint": str(task["vllm_endpoint"]),
        }


def _write_progress(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = (
        "status",
        "defense",
        "backend",
        "benchmark",
        "vllm_endpoint",
        "record_count",
        "elapsed_s",
        "out_dir",
        "log_path",
        "error",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    args = _arguments()
    budgets = _parse_budgets(args.budgets, mode=args.mode)
    defenses = tuple(args.defense or DEFAULT_DEFENSES)
    backends = tuple(args.backend or BACKENDS)
    benchmarks = tuple(args.benchmark or BENCHMARKS)
    endpoints = tuple(
        args.vllm_endpoint
        or (ENV_DEFAULTS["UMPEEK_REAL_AGENT_VLLM_BASE_URL"],)
    )
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")
    run_root = args.run_root or (
        PROJECT_ROOT / "runs" / "adaptive_defense" / f"A003_{args.mode}_external_defenses_prefix_current"
    )
    tasks = _tasks(
        defenses,
        backends,
        benchmarks,
        mode=args.mode,
        explicit_limit=args.limit,
    )
    for task_index, task in enumerate(tasks):
        task["vllm_endpoint"] = endpoints[task_index % len(endpoints)].rstrip("/")
    workers = args.max_workers if args.max_workers is not None else (1 if args.mode == "smoke" else 3)
    workers = max(1, min(int(workers), len(tasks)))
    run_root.mkdir(parents=True, exist_ok=True)
    planned_trajectories = _planned_trajectory_count(args.manifest, tasks)
    launch = {
        "schema_version": "adaptive_defense_artifact_run_v2",
        "mode": args.mode,
        "started_at": _now(),
        "manifest": str(args.manifest),
        "run_root": str(run_root),
        "defenses": list(defenses),
        "backends": list(backends),
        "benchmarks": list(benchmarks),
        "additional_followup_budgets": list(budgets),
        "explicit_limit_per_setting": args.limit,
        "benchmark_limits": {
            benchmark: next(task["limit"] for task in tasks if task["benchmark"] == benchmark)
            for benchmark in benchmarks
        },
        "setting_task_count": len(tasks),
        "planned_trajectory_count": planned_trajectories,
        "planned_record_count": planned_trajectories * len(budgets),
        "max_workers": workers,
        "vllm_endpoints": list(endpoints),
        "matched_umpeek": True,
        "shared_budget_prefix": True,
        "max_realized_followups_per_sample": max(budgets),
        "environment": ENV_DEFAULTS,
        "stateful_counterfactual_configuration": {
            "exposure_threshold": os.environ.get("UMPEEK_STATEFUL_EXPOSURE_THRESHOLD", "config_default"),
            "use_cross_request_state": os.environ.get(
                "UMPEEK_STATEFUL_USE_CROSS_REQUEST_STATE", "config_default"
            ),
            "use_counterfactual_comparison": os.environ.get(
                "UMPEEK_STATEFUL_USE_COUNTERFACTUAL_COMPARISON", "config_default"
            ),
        },
    }
    _write_json(run_root / "launch_manifest.json", launch)
    print(json.dumps({"event": "adaptive_defense_launch", **launch}, ensure_ascii=False), flush=True)

    completed: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                _run_task,
                task,
                manifest=args.manifest,
                run_root=run_root,
                resume=not args.no_resume,
                budgets=budgets,
                dry_run=args.dry_run,
            ): task
            for task in tasks
        }
        for future in as_completed(future_map):
            row = future.result()
            completed.append(row)
            _append_jsonl(run_root / "launcher_progress.jsonl", row)
            _write_progress(run_root / "launcher_progress.csv", completed)
            _write_json(
                run_root / "launcher_status.json",
                {
                    "schema_version": "adaptive_defense_artifact_run_v2",
                    "updated_at": _now(),
                    "completed_setting_tasks": len(completed),
                    "total_setting_tasks": len(tasks),
                    "failed_setting_tasks": sum(row.get("status") == "failed" for row in completed),
                    "settings": completed,
                },
            )
            print(json.dumps(row, ensure_ascii=False), flush=True)
    return int(any(row.get("status") == "failed" for row in completed))


if __name__ == "__main__":
    raise SystemExit(main())
