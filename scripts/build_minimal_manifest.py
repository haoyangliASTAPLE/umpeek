#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from umpeek.eval2.schema import METRIC_SCHEMA_VERSION  # noqa: E402
from umpeek.eval2.split_contract import (  # noqa: E402
    BENCHMARK_SLUGS,
    CURRENT_STRONG_QUERY_SPLIT_SCHEMA_VERSION,
    SPLIT_ROLE_VISIBILITY,
    attack_probe_public_path,
    behavior_heldout_path,
    eval_rows_path,
    split_id,
    split_manifest_path,
)


BACKENDS = ("Mem0", "Graphiti", "LangMem+LangGraph")
BENCHMARKS = tuple(BENCHMARK_SLUGS)
METHODS = ("UMPeek_final",)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact manifest for the current release interface.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "manifest" / "full_matrix_manifest.json",
    )
    parser.add_argument(
        "--methods",
        default="UMPeek_final",
        help="Comma-separated method names. The default builds the minimal UMPeek-only manifest.",
    )
    return parser.parse_args()


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return dict(json.load(handle))


def sample_count(benchmark: str) -> int:
    manifest_path = split_manifest_path(PROJECT_ROOT, benchmark)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {relative(manifest_path)}. Run scripts/a200_materialize_strong_query_splits.py first."
        )
    manifest = read_json(manifest_path)
    count = int(manifest.get("row_count") or manifest.get("sample_count") or 0)
    if count <= 0:
        raise ValueError(f"No rows recorded in {relative(manifest_path)}")
    return count


def split_record(benchmark: str) -> tuple[dict[str, Any], dict[str, Any]]:
    count = sample_count(benchmark)
    source_paths = [
        relative(eval_rows_path(PROJECT_ROOT, benchmark)),
        relative(attack_probe_public_path(PROJECT_ROOT, benchmark)),
        relative(behavior_heldout_path(PROJECT_ROOT, benchmark)),
        relative(split_manifest_path(PROJECT_ROOT, benchmark)),
    ]
    split = {
        "benchmark": benchmark,
        "benchmark_status": "ready",
        "sample_count": count,
        "source_paths": source_paths,
        "split_id": split_id(benchmark),
        "version_label": "current role-locked release split",
        "strong_query_split_schema_version": CURRENT_STRONG_QUERY_SPLIT_SCHEMA_VERSION,
        "split_roles": dict(SPLIT_ROLE_VISIBILITY),
        "legacy_runtime_heldout_forbidden": True,
        "legacy_dynamic_heldout_replaced": True,
        "private_eval_rows_visible_to_attackers": False,
    }
    heldout = {
        "benchmark": benchmark,
        "split_id": split_id(benchmark),
        "strategy": "precomputed_same_user_or_context_different_task_v1",
        "overlap_with_attack_probe": "fail",
        "reuse_across_all_methods_and_backends": True,
        "forbid_runtime_dynamic_heldout": True,
        "forbid_attack_access_to_heldout_gold_trace": True,
        "private_eval_rows_visible_to_attackers": False,
    }
    return split, heldout


def main() -> int:
    args = arguments()
    requested = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    unknown = sorted(set(requested).difference(METHODS))
    if not requested or unknown:
        raise ValueError(f"Unknown or empty method selection: {unknown or requested}")

    jobs: list[dict[str, Any]] = []
    for benchmark in BENCHMARKS:
        split, heldout = split_record(benchmark)
        for backend in BACKENDS:
            for method in requested:
                slug = "__".join(
                    value.lower().replace("+", "_").replace("-", "_")
                    for value in (backend, benchmark, method)
                )
                jobs.append(
                    {
                        "job_id": f"release__{slug}",
                        "backend": backend,
                        "benchmark": benchmark,
                        "method": method,
                        "status": "ready",
                        "backend_status": "ready",
                        "benchmark_status": "ready",
                        "method_status": "ready",
                        "metric_schema_version": METRIC_SCHEMA_VERSION,
                        "budget_grid": [1, 2, 4, 8, 16],
                        "expected_sample_count": split["sample_count"],
                        "input_split": split,
                        "heldout_policy": heldout,
                        "output_path": f"runs/minimal_evaluation/{slug}",
                    }
                )

    payload = {
        "artifact_schema_version": "anonymous_release_manifest_v1",
        "experiment_version": "project_a_exp2_full_comparison_metrics_v2",
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "budget_grid": [1, 2, 4, 8, 16],
        "methods": list(requested),
        "backends": list(BACKENDS),
        "benchmarks": list(BENCHMARKS),
        "setting_level_job_count": len(jobs),
        "setting_jobs": jobs,
    }
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": relative(output), "setting_jobs": len(jobs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
