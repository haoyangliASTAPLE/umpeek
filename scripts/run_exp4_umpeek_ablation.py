from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import threading
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from umpeek.attack_baselines import AttackPrediction, blank_predicted_user_model
from umpeek.attack_baselines.adapters import (
    EXP4_UMPEEK_ABLATION_VARIANTS,
    build_exp4_umpeek_ablation_adapter,
)
from umpeek.eval2.attack_adapters import (
    AttackTrajectory,
    TrajectoryVictimClient,
    _backend_project_final_reconstruction,
    _method_status,
    _rounds_from_proxy,
    validate_attack_trajectory,
)
from umpeek.eval2.matching import atoms_from_user_model
from umpeek.eval2.runner import (
    A054_TASK_ID,
    DEFAULT_BUDGET_GRID,
    EXP2_EXPERIMENT_VERSION,
    FULL_EVAL_SCHEMA_VERSION,
    METRIC_SCHEMA_VERSION,
    REQUIRED_METRIC_KEYS,
    _build_full_eval_victim_client,
    _build_full_failure_audit,
    _build_full_sample,
    _evaluate_non_ready_as_empty_reconstruction,
    _full_eval_etapp_planner_from_env,
    _method_summary_row,
    _metric_statuses,
    _require_current_strong_split,
    _row_identifier,
    _select_primary_jsonl,
    _status_metrics,
    _trajectory_reason,
    _write_method_summary_csv,
    append_jsonl,
    evaluate_smoke_metrics,
    read_json,
    read_jsonl,
    write_json,
)
from umpeek.eval2.schema import clone_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "runs"
    / "exp2_full_comparison"
    / "A200_real_agent_qwen3_full_current"
    / "manifest"
    / "full_matrix_manifest.json"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "runs" / "exp4_umpeek_evidence" / "A001_current_mechanism_subset"
DEFAULT_REUSE_FULL_UMPEEK_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "exp2_full_comparison"
    / "A200_frozen_qwen3_paper_subset_current"
)
DEFAULT_BACKENDS = ("Mem0", "Graphiti", "LangMem+LangGraph")
DEFAULT_BENCHMARKS = (
    "PersonaMem-v2",
    "PersonaLens",
    "ETAPP_150x32",
    "LoCoMo_10conv_1523QA_20speakers",
)
EXP4_TASK_ID = "EXP4"
EXP4_SCHEMA_VERSION = "exp4_umpeek_evidence_ablation_v1"
UMPEEK_METHOD = "UMPeek_final"

_FULL_SAMPLE_CACHE: dict[tuple[str, int], Any] = {}
_FULL_SAMPLE_CACHE_LOCK = threading.Lock()

STRATIFY_FIELDS = {
    "PersonaMem-v2": ("history_length_bin", "tool_category", "task_type"),
    "PersonaLens": ("task_domain", "history_length_bucket", "task_type"),
    "ETAPP_150x32": ("primary_tool_category", "personalization_strength_bin", "difficulty"),
    "LoCoMo_10conv_1523QA_20speakers": ("locomo_category_bin", "temporal_bin", "domain_bin"),
}


def _set_real_agent_defaults() -> None:
    defaults = {
        "UMPEEK_EVAL2_REAL_AGENT_MODE": "1",
        "UMPEEK_REAL_AGENT_MODEL": "Qwen/Qwen3-14B",
        "UMPEEK_REAL_AGENT_VLLM_BASE_URL": "http://127.0.0.1:8010/v1",
        "UMPEEK_REAL_AGENT_REQUIRE_LIVE_ENDPOINT": "1",
        "UMPEEK_REAL_AGENT_ENABLE_THINKING": "0",
        "UMPEEK_REAL_AGENT_STRICT_MODEL_CHECK": "1",
        "UMPEEK_EVAL2_GENERATE_MISSING_VISIBLE": "0",
        "UMPEEK_EVAL2_VISIBLE_RESPONSE_GENERATOR": "real_agent_qwen3_vllm",
        "UMPEEK_EVAL2_DISABLE_GENERATION_CACHE": "1",
        "UMPEEK_EVAL2_LATENT_GOLD_MODE": "profile",
        "UMPEEK_EVAL2_ETAPP_PLANNER_MODE": "real_agent_disabled_only",
        "UMPEEK_EVAL2_PAPER_FACING_ONLY_OUTPUTS": "1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def _stable_hash(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _slug(value: str) -> str:
    return (
        str(value)
        .replace("+", "plus")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .lower()
    )


def _job_key(backend: str, benchmark: str) -> tuple[str, str]:
    return (str(backend), str(benchmark))


def _load_umpeek_jobs(manifest: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    jobs: dict[tuple[str, str], dict[str, Any]] = {}
    for job in manifest.get("setting_jobs", []):
        if str(job.get("method")) != UMPEEK_METHOD:
            continue
        jobs[_job_key(str(job.get("backend")), str(job.get("benchmark")))] = dict(job)
    return jobs


def _load_rows_for_job(project_root: Path, job: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    _require_current_strong_split(job, project_root=project_root)
    paths = [str(path) for path in dict(job.get("input_split", {})).get("source_paths", [])]
    primary = _select_primary_jsonl(paths)
    if primary is None:
        raise FileNotFoundError(f"No eval_rows.jsonl source found for job {job.get('job_id')}")
    path = project_root / primary
    return read_jsonl(path), path.relative_to(project_root).as_posix()


def _stratify_key(row: Mapping[str, Any], benchmark: str) -> tuple[str, ...]:
    fields = STRATIFY_FIELDS.get(benchmark, ())
    values: list[str] = []
    for field in fields:
        value = row.get(field)
        if value in (None, ""):
            task_input = row.get("task_input") if isinstance(row.get("task_input"), Mapping) else {}
            value = task_input.get(field)
        values.append(str(value if value not in (None, "") else "unknown"))
    return tuple(values) or ("all",)


def select_stratified_indices(
    rows: Sequence[Mapping[str, Any]],
    *,
    benchmark: str,
    sample_count: int,
    allowed_indices: set[int] | None = None,
) -> list[int]:
    index_pool = sorted(allowed_indices) if allowed_indices is not None else list(range(len(rows)))
    if sample_count <= 0 or len(index_pool) <= sample_count:
        return list(index_pool)
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index in index_pool:
        row = rows[index]
        groups[_stratify_key(row, benchmark)].append(index)
    selected: set[int] = set()
    total = len(index_pool)
    allocations: dict[tuple[str, ...], int] = {}
    remainders: list[tuple[float, tuple[str, ...]]] = []
    for key, indices in groups.items():
        raw = len(indices) * sample_count / total
        alloc = int(raw)
        if sample_count >= len(groups) and alloc == 0:
            alloc = 1
        alloc = min(alloc, len(indices))
        allocations[key] = alloc
        remainders.append((raw - int(raw), key))
    while sum(allocations.values()) > sample_count:
        key = min(
            (key for key, alloc in allocations.items() if alloc > 0),
            key=lambda item: (allocations[item], len(groups[item])),
        )
        allocations[key] -= 1
    for _rem, key in sorted(remainders, reverse=True):
        if sum(allocations.values()) >= sample_count:
            break
        if allocations[key] < len(groups[key]):
            allocations[key] += 1
    for key, alloc in allocations.items():
        ordered = sorted(
            groups[key],
            key=lambda idx: _stable_hash(
                {
                    "scope": "exp4_mechanism_subset_v1",
                    "benchmark": benchmark,
                    "index": idx,
                    "row_id": _row_identifier(rows[idx], idx),
                }
            ),
        )
        selected.update(ordered[:alloc])
    if len(selected) < sample_count:
        remaining = sorted(
            (idx for idx in index_pool if idx not in selected),
            key=lambda idx: _stable_hash(
                {
                    "scope": "exp4_mechanism_subset_fill_v1",
                    "benchmark": benchmark,
                    "index": idx,
                    "row_id": _row_identifier(rows[idx], idx),
                }
            ),
        )
        selected.update(remaining[: sample_count - len(selected)])
    return sorted(selected)[:sample_count]


def _reuse_metric_records_path(
    reuse_root: Path | None,
    *,
    backend: str,
    benchmark: str,
) -> Path | None:
    if reuse_root is None:
        return None
    settings_root = reuse_root / "settings"
    if not settings_root.exists():
        return None
    for manifest_path in settings_root.glob("*/run_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(manifest.get("backend")) != backend or str(manifest.get("benchmark")) != benchmark:
            continue
        method_statuses = manifest.get("method_statuses")
        if isinstance(method_statuses, Mapping) and "UMPeek_final" not in method_statuses:
            continue
        records_path = dict(manifest.get("paths", {})).get("metric_records")
        if records_path:
            path = Path(str(records_path))
            if not path.is_absolute():
                path = reuse_root / path
            if path.exists():
                return path
        fallback = manifest_path.parent / "metric_records.jsonl"
        if fallback.exists():
            return fallback
    return None


def _reuse_available_sample_indices(path: Path | None) -> set[int] | None:
    if path is None or not path.exists():
        return None
    indices: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                indices.add(int(row.get("sample_index")))
            except (TypeError, ValueError):
                continue
    return indices


def _record_key(job: Mapping[str, Any], variant: str, row: Mapping[str, Any], sample_index: int) -> str:
    return _stable_hash(
        {
            "task": EXP4_TASK_ID,
            "job_id": job.get("job_id"),
            "variant": variant,
            "row_id": _row_identifier(row, sample_index),
            "sample_index": sample_index,
        }
    )[:24]


def _load_checkpoint_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("record_key")) for row in read_jsonl(path) if row.get("record_key")}


def _checkpoint_row(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_id": EXP4_TASK_ID,
        "record_key": record.get("record_key"),
        "job_id": record.get("job_id"),
        "exp4_variant": record.get("exp4_variant"),
        "sample_id": record.get("sample_id"),
        "status": record.get("run_status"),
    }


def _max_queries(budget_grid: Sequence[int]) -> int:
    return max(int(item) for item in budget_grid) if budget_grid else 16


def _run_exp4_attack(
    *,
    full_sample: Any,
    victim_client: Any,
    variant: str,
    budget: Mapping[str, Any],
) -> AttackTrajectory:
    method = f"UMPeek_{variant}"
    adapter = build_exp4_umpeek_ablation_adapter(variant, project_root=PROJECT_ROOT)
    proxy = TrajectoryVictimClient(victim_client, full_sample.attack_input, method)
    try:
        prediction = adapter.run(full_sample.attack_input, proxy, budget)
    except Exception as exc:
        prediction = AttackPrediction(
            baseline="umpeek",
            sample_id=full_sample.attack_input.sample_id,
            predicted_user_model=blank_predicted_user_model(),
            status="runtime_error",
            error_type=exc.__class__.__name__,
            notes=str(exc),
            metadata={"adapter_version": "exp4_umpeek_ablation", "exp4_exception": True},
        )

    final_reconstruction, wrapper_projected = _backend_project_final_reconstruction(
        clone_json(prediction.predicted_user_model),
        sample=full_sample.attack_input,
    )
    final_parse = atoms_from_user_model(
        final_reconstruction,
        sample_id=full_sample.sample_id,
        source=f"{method}:final_adapter_prediction",
    )
    rounds = _rounds_from_proxy(
        proxy=proxy,
        prediction=prediction,
        final_parse_status=final_parse.parse_status,
        final_reconstruction=final_reconstruction,
    )
    method_status, not_applicable_reason, blocked_reason, failed_reason = _method_status(prediction)
    actual_query_count = sum(round_record.cost.num_effective_queries for round_record in rounds)
    if actual_query_count == 0 and int(prediction.metadata.get("model_calls", 0) or 0) > 0:
        actual_query_count = int(prediction.metadata.get("model_calls", 0) or 0)
    adaptive_rounds = int(prediction.metadata.get("adaptive_rounds", 0) or 0)
    curve_mode = "adaptive_prefix" if adaptive_rounds > 0 and len(rounds) > 1 else "step_final_only"
    return AttackTrajectory(
        method=method,
        method_status=method_status,
        sample_id=full_sample.sample_id,
        backend=str(full_sample.attack_input.backend),
        benchmark=str(full_sample.attack_input.benchmark),
        prediction_status=str(prediction.status),
        not_applicable_reason=not_applicable_reason,
        blocked_reason=blocked_reason,
        failed_reason=failed_reason,
        final_reconstruction=final_reconstruction,
        recovery_parse_status=final_parse.parse_status,
        rounds=rounds,
        source_adapter="umpeek.attack_baselines.adapters.UMPeekAblationAdapter",
        source_paths=("src/umpeek/attack_baselines/adapters/schema_induced_slot_probe.py",),
        curve_mode=curve_mode,
        actual_query_count=actual_query_count,
        core_logic_modified=False,
        wrapper_modified=True,
        metadata={
            "legacy_baseline": "umpeek",
            "adapter_prediction_status": prediction.status,
            "adapter_error_type": prediction.error_type,
            "adapter_notes": prediction.notes,
            "adapter_metadata": clone_json(prediction.metadata),
            "wrapper_projected_prediction": bool(wrapper_projected),
            "source_refs": list(prediction.source_refs),
            "final_parse": final_parse.to_dict(),
            "exp4_variant": variant,
        },
    )


def _cached_build_full_sample(
    *,
    row: Mapping[str, Any],
    job: Mapping[str, Any],
    sample_index: int,
    source_path: str,
) -> Any:
    key = (str(job.get("job_id")), int(sample_index))
    with _FULL_SAMPLE_CACHE_LOCK:
        cached = _FULL_SAMPLE_CACHE.get(key)
    if cached is not None:
        return cached
    full_sample = _build_full_sample(
        row=row,
        job=job,
        sample_index=sample_index,
        source_path=source_path,
    )
    with _FULL_SAMPLE_CACHE_LOCK:
        _FULL_SAMPLE_CACHE.setdefault(key, full_sample)
    return full_sample


def _accepted_fact_precision(metrics: Mapping[str, Any]) -> Any:
    umr = metrics.get("UMR-F1")
    if isinstance(umr, Mapping):
        return umr.get("umr_precision")
    return None


def _build_exp4_record(
    *,
    job: Mapping[str, Any],
    variant: str,
    variant_label: str,
    figure_step: str,
    full_sample: Any,
    record_key: str,
    run_status: str,
    status_reason: str | None,
    trajectory: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    adapter_metadata = (
        trajectory.get("metadata", {}).get("adapter_metadata", {})
        if isinstance(trajectory.get("metadata"), Mapping)
        else {}
    )
    strict = adapter_metadata.get("strict_acceptance") if isinstance(adapter_metadata, Mapping) else {}
    schema = adapter_metadata.get("schema_induction") if isinstance(adapter_metadata, Mapping) else {}
    return {
        "task_id": EXP4_TASK_ID,
        "base_task_id": A054_TASK_ID,
        "experiment_version": EXP2_EXPERIMENT_VERSION,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "full_eval_schema_version": FULL_EVAL_SCHEMA_VERSION,
        "exp4_schema_version": EXP4_SCHEMA_VERSION,
        "record_key": record_key,
        "job_id": str(job.get("job_id")),
        "backend": str(job.get("backend")),
        "benchmark": str(job.get("benchmark")),
        "method": f"UMPeek_{variant}",
        "exp4_variant": variant,
        "exp4_variant_label": variant_label,
        "exp4_figure_step": figure_step,
        "sample_id": full_sample.sample_id,
        "sample_index": full_sample.sample_index,
        "source_path": full_sample.source_path,
        "run_status": run_status,
        "status_reason": status_reason,
        "trajectory_status": trajectory.get("method_status"),
        "prediction_status": trajectory.get("prediction_status"),
        "actual_query_count": trajectory.get("actual_query_count"),
        "metrics": clone_json(metrics),
        "metric_statuses": _metric_statuses(metrics),
        "accepted_fact_precision": _accepted_fact_precision(metrics),
        "candidate_count": schema.get("candidate_count") if isinstance(schema, Mapping) else None,
        "accepted_count": strict.get("accepted_count") if isinstance(strict, Mapping) else None,
        "metadata": {
            "paper_facing_only": True,
            "not_formal_full_comparison_result": True,
            "sample_source_is_current_a200_split": True,
            "attack_input_excludes_private_state": True,
            "exp4_variant": variant,
            "exp4_figure_step": figure_step,
        },
    }


def _write_progress_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["setting", "variant", "backend", "benchmark", "record_count", "ok", "failed", "not_applicable"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _reuse_full_umpeek_setting(
    *,
    out_root: Path,
    manifest_path: Path,
    job: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    selected_indices: Sequence[int],
    source_path: str,
    budget_grid: Sequence[int],
    reuse_records_path: Path,
) -> dict[str, Any] | None:
    variant = "full_umpeek"
    adapter_config = build_exp4_umpeek_ablation_adapter(variant).ablation_config
    backend = str(job.get("backend"))
    benchmark = str(job.get("benchmark"))
    setting_name = f"{_slug(variant)}__{_slug(backend)}__{_slug(benchmark)}"
    selected_set = {int(index) for index in selected_indices}
    source_records: dict[int, dict[str, Any]] = {}
    for record in read_jsonl(reuse_records_path):
        try:
            sample_index = int(record.get("sample_index"))
        except (TypeError, ValueError):
            continue
        if sample_index in selected_set:
            source_records[sample_index] = record
    if set(source_records) != selected_set:
        missing = sorted(selected_set - set(source_records))[:8]
        print(
            json.dumps(
                {
                    "reuse_full_umpeek": "incomplete",
                    "backend": backend,
                    "benchmark": benchmark,
                    "missing_count": len(selected_set - set(source_records)),
                    "missing_head": missing,
                    "source": str(reuse_records_path),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return None

    out_dir = out_root / "settings" / setting_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metric_records_path = out_dir / "metric_records.jsonl"
    checkpoint_path = ckpt_dir / "checkpoint.jsonl"
    if metric_records_path.exists():
        existing = read_jsonl(metric_records_path)
        if len(existing) >= len(selected_indices):
            selected_existing = existing[: len(selected_indices)]
            counts = Counter(str(row.get("run_status")) for row in selected_existing)
            return {
                "setting": setting_name,
                "variant": variant,
                "backend": backend,
                "benchmark": benchmark,
                "record_count": len(selected_existing),
                "ok": counts.get("ok", 0),
                "failed": counts.get("failed", 0),
                "not_applicable": counts.get("not_applicable", 0),
                "path": str(out_dir),
                "reused_from_a200_full_umpeek": True,
                "reuse_cache_hit": True,
            }
    metric_records_path.write_text("", encoding="utf-8")
    checkpoint_path.write_text("", encoding="utf-8")

    exp4_records: list[dict[str, Any]] = []
    for sample_index in selected_indices:
        row = rows[sample_index]
        source_record = clone_json(source_records[int(sample_index)])
        record_key = _record_key(job, variant, row, sample_index)
        metrics = source_record.get("metrics") if isinstance(source_record.get("metrics"), Mapping) else {}
        source_metadata = (
            dict(source_record.get("metadata", {}))
            if isinstance(source_record.get("metadata"), Mapping)
            else {}
        )
        record = {
            **source_record,
            "task_id": EXP4_TASK_ID,
            "base_task_id": A054_TASK_ID,
            "exp4_schema_version": EXP4_SCHEMA_VERSION,
            "record_key": record_key,
            "job_id": str(job.get("job_id")),
            "backend": backend,
            "benchmark": benchmark,
            "method": f"UMPeek_{variant}",
            "exp4_variant": variant,
            "exp4_variant_label": adapter_config.label,
            "exp4_figure_step": adapter_config.figure_step,
            "source_path": source_path,
            "metrics": clone_json(metrics),
            "metric_statuses": _metric_statuses(metrics),
            "accepted_fact_precision": _accepted_fact_precision(metrics),
            "candidate_count": None,
            "accepted_count": None,
            "metadata": {
                **source_metadata,
                "paper_facing_only": True,
                "not_formal_full_comparison_result": True,
                "sample_source_is_current_a200_split": True,
                "attack_input_excludes_private_state": True,
                "exp4_variant": variant,
                "exp4_figure_step": adapter_config.figure_step,
                "reused_from_a200_full_umpeek": True,
                "reuse_metric_records_path": str(reuse_records_path),
            },
        }
        append_jsonl(metric_records_path, record)
        append_jsonl(checkpoint_path, _checkpoint_row(record))
        exp4_records.append(record)

    summary = _method_summary_row(f"UMPeek_{variant}", backend, benchmark, exp4_records, budget_grid)
    _write_method_summary_csv(out_dir / "method_summary.csv", [summary])
    failure_audit = _build_full_failure_audit([job], exp4_records)
    failure_audit["task_id"] = EXP4_TASK_ID
    failure_audit["exp4_variant"] = variant
    failure_audit["reused_from_a200_full_umpeek"] = True
    write_json(out_dir / "failure_audit.json", failure_audit)
    run_manifest = {
        "task_id": EXP4_TASK_ID,
        "exp4_schema_version": EXP4_SCHEMA_VERSION,
        "base_task_id": A054_TASK_ID,
        "experiment_version": EXP2_EXPERIMENT_VERSION,
        "full_eval_schema_version": FULL_EVAL_SCHEMA_VERSION,
        "backend": backend,
        "benchmark": benchmark,
        "method": f"UMPeek_{variant}",
        "exp4_variant": variant,
        "exp4_variant_label": adapter_config.label,
        "exp4_figure_step": adapter_config.figure_step,
        "record_count": len(exp4_records),
        "selected_sample_count": len(selected_indices),
        "budget_grid": list(budget_grid),
        "manifest_path": str(manifest_path),
        "reused_from_a200_full_umpeek": True,
        "reuse_metric_records_path": str(reuse_records_path),
        "paths": {
            "metric_records": str(metric_records_path),
            "method_summary": str(out_dir / "method_summary.csv"),
            "failure_audit": str(out_dir / "failure_audit.json"),
            "checkpoint": str(checkpoint_path),
        },
        "paper_facing_only_outputs": True,
    }
    write_json(out_dir / "run_manifest.json", run_manifest)
    counts = Counter(str(row.get("run_status")) for row in exp4_records)
    return {
        "setting": setting_name,
        "variant": variant,
        "backend": backend,
        "benchmark": benchmark,
        "record_count": len(exp4_records),
        "ok": counts.get("ok", 0),
        "failed": counts.get("failed", 0),
        "not_applicable": counts.get("not_applicable", 0),
        "path": str(out_dir),
        "reused_from_a200_full_umpeek": True,
    }


def run_setting(
    *,
    project_root: Path,
    out_root: Path,
    manifest_path: Path,
    job: Mapping[str, Any],
    variant: str,
    rows: Sequence[Mapping[str, Any]],
    selected_indices: Sequence[int],
    source_path: str,
    budget_grid: Sequence[int],
    resume: bool,
    reuse_full_umpeek_root: Path | None = None,
) -> dict[str, Any]:
    adapter_config = build_exp4_umpeek_ablation_adapter(variant).ablation_config
    backend = str(job.get("backend"))
    benchmark = str(job.get("benchmark"))
    if variant == "full_umpeek":
        reuse_records_path = _reuse_metric_records_path(
            reuse_full_umpeek_root,
            backend=backend,
            benchmark=benchmark,
        )
        if reuse_records_path is not None:
            reused = _reuse_full_umpeek_setting(
                out_root=out_root,
                manifest_path=manifest_path,
                job=job,
                rows=rows,
                selected_indices=selected_indices,
                source_path=source_path,
                budget_grid=budget_grid,
                reuse_records_path=reuse_records_path,
            )
            if reused is not None:
                return reused
    setting_name = f"{_slug(variant)}__{_slug(backend)}__{_slug(benchmark)}"
    out_dir = out_root / "settings" / setting_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metric_records_path = out_dir / "metric_records.jsonl"
    checkpoint_path = ckpt_dir / "checkpoint.jsonl"
    completed = _load_checkpoint_keys(checkpoint_path) if resume else set()
    all_records = read_jsonl(metric_records_path) if metric_records_path.exists() and resume else []
    max_queries = _max_queries(budget_grid)

    for sample_index in selected_indices:
        row = rows[sample_index]
        record_key = _record_key(job, variant, row, sample_index)
        if record_key in completed:
            continue
        sample_id = "unknown"
        try:
            full_sample = _cached_build_full_sample(
                row=row,
                job=job,
                sample_index=sample_index,
                source_path=source_path,
            )
            sample_id = full_sample.sample_id
            victim_client = _build_full_eval_victim_client(
                full_sample,
                method=f"UMPeek_{variant}",
                max_queries=max_queries,
                project_root=project_root,
                etapp_planner=_full_eval_etapp_planner_from_env(),
            )
            trajectory = _run_exp4_attack(
                full_sample=full_sample,
                victim_client=victim_client,
                variant=variant,
                budget={"max_queries": max_queries, "max_seconds": 120.0, "smoke_run": False},
            )
            validate_attack_trajectory(trajectory)
            trajectory_dict = trajectory.to_dict()
            if trajectory.method_status != "ready":
                if trajectory.method_status == "not_applicable":
                    metrics = _evaluate_non_ready_as_empty_reconstruction(
                        full_sample,
                        trajectory_dict,
                        budget_grid=budget_grid,
                    )
                    run_status = "ok"
                    status_reason = f"scored_empty_reconstruction:{_trajectory_reason(trajectory)}"
                else:
                    run_status = "failed"
                    status_reason = _trajectory_reason(trajectory)
                    metrics = _status_metrics(run_status, status_reason, trajectory_dict, budget_grid)
            else:
                metrics = evaluate_smoke_metrics(full_sample, trajectory_dict, budget_grid=budget_grid)
                run_status = "ok"
                status_reason = None
            record = _build_exp4_record(
                job=job,
                variant=variant,
                variant_label=adapter_config.label,
                figure_step=adapter_config.figure_step,
                full_sample=full_sample,
                record_key=record_key,
                run_status=run_status,
                status_reason=status_reason,
                trajectory=trajectory_dict,
                metrics=metrics,
            )
        except Exception as exc:
            record = {
                "task_id": EXP4_TASK_ID,
                "base_task_id": A054_TASK_ID,
                "experiment_version": EXP2_EXPERIMENT_VERSION,
                "metric_schema_version": METRIC_SCHEMA_VERSION,
                "full_eval_schema_version": FULL_EVAL_SCHEMA_VERSION,
                "exp4_schema_version": EXP4_SCHEMA_VERSION,
                "record_key": record_key,
                "job_id": str(job.get("job_id")),
                "backend": backend,
                "benchmark": benchmark,
                "method": f"UMPeek_{variant}",
                "exp4_variant": variant,
                "exp4_variant_label": adapter_config.label,
                "exp4_figure_step": adapter_config.figure_step,
                "sample_id": sample_id,
                "sample_index": sample_index,
                "source_path": source_path,
                "run_status": "failed",
                "status_reason": str(exc),
                "traceback": traceback.format_exc(),
                "actual_query_count": 0,
                "metrics": _status_metrics("failed", str(exc), {}, budget_grid),
                "metric_statuses": {metric_name: "failed" for metric_name in REQUIRED_METRIC_KEYS},
                "metadata": {"exp4_exception": True, "paper_facing_only": True},
            }
        append_jsonl(metric_records_path, record)
        append_jsonl(checkpoint_path, _checkpoint_row(record))
        all_records.append(record)
        completed.add(record_key)

    summary = _method_summary_row(f"UMPeek_{variant}", backend, benchmark, all_records, budget_grid)
    _write_method_summary_csv(out_dir / "method_summary.csv", [summary])
    failure_audit = _build_full_failure_audit([job], all_records)
    failure_audit["task_id"] = EXP4_TASK_ID
    failure_audit["exp4_variant"] = variant
    write_json(out_dir / "failure_audit.json", failure_audit)
    run_manifest = {
        "task_id": EXP4_TASK_ID,
        "exp4_schema_version": EXP4_SCHEMA_VERSION,
        "base_task_id": A054_TASK_ID,
        "experiment_version": EXP2_EXPERIMENT_VERSION,
        "full_eval_schema_version": FULL_EVAL_SCHEMA_VERSION,
        "backend": backend,
        "benchmark": benchmark,
        "method": f"UMPeek_{variant}",
        "exp4_variant": variant,
        "exp4_variant_label": adapter_config.label,
        "exp4_figure_step": adapter_config.figure_step,
        "record_count": len(all_records),
        "selected_sample_count": len(selected_indices),
        "budget_grid": list(budget_grid),
        "manifest_path": str(manifest_path),
        "paths": {
            "metric_records": str(metric_records_path),
            "method_summary": str(out_dir / "method_summary.csv"),
            "failure_audit": str(out_dir / "failure_audit.json"),
            "checkpoint": str(checkpoint_path),
        },
        "paper_facing_only_outputs": True,
    }
    write_json(out_dir / "run_manifest.json", run_manifest)
    counts = Counter(str(row.get("run_status")) for row in all_records)
    return {
        "setting": setting_name,
        "variant": variant,
        "backend": backend,
        "benchmark": benchmark,
        "record_count": len(all_records),
        "ok": counts.get("ok", 0),
        "failed": counts.get("failed", 0),
        "not_applicable": counts.get("not_applicable", 0),
        "path": str(out_dir),
    }


def _parse_list(value: str | None, default: Sequence[str]) -> list[str]:
    if not value:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment 4 UMPeek evidence ablations.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--samples-per-benchmark", type=int, default=256)
    parser.add_argument("--variants", type=str, default="")
    parser.add_argument("--backends", type=str, default="")
    parser.add_argument("--benchmarks", type=str, default="")
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("UMPEEK_EXP4_MAX_WORKERS", "4")))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--smoke-samples", type=int, default=0)
    parser.add_argument("--reuse-full-umpeek-run-root", type=Path, default=DEFAULT_REUSE_FULL_UMPEEK_ROOT)
    parser.add_argument("--no-reuse-full-umpeek", action="store_true")
    args = parser.parse_args()

    _set_real_agent_defaults()
    project_root = args.project_root.resolve()
    manifest_path = args.manifest_path if args.manifest_path.is_absolute() else project_root / args.manifest_path
    out_dir = args.out_dir if args.out_dir.is_absolute() else project_root / args.out_dir
    manifest = read_json(manifest_path)
    reuse_full_root = None if args.no_reuse_full_umpeek else args.reuse_full_umpeek_run_root
    if reuse_full_root is not None and not reuse_full_root.is_absolute():
        reuse_full_root = project_root / reuse_full_root
    budget_grid = [int(item) for item in manifest.get("budget_grid", DEFAULT_BUDGET_GRID)]
    variants = _parse_list(args.variants, EXP4_UMPEEK_ABLATION_VARIANTS)
    backends = _parse_list(args.backends, DEFAULT_BACKENDS)
    benchmarks = _parse_list(args.benchmarks, DEFAULT_BENCHMARKS)
    if args.smoke_samples > 0:
        samples_per_benchmark = args.smoke_samples
    else:
        samples_per_benchmark = int(args.samples_per_benchmark)

    jobs = _load_umpeek_jobs(manifest)
    row_cache: dict[str, tuple[list[dict[str, Any]], str, list[int]]] = {}
    for benchmark in benchmarks:
        seed_job = None
        seed_backend = ""
        for backend in backends:
            seed_job = jobs.get(_job_key(backend, benchmark))
            if seed_job is not None:
                seed_backend = backend
                break
        if seed_job is None:
            raise ValueError(f"No UMPeek job found for benchmark={benchmark!r}.")
        rows, source_path = _load_rows_for_job(project_root, seed_job)
        allowed_indices = None
        if reuse_full_root is not None:
            reuse_path = _reuse_metric_records_path(
                reuse_full_root,
                backend=seed_backend,
                benchmark=benchmark,
            )
            allowed_indices = _reuse_available_sample_indices(reuse_path)
        selected = select_stratified_indices(
            rows,
            benchmark=benchmark,
            sample_count=samples_per_benchmark,
            allowed_indices=allowed_indices,
        )
        row_cache[benchmark] = (rows, source_path, selected)

    planned_tasks = []
    for variant in variants:
        for backend in backends:
            for benchmark in benchmarks:
                job = jobs.get(_job_key(backend, benchmark))
                if job is None:
                    raise ValueError(f"No UMPeek job found for backend={backend!r}, benchmark={benchmark!r}.")
                rows, source_path, selected = row_cache[benchmark]
                planned_tasks.append((variant, backend, benchmark, job, rows, source_path, selected))
    planned_records = sum(len(task[-1]) for task in planned_tasks)

    out_dir.mkdir(parents=True, exist_ok=True)
    launch_manifest = {
        "task_id": EXP4_TASK_ID,
        "exp4_schema_version": EXP4_SCHEMA_VERSION,
        "manifest_path": str(manifest_path),
        "out_dir": str(out_dir),
        "variants": variants,
        "backends": backends,
        "benchmarks": benchmarks,
        "samples_per_benchmark": samples_per_benchmark,
        "setting_tasks": len(planned_tasks),
        "planned_metric_records": planned_records,
        "budget_grid": budget_grid,
        "reuse_full_umpeek_run_root": "" if reuse_full_root is None else str(reuse_full_root),
        "row_selection": {
            benchmark: {
                "source_path": source_path,
                "available_rows": len(rows),
                "selected_rows": len(selected),
                "selected_indices_head": selected[:10],
                "selected_indices_max": max(selected) if selected else None,
            }
            for benchmark, (rows, source_path, selected) in row_cache.items()
        },
    }
    write_json(out_dir / "launch_manifest.json", launch_manifest)
    if args.dry_run:
        print(json.dumps({**launch_manifest, "dry_run": True}, ensure_ascii=False, indent=2))
        return

    progress_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as pool:
        futures = [
            pool.submit(
                run_setting,
                project_root=project_root,
                out_root=out_dir,
                manifest_path=manifest_path,
                job=job,
                variant=variant,
                rows=rows,
                selected_indices=selected,
                source_path=source_path,
                budget_grid=budget_grid,
                resume=bool(args.resume),
                reuse_full_umpeek_root=reuse_full_root,
            )
            for variant, _backend, _benchmark, job, rows, source_path, selected in planned_tasks
        ]
        for future in as_completed(futures):
            row = future.result()
            progress_rows.append(row)
            append_jsonl(out_dir / "launcher_progress.jsonl", row)
            _write_progress_csv(out_dir / "launcher_progress.csv", progress_rows)
            done = len(progress_rows)
            print(
                json.dumps(
                    {
                        "done_settings": done,
                        "total_settings": len(planned_tasks),
                        "last": row,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    status = {
        "task_id": EXP4_TASK_ID,
        "setting_tasks": len(planned_tasks),
        "completed_settings": len(progress_rows),
        "planned_metric_records": planned_records,
        "observed_metric_records": sum(int(row.get("record_count", 0) or 0) for row in progress_rows),
        "status_counts": {
            "ok": sum(int(row.get("ok", 0) or 0) for row in progress_rows),
            "failed": sum(int(row.get("failed", 0) or 0) for row in progress_rows),
            "not_applicable": sum(int(row.get("not_applicable", 0) or 0) for row in progress_rows),
        },
    }
    write_json(out_dir / "launcher_status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
