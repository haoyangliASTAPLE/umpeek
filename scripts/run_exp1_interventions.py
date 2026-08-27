#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from umpeek.eval2.matching import atoms_from_user_model
from umpeek.eval2.runner import build_smoke_sample
from umpeek.exp12_true_interventions import (
    BACKEND_ORDER,
    BENCHMARK_ORDER,
    SCHEMA_VERSION,
    EndpointPool,
    QwenTokenCounter,
    SampleSpec,
    SourceEntry,
    SourceRepository,
    append_jsonl,
    atom_memory_items,
    atom_recovery_scores,
    benchmark_display,
    canonical_backend,
    choose_swap_donors,
    empty_state_context,
    exp1_runtime_state_context,
    exp1_runtime_state_items,
    full_memory_entries,
    generate_history_summary,
    history_text,
    load_completed_keys,
    match_memory_items_to_context_budget,
    metric_atoms,
    prepare_exp1_task_surface,
    public_retrieval_query,
    read_jsonl,
    record_key,
    relevant_history,
    runtime_context_from_items,
    score_task_behavior,
    scoped_state_context,
    source_memory_items,
    stable_hash,
    task_decision_signature,
    victim_input_token_count,
)
from umpeek.real_agent.backends import MemoryItem


DEFAULT_MANIFEST = PROJECT_ROOT / "runs/exp2_full_comparison/A200_real_agent_qwen3_full_current/manifest/full_matrix_manifest.json"
DEFAULT_METRIC_ROOT = PROJECT_ROOT / "runs/exp2_full_comparison/A200_frozen_qwen3_paper_subset_current"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "runs/exp1_exp2_current_completion/A204_exp1_causal_repair"
CONDITION_NAMES = ("S", "no_memory", "delete_S", "swap_S", "M_u", "H_rel", "H_sum")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Experiment 1 real-Qwen state interventions.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--metric-root", type=Path, default=DEFAULT_METRIC_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--per-setting-limit", type=int, default=512)
    parser.add_argument("--base-urls", default="http://127.0.0.1:8010/v1,http://127.0.0.1:8011/v1,http://127.0.0.1:8012/v1")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--max-inflight-per-endpoint", type=int, default=8)
    parser.add_argument("--benchmarks", default=",".join(BENCHMARK_ORDER))
    parser.add_argument("--backends", default=",".join(BACKEND_ORDER))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-finalize", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.set_defaults(skip_atom_interventions=True)
    return parser.parse_args()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _job_slug(job: Mapping[str, Any]) -> str:
    job_id = str(job.get("job_id") or "")
    return job_id.removeprefix("exp2__")


def _primary_eval_path(job: Mapping[str, Any]) -> Path:
    paths = [str(path) for path in (job.get("input_split") or {}).get("source_paths", [])]
    preferred = next((path for path in paths if path.endswith("/eval_rows.jsonl")), None)
    if preferred is None:
        raise FileNotFoundError(f"No eval_rows.jsonl in job {job.get('job_id')}")
    return PROJECT_ROOT / preferred


def _selected_jobs(manifest: Mapping[str, Any], benchmarks: set[str], backends: set[str]) -> list[dict[str, Any]]:
    jobs = []
    for job in manifest.get("setting_jobs", []):
        if not isinstance(job, Mapping) or str(job.get("method")) != "UMPeek_final":
            continue
        benchmark = benchmark_display(job.get("benchmark"))
        backend = canonical_backend(job.get("backend"))
        if benchmark in benchmarks and backend in backends:
            jobs.append(dict(job))
    order = {(backend, benchmark): index for index, (benchmark, backend) in enumerate((b, k) for b in BENCHMARK_ORDER for k in BACKEND_ORDER)}
    return sorted(jobs, key=lambda job: order.get((canonical_backend(job["backend"]), benchmark_display(job["benchmark"])), 999))


def _audit_value_present(text: str, value: Any) -> bool:
    normalized_text = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()
    normalized_value = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
    if not normalized_value:
        return False
    if normalized_value in normalized_text:
        return True
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        numeric_value = None
    if numeric_value is not None:
        for lower, upper in re.findall(r"(\d+(?:\.\d+)?)\s*[% ]*(?:to|through|[-~])\s*(\d+(?:\.\d+)?)", str(text).lower()):
            if float(lower) <= numeric_value <= float(upper):
                return True
    value_tokens = set(normalized_value.split())
    text_tokens = set(normalized_text.split())
    return bool(value_tokens) and len(value_tokens & text_tokens) / len(value_tokens) >= 0.8


def _state_relevance_audit(
    setting_specs: Sequence[tuple[dict[str, Any], list[SampleSpec], list[SampleSpec], dict[str, SampleSpec], str]],
    source_repo: SourceRepository,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for _job, specs, _pool, _donors, _path in setting_specs:
        for sample in specs:
            context = exp1_runtime_state_context(sample, source_repo)
            text = context.prompt_context
            expected: list[Any] = []
            if sample.benchmark == "PersonaMem-v2":
                metadata = sample.source_row.get("metadata") if isinstance(sample.source_row.get("metadata"), Mapping) else {}
                if str(metadata.get("pref_type") or "").strip().lower() == "ask_to_forget":
                    expected = [metadata.get("prev_pref") or metadata.get("preference")]
                else:
                    expected = [metadata.get("preference")] if metadata.get("preference") else []
            elif sample.benchmark == "PersonaLens":
                gold = sample.source_row.get("gold") if isinstance(sample.source_row.get("gold"), Mapping) else {}
                expected = [
                    value
                    for affinity in gold.get("expected_affinities", []) or []
                    if isinstance(affinity, Mapping)
                    for value in affinity.get("values", []) or []
                ]
            elif sample.benchmark == "ETAPP":
                actions = sample.source_row.get("action_sequence") or []
                expected = [
                    value
                    for action in actions
                    if isinstance(action, Mapping)
                    for key, value in (action.get("normalized_args") or {}).items()
                    if str(key).lower() not in {"time", "date", "timestamp"}
                    if value not in (None, "")
                ]
            if sample.benchmark == "LoCoMo":
                expected_refs = {str(value) for value in sample.source_row.get("evidence", []) if str(value).strip()}
                retrieved_refs = {
                    str(item.metadata.get("source_ref") or "").rsplit("locomo:", 1)[-1].split(":", 1)[-1]
                    for item in context.retrieved_items
                    if item.metadata.get("source_ref")
                }
                matched = len(expected_refs & retrieved_refs)
                expected = sorted(expected_refs)
                coverage = matched / len(expected_refs) if expected_refs else float(bool(context.retrieved_items))
            else:
                matched = sum(int(_audit_value_present(text, value)) for value in expected)
                coverage = matched / len(expected) if expected else float(bool(context.retrieved_items))
            rows.append(
                {
                    "benchmark": sample.benchmark,
                    "backend": sample.backend,
                    "sample_id_hash": stable_hash(sample.sample_id)[:16],
                    "expected_values": len(expected),
                    "matched_values": matched,
                    "coverage": coverage,
                    "retrieved_items": len(context.retrieved_items),
                }
            )
    setting_rows: list[dict[str, Any]] = []
    for benchmark in BENCHMARK_ORDER:
        for backend in BACKEND_ORDER:
            selected = [row for row in rows if row["benchmark"] == benchmark and row["backend"] == backend]
            if not selected:
                continue
            setting_rows.append(
                {
                    "benchmark": benchmark,
                    "backend": backend,
                    "samples": len(selected),
                    "mean_expected_value_coverage": sum(row["coverage"] for row in selected) / len(selected),
                    "empty_state_count": sum(int(row["retrieved_items"] == 0) for row in selected),
                }
            )
    minimum = {"PersonaMem-v2": 0.80, "PersonaLens": 0.85, "ETAPP": 0.60, "LoCoMo": 0.70}
    failures = [
        row
        for row in setting_rows
        if row["mean_expected_value_coverage"] < minimum[row["benchmark"]] or row["empty_state_count"] > 0
    ]
    return {
        "status": "pass" if not failures else "fail",
        "definition": "evaluator-only pre-run check; expected values are never used to build runtime state",
        "minimum_mean_coverage": minimum,
        "settings": setting_rows,
        "failures": failures,
    }


def _load_setting_specs(
    job: Mapping[str, Any],
    *,
    metric_root: Path,
    limit: int,
    include_memory_atoms: bool = False,
) -> tuple[list[SampleSpec], str]:
    source_path = _primary_eval_path(job)
    rows = read_jsonl(source_path, limit=limit)
    setting_slug = _job_slug(job)
    metric_path = metric_root / "settings" / setting_slug / "metric_records.jsonl"
    metric_rows = read_jsonl(metric_path, limit=limit)
    if len(rows) != len(metric_rows):
        raise RuntimeError(f"row/metric count mismatch for {setting_slug}: {len(rows)} != {len(metric_rows)}")
    specs: list[SampleSpec] = []
    backend = canonical_backend(job.get("backend"))
    benchmark = benchmark_display(job.get("benchmark"))
    for index, (row, metric_record) in enumerate(zip(rows, metric_rows)):
        source_rel = source_path.relative_to(PROJECT_ROOT).as_posix()
        sample = build_smoke_sample(row=row, job=job, sample_index=index, source_path=source_rel)
        prepared_attack_input, prepared_row = prepare_exp1_task_surface(benchmark, sample.attack_input, row)
        gold_atoms, fixed_atoms, recovered_atoms = metric_atoms(metric_record)
        parsed_memory = (
            atoms_from_user_model(
                sample.gold_user_model,
                sample_id=str(metric_record.get("sample_id") or sample.sample_id),
                source="backend_memory",
            )
            if include_memory_atoms
            else None
        )
        metrics = metric_record.get("metrics") if isinstance(metric_record.get("metrics"), Mapping) else {}
        crs = metrics.get("CRS") if isinstance(metrics.get("CRS"), Mapping) else {}
        base_behavior = crs.get("target_behavior")
        if base_behavior in (None, "", [], {}):
            base_behavior = sample.original_behavior
        task_type = str(crs.get("task_type") or sample.replay_context.task_type or row.get("task_type") or "open")
        base_score, score_audit = score_task_behavior(benchmark, base_behavior, prepared_row, task_type)
        if score_audit.get("status") not in {"ok", "fallback"}:
            raise RuntimeError(
                f"base TaskScore is not scorable for {setting_slug} sample={index}: {score_audit}"
            )
        specs.append(
            SampleSpec(
                backend=backend,
                benchmark=benchmark,
                setting_key=setting_slug,
                sample_index=index,
                sample_id=str(metric_record.get("sample_id") or sample.sample_id),
                user_id=str(sample.attack_input.user_id or row.get("user_id") or index),
                task_id=str(sample.attack_input.task_id or row.get("task_id") or index),
                task_type=task_type,
                attack_input=prepared_attack_input,
                source_row=prepared_row,
                metric_record=dict(metric_record),
                gold_atoms=gold_atoms,
                fixed_atoms=fixed_atoms,
                recovered_atoms=recovered_atoms,
                base_behavior=base_behavior,
                base_score=float(base_score),
                memory_atoms=(tuple(atom.to_dict() for atom in parsed_memory.atoms) if parsed_memory is not None else ()),
            )
        )
    return specs, source_path.relative_to(PROJECT_ROOT).as_posix()


def _stratified_specs(samples: Sequence[SampleSpec], limit: int) -> list[SampleSpec]:
    if limit >= len(samples):
        return list(samples)
    groups: dict[tuple[str, str, str], list[SampleSpec]] = {}
    for sample in samples:
        groups.setdefault(sample.task_group, []).append(sample)
    for group in groups.values():
        group.sort(
            key=lambda sample: stable_hash(
                [
                    sample.source_row.get("sample_key") or sample.source_row.get("task_id") or sample.task_id,
                    sample.user_id,
                ]
            )
        )
    selected: list[SampleSpec] = []
    depth = 0
    ordered_groups = sorted(groups, key=lambda group: stable_hash(["exp1_stratum_v2", group]))
    while len(selected) < limit:
        added = False
        for key in ordered_groups:
            group = groups[key]
            if depth < len(group):
                selected.append(group[depth])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        depth += 1
    return selected


def _condition_record_base(sample: SampleSpec, condition: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_key": record_key("condition", sample.setting_key, sample.sample_id, condition),
        "setting_key": sample.setting_key,
        "backend": sample.backend,
        "benchmark": sample.benchmark,
        "sample_index": sample.sample_index,
        "sample_id": sample.sample_id,
        "user_id_hash": stable_hash(sample.user_id)[:16],
        "task_id": sample.task_id,
        "task_type": sample.task_type,
        "condition": condition,
    }


def _ok_condition_record(
    sample: SampleSpec,
    condition: str,
    *,
    behavior: Any,
    score: float,
    score_audit: Mapping[str, Any],
    context_text: str,
    state_tokens: int,
    llm_usage: Mapping[str, Any] | None,
    retry_count: int,
    endpoint_index: int | None,
    call_executed: bool,
    include_memory_section: bool,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    decision_signature = task_decision_signature(sample.benchmark, behavior, score_audit)
    record = {
        **_condition_record_base(sample, condition),
        "status": "ok",
        "call_executed": bool(call_executed),
        "include_memory_section": bool(include_memory_section),
        "state_token_count": int(state_tokens),
        "task_score": round(float(score), 6),
        "task_score_audit": dict(score_audit),
        "behavior_hash": stable_hash(behavior)[:24],
        "decision_signature_hash": stable_hash(decision_signature)[:24],
        "private_decision_signature": decision_signature,
        "context_hash": stable_hash(context_text)[:24],
        "llm_usage": dict(llm_usage or {}),
        "retry_count": int(retry_count),
        "endpoint_index": endpoint_index,
        "metadata": dict(metadata or {}),
        "private_behavior": behavior,
    }
    record["integrity_hash"] = stable_hash(record)
    return record


def _error_record(sample: SampleSpec, condition: str, exc: Exception) -> dict[str, Any]:
    record = {
        **_condition_record_base(sample, condition),
        "status": "error",
        "error_type": exc.__class__.__name__,
        "error": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-8:],
    }
    record["integrity_hash"] = stable_hash(record)
    return record


def _run_condition(
    sample: SampleSpec,
    condition: str,
    *,
    pool: EndpointPool,
    token_counter: QwenTokenCounter,
    source_repo: SourceRepository,
    donors: Mapping[str, SampleSpec],
) -> dict[str, Any]:
    try:
        history = source_repo.history(sample.benchmark, sample.source_row)
        base_context = exp1_runtime_state_context(sample, source_repo)
        base_tokens = token_counter.count(base_context.prompt_context)
        metadata: dict[str, Any] = {}
        summary_usage: Mapping[str, Any] | None = None
        summary_text = ""
        if condition == "S":
            context = base_context
            include_memory = True
            metadata.update(
                {
                    "state_builder": "task_conditioned_runtime_state_v2",
                    "retrieval_query_hash": stable_hash(public_retrieval_query(sample))[:24],
                    "retrieved_item_count": len(context.retrieved_items),
                }
            )
        elif condition == "no_memory":
            context = empty_state_context(sample, mode="no_memory")
            include_memory = False
        elif condition == "delete_S":
            context = empty_state_context(sample, mode="runtime_state_deleted")
            include_memory = True
        elif condition == "swap_S":
            donor = donors.get(sample.sample_id)
            if donor is None:
                raise RuntimeError("no eligible different-user swap donor")
            donor_items = exp1_runtime_state_items(sample, source_repo, owner=donor)
            donor_context = runtime_context_from_items(
                backend=sample.backend,
                sample=sample,
                items=donor_items,
                adapter_mode="runtime_state_swap_length_probe",
            )
            matched_items = match_memory_items_to_context_budget(
                sample=sample,
                items=donor_items,
                target_tokens=base_tokens,
                token_counter=token_counter,
            )
            context = runtime_context_from_items(
                backend=sample.backend,
                sample=sample,
                items=matched_items,
                adapter_mode="runtime_state_swapped_different_user",
            )
            include_memory = True
            metadata.update(
                {
                    "donor_sample_hash": stable_hash(donor.sample_id)[:16],
                    "donor_user_hash": stable_hash(donor.user_id)[:16],
                    "donor_task_group": list(donor.task_group),
                    "recipient_task_group": list(sample.task_group),
                    "target_state_tokens": base_tokens,
                    "donor_state_tokens_before_matching": token_counter.count(donor_context.prompt_context),
                    "donor_state_tokens_after_matching": token_counter.count(context.prompt_context),
                    "donor_state_length_matched": True,
                }
            )
        elif condition == "M_u":
            selected = full_memory_entries(sample, history, token_counter)
            context = runtime_context_from_items(
                backend=sample.backend,
                sample=sample,
                items=source_memory_items(selected),
                adapter_mode="full_pre_task_backend_memory",
            )
            include_memory = True
            metadata.update(
                {
                    "source_entry_count": len(selected),
                    "history_entry_count": len(history),
                    "context_window_capped": len(selected) < len(history),
                    "victim_input_token_estimate": victim_input_token_count(sample, context, token_counter),
                }
            )
        elif condition == "H_rel":
            selected = relevant_history(sample, history, token_counter, base_tokens)
            matched_items = match_memory_items_to_context_budget(
                sample=sample,
                items=source_memory_items(selected),
                target_tokens=base_tokens,
                token_counter=token_counter,
            )
            context = runtime_context_from_items(
                backend=sample.backend,
                sample=sample,
                items=matched_items,
                adapter_mode="bm25_relevant_history_state_budget_matched",
            )
            include_memory = True
            metadata.update({"source_entry_count": len(selected), "history_entry_count": len(history), "budget_tokens": base_tokens})
        elif condition == "H_sum":
            summary_result = generate_history_summary(
                pool,
                sample=sample,
                history=history,
                token_counter=token_counter,
                output_budget=max(1, base_tokens),
            )
            summary_text = str(summary_result["summary"])
            summary_usage = summary_result.get("llm_usage") if isinstance(summary_result.get("llm_usage"), Mapping) else {}
            summary_items = match_memory_items_to_context_budget(
                sample=sample,
                items=(MemoryItem(text=summary_text, category="facts", source="qwen_history_summary"),) if summary_text else (),
                target_tokens=base_tokens,
                token_counter=token_counter,
            )
            context = runtime_context_from_items(
                backend=sample.backend,
                sample=sample,
                items=summary_items,
                adapter_mode="task_conditioned_history_summary_state_budget_matched",
            )
            include_memory = True
            metadata.update(
                {
                    "summary_text": summary_text,
                    "summary_text_hash": stable_hash(summary_text)[:24],
                    "summary_generation_usage": dict(summary_usage or {}),
                    "summary_retry_count": int(summary_result.get("retry_count", 0)),
                    "summary_input_history_tokens": int(summary_result.get("input_history_tokens", 0)),
                    "budget_tokens": base_tokens,
                }
            )
        else:
            raise ValueError(f"Unsupported live condition: {condition}")
        result = pool.run(sample=sample, context=context, include_memory_section=include_memory)
        behavior = result["behavior"]
        score, score_audit = score_task_behavior(sample.benchmark, behavior, sample.source_row, sample.task_type)
        return _ok_condition_record(
            sample,
            condition,
            behavior=behavior,
            score=score,
            score_audit=score_audit,
            context_text=context.prompt_context,
            state_tokens=token_counter.count(context.prompt_context),
            llm_usage=result.get("llm_usage") if isinstance(result.get("llm_usage"), Mapping) else {},
            retry_count=int(result.get("retry_count", 0)),
            endpoint_index=int(result.get("endpoint_index", 0)),
            call_executed=True,
            include_memory_section=include_memory,
            metadata=metadata,
        )
    except Exception as exc:
        return _error_record(sample, condition, exc)


def _atom_record_base(sample: SampleSpec, atom: Mapping[str, Any]) -> dict[str, Any]:
    atom_id = str(atom.get("atom_id") or stable_hash(atom)[:16])
    return {
        "schema_version": SCHEMA_VERSION,
        "record_key": record_key("atom", sample.setting_key, sample.sample_id, atom_id),
        "setting_key": sample.setting_key,
        "backend": sample.backend,
        "benchmark": sample.benchmark,
        "sample_index": sample.sample_index,
        "sample_id": sample.sample_id,
        "task_id": sample.task_id,
        "task_type": sample.task_type,
        "atom_id": atom_id,
        "atom_text_hash": stable_hash(str(atom.get("text") or atom.get("typed_text") or ""))[:24],
        "atom_category": str(atom.get("category") or "facts"),
        "atom_type": str(atom.get("atom_type") or "semantic"),
    }


def _run_atom_intervention(
    sample: SampleSpec,
    atom: Mapping[str, Any],
    *,
    pool: EndpointPool,
    token_counter: QwenTokenCounter,
    recovery_scores: Mapping[str, float],
) -> dict[str, Any]:
    base = _atom_record_base(sample, atom)
    atom_id = str(base["atom_id"])
    try:
        context = scoped_state_context(sample, removed_atom_id=atom_id)
        result = pool.run(sample=sample, context=context, include_memory_section=True)
        score, score_audit = score_task_behavior(sample.benchmark, result["behavior"], sample.source_row, sample.task_type)
        strength = float(recovery_scores.get(atom_id, 0.0))
        record = {
            **base,
            "status": "ok",
            "base_task_score": round(sample.base_score, 6),
            "deleted_task_score": round(float(score), 6),
            "fact_effect": round(float(sample.base_score) - float(score), 6),
            "recovery_strength": round(strength, 6),
            "recovered": int(strength >= 0.72),
            "task_score_audit": score_audit,
            "gold_atom_count_before": len(sample.gold_atoms),
            "gold_atom_count_after": len(sample.gold_atoms) - 1,
            "removed_atom_count": 1,
            "context_hash": stable_hash(context.prompt_context)[:24],
            "state_token_count": token_counter.count(context.prompt_context),
            "behavior_hash": stable_hash(result["behavior"])[:24],
            "llm_usage": dict(result.get("llm_usage") or {}),
            "retry_count": int(result.get("retry_count", 0)),
            "endpoint_index": int(result.get("endpoint_index", 0)),
        }
        record["integrity_hash"] = stable_hash(record)
        return record
    except Exception as exc:
        record = {
            **base,
            "status": "error",
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-8:],
        }
        record["integrity_hash"] = stable_hash(record)
        return record


def _run_parallel(
    tasks: Sequence[tuple[Any, ...]],
    worker: Callable[..., dict[str, Any]],
    *,
    workers: int,
    output_path: Path,
    output_lock: threading.Lock,
    completed: set[str],
    progress_callback: Callable[[int, int, int], None],
    progress_every: int,
) -> tuple[int, int]:
    pending = [task for task in tasks if str(task[0]) not in completed]
    ok = 0
    failed = 0
    if not pending:
        progress_callback(0, 0, 0)
        return ok, failed
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, *task[1:]) for task in pending]
        for done, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            append_jsonl(output_path, record, output_lock)
            if record.get("status") == "ok":
                completed.add(str(record["record_key"]))
                ok += 1
            else:
                failed += 1
            if done % progress_every == 0 or done == len(futures):
                progress_callback(done, len(futures), failed)
    return ok, failed


def main() -> int:
    args = parse_args()
    benchmarks = {item.strip() for item in args.benchmarks.split(",") if item.strip()}
    backends = {item.strip() for item in args.backends.split(",") if item.strip()}
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs = _selected_jobs(manifest, benchmarks, backends)
    if not jobs:
        raise RuntimeError("No UMPeek settings selected.")

    run_root = args.run_root.resolve()
    records_dir = run_root / "records"
    logs_dir = run_root / "logs"
    records_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    condition_path = records_dir / "condition_records.jsonl"
    atom_path = records_dir / "atom_intervention_records.jsonl"
    condition_completed = load_completed_keys(condition_path)
    atom_completed = load_completed_keys(atom_path)
    token_counter = QwenTokenCounter()
    source_repo = SourceRepository(PROJECT_ROOT)
    output_lock = threading.Lock()

    setting_specs: list[
        tuple[dict[str, Any], list[SampleSpec], list[SampleSpec], dict[str, SampleSpec], str]
    ] = []
    expected_atoms = 0
    expected_samples = 0
    for job in jobs:
        donor_pool_limit = max(args.per_setting_limit, 512)
        donor_pool, source_path = _load_setting_specs(
            job,
            metric_root=args.metric_root,
            limit=donor_pool_limit,
        )
        specs = _stratified_specs(donor_pool, args.per_setting_limit)
        state_tokens = {
            spec.sample_id: token_counter.count(exp1_runtime_state_context(spec, source_repo).prompt_context)
            for spec in donor_pool
        }
        donors = choose_swap_donors(
            specs,
            state_tokens,
            candidate_samples=donor_pool,
        )
        if len(donors) != len(specs):
            missing = [spec.sample_id for spec in specs if spec.sample_id not in donors]
            raise RuntimeError(
                f"swap donor coverage failed for {_job_slug(job)}: {len(missing)} missing "
                "after requiring a different user with the same task type and domain"
            )
        setting_specs.append((job, specs, donor_pool, donors, source_path))
        expected_samples += len(specs)
        expected_atoms += sum(len(spec.gold_atoms) for spec in specs)

    state_relevance_audit = _state_relevance_audit(setting_specs, source_repo)
    if state_relevance_audit["status"] != "pass":
        write_json(run_root / "state_relevance_audit.json", state_relevance_audit)
        raise RuntimeError(
            "runtime-state relevance audit failed before victim calls: "
            f"{state_relevance_audit['failures']}"
        )

    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "planned" if args.dry_run else "running",
        "created_at_unix": time.time(),
        "source_manifest": args.manifest.relative_to(PROJECT_ROOT).as_posix(),
        "source_metric_root": args.metric_root.relative_to(PROJECT_ROOT).as_posix(),
        "per_setting_limit": args.per_setting_limit,
        "setting_count": len(setting_specs),
        "base_sample_records": expected_samples,
        "expected_condition_records": expected_samples * len(CONDITION_NAMES),
        "expected_atom_intervention_records": 0 if args.skip_atom_interventions else expected_atoms,
        "estimated_live_calls": expected_samples * 7 + (0 if args.skip_atom_interventions else expected_atoms),
        "condition_names": list(CONDITION_NAMES),
        "base_urls": [item.strip() for item in args.base_urls.split(",") if item.strip()],
        "workers": args.workers,
        "non_thinking": True,
        "temperature": 0.0,
        "heldout_visible_to_attack": False,
        "proxy_conditions_forbidden": True,
        "all_exp1_conditions_use_live_qwen": True,
        "runtime_state_builder": "task_conditioned_runtime_state_v3",
        "personamem_choice_order": "deterministically_shuffled_per_task",
        "sample_selection": "deterministic_round_robin_over_task_type_domain_signature_v2",
        "state_relevance_audit": state_relevance_audit,
        "settings": [
            {
                "job_id": job["job_id"],
                "backend": canonical_backend(job["backend"]),
                "benchmark": benchmark_display(job["benchmark"]),
                "samples": len(specs),
                "atoms": sum(len(spec.gold_atoms) for spec in specs),
                "swap_donor_pool_samples": len(donor_pool),
                "swap_donor_coverage": len(donors),
                "source_path": source_path,
            }
            for job, specs, donor_pool, donors, source_path in setting_specs
        ],
    }
    write_json(run_root / "run_manifest.json", run_manifest)
    print(json.dumps(run_manifest, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0

    pool = EndpointPool(
        [item.strip() for item in args.base_urls.split(",") if item.strip()],
        max_inflight_per_endpoint=args.max_inflight_per_endpoint,
    )
    started = time.monotonic()
    total_ok = 0
    total_failed = 0

    for setting_index, (job, specs, _donor_pool, donors, source_path) in enumerate(setting_specs, start=1):
        setting_key = _job_slug(job)

        condition_tasks: list[tuple[Any, ...]] = []
        for sample in specs:
            for condition in CONDITION_NAMES:
                key = record_key("condition", sample.setting_key, sample.sample_id, condition)
                condition_tasks.append((key, sample, condition))

        def condition_worker(sample: SampleSpec, condition: str) -> dict[str, Any]:
            return _run_condition(
                sample,
                condition,
                pool=pool,
                token_counter=token_counter,
                source_repo=source_repo,
                donors=donors,
            )

        def progress(done: int, pending: int, failed: int) -> None:
            elapsed = max(time.monotonic() - started, 1e-6)
            completed_total = len(condition_completed) + len(atom_completed)
            rate = completed_total / elapsed
            expected_total = run_manifest["expected_condition_records"] + run_manifest["expected_atom_intervention_records"]
            eta = (expected_total - completed_total) / rate if rate > 0 else None
            payload = {
                "schema_version": SCHEMA_VERSION,
                "status": "running",
                "current_setting": setting_key,
                "setting_index": setting_index,
                "setting_count": len(setting_specs),
                "phase_done": done,
                "phase_pending_at_start": pending,
                "phase_failures": failed,
                "condition_completed": len(condition_completed),
                "atom_completed": len(atom_completed),
                "expected_total_records": expected_total,
                "records_per_second": rate,
                "eta_seconds": eta,
                "updated_at_unix": time.time(),
            }
            write_json(run_root / "progress.json", payload)
            print(json.dumps(payload, ensure_ascii=False), flush=True)

        ok, failed = _run_parallel(
            condition_tasks,
            condition_worker,
            workers=args.workers,
            output_path=condition_path,
            output_lock=output_lock,
            completed=condition_completed,
            progress_callback=progress,
            progress_every=args.progress_every,
        )
        total_ok += ok
        total_failed += failed

        if not args.skip_atom_interventions:
            atom_tasks: list[tuple[Any, ...]] = []
            score_maps = {sample.sample_id: atom_recovery_scores(sample.gold_atoms, sample.recovered_atoms) for sample in specs}
            for sample in specs:
                for atom in sample.gold_atoms:
                    atom_id = str(atom.get("atom_id") or stable_hash(atom)[:16])
                    key = record_key("atom", sample.setting_key, sample.sample_id, atom_id)
                    atom_tasks.append((key, sample, atom, score_maps[sample.sample_id]))

            def atom_worker(sample: SampleSpec, atom: Mapping[str, Any], scores: Mapping[str, float]) -> dict[str, Any]:
                return _run_atom_intervention(
                    sample,
                    atom,
                    pool=pool,
                    token_counter=token_counter,
                    recovery_scores=scores,
                )

            ok, failed = _run_parallel(
                atom_tasks,
                atom_worker,
                workers=args.workers,
                output_path=atom_path,
                output_lock=output_lock,
                completed=atom_completed,
                progress_callback=progress,
                progress_every=args.progress_every,
            )
            total_ok += ok
            total_failed += failed

    complete = (
        len(condition_completed) == run_manifest["expected_condition_records"]
        and len(atom_completed) == run_manifest["expected_atom_intervention_records"]
        and total_failed == 0
    )
    final_progress = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if complete else "incomplete",
        "condition_completed": len(condition_completed),
        "condition_expected": run_manifest["expected_condition_records"],
        "atom_completed": len(atom_completed),
        "atom_expected": run_manifest["expected_atom_intervention_records"],
        "new_failures": total_failed,
        "elapsed_seconds": time.monotonic() - started,
        "updated_at_unix": time.time(),
    }
    write_json(run_root / "progress.json", final_progress)
    run_manifest.update(final_progress)
    write_json(run_root / "run_manifest.json", run_manifest)
    if complete and not args.no_finalize:
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "export_exp1_causal_interventions.py"),
                "--run-root",
                str(run_root),
            ],
            cwd=PROJECT_ROOT,
            check=True,
            env={**os.environ, "PYTHONPATH": str(SRC_ROOT)},
        )
    print(json.dumps(final_progress, ensure_ascii=False, indent=2), flush=True)
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
