#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from umpeek.eval2.runner import (  # noqa: E402
    _heldout_group_keys,
    _row_identifier,
    precompute_independent_heldout_tasks_for_split,
)
from umpeek.eval2.split_contract import (  # noqa: E402
    BENCHMARK_SLUGS,
    CURRENT_STRONG_QUERY_SPLIT_SCHEMA_VERSION,
    DEPRECATED_STRONG_QUERY_SPLIT_SCHEMA_VERSIONS,
    FORBIDDEN_LEGACY_SPLIT_GLOBS,
    SPLIT_ROLE_VISIBILITY,
    attack_probe_public_path,
    behavior_heldout_path,
    eval_rows_path,
    split_manifest_path,
)


SPLIT_SCHEMA_VERSION = CURRENT_STRONG_QUERY_SPLIT_SCHEMA_VERSION
OUT_ROOT = PROJECT_ROOT / "data" / "interim" / "eval2_splits" / SPLIT_SCHEMA_VERSION
SPLIT_PARENT = OUT_ROOT.parent
A200_MANIFEST = (
    PROJECT_ROOT
    / "runs"
    / "exp2_full_comparison"
    / "A200_real_agent_qwen3_full_current"
    / "manifest"
    / "full_matrix_manifest.json"
)

BENCHMARK_SOURCES = {
    "PersonaMem-v2": PROJECT_ROOT / "data" / "interim" / "exp1_whitebox" / "PersonaMemv2" / "task_records.jsonl",
    "PersonaLens": PROJECT_ROOT / "data" / "interim" / "exp1_whitebox" / "PersonaLens" / "task_records.jsonl",
    "ETAPP_150x32": PROJECT_ROOT / "data" / "interim" / "benchmarks" / "ETAPP" / "expanded_v1" / "examples.jsonl",
    "LoCoMo_10conv_1523QA_20speakers": PROJECT_ROOT / "data" / "benchmarks" / "LoCoMo" / "task_rows.jsonl",
}

def rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(dict(json.loads(line)))
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def remove_path(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "status": "already_absent"}
    if path.is_dir():
        shutil.rmtree(path)
        return {"path": rel(path), "status": "removed_dir"}
    path.unlink()
    return {"path": rel(path), "status": "removed_file"}


def cleanup_legacy_split_artifacts() -> list[dict[str, Any]]:
    cleanup_rows: list[dict[str, Any]] = []
    if OUT_ROOT.exists():
        cleanup_rows.append(remove_path(OUT_ROOT))
    for version in DEPRECATED_STRONG_QUERY_SPLIT_SCHEMA_VERSIONS:
        cleanup_rows.append(remove_path(SPLIT_PARENT / version))
    for pattern in FORBIDDEN_LEGACY_SPLIT_GLOBS:
        matches = sorted(PROJECT_ROOT.glob(pattern))
        if not matches:
            cleanup_rows.append({"path": pattern, "status": "already_absent_glob"})
            continue
        for match in matches:
            cleanup_rows.append(remove_path(match))
    return cleanup_rows


def group_summary(benchmark: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    no_group = 0
    for row in rows:
        keys = _heldout_group_keys(benchmark, row)
        if not keys:
            no_group += 1
            continue
        counts[keys[0]] += 1
    if not counts:
        return {
            "group_count": 0,
            "no_group_row_count": no_group,
            "min_rows_per_group": 0,
            "max_rows_per_group": 0,
            "groups_with_at_least_2": 0,
        }
    return {
        "group_count": len(counts),
        "no_group_row_count": no_group,
        "min_rows_per_group": min(counts.values()),
        "max_rows_per_group": max(counts.values()),
        "groups_with_at_least_2": sum(1 for value in counts.values() if value >= 2),
    }


def split_source_paths(benchmark: str, source_path: Path) -> list[str]:
    return [
        rel(eval_rows_path(PROJECT_ROOT, benchmark)),
        rel(attack_probe_public_path(PROJECT_ROOT, benchmark)),
        rel(behavior_heldout_path(PROJECT_ROOT, benchmark)),
        rel(source_path),
        rel(split_manifest_path(PROJECT_ROOT, benchmark)),
    ]


def public_attack_probe_row(benchmark: str, row: Mapping[str, Any], *, row_id: str, index: int) -> dict[str, Any]:
    task_input = row.get("task_input") if isinstance(row.get("task_input"), Mapping) else {}
    if benchmark == "PersonaMem-v2":
        task_prompt = str(task_input.get("user_query") or "Personalized choice task.")
        task_domain = str(task_input.get("topic_query") or row.get("task_domain") or "PersonaMemv2")
        task_type = "choice"
        visible_tools: Any = []
    elif benchmark == "PersonaLens":
        task_prompt = str(task_input.get("prompt") or task_input.get("task_description") or "PersonaLens task.")
        task_domain = str(row.get("task_domain") or "PersonaLens")
        task_type = "open"
        visible_tools = []
    elif benchmark == "ETAPP_150x32":
        task_prompt = str(row.get("query") or "ETAPP action task.")
        task_domain = str(row.get("dialogue_domain") or "ETAPP")
        task_type = "action"
        visible_tools = list(row.get("available_tools", [])) if isinstance(row.get("available_tools"), list) else []
    elif benchmark == "LoCoMo_10conv_1523QA_20speakers":
        task_prompt = str(row.get("question") or "LoCoMo question.")
        task_domain = str(row.get("domain_bin") or row.get("question_family") or "LoCoMo")
        task_type = "open"
        visible_tools = []
    else:
        raise ValueError(f"Unsupported benchmark for public split projection: {benchmark!r}")
    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "split_id": f"{SPLIT_SCHEMA_VERSION}__{BENCHMARK_SLUGS[benchmark]}",
        "benchmark": benchmark,
        "attack_probe_id": row_id,
        "source_row_index": index,
        "source_row_id": row_id,
        "user_id": str(row.get("user_id") or row.get("source_user_id") or ""),
        "task_id": str(row.get("task_id") or row.get("example_id") or row_id),
        "task_prompt": task_prompt,
        "task_type": task_type,
        "task_domain": task_domain,
        "tool_action_category": str(row.get("primary_tool_category") or ""),
        "visible_tools": visible_tools,
        "public_context": {
            "benchmark_canonical": benchmark,
            "split_role": "attack_probe",
            "private_memory_seed_visible": False,
            "behavior_heldout_visible": False,
            "private_eval_row_visible": False,
        },
    }


def materialize_benchmark(benchmark: str, source_path: Path) -> dict[str, Any]:
    rows = read_jsonl(source_path)
    out_dir = OUT_ROOT / BENCHMARK_SLUGS[benchmark]
    eval_path = eval_rows_path(PROJECT_ROOT, benchmark)
    attack_public_path = attack_probe_public_path(PROJECT_ROOT, benchmark)
    heldout_path = behavior_heldout_path(PROJECT_ROOT, benchmark)
    manifest_path = out_dir / "split_manifest.json"
    rel_source = rel(source_path)

    out_rows: list[dict[str, Any]] = []
    public_rows: list[dict[str, Any]] = []
    flattened_heldout_rows: list[dict[str, Any]] = []
    heldout_counts: Counter[int] = Counter()
    heldout_source_rows: set[str] = set()
    for index, row in enumerate(rows):
        row_id = _row_identifier(row, index)
        heldout_tasks = list(
            precompute_independent_heldout_tasks_for_split(
                row=row,
                benchmark=benchmark,
                sample_index=index,
                source_path=rel_source,
            )
        )
        for task in heldout_tasks:
            metadata = task.get("metadata") if isinstance(task.get("metadata"), Mapping) else {}
            task["metadata"] = {
                **dict(metadata),
                "heldout_selection": "precomputed_same_user_or_context_different_task_v1",
                "strong_query_split_schema_version": SPLIT_SCHEMA_VERSION,
            }
        heldout_counts[len(heldout_tasks)] += 1
        for task in heldout_tasks:
            metadata = task.get("metadata") if isinstance(task.get("metadata"), Mapping) else {}
            heldout_row = str(metadata.get("heldout_row_id") or "").strip()
            if heldout_row:
                heldout_source_rows.add(heldout_row)
        patched = dict(row)
        patched["benchmark"] = benchmark
        patched["_umpeek_split"] = {
            "schema_version": SPLIT_SCHEMA_VERSION,
            "benchmark": benchmark,
            "split_id": f"{SPLIT_SCHEMA_VERSION}__{BENCHMARK_SLUGS[benchmark]}",
            "split_source_paths": split_source_paths(benchmark, source_path),
            "attack_probe_id": row_id,
            "source_row_index": index,
            "source_row_id": row_id,
            "roles": {
                "memory_seed": {
                    "source": "benchmark_profile_history_or_evidence",
                    "visible_to_attacker": SPLIT_ROLE_VISIBILITY["memory_seed"]["visible_to_attacker"],
                    "used_to_materialize_backend_memory": True,
                },
                "attack_probe": {
                    "source": "attack_probe_public.jsonl public task projection",
                    "visible_to_attacker": SPLIT_ROLE_VISIBILITY["attack_probe"]["visible_to_attacker"],
                    "used_by_attack_methods": True,
                },
                "behavior_heldout": {
                    "source": "behavior_heldout.jsonl plus row-local precomputed tasks",
                    "visible_to_attacker": SPLIT_ROLE_VISIBILITY["behavior_heldout"]["visible_to_attacker"],
                    "used_only_by_hbps": True,
                },
                "evaluator_row": {
                    "source": "eval_rows.jsonl private evaluator join row",
                    "visible_to_attacker": SPLIT_ROLE_VISIBILITY["evaluator_row"]["visible_to_attacker"],
                    "used_by_runner_only": True,
                },
            },
            "behavior_heldout_tasks": heldout_tasks,
            "behavior_heldout_count": len(heldout_tasks),
            "legacy_dynamic_heldout_replaced": True,
            "forbid_runtime_dynamic_heldout": True,
            "private_eval_rows_visible_to_attackers": False,
        }
        out_rows.append(patched)
        public_rows.append(public_attack_probe_row(benchmark, row, row_id=row_id, index=index))
        for heldout_rank, task in enumerate(heldout_tasks):
            flattened_heldout_rows.append(
                {
                    "schema_version": SPLIT_SCHEMA_VERSION,
                    "split_id": f"{SPLIT_SCHEMA_VERSION}__{BENCHMARK_SLUGS[benchmark]}",
                    "benchmark": benchmark,
                    "attack_probe_id": row_id,
                    "source_row_index": index,
                    "heldout_rank": heldout_rank,
                    "visible_to_attacker": False,
                    "used_only_by_hbps": True,
                    "heldout_task": task,
                }
            )

    write_jsonl(eval_path, out_rows)
    write_jsonl(attack_public_path, public_rows)
    write_jsonl(heldout_path, flattened_heldout_rows)
    manifest = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "benchmark": benchmark,
        "split_id": f"{SPLIT_SCHEMA_VERSION}__{BENCHMARK_SLUGS[benchmark]}",
        "source_path": rel_source,
        "eval_rows_path": rel(eval_path),
        "attack_probe_public_path": rel(attack_public_path),
        "behavior_heldout_path": rel(heldout_path),
        "source_paths": split_source_paths(benchmark, source_path),
        "row_count": len(out_rows),
        "public_attack_probe_row_count": len(public_rows),
        "behavior_heldout_row_count": len(flattened_heldout_rows),
        "split_roles": {
            role: dict(payload) for role, payload in SPLIT_ROLE_VISIBILITY.items()
        },
        "threat_model_visibility": {
            "memory_seed_visible_to_attacker": False,
            "attack_probe_visible_to_attacker": True,
            "behavior_heldout_visible_to_attacker": False,
            "private_eval_rows_visible_to_attackers": False,
        },
        "forbidden_legacy_split_sources": [
            *FORBIDDEN_LEGACY_SPLIT_GLOBS,
            *(
                f"data/interim/eval2_splits/{version}"
                for version in DEPRECATED_STRONG_QUERY_SPLIT_SCHEMA_VERSIONS
            ),
        ],
        "heldout_policy": {
            "strategy": "precomputed_same_user_or_context_different_task_v1",
            "memory_seed_visible_to_attacker": False,
            "attack_probe_visible_to_attacker": True,
            "behavior_heldout_visible_to_attacker": False,
            "private_eval_rows_visible_to_attackers": False,
            "forbid_runtime_dynamic_heldout": True,
            "heldout_tasks_per_attack_probe_target": int(__import__("os").environ.get("UMPEEK_EVAL2_HBPS_HELDOUT_PER_SAMPLE", "3")),
        },
        "heldout_count_distribution": {str(key): value for key, value in sorted(heldout_counts.items())},
        "attack_rows_with_any_heldout": sum(value for key, value in heldout_counts.items() if key > 0),
        "unique_heldout_source_row_count": len(heldout_source_rows),
        "group_summary": group_summary(benchmark, rows),
    }
    write_json(manifest_path, manifest)
    return manifest


def patch_a200_manifest(split_manifests: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    manifest = json.loads(A200_MANIFEST.read_text(encoding="utf-8"))
    for audit in manifest.get("benchmark_audits", []):
        benchmark = str(audit.get("benchmark"))
        split = split_manifests.get(benchmark)
        if not split:
            continue
        heldout_policy = {
            **dict(audit.get("heldout_policy", {})),
            **dict(split.get("heldout_policy", {})),
            "benchmark": benchmark,
            "split_id": split["split_id"],
            "matrix_fixed": True,
            "reuse_across_all_methods_and_backends": True,
            "strong_query_split_schema_version": SPLIT_SCHEMA_VERSION,
        }
        heldout_policy.pop("fallback_strategy", None)
        audit["split_id"] = split["split_id"]
        audit["source_paths"] = list(split["source_paths"])
        audit["expected_sample_count"] = int(split["row_count"])
        audit["observed_sample_count"] = int(split["row_count"])
        audit["heldout_policy"] = heldout_policy
        audit["hbps_precheck"] = {
            "hbps_precheck_status": "ok" if int(split.get("attack_rows_with_any_heldout", 0)) > 0 else "no_precomputed_heldout",
            "row_count": int(split["row_count"]),
            "attack_rows_with_any_heldout": int(split.get("attack_rows_with_any_heldout", 0)),
            "heldout_count_distribution": split.get("heldout_count_distribution", {}),
        }
        audit.setdefault("metadata", {})
        audit_metadata = {
            **dict(audit.get("metadata", {})),
            "strong_query_split_manifest": rel(OUT_ROOT / BENCHMARK_SLUGS[benchmark] / "split_manifest.json"),
            "strong_query_eval_rows": split["eval_rows_path"],
            "strong_query_public_attack_probe": split["attack_probe_public_path"],
            "strong_query_behavior_heldout": split["behavior_heldout_path"],
            "old_weak_runtime_split_replaced": True,
        }
        audit_metadata.pop("deprecated_split_versions_removed", None)
        audit["metadata"] = audit_metadata

    for job in manifest.get("setting_jobs", []):
        benchmark = str(job.get("benchmark"))
        split = split_manifests.get(benchmark)
        if not split:
            continue
        heldout_policy = {
            **dict(job.get("heldout_policy", {})),
            **dict(split.get("heldout_policy", {})),
            "benchmark": benchmark,
            "split_id": split["split_id"],
            "matrix_fixed": True,
            "reuse_across_all_methods_and_backends": True,
            "strong_query_split_schema_version": SPLIT_SCHEMA_VERSION,
        }
        heldout_policy.pop("fallback_strategy", None)
        job["input_split"] = {
            **dict(job.get("input_split", {})),
            "split_id": split["split_id"],
            "source_paths": list(split["source_paths"]),
            "sample_count": int(split["row_count"]),
            "version_label": f"{SPLIT_SCHEMA_VERSION} private evaluator rows plus public attack probe",
            "strong_query_split_schema_version": SPLIT_SCHEMA_VERSION,
            "split_roles": {
                "memory_seed": "private benchmark profile/history/evidence; backend materialization only",
                "attack_probe": "public task row visible to UMPeek/baselines",
                "behavior_heldout": "precomputed private heldout tasks for HBPS only",
                "evaluator_row": "private runner row with hidden gold/profile and precomputed heldout",
            },
            "legacy_runtime_heldout_forbidden": True,
            "legacy_dynamic_heldout_replaced": True,
            "private_eval_rows_visible_to_attackers": False,
        }
        job["expected_sample_count"] = int(split["row_count"])
        job["heldout_policy"] = heldout_policy
        job.setdefault("metadata", {})
        job_metadata = {
            **dict(job.get("metadata", {})),
            "strong_query_split_manifest": rel(OUT_ROOT / BENCHMARK_SLUGS[benchmark] / "split_manifest.json"),
            "strong_query_eval_rows": split["eval_rows_path"],
            "strong_query_public_attack_probe": split["attack_probe_public_path"],
            "strong_query_behavior_heldout": split["behavior_heldout_path"],
            "old_weak_runtime_split_replaced": True,
        }
        job_metadata.pop("deprecated_split_versions_removed", None)
        job["metadata"] = job_metadata

    for shard in manifest.get("shard_jobs", []):
        benchmark = str(shard.get("benchmark"))
        split = split_manifests.get(benchmark)
        if not split:
            continue
        heldout_policy = {
            **dict(shard.get("heldout_policy", {})),
            **dict(split.get("heldout_policy", {})),
            "benchmark": benchmark,
            "split_id": split["split_id"],
            "matrix_fixed": True,
            "reuse_across_all_methods_and_backends": True,
            "strong_query_split_schema_version": SPLIT_SCHEMA_VERSION,
        }
        heldout_policy.pop("fallback_strategy", None)
        shard["input_split"] = {
            **dict(shard.get("input_split", {})),
            "split_id": split["split_id"],
            "source_paths": list(split["source_paths"]),
            "sample_count": int(split["row_count"]),
            "version_label": f"{SPLIT_SCHEMA_VERSION} private evaluator rows plus public attack probe",
            "strong_query_split_schema_version": SPLIT_SCHEMA_VERSION,
            "legacy_runtime_heldout_forbidden": True,
            "legacy_dynamic_heldout_replaced": True,
            "private_eval_rows_visible_to_attackers": False,
        }
        shard["heldout_policy"] = heldout_policy
        shard["parent_resume_key"] = str(shard.get("parent_resume_key") or "")
        shard_metadata = dict(shard.get("metadata", {})) if isinstance(shard.get("metadata"), Mapping) else {}
        shard_metadata = {
            **shard_metadata,
            "strong_query_split_manifest": rel(OUT_ROOT / BENCHMARK_SLUGS[benchmark] / "split_manifest.json"),
            "strong_query_eval_rows": split["eval_rows_path"],
            "strong_query_public_attack_probe": split["attack_probe_public_path"],
            "strong_query_behavior_heldout": split["behavior_heldout_path"],
            "old_weak_runtime_split_replaced": True,
        }
        shard_metadata.pop("deprecated_split_versions_removed", None)
        shard["metadata"] = shard_metadata

    manifest.setdefault("metadata", {})
    manifest_metadata = {
        **dict(manifest.get("metadata", {})),
        "strong_query_split_schema_version": SPLIT_SCHEMA_VERSION,
        "strong_query_split_root": rel(OUT_ROOT),
        "old_weak_runtime_split_replaced": True,
    }
    manifest_metadata.pop("deprecated_split_versions_removed", None)
    manifest["metadata"] = manifest_metadata
    A200_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    A200_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create role-separated evaluator, attack-probe, and held-out rows."
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not remove an existing current split before rewriting it.",
    )
    return parser.parse_args()


def main() -> int:
    args = arguments()
    missing_sources = [rel(path) for path in BENCHMARK_SOURCES.values() if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(
            "Canonical benchmark rows are missing: "
            + ", ".join(missing_sources)
            + ". Run scripts/preprocess_benchmarks.py first."
        )
    cleanup_rows = [] if args.no_clean else cleanup_legacy_split_artifacts()
    split_manifests = {
        benchmark: materialize_benchmark(benchmark, source)
        for benchmark, source in BENCHMARK_SOURCES.items()
    }
    patched = patch_a200_manifest(split_manifests) if A200_MANIFEST.is_file() else {"setting_jobs": []}
    summary = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "split_root": rel(OUT_ROOT),
        "a200_manifest": rel(A200_MANIFEST) if A200_MANIFEST.is_file() else None,
        "benchmarks": {
            benchmark: {
                "row_count": split["row_count"],
                "eval_rows_path": split["eval_rows_path"],
                "attack_probe_public_path": split["attack_probe_public_path"],
                "behavior_heldout_path": split["behavior_heldout_path"],
                "heldout_count_distribution": split["heldout_count_distribution"],
                "attack_rows_with_any_heldout": split["attack_rows_with_any_heldout"],
            }
            for benchmark, split in split_manifests.items()
        },
        "setting_jobs": len(patched.get("setting_jobs", [])),
        "next_step": (
            "The existing A200 manifest was updated."
            if A200_MANIFEST.is_file()
            else "Run scripts/build_minimal_manifest.py to create a release manifest."
        ),
        "cleanup": cleanup_rows,
    }
    write_json(OUT_ROOT / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
