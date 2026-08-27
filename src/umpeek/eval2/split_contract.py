from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


CURRENT_STRONG_QUERY_SPLIT_SCHEMA_VERSION = "a200_role_locked_benchmark_split_v4"
DEPRECATED_STRONG_QUERY_SPLIT_SCHEMA_VERSIONS = (
    "a200_strong_query_split_v1",
    "a200_strong_query_split_v2",
    "a200_strong_query_split_v3",
)

SPLIT_ROOT_REL = "data/interim/eval2_splits"
REQUIRED_SPLIT_ROLES = ("memory_seed", "attack_probe", "behavior_heldout", "evaluator_row")
FORBIDDEN_LEGACY_SPLIT_GLOBS = (
    "data/interim/exp1_whitebox/*/representative_subset.jsonl",
    "data/interim/benchmarks/ETAPP/expanded_v1/quick_subset.jsonl",
)
SPLIT_ROLE_VISIBILITY = {
    "memory_seed": {
        "visible_to_attacker": False,
        "used_by": "backend_materializer",
        "description": "private benchmark profile/history/evidence used to seed backend memory",
    },
    "attack_probe": {
        "visible_to_attacker": True,
        "used_by": "attack_methods",
        "description": "public task/query projection exposed through AttackInput",
    },
    "behavior_heldout": {
        "visible_to_attacker": False,
        "used_by": "hbps_only",
        "description": "private heldout behavior tasks used only by HBPS",
    },
    "evaluator_row": {
        "visible_to_attacker": False,
        "used_by": "eval_runner_only",
        "description": "private row joining source fields, gold state, split ids, and heldout tasks",
    },
}

BENCHMARK_SLUGS = {
    "PersonaMem-v2": "personamem_v2",
    "PersonaLens": "personalens",
    "ETAPP_150x32": "etapp_150x32",
    "LoCoMo_10conv_1523QA_20speakers": "locomo_10conv_1523qa_20speakers",
}
BENCHMARK_ALIASES = {
    "PersonaMem-v2": "PersonaMem-v2",
    "PersonaMemv2": "PersonaMem-v2",
    "personamem-v2": "PersonaMem-v2",
    "personamemv2": "PersonaMem-v2",
    "PersonaLens": "PersonaLens",
    "personalens": "PersonaLens",
    "ETAPP_150x32": "ETAPP_150x32",
    "ETAPP": "ETAPP_150x32",
    "etapp": "ETAPP_150x32",
    "LoCoMo_10conv_1523QA_20speakers": "LoCoMo_10conv_1523QA_20speakers",
    "LoCoMo": "LoCoMo_10conv_1523QA_20speakers",
    "locomo": "LoCoMo_10conv_1523QA_20speakers",
}
LEGACY_ATTACK_BENCHMARK_NAMES = {
    "PersonaMem-v2": "PersonaMemv2",
    "PersonaLens": "PersonaLens",
    "ETAPP_150x32": "ETAPP",
    "LoCoMo_10conv_1523QA_20speakers": "LoCoMo",
}


def canonical_benchmark_name(benchmark: str) -> str:
    raw = str(benchmark or "").strip()
    canonical = BENCHMARK_ALIASES.get(raw)
    if canonical is not None:
        return canonical
    lowered = raw.lower()
    for alias, target in BENCHMARK_ALIASES.items():
        if alias.lower() == lowered:
            return target
    raise ValueError(f"Unsupported benchmark for strong split: {benchmark!r}")


def benchmark_slug(benchmark: str) -> str:
    benchmark = canonical_benchmark_name(benchmark)
    try:
        return BENCHMARK_SLUGS[benchmark]
    except KeyError as exc:
        raise ValueError(f"Unsupported benchmark for strong split: {benchmark!r}") from exc


def split_root(project_root: Path) -> Path:
    return project_root / SPLIT_ROOT_REL / CURRENT_STRONG_QUERY_SPLIT_SCHEMA_VERSION


def benchmark_split_dir(project_root: Path, benchmark: str) -> Path:
    return split_root(project_root) / benchmark_slug(benchmark)


def split_manifest_path(project_root: Path, benchmark: str) -> Path:
    return benchmark_split_dir(project_root, benchmark) / "split_manifest.json"


def eval_rows_path(project_root: Path, benchmark: str) -> Path:
    return benchmark_split_dir(project_root, benchmark) / "eval_rows.jsonl"


def attack_probe_public_path(project_root: Path, benchmark: str) -> Path:
    return benchmark_split_dir(project_root, benchmark) / "attack_probe_public.jsonl"


def behavior_heldout_path(project_root: Path, benchmark: str) -> Path:
    return benchmark_split_dir(project_root, benchmark) / "behavior_heldout.jsonl"


def attack_probe_path(project_root: Path, benchmark: str) -> Path:
    """Backward-compatible name for the private evaluator row file.

    Current role-locked splits intentionally separate private evaluator rows
    from the public attack-probe projection. New code should call
    eval_rows_path() or attack_probe_public_path() explicitly.
    """

    return eval_rows_path(project_root, benchmark)


def split_id(benchmark: str) -> str:
    benchmark = canonical_benchmark_name(benchmark)
    return f"{CURRENT_STRONG_QUERY_SPLIT_SCHEMA_VERSION}__{benchmark_slug(benchmark)}"


def relative_to_project(project_root: Path, path: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def expected_attack_probe_rel(project_root: Path, benchmark: str) -> str:
    return relative_to_project(project_root, attack_probe_public_path(project_root, benchmark))


def expected_eval_rows_rel(project_root: Path, benchmark: str) -> str:
    return relative_to_project(project_root, eval_rows_path(project_root, benchmark))


def expected_behavior_heldout_rel(project_root: Path, benchmark: str) -> str:
    return relative_to_project(project_root, behavior_heldout_path(project_root, benchmark))


def expected_manifest_rel(project_root: Path, benchmark: str) -> str:
    return relative_to_project(project_root, split_manifest_path(project_root, benchmark))


def validate_current_job_split(job: Mapping[str, Any], *, project_root: Path | None = None) -> list[str]:
    raw_benchmark = str(job.get("benchmark") or "")
    try:
        benchmark = canonical_benchmark_name(raw_benchmark)
    except ValueError:
        benchmark = raw_benchmark
    split = job.get("input_split")
    heldout_policy = job.get("heldout_policy")
    split = dict(split) if isinstance(split, Mapping) else {}
    heldout_policy = dict(heldout_policy) if isinstance(heldout_policy, Mapping) else {}
    errors: list[str] = []

    schema = str(split.get("strong_query_split_schema_version") or "")
    if schema != CURRENT_STRONG_QUERY_SPLIT_SCHEMA_VERSION:
        errors.append(f"strong_query_split_schema_version={schema!r}")

    observed_split_id = str(split.get("split_id") or "")
    expected_split_id = split_id(benchmark) if benchmark in BENCHMARK_SLUGS else ""
    if observed_split_id != expected_split_id:
        errors.append(f"split_id={observed_split_id!r}, expected={expected_split_id!r}")

    raw_source_paths = split.get("source_paths", [])
    source_paths = (
        [str(item) for item in raw_source_paths]
        if isinstance(raw_source_paths, Sequence) and not isinstance(raw_source_paths, (str, bytes, bytearray))
        else []
    )
    if project_root is not None and benchmark in BENCHMARK_SLUGS:
        expected_eval = expected_eval_rows_rel(project_root, benchmark)
        expected_attack = expected_attack_probe_rel(project_root, benchmark)
        expected_heldout = expected_behavior_heldout_rel(project_root, benchmark)
        expected_manifest = expected_manifest_rel(project_root, benchmark)
        if expected_eval not in source_paths:
            errors.append(f"missing_current_eval_rows_path={expected_eval!r}")
        if expected_attack not in source_paths:
            errors.append(f"missing_current_public_attack_probe_path={expected_attack!r}")
        if expected_heldout not in source_paths:
            errors.append(f"missing_current_behavior_heldout_path={expected_heldout!r}")
        if expected_manifest not in source_paths:
            errors.append(f"missing_current_split_manifest_path={expected_manifest!r}")
    elif not any(Path(path).name == "eval_rows.jsonl" for path in source_paths):
        errors.append("missing_eval_rows_jsonl")

    roles = split.get("split_roles")
    roles = dict(roles) if isinstance(roles, Mapping) else {}
    missing_roles = [role for role in REQUIRED_SPLIT_ROLES if role not in roles]
    if missing_roles:
        errors.append(f"missing_split_roles={missing_roles!r}")

    if split.get("legacy_runtime_heldout_forbidden") is not True:
        errors.append("legacy_runtime_heldout_forbidden_not_true")
    if split.get("legacy_dynamic_heldout_replaced") is not True:
        errors.append("legacy_dynamic_heldout_replaced_not_true")
    if split.get("private_eval_rows_visible_to_attackers") is not False:
        errors.append("private_eval_rows_visibility_not_false")
    if heldout_policy.get("forbid_runtime_dynamic_heldout") is not True:
        errors.append("heldout_policy_forbid_runtime_dynamic_heldout_not_true")
    if heldout_policy.get("reuse_across_all_methods_and_backends") is not True:
        errors.append("heldout_policy_not_reused_across_matrix")

    sample_count = int(split.get("sample_count") or 0)
    if sample_count <= 0:
        errors.append(f"non_positive_sample_count={sample_count}")
    return errors
