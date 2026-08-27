from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from ..exp1.personalens_data import (
    PERSONALENS_BENCHMARK,
    PERSONALENS_DATASET_REPO,
    PERSONALENS_LICENSE,
    build_personalens_local_dataset,
)
from ..exp1.schema import TaskRecord, UserRecord
from .schema import EXPERIMENT_VERSION


DEFAULT_PERSONALENS_SUBSET_CONFIG = Path("configs/exp1_whitebox_personalens_subset_a007.json")
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")

_EXPLICIT_EVIDENCE_TERMS = (
    "preferred",
    "preference",
    "preferences",
    "favorite",
    "favourite",
    "tailored",
    "aligned with their preferences",
    "aligned with your preferences",
    "according to your preferences",
    "match their preferences",
    "match your preferences",
)
_IMPLICIT_EVIDENCE_TERMS = (
    "based on",
    "existing",
    "current",
    "usually",
    "typical",
    "habit",
    "history",
    "previously enjoyed",
    "in the past",
    "reading habits",
    "travel frequency",
    "gaming frequency",
)
_AMBIGUOUS_EVIDENCE_TERMS = (
    "if needed",
    "if applicable",
    "if you'd like",
    "potentially",
    "explore",
    "considering",
)


@dataclass(frozen=True, slots=True)
class PersonaLensSamplingConfig:
    task_id: str
    benchmark: str
    dataset_root: Path
    output_dir: Path
    seed: int
    sample_target: int
    sample_min: int
    max_tasks_per_user: int
    task_type_min: int
    history_bucket_min: int
    evidence_bucket_min: int
    domain_min: int
    tool_action_min: int


@dataclass(slots=True)
class PersonaLensCandidate:
    user: UserRecord
    task: TaskRecord
    domain: str
    task_template_id: str
    task_type: str
    tool_action_category: str
    history_length_words: int
    history_length_bucket: str = "unknown"
    evidence_label: str = "ambiguous"
    evidence_reason: str = ""
    evidence_signals: tuple[str, ...] = ()

    @property
    def user_id(self) -> str:
        return self.task.user_id

    @property
    def task_id(self) -> str:
        return self.task.task_id

    @property
    def sample_key(self) -> str:
        return str(self.task.metadata.get("sample_key", f"{self.user_id}::{self.task_id}"))


@dataclass(frozen=True, slots=True)
class PersonaLensSubsetBuildResult:
    task_records_path: Path
    representative_subset_path: Path
    sampling_report_path: Path
    total_candidates: int
    selected_count: int
    sample_target: int
    unique_users: int
    dataset_root: Path


def load_personalens_sampling_config(
    project_root: Path,
    config_path: Path | None = None,
) -> PersonaLensSamplingConfig:
    config_file = config_path or (project_root / DEFAULT_PERSONALENS_SUBSET_CONFIG)
    payload = json.loads(config_file.read_text(encoding="utf-8")) if config_file.is_file() else {}
    benchmark = str(payload.get("benchmark", PERSONALENS_BENCHMARK))
    if benchmark != PERSONALENS_BENCHMARK:
        raise ValueError(f"Unsupported benchmark in {config_file}: {benchmark}")
    return PersonaLensSamplingConfig(
        task_id=str(payload.get("task_id", "A007")),
        benchmark=benchmark,
        dataset_root=_resolve_project_path(project_root, payload.get("dataset_root", "data/benchmarks/PersonaLens")),
        output_dir=_resolve_project_path(
            project_root,
            payload.get("output_dir", "data/interim/exp1_whitebox/PersonaLens"),
        ),
        seed=int(payload.get("seed", 20260515)),
        sample_target=int(payload.get("sample_target", 240)),
        sample_min=int(payload.get("sample_min", 150)),
        max_tasks_per_user=int(payload.get("max_tasks_per_user", 4)),
        task_type_min=int(payload.get("task_type_min", 20)),
        history_bucket_min=int(payload.get("history_bucket_min", 20)),
        evidence_bucket_min=int(payload.get("evidence_bucket_min", 20)),
        domain_min=int(payload.get("domain_min", 5)),
        tool_action_min=int(payload.get("tool_action_min", 5)),
    )


def build_personalens_whitebox_subset(
    project_root: Path,
    *,
    config_path: Path | None = None,
    dataset_root: Path | None = None,
    output_dir: Path | None = None,
) -> PersonaLensSubsetBuildResult:
    config = load_personalens_sampling_config(project_root=project_root, config_path=config_path)
    if dataset_root is not None:
        config = replace(config, dataset_root=dataset_root)
    if output_dir is not None:
        config = replace(config, output_dir=output_dir)

    dataset = build_personalens_local_dataset(config.dataset_root)
    user_by_id = {user.user_id: user for user in dataset.users}
    candidates = _build_candidates(dataset.tasks, user_by_id)
    bucket_summary = _assign_history_length_buckets(candidates)

    target = _resolve_sample_target(candidates, config)
    quota_plan = _build_quota_plan(candidates, target=target, config=config)
    selected = _select_candidates(
        candidates,
        target=target,
        quota_plan=quota_plan,
        max_tasks_per_user=config.max_tasks_per_user,
        seed=config.seed,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    task_records_path = config.output_dir / "task_records.jsonl"
    representative_subset_path = config.output_dir / "representative_subset.jsonl"
    sampling_report_path = config.output_dir / "sampling_report.json"

    ordered_candidates = sorted(candidates, key=lambda item: (item.user_id, item.task_id))
    selected_keys = {candidate.sample_key for candidate in selected}
    ordered_selected = [candidate for candidate in ordered_candidates if candidate.sample_key in selected_keys]

    _write_jsonl(task_records_path, (_serialize_task_record(candidate) for candidate in ordered_candidates))
    _write_jsonl(representative_subset_path, (_serialize_task_record(candidate) for candidate in ordered_selected))

    sampling_report = _build_sampling_report(
        candidates=candidates,
        selected=ordered_selected,
        quota_plan=quota_plan,
        config=config,
        target=target,
        bucket_summary=bucket_summary,
    )
    sampling_report_path.write_text(json.dumps(sampling_report, indent=2, sort_keys=True), encoding="utf-8")

    return PersonaLensSubsetBuildResult(
        task_records_path=task_records_path,
        representative_subset_path=representative_subset_path,
        sampling_report_path=sampling_report_path,
        total_candidates=len(candidates),
        selected_count=len(ordered_selected),
        sample_target=target,
        unique_users=len({candidate.user_id for candidate in ordered_selected}),
        dataset_root=config.dataset_root,
    )


def _resolve_project_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return project_root / path


def _build_candidates(
    tasks: list[TaskRecord],
    user_by_id: dict[str, UserRecord],
) -> list[PersonaLensCandidate]:
    candidates: list[PersonaLensCandidate] = []
    for task in tasks:
        user = user_by_id[task.user_id]
        evidence_label, evidence_reason, evidence_signals = _classify_personalization_evidence(task)
        candidates.append(
            PersonaLensCandidate(
                user=user,
                task=task,
                domain=str(task.metadata.get("dialogue_domain", "unknown") or "unknown"),
                task_template_id=task.task_id,
                task_type=task.task_type,
                tool_action_category="not_applicable",
                history_length_words=_relevant_history_length_words(user, task),
                evidence_label=evidence_label,
                evidence_reason=evidence_reason,
                evidence_signals=evidence_signals,
            )
        )
    return candidates


def _classify_personalization_evidence(task: TaskRecord) -> tuple[str, str, tuple[str, ...]]:
    text = " ".join(
        str(task.metadata.get(field_name, ""))
        for field_name in ("task_description", "user_intent", "task_goal")
    )
    text = f"{task.prompt} {text}".strip().lower()
    expected_affinities = list(task.metadata.get("expected_affinities", []))
    affinity_count = len(expected_affinities)
    value_count = sum(len(item.get("values", [])) for item in expected_affinities)
    explicit_hits = _matched_terms(text, _EXPLICIT_EVIDENCE_TERMS)
    implicit_hits = _matched_terms(text, _IMPLICIT_EVIDENCE_TERMS)
    ambiguous_hits = _matched_terms(text, _AMBIGUOUS_EVIDENCE_TERMS)

    if len(explicit_hits) >= 2 and affinity_count >= 2:
        return (
            "explicit",
            f"direct preference wording with {affinity_count} affinity anchors",
            tuple(sorted(explicit_hits)),
        )
    if explicit_hits and affinity_count >= 3:
        return (
            "explicit",
            f"explicit preference cues with broad multi-attribute grounding ({affinity_count})",
            tuple(sorted(explicit_hits)),
        )
    if not explicit_hits and implicit_hits and affinity_count >= 2:
        return (
            "implicit",
            f"history or habit cues without direct preference wording ({affinity_count} anchors)",
            tuple(sorted(implicit_hits)),
        )
    if ambiguous_hits and affinity_count <= 2:
        return (
            "ambiguous",
            "task wording leaves preference application optional or underspecified",
            tuple(sorted(ambiguous_hits)),
        )
    if affinity_count <= 1 or value_count <= 2:
        reason = "single narrow affinity anchor" if affinity_count <= 1 else "limited concrete affinity values"
        signals = explicit_hits or implicit_hits or ambiguous_hits
        return ("weak", reason, tuple(sorted(signals)))
    if explicit_hits:
        return (
            "implicit",
            "preference language is present but not broad enough for explicit multi-anchor evidence",
            tuple(sorted(explicit_hits)),
        )
    return (
        "ambiguous",
        "benchmark metadata carries affinities, but prompt wording does not expose a clear evidence style",
        tuple(sorted(implicit_hits or ambiguous_hits)),
    )


def _matched_terms(text: str, terms: Iterable[str]) -> set[str]:
    return {term for term in terms if term in text}


def _relevant_history_length_words(user: UserRecord, task: TaskRecord) -> int:
    relevant_domains = list(task.metadata.get("relevant_domains", []))
    affinities = dict(user.metadata.get("affinities", {}))
    interactions = dict(user.metadata.get("interactions", {}))
    profile_parts: list[str] = []
    for domain in relevant_domains:
        domain_affinities = affinities.get(domain, {})
        if isinstance(domain_affinities, dict):
            for affinity_key, values in domain_affinities.items():
                profile_parts.append(str(affinity_key))
                if isinstance(values, (list, tuple)):
                    profile_parts.extend(str(item) for item in values)
                else:
                    profile_parts.append(str(values))
        interaction_text = interactions.get(domain)
        if interaction_text:
            profile_parts.append(str(interaction_text))
    if not profile_parts:
        profile_parts.extend(str(value) for value in user.profile.values())
    return sum(_word_count(part) for part in profile_parts)


def _word_count(text: Any) -> int:
    return len(_WORD_RE.findall(str(text or "")))


def _assign_history_length_buckets(candidates: list[PersonaLensCandidate]) -> dict[str, dict[str, int]]:
    ordered = sorted(candidates, key=lambda item: (item.history_length_words, item.user_id, item.task_id))
    if not ordered:
        return {bucket: {"count": 0, "min_words": 0, "max_words": 0} for bucket in ("short", "medium", "long")}

    bucket_names = ("short", "medium", "long")
    total = len(ordered)
    for index, candidate in enumerate(ordered):
        bucket_name = bucket_names[min(2, (index * 3) // total)]
        candidate.history_length_bucket = bucket_name

    summary: dict[str, dict[str, int]] = {}
    for bucket_name in bucket_names:
        values = [candidate.history_length_words for candidate in ordered if candidate.history_length_bucket == bucket_name]
        summary[bucket_name] = {
            "count": len(values),
            "min_words": min(values) if values else 0,
            "max_words": max(values) if values else 0,
        }
    return summary


def _resolve_sample_target(candidates: list[PersonaLensCandidate], config: PersonaLensSamplingConfig) -> int:
    total = len(candidates)
    feasible_total = sum(
        min(count, config.max_tasks_per_user)
        for count in Counter(candidate.user_id for candidate in candidates).values()
    )
    if total <= 300:
        return total
    target = min(config.sample_target, feasible_total)
    if target < config.sample_min:
        raise ValueError(
            f"Feasible PersonaLens sample target {target} falls below required minimum {config.sample_min}."
        )
    return target


def _build_quota_plan(
    candidates: list[PersonaLensCandidate],
    *,
    target: int,
    config: PersonaLensSamplingConfig,
) -> dict[str, dict[str, dict[str, Any]]]:
    plan: dict[str, dict[str, dict[str, Any]]] = {}
    task_type_floor = min(config.task_type_min, math.ceil(target * 0.2))
    plan["task_type"] = _quota_entries(
        candidates,
        dimension="task_type",
        desired_values=sorted({candidate.task_type for candidate in candidates}),
        base_required=task_type_floor,
        max_tasks_per_user=config.max_tasks_per_user,
    )
    plan["history_length_bucket"] = _quota_entries(
        candidates,
        dimension="history_length_bucket",
        desired_values=["short", "medium", "long"],
        base_required=config.history_bucket_min,
        max_tasks_per_user=config.max_tasks_per_user,
    )
    plan["personalization_evidence"] = _quota_entries(
        candidates,
        dimension="personalization_evidence",
        desired_values=["explicit", "implicit", "weak", "ambiguous"],
        base_required=config.evidence_bucket_min,
        max_tasks_per_user=config.max_tasks_per_user,
    )
    plan["dialogue_domain"] = _quota_entries(
        candidates,
        dimension="dialogue_domain",
        desired_values=sorted({candidate.domain for candidate in candidates}),
        base_required=config.domain_min,
        max_tasks_per_user=config.max_tasks_per_user,
    )
    tool_values = sorted(
        {
            candidate.tool_action_category
            for candidate in candidates
            if candidate.tool_action_category != "not_applicable"
        }
    )
    if tool_values:
        plan["tool_action_category"] = _quota_entries(
            candidates,
            dimension="tool_action_category",
            desired_values=tool_values,
            base_required=config.tool_action_min,
            max_tasks_per_user=config.max_tasks_per_user,
        )
    else:
        plan["tool_action_category"] = {
            "not_applicable": {
                "available": 0,
                "feasible_capacity": 0,
                "base_required": 0,
                "required": 0,
                "reason": "PersonaLens public profile tasks are open-dialogue records with no released tool schema.",
            }
        }
    return plan


def _quota_entries(
    candidates: list[PersonaLensCandidate],
    *,
    dimension: str,
    desired_values: Iterable[str],
    base_required: int,
    max_tasks_per_user: int,
) -> dict[str, dict[str, Any]]:
    available = Counter(_candidate_value(candidate, dimension) for candidate in candidates)
    per_user: dict[str, Counter[str]] = defaultdict(Counter)
    for candidate in candidates:
        per_user[_candidate_value(candidate, dimension)][candidate.user_id] += 1

    entries: dict[str, dict[str, Any]] = {}
    for value in desired_values:
        user_counts = per_user.get(value, Counter())
        feasible_capacity = sum(min(count, max_tasks_per_user) for count in user_counts.values())
        available_count = available.get(value, 0)
        required = min(base_required, available_count, feasible_capacity)
        reason = None
        if available_count == 0:
            reason = "no_candidates_in_source"
        elif required < base_required:
            if available_count < base_required:
                reason = "available_count_below_base_requirement"
            elif feasible_capacity < base_required:
                reason = "user_cap_reduces_feasible_capacity"
        entries[value] = {
            "available": available_count,
            "feasible_capacity": feasible_capacity,
            "base_required": base_required,
            "required": required,
            "reason": reason,
        }
    return entries


def _candidate_value(candidate: PersonaLensCandidate, dimension: str) -> str:
    if dimension == "task_type":
        return candidate.task_type
    if dimension == "history_length_bucket":
        return candidate.history_length_bucket
    if dimension == "personalization_evidence":
        return candidate.evidence_label
    if dimension == "dialogue_domain":
        return candidate.domain
    if dimension == "tool_action_category":
        return candidate.tool_action_category
    raise KeyError(f"Unsupported quota dimension: {dimension}")


def _select_candidates(
    candidates: list[PersonaLensCandidate],
    *,
    target: int,
    quota_plan: dict[str, dict[str, dict[str, Any]]],
    max_tasks_per_user: int,
    seed: int,
) -> list[PersonaLensCandidate]:
    if len(candidates) <= target:
        return sorted(candidates, key=lambda item: (item.user_id, item.task_id))

    candidate_pool = list(candidates)
    random.Random(seed).shuffle(candidate_pool)
    selected: list[PersonaLensCandidate] = []
    selected_keys: set[str] = set()
    user_counts: Counter[str] = Counter()
    task_template_counts: Counter[str] = Counter()
    counts_by_dimension: dict[str, Counter[str]] = defaultdict(Counter)

    while len(selected) < target:
        best_candidate: PersonaLensCandidate | None = None
        best_score: tuple[int, int, int, int, int, int] | None = None
        for candidate in candidate_pool:
            if candidate.sample_key in selected_keys:
                continue
            if user_counts[candidate.user_id] >= max_tasks_per_user:
                continue
            score = _candidate_score(
                candidate,
                quota_plan=quota_plan,
                counts_by_dimension=counts_by_dimension,
                user_counts=user_counts,
                task_template_counts=task_template_counts,
            )
            if best_score is None or score > best_score:
                best_candidate = candidate
                best_score = score
        if best_candidate is None:
            break
        selected.append(best_candidate)
        selected_keys.add(best_candidate.sample_key)
        user_counts[best_candidate.user_id] += 1
        task_template_counts[best_candidate.task_template_id] += 1
        for dimension in quota_plan:
            counts_by_dimension[dimension][_candidate_value(best_candidate, dimension)] += 1

    return sorted(selected, key=lambda item: (item.user_id, item.task_id))


def _candidate_score(
    candidate: PersonaLensCandidate,
    *,
    quota_plan: dict[str, dict[str, dict[str, Any]]],
    counts_by_dimension: dict[str, Counter[str]],
    user_counts: Counter[str],
    task_template_counts: Counter[str],
) -> tuple[int, int, int, int, int, int]:
    unmet_dimensions = 0
    remaining_gap = 0
    for dimension, entries in quota_plan.items():
        value = _candidate_value(candidate, dimension)
        required = int(entries.get(value, {}).get("required", 0))
        if required <= 0:
            continue
        current_count = counts_by_dimension[dimension][value]
        if current_count < required:
            unmet_dimensions += 1
            remaining_gap += required - current_count
    return (
        unmet_dimensions,
        remaining_gap,
        int(task_template_counts[candidate.task_template_id] == 0),
        int(user_counts[candidate.user_id] == 0),
        -counts_by_dimension["dialogue_domain"][candidate.domain],
        -user_counts[candidate.user_id],
    )


def _serialize_task_record(candidate: PersonaLensCandidate) -> dict[str, Any]:
    task = candidate.task
    user = candidate.user
    expected_affinities = _json_safe(task.metadata.get("expected_affinities", []))
    relevant_domains = [str(item) for item in task.metadata.get("relevant_domains", [])]
    relevant_affinity_types = [str(item) for item in task.metadata.get("relevant_affinity_types", [])]
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "record_type": "TaskRecord",
        "benchmark": task.benchmark,
        "user_id": task.user_id,
        "task_id": task.task_id,
        "sample_key": candidate.sample_key,
        "task_domain": candidate.domain,
        "task_template_id": candidate.task_template_id,
        "task_type": task.task_type,
        "pre_history_ref": {
            "profile_source": str(user.metadata.get("profile_source", "")),
            "tasks_source": str(user.metadata.get("tasks_source", "")),
            "history_scope": "relevant_domain_affinities_and_interactions",
            "relevant_domains": relevant_domains,
            "relevant_affinity_types": relevant_affinity_types,
        },
        "task_input": {
            "prompt": task.prompt,
            "task_description": str(task.metadata.get("task_description", "")),
            "user_intent": str(task.metadata.get("user_intent", "")),
            "task_goal": str(task.metadata.get("task_goal", "")),
            "situations": _json_safe(task.metadata.get("situations", {})),
        },
        "gold": {
            "task_goal": str(task.metadata.get("task_goal", "")),
            "expected_affinities": expected_affinities,
            "evaluation_dimensions": ["task_completion", "personalization"],
            "reference_mode": "official_judge_criteria",
        },
        "personalization_evidence": {
            "label": candidate.evidence_label,
            "reason": candidate.evidence_reason,
            "signals": list(candidate.evidence_signals),
            "relevant_domains": relevant_domains,
            "relevant_affinity_types": relevant_affinity_types,
            "expected_affinity_count": len(expected_affinities),
        },
        "tool_schema": [],
        "tool_action_category": candidate.tool_action_category,
        "scoring_method": {
            "type": "official_llm_as_judge",
            "source_repo": "https://github.com/amazon-science/personalens",
            "source_paper": "https://aclanthology.org/2025.findings-acl.927/",
            "judge_dimensions": ["task_completion", "personalization"],
            "official_judge_backend": "amazon_bedrock",
            "judge_execution_status": "not_run_in_a007",
            "blocked_reason": "Official PersonaLens generation/evaluation requires Amazon Bedrock; A007 only prepares benchmark records and subset manifests.",
        },
        "history_length_words": candidate.history_length_words,
        "history_length_bucket": candidate.history_length_bucket,
        "swap_user_id": task.metadata.get("swap_user_id"),
        "swap_status": str(task.metadata.get("swap_status", "swap_unavailable")),
        "trace_status": "pending_runtime_capture",
        "source_dataset": {
            "dataset_repo": PERSONALENS_DATASET_REPO,
            "license": PERSONALENS_LICENSE,
            "benchmark_split": str(task.metadata.get("benchmark_split", "")),
        },
    }


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _build_sampling_report(
    *,
    candidates: list[PersonaLensCandidate],
    selected: list[PersonaLensCandidate],
    quota_plan: dict[str, dict[str, dict[str, Any]]],
    config: PersonaLensSamplingConfig,
    target: int,
    bucket_summary: dict[str, dict[str, int]],
) -> dict[str, Any]:
    full_distributions = _distribution_summary(candidates)
    selected_distributions = _distribution_summary(selected)
    selected_counts_by_dimension: dict[str, Counter[str]] = defaultdict(Counter)
    for candidate in selected:
        for dimension in quota_plan:
            selected_counts_by_dimension[dimension][_candidate_value(candidate, dimension)] += 1

    quota_status: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    for dimension, entries in quota_plan.items():
        for value, entry in entries.items():
            selected_count = int(selected_counts_by_dimension[dimension][value])
            available = int(entry.get("available", 0))
            required = int(entry.get("required", 0))
            status = "met"
            reason = entry.get("reason")
            if required == 0 and dimension == "tool_action_category":
                status = "not_applicable"
            elif available == 0:
                status = "unavailable_in_source"
            elif selected_count < required:
                status = "underfilled"
                reason = reason or "selection_stalled_before_required_quota"
            quota_item = {
                "dimension": dimension,
                "value": value,
                "available": available,
                "feasible_capacity": int(entry.get("feasible_capacity", 0)),
                "base_required": int(entry.get("base_required", 0)),
                "required": required,
                "selected": selected_count,
                "status": status,
                "reason": reason,
            }
            quota_status.append(quota_item)
            if status != "met":
                gaps.append(quota_item)

    return {
        "task_id": config.task_id,
        "benchmark": config.benchmark,
        "experiment_version": EXPERIMENT_VERSION,
        "seed": config.seed,
        "dataset_root": str(config.dataset_root),
        "output_dir": str(config.output_dir),
        "total_candidates": len(candidates),
        "selected_count": len(selected),
        "sample_target": target,
        "sample_min": config.sample_min,
        "max_tasks_per_user": config.max_tasks_per_user,
        "selection_mode": "full_include" if len(candidates) <= 300 else "representative_subset",
        "source_dataset": {
            "dataset_repo": PERSONALENS_DATASET_REPO,
            "license": PERSONALENS_LICENSE,
            "official_data_free": True,
            "requires_manual_access": False,
        },
        "history_length_bucket_ranges": bucket_summary,
        "distributions": {
            "full": full_distributions,
            "selected": selected_distributions,
            "selected_unique_users": len({candidate.user_id for candidate in selected}),
            "selected_unique_task_templates": len({candidate.task_template_id for candidate in selected}),
        },
        "quota_status": quota_status,
        "gaps": gaps,
        "stratification_rules": {
            "task_type": "IF task_type exists THEN each class covers min(20, 20%) capped by available count and per-user feasible capacity.",
            "history_length_bucket": "IF user history length exists THEN rank-based short/medium/long buckets each target >= 20 samples when available.",
            "personalization_evidence": "IF personalization evidence is identifiable THEN explicit/implicit/weak/ambiguous each target >= 20 samples when available.",
            "tool_action_category": "IF PersonaLens released data exposes tool/action categories THEN each major category targets >= 5 samples; otherwise mark not_applicable.",
        },
        "notes": [
            "PersonaLens local source data already exists under data/benchmarks/PersonaLens, so A007 did not re-download official files.",
            "Official PersonaLens generation/evaluation depends on Amazon Bedrock, so A007 only records scoring metadata and does not invoke judge or user agents.",
            "History buckets are computed from task-relevant profile affinities and interaction summaries, not the full profile dump, to avoid collapsing all tasks into a single long bucket.",
        ],
    }


def _distribution_summary(candidates: list[PersonaLensCandidate]) -> dict[str, dict[str, int]]:
    return {
        "task_type": dict(sorted(Counter(candidate.task_type for candidate in candidates).items())),
        "history_length_bucket": dict(
            sorted(Counter(candidate.history_length_bucket for candidate in candidates).items())
        ),
        "personalization_evidence": dict(
            sorted(Counter(candidate.evidence_label for candidate in candidates).items())
        ),
        "dialogue_domain": dict(sorted(Counter(candidate.domain for candidate in candidates).items())),
        "tool_action_category": dict(sorted(Counter(candidate.tool_action_category for candidate in candidates).items())),
    }
