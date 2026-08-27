from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from huggingface_hub import HfApi, hf_hub_download

from .schema import TaskRecord, UserRecord


PERSONALENS_BENCHMARK = "PersonaLens"
PERSONALENS_DATASET_REPO = "AmazonScience/PersonaLens"
PERSONALENS_LICENSE = "CC-BY-NC-4.0"
PERSONALENS_SMOKE_TASK_IDS = ("SD-Buses-task-1",)
PERSONALENS_PUBLIC_SPLIT_LABEL = "public_hf_profile_tasks"

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class PersonaLensSmokeDataset:
    users: list[UserRecord]
    tasks: list[TaskRecord]
    swap_user_map: dict[str, str]
    swap_user_by_sample: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def personalens_sample_key(user_id: str, task_id: str) -> str:
    return f"{user_id}::{task_id}"


def list_personalens_local_user_ids(dataset_root: Path) -> list[str]:
    profile_root = dataset_root / "data" / "profile"
    if not profile_root.exists():
        return []
    user_ids: list[str] = []
    for user_dir in profile_root.iterdir():
        if not user_dir.is_dir():
            continue
        if (user_dir / "profile.json").exists() and (user_dir / "tasks.json").exists():
            user_ids.append(user_dir.name)
    return sorted(user_ids, key=_user_sort_key)


def download_personalens_dataset_to_local(
    dataset_root: Path,
    *,
    dataset_repo: str = PERSONALENS_DATASET_REPO,
    user_limit: int | None = None,
) -> list[str]:
    dataset_root.mkdir(parents=True, exist_ok=True)
    downloaded_files: list[str] = []
    user_ids = list_personalens_repo_user_ids(dataset_repo)
    if user_limit is not None and user_limit > 0:
        user_ids = user_ids[:user_limit]
    for user_id in user_ids:
        for filename in (
            f"data/profile/{user_id}/profile.json",
            f"data/profile/{user_id}/tasks.json",
        ):
            destination = dataset_root / filename
            if destination.exists():
                continue
            source_path = Path(
                hf_hub_download(
                    repo_id=dataset_repo,
                    filename=filename,
                    repo_type="dataset",
                )
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, destination)
            downloaded_files.append(str(destination))
    return downloaded_files


def build_personalens_local_dataset(
    dataset_root: Path,
    *,
    dataset_repo: str = PERSONALENS_DATASET_REPO,
    user_ids: Sequence[str] | None = None,
    task_ids: Sequence[str] | None = None,
    max_tasks_per_user: int | None = None,
) -> PersonaLensSmokeDataset:
    selected_user_ids = list(user_ids) if user_ids is not None else list_personalens_local_user_ids(dataset_root)
    selected_task_ids = {str(task_id).strip() for task_id in task_ids or () if str(task_id).strip()}
    users: list[UserRecord] = []
    task_records: list[TaskRecord] = []

    for user_id in selected_user_ids:
        profile_path = dataset_root / "data" / "profile" / user_id / "profile.json"
        tasks_path = dataset_root / "data" / "profile" / user_id / "tasks.json"
        if not profile_path.exists() or not tasks_path.exists():
            continue
        profile_payload = _load_local_json(profile_path)
        tasks_payload = _load_local_json(tasks_path)
        normalized_affinities = _normalize_affinities(profile_payload.get("affinities", {}))
        interactions = {
            str(domain): str(text)
            for domain, text in dict(profile_payload.get("interactions", {})).items()
            if str(text).strip()
        }
        users.append(
            UserRecord(
                user_id=user_id,
                profile={
                    "demographics": dict(profile_payload.get("demographics", {})),
                    "interests": dict(profile_payload.get("interests", {})),
                },
                metadata={
                    "affinities": normalized_affinities,
                    "interactions": interactions,
                    "profile_source": str(profile_path),
                    "tasks_source": str(tasks_path),
                },
            )
        )

        raw_task_specs = [
            dict(task_spec)
            for task_spec in dict(tasks_payload).values()
            if isinstance(task_spec, dict) and str(task_spec.get("task_id", "")).strip()
        ]
        raw_task_specs.sort(key=lambda item: str(item.get("task_id", "")).strip())
        if max_tasks_per_user is not None and max_tasks_per_user > 0:
            raw_task_specs = raw_task_specs[:max_tasks_per_user]

        for task_spec in raw_task_specs:
            task_id = str(task_spec.get("task_id", "")).strip()
            if selected_task_ids and task_id not in selected_task_ids:
                continue
            expected_affinities = _resolve_expected_affinities(normalized_affinities, task_spec)
            if not expected_affinities:
                continue
            relevant_domains = _relevant_domains(task_spec)
            relevant_affinity_types = [
                str(item).strip()
                for item in task_spec.get("Relevant Affinity Types", [])
                if str(item).strip()
            ]
            situations = {
                str(key): value
                for key, value in dict(task_spec.get("situations", {})).items()
                if str(value).strip()
            }
            search_terms = _build_search_terms(
                relevant_domains=relevant_domains,
                relevant_affinity_types=relevant_affinity_types,
                expected_affinities=expected_affinities,
                situations=situations,
            )
            prompt = _build_prompt(task_spec, situations)
            dialogue_domain = _dialogue_domain_label(task_spec, relevant_domains)
            task_oriented_intent = _task_oriented_intent_label(task_spec)
            profile_attribute_type = _profile_attribute_type_label(relevant_affinity_types)
            history_length_estimate = _estimate_history_length(profile_payload, task_spec)
            personalization_strength_score = _personalization_strength_score(
                expected_affinities=expected_affinities,
                relevant_domains=relevant_domains,
                situations=situations,
            )
            task_records.append(
                TaskRecord(
                    user_id=user_id,
                    task_id=task_id,
                    benchmark=PERSONALENS_BENCHMARK,
                    task_type="open",
                    prompt=prompt,
                    gold_label=None,
                    metadata={
                        "task_description": str(task_spec.get("Task Description", "")),
                        "user_intent": str(task_spec.get("User Intent", "")),
                        "task_goal": str(task_spec.get("Task Goal", "")),
                        "situations": situations,
                        "relevant_domains": relevant_domains,
                        "relevant_affinity_types": relevant_affinity_types,
                        "expected_affinities": expected_affinities,
                        "search_terms": search_terms,
                        "dialogue_domain": dialogue_domain,
                        "task_oriented_intent": task_oriented_intent,
                        "profile_attribute_type": profile_attribute_type,
                        "benchmark_split": PERSONALENS_PUBLIC_SPLIT_LABEL,
                        "history_length_estimate": history_length_estimate,
                        "history_length_bin": _history_length_bin(history_length_estimate),
                        "personalization_strength_score": personalization_strength_score,
                        "personalization_strength_bin": _personalization_strength_bin(personalization_strength_score),
                        "answer_judge_label_bin": _answer_judge_label_bin(expected_affinities),
                        "profile_source": str(profile_path),
                        "tasks_source": str(tasks_path),
                    },
                )
            )

    task_records.sort(key=lambda task: (task.user_id, task.task_id))
    swap_user_by_sample = _build_swap_user_by_sample(task_records)
    normalized_tasks: list[TaskRecord] = []
    for task in task_records:
        sample_key = personalens_sample_key(task.user_id, task.task_id)
        swap_user_id = swap_user_by_sample.get(sample_key)
        normalized_tasks.append(
            TaskRecord(
                user_id=task.user_id,
                task_id=task.task_id,
                benchmark=task.benchmark,
                task_type=task.task_type,
                prompt=task.prompt,
                gold_label=task.gold_label,
                metadata={
                    **task.metadata,
                    "sample_key": sample_key,
                    "swap_user_id": swap_user_id,
                    "swap_status": "available" if swap_user_id else "swap_unavailable",
                },
            )
        )

    swap_user_map: dict[str, str] = {}
    for sample_key, swap_user_id in swap_user_by_sample.items():
        user_id = sample_key.split("::", 1)[0]
        swap_user_map.setdefault(user_id, swap_user_id)

    metadata = {
        "dataset_repo": dataset_repo,
        "dataset_root": str(dataset_root),
        "license": PERSONALENS_LICENSE,
        "benchmark_split": PERSONALENS_PUBLIC_SPLIT_LABEL,
        "selected_user_ids": [user.user_id for user in users],
        "selected_task_ids": sorted({task.task_id for task in normalized_tasks}),
        "user_count": len(users),
        "task_count": len(normalized_tasks),
        "swap_available_count": len(swap_user_by_sample),
        "swap_unavailable_count": len(normalized_tasks) - len(swap_user_by_sample),
    }
    return PersonaLensSmokeDataset(
        users=users,
        tasks=normalized_tasks,
        swap_user_map=swap_user_map,
        swap_user_by_sample=swap_user_by_sample,
        metadata=metadata,
    )


def build_personalens_task_rows_from_local(
    dataset_root: Path,
    *,
    dataset_repo: str = PERSONALENS_DATASET_REPO,
    user_ids: Sequence[str] | None = None,
    task_ids: Sequence[str] | None = None,
    max_tasks_per_user: int | None = None,
) -> list[dict[str, Any]]:
    dataset = build_personalens_local_dataset(
        dataset_root,
        dataset_repo=dataset_repo,
        user_ids=user_ids,
        task_ids=task_ids,
        max_tasks_per_user=max_tasks_per_user,
    )
    rows: list[dict[str, Any]] = []
    for row_index, task in enumerate(dataset.tasks):
        rows.append(
            {
                "__row_index": row_index,
                "sample_key": personalens_sample_key(task.user_id, task.task_id),
                "user_id": task.user_id,
                "task_id": task.task_id,
                "benchmark": task.benchmark,
                "task_type": task.task_type,
                "prompt": task.prompt,
                "benchmark_split": str(task.metadata.get("benchmark_split", PERSONALENS_PUBLIC_SPLIT_LABEL)),
                "dialogue_domain": str(task.metadata.get("dialogue_domain", "unknown")),
                "task_oriented_intent": str(task.metadata.get("task_oriented_intent", "unknown")),
                "profile_attribute_type": str(task.metadata.get("profile_attribute_type", "unknown")),
                "history_length_estimate": int(task.metadata.get("history_length_estimate", 0) or 0),
                "history_length_bin": str(task.metadata.get("history_length_bin", "unknown")),
                "personalization_strength_score": float(task.metadata.get("personalization_strength_score", 0.0) or 0.0),
                "personalization_strength_bin": str(task.metadata.get("personalization_strength_bin", "unknown")),
                "answer_judge_label_bin": str(task.metadata.get("answer_judge_label_bin", "unknown")),
                "swap_user_id": task.metadata.get("swap_user_id"),
                "swap_status": str(task.metadata.get("swap_status", "swap_unavailable")),
                "relevant_domains": list(task.metadata.get("relevant_domains", [])),
                "relevant_affinity_types": list(task.metadata.get("relevant_affinity_types", [])),
                "expected_affinity_count": len(task.metadata.get("expected_affinities", [])),
            }
        )
    return rows


def canonicalize_personalens_label(text: str) -> str:
    normalized = text.strip().lower()
    normalized = normalized.replace("favourite", "favorite")
    normalized = normalized.replace("behaviour", "behavior")
    return _NON_ALNUM_RE.sub("", normalized)


def build_personalens_smoke_dataset(
    *,
    dataset_repo: str = PERSONALENS_DATASET_REPO,
    smoke_user_count: int = 4,
    task_ids: Sequence[str] = PERSONALENS_SMOKE_TASK_IDS,
) -> PersonaLensSmokeDataset:
    if smoke_user_count <= 1 or smoke_user_count % 2 != 0:
        raise ValueError("PersonaLens smoke runs require an even smoke_user_count >= 2.")
    selected_task_ids = tuple(task_ids)
    if not selected_task_ids:
        raise ValueError("At least one PersonaLens task id is required.")

    user_ids = _list_personalens_user_ids(dataset_repo)
    selected_rows: list[tuple[UserRecord, dict[str, Any]]] = []
    seen_signatures: set[str] = set()

    for user_id in user_ids:
        profile_payload = _download_json(dataset_repo, f"data/profile/{user_id}/profile.json")
        tasks_payload = _download_json(dataset_repo, f"data/profile/{user_id}/tasks.json")
        tasks_by_id = {
            task_spec.get("task_id"): task_spec
            for task_spec in tasks_payload.values()
            if isinstance(task_spec, dict) and task_spec.get("task_id")
        }
        if not all(task_id in tasks_by_id for task_id in selected_task_ids):
            continue

        normalized_affinities = _normalize_affinities(profile_payload.get("affinities", {}))
        selected_specs: dict[str, Any] = {}
        signature_parts: list[str] = []
        for task_id in selected_task_ids:
            task_spec = dict(tasks_by_id[task_id])
            expected_affinities = _resolve_expected_affinities(normalized_affinities, task_spec)
            if not expected_affinities:
                break
            task_spec["_expected_affinities"] = expected_affinities
            selected_specs[task_id] = task_spec
            signature_parts.append(_signature_from_expected_affinities(expected_affinities))
        if len(selected_specs) != len(selected_task_ids):
            continue

        signature = "||".join(signature_parts)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        user_record = UserRecord(
            user_id=user_id,
            profile={
                "demographics": dict(profile_payload.get("demographics", {})),
                "interests": dict(profile_payload.get("interests", {})),
            },
            metadata={
                "affinities": normalized_affinities,
                "interactions": {
                    str(domain): str(text)
                    for domain, text in dict(profile_payload.get("interactions", {})).items()
                },
                "profile_source": f"hf://{dataset_repo}/data/profile/{user_id}/profile.json",
            },
        )
        selected_rows.append((user_record, selected_specs))
        if len(selected_rows) == smoke_user_count:
            break

    if len(selected_rows) < smoke_user_count:
        raise RuntimeError(
            f"Only found {len(selected_rows)} PersonaLens users with unique signatures for {selected_task_ids}."
        )

    users = [user for user, _ in selected_rows]
    tasks: list[TaskRecord] = []
    for user, task_specs in selected_rows:
        for task_id in selected_task_ids:
            task_spec = task_specs[task_id]
            expected_affinities = list(task_spec.get("_expected_affinities", []))
            relevant_domains = _relevant_domains(task_spec)
            relevant_affinity_types = [
                str(item) for item in task_spec.get("Relevant Affinity Types", [])
            ]
            situations = dict(task_spec.get("situations", {}))
            search_terms = _build_search_terms(
                relevant_domains=relevant_domains,
                relevant_affinity_types=relevant_affinity_types,
                expected_affinities=expected_affinities,
                situations=situations,
            )
            prompt = _build_prompt(task_spec, situations)
            tasks.append(
                TaskRecord(
                    user_id=user.user_id,
                    task_id=task_id,
                    benchmark=PERSONALENS_BENCHMARK,
                    task_type="open",
                    prompt=prompt,
                    gold_label=None,
                    metadata={
                        "task_description": str(task_spec.get("Task Description", "")),
                        "user_intent": str(task_spec.get("User Intent", "")),
                        "task_goal": str(task_spec.get("Task Goal", "")),
                        "situations": situations,
                        "relevant_domains": relevant_domains,
                        "relevant_affinity_types": relevant_affinity_types,
                        "expected_affinities": expected_affinities,
                        "search_terms": search_terms,
                        "evaluation_mode": "offline_dry_run",
                        "judge_backend": "amazon_bedrock_required",
                        "judge_status": "not_run",
                    },
                )
            )

    swap_user_map: dict[str, str] = {}
    for index in range(0, len(users), 2):
        left_user = users[index]
        right_user = users[index + 1]
        swap_user_map[left_user.user_id] = right_user.user_id
        swap_user_map[right_user.user_id] = left_user.user_id

    metadata = {
        "dataset_repo": dataset_repo,
        "license": PERSONALENS_LICENSE,
        "evaluation_mode": "offline_dry_run",
        "judge_backend": "amazon_bedrock_required",
        "judge_status": "not_run",
        "selected_user_ids": [user.user_id for user in users],
        "selected_task_ids": list(selected_task_ids),
        "smoke_user_count": smoke_user_count,
    }
    swap_user_by_sample = {
        personalens_sample_key(task.user_id, task.task_id): swap_user_map[task.user_id]
        for task in tasks
        if task.user_id in swap_user_map
    }
    return PersonaLensSmokeDataset(
        users=users,
        tasks=tasks,
        swap_user_map=swap_user_map,
        swap_user_by_sample=swap_user_by_sample,
        metadata=metadata,
    )


def list_personalens_repo_user_ids(dataset_repo: str) -> list[str]:
    api = HfApi()
    user_files: dict[str, set[str]] = {}
    for file_path in api.list_repo_files(dataset_repo, repo_type="dataset"):
        parts = file_path.split("/")
        if len(parts) == 4 and parts[:2] == ["data", "profile"] and parts[3] in {"profile.json", "tasks.json"}:
            user_files.setdefault(parts[2], set()).add(parts[3])
    return sorted(
        [user_id for user_id, files in user_files.items() if {"profile.json", "tasks.json"}.issubset(files)],
        key=_user_sort_key,
    )


def _list_personalens_user_ids(dataset_repo: str) -> list[str]:
    return list_personalens_repo_user_ids(dataset_repo)


def _user_sort_key(user_id: str) -> tuple[int, str]:
    numeric_part = "".join(character for character in user_id if character.isdigit())
    return (int(numeric_part or 0), user_id)


def _download_json(dataset_repo: str, filename: str) -> dict[str, Any]:
    file_path = Path(
        hf_hub_download(
            repo_id=dataset_repo,
            filename=filename,
            repo_type="dataset",
        )
    )
    return json.loads(file_path.read_text(encoding="utf-8"))


def _load_local_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_affinities(raw_affinities: dict[str, Any]) -> dict[str, dict[str, tuple[str, ...]]]:
    normalized: dict[str, dict[str, tuple[str, ...]]] = {}
    for domain, affinity_map in dict(raw_affinities).items():
        if not isinstance(affinity_map, dict):
            continue
        domain_entries: dict[str, tuple[str, ...]] = {}
        for key, value in affinity_map.items():
            values = _coerce_values(value)
            if values:
                domain_entries[str(key)] = values
        if domain_entries:
            normalized[str(domain)] = domain_entries
    return normalized


def _coerce_values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, dict):
        flattened: list[str] = []
        for key, item in value.items():
            rendered = str(item).strip()
            if rendered:
                flattened.append(f"{key}: {rendered}")
        return tuple(flattened)
    rendered = str(value).strip()
    return (rendered,) if rendered else ()


def _relevant_domains(task_spec: dict[str, Any]) -> list[str]:
    domains = [str(item) for item in task_spec.get("Relevant Domains", []) if str(item).strip()]
    if domains:
        return domains
    task_id = str(task_spec.get("task_id", ""))
    parts = task_id.split("-")
    if len(parts) >= 3 and parts[1].strip():
        return [parts[1].strip()]
    return []


def _resolve_expected_affinities(
    normalized_affinities: dict[str, dict[str, tuple[str, ...]]],
    task_spec: dict[str, Any],
) -> list[dict[str, Any]]:
    relevant_affinity_types = [
        str(item) for item in task_spec.get("Relevant Affinity Types", []) if str(item).strip()
    ]
    expected_affinities: list[dict[str, Any]] = []
    for domain in _relevant_domains(task_spec):
        domain_affinities = normalized_affinities.get(domain, {})
        canonical_index = {
            canonicalize_personalens_label(original_key): (original_key, values)
            for original_key, values in domain_affinities.items()
        }
        for affinity_type in relevant_affinity_types:
            canonical_key = canonicalize_personalens_label(affinity_type)
            original_pair = canonical_index.get(canonical_key)
            if original_pair is None:
                continue
            original_key, values = original_pair
            expected_affinities.append(
                {
                    "domain": domain,
                    "affinity_key": original_key,
                    "affinity_key_canonical": canonical_key,
                    "values": list(values),
                }
            )
    return expected_affinities


def _signature_from_expected_affinities(expected_affinities: list[dict[str, Any]]) -> str:
    signature_parts = []
    for affinity in expected_affinities:
        signature_parts.append(
            "{domain}:{key}:{values}".format(
                domain=canonicalize_personalens_label(str(affinity.get("domain", ""))),
                key=str(affinity.get("affinity_key_canonical", "")),
                values="|".join(
                    canonicalize_personalens_label(str(value))
                    for value in affinity.get("values", [])
                ),
            )
        )
    return ";".join(sorted(signature_parts))


def _build_search_terms(
    *,
    relevant_domains: list[str],
    relevant_affinity_types: list[str],
    expected_affinities: list[dict[str, Any]],
    situations: dict[str, Any],
) -> list[str]:
    terms: list[str] = []
    terms.extend(relevant_domains)
    terms.extend(relevant_affinity_types)
    for affinity in expected_affinities:
        terms.append(str(affinity.get("affinity_key", "")))
        terms.extend(str(value) for value in affinity.get("values", []))
    terms.extend(str(value) for value in situations.values())
    return [term for term in terms if term]


def _build_prompt(task_spec: dict[str, Any], situations: dict[str, Any]) -> str:
    sections = [
        f"Task Description: {task_spec.get('Task Description', '')}",
        f"User Intent: {task_spec.get('User Intent', '')}",
        f"Task Goal: {task_spec.get('Task Goal', '')}",
    ]
    if situations:
        situation_text = ", ".join(
            f"{key}={value}" for key, value in situations.items() if str(value).strip()
        )
        sections.append(f"Situation: {situation_text}")
    return "\n".join(section for section in sections if section.strip())


def _normalize_label(text: Any, *, fallback: str = "unknown") -> str:
    rendered = str(text or "").strip().lower()
    if not rendered:
        return fallback
    normalized = _NON_ALNUM_RE.sub("_", rendered).strip("_")
    return normalized or fallback


def _word_count(text: Any) -> int:
    return len(re.findall(r"[A-Za-z0-9_]+", str(text or "")))


def _dialogue_domain_label(task_spec: dict[str, Any], relevant_domains: list[str]) -> str:
    if relevant_domains:
        return _normalize_label(relevant_domains[0])
    task_id = str(task_spec.get("task_id", "")).strip()
    parts = task_id.split("-")
    if len(parts) >= 3:
        return _normalize_label(parts[1])
    return "unknown"


def _task_oriented_intent_label(task_spec: dict[str, Any]) -> str:
    for value in (task_spec.get("User Intent"), task_spec.get("Task Goal"), task_spec.get("Task Description")):
        normalized = _normalize_label(value)
        if normalized != "unknown":
            return normalized
    return "unknown"


def _profile_attribute_type_label(relevant_affinity_types: list[str]) -> str:
    if not relevant_affinity_types:
        return "unknown"
    return _normalize_label(relevant_affinity_types[0])


def _estimate_history_length(profile_payload: dict[str, Any], task_spec: dict[str, Any]) -> int:
    interactions = " ".join(str(value) for value in dict(profile_payload.get("interactions", {})).values())
    affinities = " ".join(
        str(item)
        for domain in dict(profile_payload.get("affinities", {})).values()
        for item in (domain.values() if isinstance(domain, dict) else ())
    )
    task_text = " ".join(
        str(task_spec.get(field_name, ""))
        for field_name in ("Task Description", "User Intent", "Task Goal")
    )
    return _word_count(interactions) + _word_count(affinities) + _word_count(task_text)


def _history_length_bin(history_length_estimate: int) -> str:
    if history_length_estimate < 120:
        return "short"
    if history_length_estimate < 260:
        return "medium"
    return "long"


def _personalization_strength_score(
    *,
    expected_affinities: list[dict[str, Any]],
    relevant_domains: list[str],
    situations: dict[str, Any],
) -> float:
    value_count = sum(len(item.get("values", [])) for item in expected_affinities)
    return float(len(expected_affinities) + value_count + len(relevant_domains) + len(situations))


def _personalization_strength_bin(score: float) -> str:
    if score >= 7.0:
        return "high"
    if score >= 4.0:
        return "medium"
    return "low"


def _answer_judge_label_bin(expected_affinities: list[dict[str, Any]]) -> str:
    value_count = sum(len(item.get("values", [])) for item in expected_affinities)
    if value_count <= 1:
        return "single_anchor"
    if value_count <= 3:
        return "focused_multi"
    return "broad_multi"


def _build_swap_user_by_sample(tasks: list[TaskRecord]) -> dict[str, str]:
    exact_groups: dict[tuple[str, str, str, str], list[TaskRecord]] = {}
    task_groups: dict[str, list[TaskRecord]] = {}
    domain_groups: dict[tuple[str, str], list[TaskRecord]] = {}
    for task in tasks:
        exact_key = (
            task.task_id,
            str(task.metadata.get("dialogue_domain", "unknown")),
            str(task.metadata.get("profile_attribute_type", "unknown")),
            str(task.metadata.get("task_oriented_intent", "unknown")),
        )
        exact_groups.setdefault(exact_key, []).append(task)
        task_groups.setdefault(task.task_id, []).append(task)
        domain_groups.setdefault(
            (
                str(task.metadata.get("dialogue_domain", "unknown")),
                str(task.metadata.get("profile_attribute_type", "unknown")),
            ),
            [],
        ).append(task)

    for group in (*exact_groups.values(), *task_groups.values(), *domain_groups.values()):
        group.sort(key=lambda task: (task.user_id, task.task_id))

    swap_map: dict[str, str] = {}
    for task in tasks:
        candidate_buckets = [
            exact_groups.get(
                (
                    task.task_id,
                    str(task.metadata.get("dialogue_domain", "unknown")),
                    str(task.metadata.get("profile_attribute_type", "unknown")),
                    str(task.metadata.get("task_oriented_intent", "unknown")),
                ),
                [],
            ),
            task_groups.get(task.task_id, []),
            domain_groups.get(
                (
                    str(task.metadata.get("dialogue_domain", "unknown")),
                    str(task.metadata.get("profile_attribute_type", "unknown")),
                ),
                [],
            ),
        ]
        swap_user_id = None
        for bucket in candidate_buckets:
            options = [candidate for candidate in bucket if candidate.user_id != task.user_id]
            if not options:
                continue
            swap_user_id = options[0].user_id
            break
        if swap_user_id is not None:
            swap_map[personalens_sample_key(task.user_id, task.task_id)] = swap_user_id
    return swap_map
