from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOCOMO_BENCHMARK = "LoCoMo"
LOCOMO_BENCHMARK_SPLIT = "public_locomo10"
LOCOMO_RELATIVE_DATA_PATH = Path(".vendor/locomo/data/locomo10.json")
LOCOMO_PREPARED_RELATIVE_DATA_PATH = Path("data/benchmarks/LoCoMo/locomo10.json")
_QUESTION_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_LOCOMO_CATEGORY_BIN = {
    1: "profile_fact",
    2: "temporal_when",
    3: "counterfactual_inference",
    4: "episodic_detail",
    5: "binary_visual",
}


@dataclass(frozen=True, slots=True)
class LocomoTaskDescriptor:
    sample_id: str
    qa_index: int
    user_id: str
    task_id: str
    speaker_name: str
    swap_user_id: str | None
    question: str
    question_template: str
    gold_answer: str
    locomo_category: int | None
    question_family: str
    answer_type_bin: str
    temporal_bin: str
    history_tokens: int
    history_bin: str
    categories: tuple[str, ...]
    evidence: tuple[str, ...]
    attribute_key: str | None = None
    importance: float = 1.0


def resolve_locomo_local_data_path(data_root_or_file: Path) -> Path:
    candidate = Path(data_root_or_file)
    if candidate.is_file():
        return candidate.resolve()
    for path in (candidate / "data" / "locomo10.json", candidate / "locomo10.json"):
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(
        f"LoCoMo data file not found under {data_root_or_file}. Expected data/locomo10.json or locomo10.json."
    )


def resolve_locomo_data_path(*, project_root: Path, data_path: Path | None = None) -> Path:
    if data_path is not None:
        return resolve_locomo_local_data_path(data_path)
    for candidate in (
        project_root / LOCOMO_PREPARED_RELATIVE_DATA_PATH,
        project_root / LOCOMO_RELATIVE_DATA_PATH,
    ):
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "LoCoMo data file not found in either the prepared benchmark root or .vendor/locomo."
    )


def build_locomo_task_rows(
    *,
    project_root: Path,
    data_path: Path | None = None,
) -> list[dict[str, Any]]:
    dataset_path = resolve_locomo_data_path(project_root=project_root, data_path=data_path)
    return build_locomo_task_rows_from_local(dataset_path)


def build_locomo_task_rows_from_local(data_root_or_file: Path) -> list[dict[str, Any]]:
    dataset_path = resolve_locomo_local_data_path(data_root_or_file)
    samples = json.loads(dataset_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for sample in samples:
        rows.extend(_build_locomo_task_rows_for_sample(sample))
    return rows


def _build_dialog_index(conversation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    dialog_index: dict[str, dict[str, Any]] = {}
    for session_idx in _session_indices(conversation):
        session_key = f"session_{session_idx}"
        session_date = conversation[f"session_{session_idx}_date_time"]
        for order, dialog in enumerate(conversation[session_key], start=1):
            dialog_index[dialog["dia_id"]] = {
                "session_idx": session_idx,
                "session_date": session_date,
                "order": order,
                "dialog": dialog,
            }
    return dialog_index


def _normalize_question_and_speaker(question: str, speakers: tuple[str, str]) -> tuple[str, str | None]:
    normalized = question.replace("`", "'").replace("’", "'")
    matched_speaker: str | None = None
    for speaker_name in speakers:
        if speaker_name in normalized:
            matched_speaker = speaker_name
            normalized = normalized.replace(speaker_name, "{USER}")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized, matched_speaker


def _session_indices(conversation: dict[str, Any]) -> list[int]:
    indices = []
    for key in conversation:
        if key.startswith("session_") and not key.endswith("_date_time"):
            suffix = key.split("_")[-1]
            if suffix.isdigit():
                indices.append(int(suffix))
    return sorted(indices)


def _user_id(sample_id: str, speaker_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", speaker_name.lower()).strip("_")
    return f"{sample_id}__{slug}"


def _build_locomo_task_rows_for_sample(sample: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for descriptor, _ in _iter_locomo_task_descriptors(sample):
        rows.append(
            {
                "sample_id": descriptor.sample_id,
                "qa_index": descriptor.qa_index,
                "user_id": descriptor.user_id,
                "task_id": descriptor.task_id,
                "speaker_name": descriptor.speaker_name,
                "swap_user_id": descriptor.swap_user_id,
                "swap_status": "available" if descriptor.swap_user_id else "swap_unavailable",
                "benchmark_split": LOCOMO_BENCHMARK_SPLIT,
                "task_type": "open",
                "choice_type": "open_ended",
                "question": descriptor.question,
                "gold_answer": descriptor.gold_answer,
                "question_template": descriptor.question_template,
                "locomo_category": descriptor.locomo_category,
                "locomo_category_bin": _locomo_category_bin(descriptor.locomo_category),
                "question_family": descriptor.question_family,
                "answer_type_bin": descriptor.answer_type_bin,
                "temporal_bin": descriptor.temporal_bin,
                "history_tokens": descriptor.history_tokens,
                "history_length_bin": descriptor.history_bin,
                "evidence_count": len(descriptor.evidence),
                "evidence": list(descriptor.evidence),
                "domain_bin": "temporal_memory",
            }
        )
    return rows


def _iter_locomo_task_descriptors(sample: dict[str, Any]) -> list[tuple[LocomoTaskDescriptor, dict[str, Any]]]:
    conversation = sample["conversation"]
    speakers = (conversation["speaker_a"], conversation["speaker_b"])
    dialog_index = _build_dialog_index(conversation)
    max_session_idx = max(dialog_index[dialog_id]["session_idx"] for dialog_id in dialog_index)
    history_tokens_by_speaker = {
        speaker_name: _locomo_user_history_tokens(sample, speaker_name)
        for speaker_name in speakers
    }
    user_ids = {speaker_name: _user_id(sample["sample_id"], speaker_name) for speaker_name in speakers}
    paired: list[tuple[LocomoTaskDescriptor, dict[str, Any]]] = []
    for qa_index, qa_entry in enumerate(sample.get("qa", []), start=1):
        descriptor = _build_locomo_task_descriptor(
            sample=sample,
            qa_index=qa_index,
            qa_entry=qa_entry,
            speakers=speakers,
            dialog_index=dialog_index,
            max_session_idx=max_session_idx,
            history_tokens_by_speaker=history_tokens_by_speaker,
            user_ids=user_ids,
        )
        if descriptor is None:
            continue
        paired.append((descriptor, qa_entry))
    return paired


def _build_locomo_task_descriptor(
    *,
    sample: dict[str, Any],
    qa_index: int,
    qa_entry: dict[str, Any],
    speakers: tuple[str, str],
    dialog_index: dict[str, dict[str, Any]],
    max_session_idx: int,
    history_tokens_by_speaker: dict[str, int],
    user_ids: dict[str, str],
) -> LocomoTaskDescriptor | None:
    if "answer" not in qa_entry:
        return None
    question = str(qa_entry.get("question", "")).strip()
    if not question:
        return None
    question_template, speaker_name = _normalize_question_and_speaker(question, speakers)
    if speaker_name is None:
        return None
    gold_answer = _normalize_locomo_answer(qa_entry.get("answer"))
    if not gold_answer:
        return None
    locomo_category = _safe_int(qa_entry.get("category"))
    question_family = _locomo_question_family(question, locomo_category)
    answer_type_bin = _locomo_answer_type_bin(gold_answer)
    evidence = tuple(str(dialog_id) for dialog_id in qa_entry.get("evidence", []) if str(dialog_id).strip())
    temporal_bin = _locomo_temporal_bin(evidence, dialog_index, max_session_idx)
    categories = _locomo_memory_categories(
        question_family=question_family,
        answer_type_bin=answer_type_bin,
        locomo_category=locomo_category,
    )
    return LocomoTaskDescriptor(
        sample_id=str(sample["sample_id"]),
        qa_index=qa_index,
        user_id=user_ids[speaker_name],
        task_id=_locomo_task_id(str(sample["sample_id"]), qa_index),
        speaker_name=speaker_name,
        swap_user_id=next((user_id for other_speaker, user_id in user_ids.items() if other_speaker != speaker_name), None),
        question=question,
        question_template=question_template,
        gold_answer=gold_answer,
        locomo_category=locomo_category,
        question_family=question_family,
        answer_type_bin=answer_type_bin,
        temporal_bin=temporal_bin,
        history_tokens=history_tokens_by_speaker.get(speaker_name, 0),
        history_bin=_locomo_history_bin(history_tokens_by_speaker.get(speaker_name, 0)),
        categories=categories,
        evidence=evidence,
        attribute_key=_locomo_attribute_key(question_template, categories),
        importance=_locomo_importance(categories=categories, temporal_bin=temporal_bin),
    )


def _locomo_task_id(sample_id: str, qa_index: int) -> str:
    sample_slug = re.sub(r"[^a-z0-9]+", "_", sample_id.lower()).strip("_")
    return f"locomo_{sample_slug}_{qa_index:04d}"


def _normalize_locomo_answer(value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value).strip()


def _locomo_category_bin(locomo_category: int | None) -> str:
    if locomo_category is None:
        return "unknown"
    return _LOCOMO_CATEGORY_BIN.get(locomo_category, f"category_{locomo_category}")


def _locomo_question_family(question: str, locomo_category: int | None) -> str:
    lowered = question.strip().lower()
    if locomo_category == 2 or lowered.startswith("when "):
        return "when"
    if locomo_category == 1:
        return "profile"
    if locomo_category == 3:
        return "counterfactual"
    if locomo_category == 5:
        return "yes_no"
    if lowered.startswith("who "):
        return "who"
    if lowered.startswith("where "):
        return "where"
    if lowered.startswith("why "):
        return "why"
    if lowered.startswith("how "):
        return "how"
    return "detail"


def _locomo_answer_type_bin(gold_answer: str) -> str:
    lowered = gold_answer.strip().lower()
    if lowered in {"yes", "no"}:
        return "yes_no"
    if lowered.startswith("likely yes") or lowered.startswith("likely no"):
        return "yes_no"
    return "open"


def _locomo_temporal_bin(
    evidence: tuple[str, ...],
    dialog_index: dict[str, dict[str, Any]],
    max_session_idx: int,
) -> str:
    if not evidence:
        return "unknown"
    evidence_sessions = [int(dialog_index[dialog_id]["session_idx"]) for dialog_id in evidence if dialog_id in dialog_index]
    if not evidence_sessions:
        return "unknown"
    gap = max_session_idx - min(evidence_sessions)
    if gap <= 1:
        return "recent"
    if gap <= 3:
        return "mid"
    return "far"


def _locomo_user_history_tokens(sample: dict[str, Any], speaker_name: str) -> int:
    parts: list[str] = []
    conversation = sample["conversation"]
    for session_idx in _session_indices(conversation):
        for dialog in conversation.get(f"session_{session_idx}", []):
            parts.append(str(dialog.get("text", "")))
            if dialog.get("blip_caption"):
                parts.append(str(dialog.get("blip_caption", "")))
        event_bucket = sample.get(f"events_session_{session_idx}", {})
        for event_text in event_bucket.get(speaker_name, []):
            parts.append(str(event_text))
    return len(_QUESTION_TOKEN_RE.findall(" ".join(parts)))


def _locomo_history_bin(history_tokens: int) -> str:
    if history_tokens < 250:
        return "short"
    if history_tokens < 500:
        return "medium"
    return "long"


def _locomo_memory_categories(
    *,
    question_family: str,
    answer_type_bin: str,
    locomo_category: int | None,
) -> tuple[str, ...]:
    if locomo_category == 2 or question_family == "when":
        return ("episodic", "temporal")
    if locomo_category == 1 or question_family == "profile":
        return ("profile", "entity")
    if locomo_category == 3 or question_family == "counterfactual":
        return ("profile", "inference", "preference")
    if locomo_category == 5 or answer_type_bin == "yes_no":
        return ("binary", "preference")
    return ("episodic", "detail")


def _locomo_attribute_key(question_template: str, categories: tuple[str, ...]) -> str | None:
    if not ({"profile", "preference", "entity"} & set(categories)):
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", question_template.lower().replace("{user}", "user")).strip("_")
    return normalized[:64] or None


def _locomo_importance(*, categories: tuple[str, ...], temporal_bin: str) -> float:
    importance = 1.0
    if "temporal" in categories:
        importance += 0.2
    if {"profile", "preference", "entity"} & set(categories):
        importance += 0.1
    if temporal_bin == "recent":
        importance += 0.05
    return importance


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
