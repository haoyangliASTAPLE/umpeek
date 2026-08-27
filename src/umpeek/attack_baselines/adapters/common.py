from __future__ import annotations

import json
import re
from typing import Any, Iterable, Mapping, Sequence

from umpeek.attack_baselines.schema import blank_predicted_user_model


_TOKEN_RE = re.compile(r"\w+|[^\s\w]", re.UNICODE)


def normalize_text(value: Any) -> str:
    text = str(value or "").strip()
    return re.sub(r"\s+", " ", text)


def estimate_token_count(text: str) -> int:
    return len(_TOKEN_RE.findall(text or ""))


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        normalized = normalize_text(item)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(normalized)
    return output


def visible_assistant_text(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in reversed(list(messages)):
        if str(message.get("role") or "").lower() == "assistant":
            content = normalize_text(message.get("content"))
            if content:
                return content
    return ""


def _stringify_nested(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        rows: list[str] = []
        text_value = value.get("text")
        if text_value not in (None, ""):
            rows.append(str(text_value))
        if "normalized_args" in value and isinstance(value["normalized_args"], Mapping):
            for key, item in value["normalized_args"].items():
                rows.append(f"{key}={item}")
        for key in ("tool_name", "memory_id", "pref_type", "query", "reason"):
            if key in value and value[key] not in (None, ""):
                rows.append(f"{key}={value[key]}")
        if not rows:
            rows.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
        return rows
    if isinstance(value, Sequence):
        rows: list[str] = []
        for item in value:
            rows.extend(_stringify_nested(item))
        return rows
    return [str(value)]


def extract_evidence_snippets(
    values: Sequence[Any],
    *,
    max_items: int = 12,
    min_length: int = 8,
) -> list[str]:
    raw_rows: list[str] = []
    for value in values:
        for row in _stringify_nested(value):
            text = normalize_text(row)
            if not text:
                continue
            segments = re.split(r"[\n;]+", text)
            for segment in segments:
                normalized = normalize_text(segment.strip(" -*"))
                if len(normalized) < min_length:
                    continue
                raw_rows.append(normalized)
                if len(raw_rows) >= max_items * 3:
                    break
            if len(raw_rows) >= max_items * 3:
                break
        if len(raw_rows) >= max_items * 3:
            break
    return dedupe_preserve_order(raw_rows)[:max_items]


def classify_snippet(snippet: str) -> str:
    lowered = snippet.lower()
    if lowered.count("|") == 2:
        return "relations"
    if any(token in lowered for token in ("memory_id=", "tool_name=", "query=", "selected_", "valid_fact_ids")):
        return "tool_state"
    if any(token in lowered for token in ("avoid ", "cannot", "can't", "history of", "allergy", "condition", "constraint")):
        return "constraints"
    if any(token in lowered for token in ("prefer", "like ", "likes ", "favorite", "favourite", "enjoy", "usually", "often")):
        return "preferences"
    return "facts"


def build_prediction_payload(
    snippets: Sequence[str],
    *,
    task_prompt: str,
    visible_tool_results: Sequence[Mapping[str, Any]] = (),
    max_items_per_section: int = 6,
    include_task_context_in_raw_text: bool = False,
) -> dict[str, Any]:
    payload = blank_predicted_user_model()
    sections = {
        "facts": [],
        "preferences": [],
        "constraints": [],
        "relations": [],
        "tool_state": [],
    }
    for snippet in dedupe_preserve_order(snippets):
        category = classify_snippet(snippet)
        bucket = sections[category]
        if snippet not in bucket and len(bucket) < max_items_per_section:
            bucket.append(snippet)

    if visible_tool_results and len(sections["tool_state"]) < max_items_per_section:
        for item in extract_evidence_snippets(list(visible_tool_results), max_items=max_items_per_section):
            if item not in sections["tool_state"]:
                sections["tool_state"].append(item)
            if len(sections["tool_state"]) >= max_items_per_section:
                break

    raw_lines = []
    if include_task_context_in_raw_text:
        raw_lines.extend(["Task:", f"- {normalize_text(task_prompt)}"])
    for section_name, title in (
        ("facts", "Facts"),
        ("preferences", "Preferences"),
        ("constraints", "Constraints"),
        ("relations", "Relations"),
        ("tool_state", "Tool state"),
    ):
        if not sections[section_name]:
            continue
        raw_lines.append(f"{title}:")
        raw_lines.extend(f"- {item}" for item in sections[section_name])

    payload.update(sections)
    payload["raw_text"] = "\n".join(raw_lines)
    recovered_count = sum(len(sections[name]) for name in sections)
    payload["confidence"] = round(min(0.95, 0.2 + 0.12 * recovered_count), 3) if recovered_count else 0.05
    return payload
