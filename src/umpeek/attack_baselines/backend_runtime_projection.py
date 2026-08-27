from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from umpeek.attack_baselines.metrics import normalize_user_semantic_state
from umpeek.exp1_whitebox.schema import clone_json


BACKEND_RUNTIME_PROJECTION_VERSION = "backend_runtime_projection_v1"


def project_user_model_for_backend(
    user_model: Any,
    *,
    backend: str,
    benchmark: str = "",
    task_domain: str = "",
    task_id: str = "",
    user_id: str = "",
) -> dict[str, Any]:
    """Project latent user-model atoms into a backend-shaped runtime S(u,x)."""

    backend_key = _backend_key(backend)
    normalized = normalize_user_semantic_state(user_model)
    facts = _items(normalized, "facts")
    preferences = _items(normalized, "preferences")
    constraints = _items(normalized, "constraints")
    relations = _items(normalized, "relations")
    tool_state = _items(normalized, "tool_state")
    salt = "|".join([backend_key, str(benchmark), str(task_domain), str(task_id), str(user_id)])

    if backend_key == "mem0":
        projected = _project_mem0(
            facts=facts,
            preferences=preferences,
            constraints=constraints,
            relations=relations,
            tool_state=tool_state,
            salt=salt,
            task_domain=task_domain,
        )
    elif backend_key == "graphiti":
        projected = _project_graphiti(
            facts=facts,
            preferences=preferences,
            constraints=constraints,
            relations=relations,
            tool_state=tool_state,
            salt=salt,
            task_domain=task_domain,
            user_id=user_id,
            task_id=task_id,
        )
    else:
        projected = _project_langmem(
            facts=facts,
            preferences=preferences,
            constraints=constraints,
            relations=relations,
            tool_state=tool_state,
            salt=salt,
            task_domain=task_domain,
        )

    projected["confidence"] = _confidence(user_model)
    if isinstance(user_model, Mapping) and "replayed_behavior" in user_model:
        projected["replayed_behavior"] = clone_json(user_model["replayed_behavior"])
    projected["metadata"] = {
        "projection_version": BACKEND_RUNTIME_PROJECTION_VERSION,
        "backend": backend_key,
        "benchmark": str(benchmark),
        "task_domain": str(task_domain),
        "task_id": str(task_id),
        "source_item_count": int(normalized.get("item_count", 0) or 0),
        "excluded_count": int(normalized.get("excluded_count", 0) or 0),
    }
    projected["raw_text"] = _render(projected)
    return projected


def merge_user_models(*models: Any) -> dict[str, Any]:
    merged = {"facts": [], "preferences": [], "constraints": [], "relations": [], "tool_state": []}
    for model in models:
        normalized = normalize_user_semantic_state(model)
        for category in ("facts", "preferences", "constraints", "relations", "tool_state"):
            for item in _items(normalized, category):
                if item not in merged[category]:
                    merged[category].append(item)
    merged["confidence"] = 1.0
    merged["raw_text"] = _render(merged)
    return merged


def _backend_key(backend: str) -> str:
    text = str(backend or "").strip().lower()
    if text in {"mem0", "memory0"}:
        return "mem0"
    if text in {"graphiti", "zep", "zep/graphiti"}:
        return "graphiti"
    return "langmem"


def _items(model: Mapping[str, Any], category: str) -> list[str]:
    value = model.get(category)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    out: list[str] = []
    for item in value:
        text = _canonical_source_item(" ".join(str(item or "").strip().split()))
        if text and text not in out:
            out.append(text)
    return out


def _canonical_source_item(text: str) -> str:
    text = str(text or "").strip()
    lowered = text.lower()
    if lowered.startswith("personalens_affinity="):
        value = text.split("=", 1)[1].strip()
        if "|" in value:
            value = value.split("|", 1)[1].strip()
        return value
    if lowered.startswith("personalens_response="):
        return text.split("=", 1)[1].strip()
    return text


def _select(items: Sequence[str], *, cap: int, salt: str, category: str) -> list[str]:
    ranked = sorted(
        [str(item) for item in items if str(item).strip()],
        key=lambda item: (
            hashlib.sha256(f"{salt}|{category}|{item}".encode("utf-8")).hexdigest(),
            item,
        ),
    )
    return ranked[: max(0, cap)]


def _label(task_domain: str) -> str:
    text = " ".join(str(task_domain or "current task").strip().split())
    return text if text else "current task"


def _project_mem0(
    *,
    facts: Sequence[str],
    preferences: Sequence[str],
    constraints: Sequence[str],
    relations: Sequence[str],
    tool_state: Sequence[str],
    salt: str,
    task_domain: str,
) -> dict[str, Any]:
    domain = _label(task_domain)
    selected_facts = _select(facts, cap=4, salt=salt, category="mem0_facts")
    selected_preferences = _select(preferences, cap=7, salt=salt, category="mem0_preferences")
    selected_constraints = _select(constraints, cap=3, salt=salt, category="mem0_constraints")
    selected_relations = _select(relations, cap=2, salt=salt, category="mem0_relations")
    selected_tool_state = _select_tool_state(tool_state, cap=4, salt=salt, category="mem0_tool_state")
    replay_tool_state = _replay_tool_state(selected_tool_state)
    return {
        "facts": [f"retrieved memory fact for {domain}: {item}" for item in selected_facts],
        "preferences": [f"retrieved memory preference for {domain}: {item}" for item in selected_preferences],
        "constraints": [f"retrieved memory constraint for {domain}: {item}" for item in selected_constraints],
        "relations": [f"(user,retrieved_memory_relation,{item})" for item in selected_relations],
        "tool_state": [
            *[f"mem0 recovered action cue {_safe_carrier_text(item)}" for item in selected_tool_state if not _is_nonsemantic_tool_item(item)],
            *replay_tool_state,
        ],
    }


def _project_graphiti(
    *,
    facts: Sequence[str],
    preferences: Sequence[str],
    constraints: Sequence[str],
    relations: Sequence[str],
    tool_state: Sequence[str],
    salt: str,
    task_domain: str,
    user_id: str,
    task_id: str,
) -> dict[str, Any]:
    domain = _label(task_domain)
    subject = _safe_node(user_id, fallback="user")
    task = _safe_node(task_id, fallback="task")
    selected_facts = _select(facts, cap=5, salt=salt, category="graphiti_facts")
    selected_preferences = _select(preferences, cap=5, salt=salt, category="graphiti_preferences")
    selected_constraints = _select(constraints, cap=2, salt=salt, category="graphiti_constraints")
    selected_relations = _select(relations, cap=5, salt=salt, category="graphiti_relations")
    selected_tool_state = _select_tool_state(tool_state, cap=5, salt=salt, category="graphiti_tool_state")
    replay_tool_state = _replay_tool_state(selected_tool_state)
    graph_relations = [
        f"({subject},current_{domain}_fact,{item})" for item in selected_facts[:3]
    ] + [
        f"({subject},current_{domain}_preference,{item})" for item in selected_preferences[:3]
    ] + [
        f"({subject},current_{task}_relation,{item})" for item in selected_relations
    ]
    return {
        "facts": [f"current graph fact for {domain}: {item}" for item in selected_facts],
        "preferences": [f"current graph preference for {domain}: {item}" for item in selected_preferences],
        "constraints": [f"current graph constraint for {domain}: {item}" for item in selected_constraints],
        "relations": graph_relations,
        "tool_state": [
            *[f"graphiti visible action support {_safe_carrier_text(item)}" for item in selected_tool_state if not _is_nonsemantic_tool_item(item)],
            *replay_tool_state,
        ],
    }


def _project_langmem(
    *,
    facts: Sequence[str],
    preferences: Sequence[str],
    constraints: Sequence[str],
    relations: Sequence[str],
    tool_state: Sequence[str],
    salt: str,
    task_domain: str,
) -> dict[str, Any]:
    domain = _label(task_domain)
    selected_facts = _select(facts, cap=4, salt=salt, category="langmem_facts")
    selected_preferences = _select(preferences, cap=5, salt=salt, category="langmem_preferences")
    selected_constraints = _select(constraints, cap=5, salt=salt, category="langmem_constraints")
    selected_relations = _select(relations, cap=3, salt=salt, category="langmem_relations")
    selected_tool_state = _select_tool_state(tool_state, cap=5, salt=salt, category="langmem_tool_state")
    replay_tool_state = _replay_tool_state(selected_tool_state)
    return {
        "facts": [f"semantic profile fact in {domain}: {item}" for item in selected_facts],
        "preferences": [f"semantic profile preference in {domain}: {item}" for item in selected_preferences],
        "constraints": [f"procedural memory constraint in {domain}: {item}" for item in selected_constraints],
        "relations": [f"(semantic_memory,links_to_user_state,{item})" for item in selected_relations],
        "tool_state": [
            *[f"langmem active section cue {_safe_carrier_text(item)}" for item in selected_tool_state if not _is_nonsemantic_tool_item(item)],
            *replay_tool_state,
        ],
    }


def _replay_tool_state(items: Sequence[str]) -> list[str]:
    def sort_key(item: str) -> tuple[int, str]:
        text = str(item).strip().lower()
        if text.startswith("tool_name=") or text.startswith("action_name="):
            return (0, text)
        return (1, text)

    return sorted(
        [str(item) for item in items if str(item).strip() and not _is_nonsemantic_tool_item(str(item))],
        key=sort_key,
    )


def _select_tool_state(items: Sequence[str], *, cap: int, salt: str, category: str) -> list[str]:
    critical_prefixes = (
        "tool_name=",
        "action_name=",
        "music_name=",
        "volume_level=",
        "location=",
        "city=",
        "destination_city=",
        "origin_city=",
        "category=",
        "query=",
    )
    cleaned = [
        str(item).strip()
        for item in items
        if str(item).strip() and not _is_nonsemantic_tool_item(str(item))
    ]
    critical = [
        item
        for item in cleaned
        if any(item.lower().startswith(prefix) for prefix in critical_prefixes)
    ]
    rest = [item for item in cleaned if item not in critical]
    selected: list[str] = []
    for item in sorted(critical, key=lambda value: (not value.lower().startswith("tool_name="), value.lower())):
        if item not in selected:
            selected.append(item)
    for item in _select(rest, cap=max(0, cap - len(selected)), salt=salt, category=category):
        if item not in selected:
            selected.append(item)
    return selected


def _safe_carrier_text(item: str) -> str:
    return str(item).replace("=", " is ").replace(":", " ").strip()


def _is_nonsemantic_tool_item(item: str) -> bool:
    text = str(item or "").strip().lower()
    return (
        not text
        or "injection_enabled" in text
        or "mem_eval2_" in text
        or text.startswith("memory_id=")
        or text.startswith("selected_memory_ids")
    )


def _safe_node(value: str, *, fallback: str) -> str:
    text = str(value or fallback).strip().lower()
    cleaned = "".join(char if char.isalnum() else "_" for char in text).strip("_")
    return cleaned or fallback


def _confidence(user_model: Any) -> float:
    if isinstance(user_model, Mapping):
        try:
            return float(user_model.get("confidence", 1.0))
        except (TypeError, ValueError):
            return 1.0
    return 1.0


def _render(model: Mapping[str, Any]) -> str:
    lines: list[str] = []
    for category in ("facts", "preferences", "constraints", "relations", "tool_state"):
        values = model.get(category)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            continue
        if values:
            lines.append(f"{category}:")
            lines.extend(f"- {item}" for item in values if str(item).strip())
    return "\n".join(lines)
