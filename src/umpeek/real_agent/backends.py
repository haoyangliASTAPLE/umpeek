from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Mapping, Sequence


BACKEND_ADAPTER_SCHEMA_VERSION = "real_agent_backend_adapter_v1"


@dataclass(frozen=True, slots=True)
class MemoryItem:
    text: str
    category: str = "semantic"
    source: str = "benchmark_gold"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RuntimeMemoryContext:
    backend: str
    user_id: str
    task_id: str
    query: str
    adapter_mode: str
    retrieved_items: tuple[MemoryItem, ...]
    prompt_context: str
    gold_user_model: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_public_metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "adapter_mode": self.adapter_mode,
            "retrieved_item_count": len(self.retrieved_items),
            "runtime_context_hash": _stable_hash(self.prompt_context),
            "backend_adapter_schema_version": BACKEND_ADAPTER_SCHEMA_VERSION,
        }


class MemoryBackendAdapter:
    backend_name = "backend"
    adapter_mode = "local_compatible"

    def materialize(self, *, user_id: str, task_id: str, gold_model: Mapping[str, Any], row: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def retrieve(self, *, user_id: str, task_id: str, query: str, limit: int = 12) -> RuntimeMemoryContext:
        raise NotImplementedError

    def render_items(self, items: Sequence[MemoryItem], *, query: str) -> str:
        """Render evaluator-supplied memory items in this backend's prompt format."""
        raise NotImplementedError


class LocalMemoryBackendAdapter(MemoryBackendAdapter):
    """Local faithful adapter with the same add/search shape as memory backends.

    This is used when the official package needs a managed service, an API key, or an
    unavailable dependency. It preserves backend-specific memory shape and retrieval
    surfaces instead of replaying benchmark behavior.
    """

    backend_name = "local"
    adapter_mode = "local_compatible"

    def __init__(self) -> None:
        self._items_by_user: dict[str, list[MemoryItem]] = {}

    def materialize(self, *, user_id: str, task_id: str, gold_model: Mapping[str, Any], row: Mapping[str, Any]) -> None:
        del task_id, row
        self._items_by_user[user_id] = self._memory_items_from_gold(gold_model)

    def retrieve(self, *, user_id: str, task_id: str, query: str, limit: int = 12) -> RuntimeMemoryContext:
        items = self._items_by_user.get(user_id, [])
        ranked = _rank_items(items, query)[: max(1, int(limit))]
        prompt_context = self._render_prompt_context(ranked, query=query)
        return RuntimeMemoryContext(
            backend=self.backend_name,
            user_id=user_id,
            task_id=task_id,
            query=query,
            adapter_mode=self.adapter_mode,
            retrieved_items=tuple(ranked),
            prompt_context=prompt_context,
            gold_user_model=_gold_model_from_items(ranked, source_backend=self.backend_name),
            metadata={"ranking": "lexical_overlap_plus_category_prior"},
        )

    def _memory_items_from_gold(self, gold_model: Mapping[str, Any]) -> list[MemoryItem]:
        items: list[MemoryItem] = []
        for category in ("facts", "preferences", "constraints", "relations"):
            value = gold_model.get(category)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                for item in value:
                    text = " ".join(str(item or "").split())
                    if text:
                        items.append(MemoryItem(text=text, category=category, metadata={"gold_category": category}))
        tool_state = gold_model.get("tool_state")
        if isinstance(tool_state, Mapping):
            text = json.dumps(tool_state, ensure_ascii=False, sort_keys=True)
            items.append(MemoryItem(text=text, category="tool_state", metadata={"gold_category": "tool_state"}))
        elif isinstance(tool_state, Sequence) and not isinstance(tool_state, (str, bytes, bytearray)):
            for item in tool_state:
                text = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, Mapping) else str(item)
                if text.strip():
                    items.append(MemoryItem(text=text, category="tool_state", metadata={"gold_category": "tool_state"}))
        raw_text = str(gold_model.get("raw_text") or "").strip()
        if raw_text and not items:
            for line in raw_text.splitlines():
                text = " ".join(line.split())
                if text:
                    items.append(MemoryItem(text=text, category="facts", metadata={"gold_category": "raw_text"}))
        return _dedupe_items(items)

    def _render_prompt_context(self, items: Sequence[MemoryItem], *, query: str) -> str:
        del query
        return "\n".join(f"- {item.category}: {item.text}" for item in items)

    def render_items(self, items: Sequence[MemoryItem], *, query: str) -> str:
        return self._render_prompt_context(items, query=query)


class Mem0BackendAdapter(LocalMemoryBackendAdapter):
    backend_name = "Mem0"

    def _render_prompt_context(self, items: Sequence[MemoryItem], *, query: str) -> str:
        del query
        return "\n".join(
            f"retrieved memory fact [{idx + 1}] ({item.category}): {item.text}"
            for idx, item in enumerate(items)
        )


class GraphitiBackendAdapter(LocalMemoryBackendAdapter):
    backend_name = "Graphiti"

    def _render_prompt_context(self, items: Sequence[MemoryItem], *, query: str) -> str:
        del query
        lines = []
        for idx, item in enumerate(items):
            node_id = f"user_memory_{idx + 1}"
            lines.append(
                f"temporal knowledge edge: User -> {item.category} -> {node_id}; fact=\"{item.text}\""
            )
        return "\n".join(lines)


class LangMemBackendAdapter(LocalMemoryBackendAdapter):
    backend_name = "LangMem+LangGraph"

    def _render_prompt_context(self, items: Sequence[MemoryItem], *, query: str) -> str:
        del query
        return "\n".join(
            f"LangGraph store namespace=user_profile semantic_memory.{idx + 1}: {item.category}={item.text}"
            for idx, item in enumerate(items)
        )


def build_backend_adapter(backend: str) -> MemoryBackendAdapter:
    normalized = str(backend or "").strip().lower()
    if normalized == "mem0":
        return Mem0BackendAdapter()
    if normalized == "graphiti":
        return GraphitiBackendAdapter()
    if normalized in {"langmem+langgraph", "langmem", "langgraph"}:
        return LangMemBackendAdapter()
    raise ValueError(f"Unsupported real-agent backend: {backend!r}")


def _rank_items(items: Sequence[MemoryItem], query: str) -> list[MemoryItem]:
    query_tokens = _tokens(query)
    scored: list[tuple[float, int, MemoryItem]] = []
    category_prior = {"tool_state": 2.0, "preferences": 1.5, "facts": 1.2, "constraints": 1.1, "relations": 1.0}
    for index, item in enumerate(items):
        item_tokens = _tokens(item.text)
        overlap = len(query_tokens & item_tokens)
        score = float(overlap) + category_prior.get(item.category, 1.0)
        scored.append((score, -index, item))
    return [item for _score, _idx, item in sorted(scored, key=lambda row: (row[0], row[1]), reverse=True)]


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9_]{2,}", str(text or "").lower())
        if token not in {"the", "and", "for", "with", "that", "this", "you", "your", "user"}
    }


def _stable_hash(value: Any) -> str:
    payload = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _dedupe_items(items: Sequence[MemoryItem]) -> list[MemoryItem]:
    out: list[MemoryItem] = []
    seen: set[str] = set()
    for item in items:
        key = f"{item.category}:{item.text}".lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _gold_model_from_items(items: Sequence[MemoryItem], *, source_backend: str) -> dict[str, Any]:
    model: dict[str, Any] = {
        "facts": [],
        "preferences": [],
        "constraints": [],
        "relations": [],
        "tool_state": [],
        "raw_text": "\n".join(item.text for item in items),
        "metadata": {
            "gold_scope": "runtime_backend_context",
            "source_backend": source_backend,
            "backend_adapter_schema_version": BACKEND_ADAPTER_SCHEMA_VERSION,
        },
    }
    for item in items:
        if item.category in {"facts", "preferences", "constraints", "relations"}:
            model[item.category].append(item.text)
        elif item.category == "tool_state":
            try:
                model["tool_state"].append(json.loads(item.text))
            except json.JSONDecodeError:
                model["tool_state"].append(item.text)
        else:
            model["facts"].append(item.text)
    return clone_json(model)


def clone_json(value: Any) -> Any:
    if is_dataclass(value):
        return clone_json(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): clone_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clone_json(item) for item in value]
    return json.loads(json.dumps(value, ensure_ascii=False))
