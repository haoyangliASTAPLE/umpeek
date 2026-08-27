from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from nltk.stem import PorterStemmer
from transformers import AutoTokenizer

from umpeek.attack_baselines import AttackInput
from umpeek.eval2.matching import atoms_from_user_model, normalize_atom_text, semantic_atom_score
from umpeek.eval2.replay import score_behavior
from umpeek.eval2.runner import _etapp_field_value_matches, _load_etapp_profile_index
from umpeek.eval2.schema import MetricAtom, clone_json
from umpeek.real_agent.backends import MemoryItem, RuntimeMemoryContext, build_backend_adapter
from umpeek.real_agent.llm import QwenVLLMClient, QwenVLLMConfig
from umpeek.real_agent.materializer import _system_prompt, _victim_prompt, run_victim_once


SCHEMA_VERSION = "exp1_exp2_true_interventions_v2"
MODEL_NAME = "Qwen/Qwen3-14B"
BENCHMARK_ORDER = ("PersonaMem-v2", "PersonaLens", "ETAPP", "LoCoMo")
BACKEND_ORDER = ("Mem0", "Graphiti", "LangMem+LangGraph")

_BENCHMARK_DISPLAY = {
    "PersonaMem-v2": "PersonaMem-v2",
    "PersonaMemv2": "PersonaMem-v2",
    "personamem_v2": "PersonaMem-v2",
    "PersonaLens": "PersonaLens",
    "personalens": "PersonaLens",
    "ETAPP": "ETAPP",
    "ETAPP_150x32": "ETAPP",
    "etapp_150x32": "ETAPP",
    "LoCoMo": "LoCoMo",
    "LoCoMo_10conv_1523QA_20speakers": "LoCoMo",
    "locomo_10conv_1523qa_20speakers": "LoCoMo",
}

_SOURCE_CATEGORY_TERMS = {
    "preferences": ("preference", "affinity", "interest", "favorite", "hobby", "habit", "routine"),
    "constraints": ("constraint", "allerg", "avoid", "restriction", "cannot", "must not"),
    "relations": ("relation", "spouse", "family", "friend", "parent", "child", "colleague"),
}

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_PORTER_STEMMER = PorterStemmer()


def stable_hash(value: Any) -> str:
    payload = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def benchmark_display(value: Any) -> str:
    return _BENCHMARK_DISPLAY.get(str(value), str(value))


def canonical_backend(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", "+")
    if text == "mem0":
        return "Mem0"
    if text == "graphiti":
        return "Graphiti"
    if text in {"langmem+langgraph", "langmem", "langgraph", "langmem+langgraph"}:
        return "LangMem+LangGraph"
    raise ValueError(f"Unsupported backend: {value!r}")


class QwenTokenCounter:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self._lock = threading.Lock()

    def count(self, value: Any) -> int:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
        if not text:
            return 0
        with self._lock:
            return len(self._tokenizer.encode(text, add_special_tokens=False))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0 or not text:
            return ""
        with self._lock:
            ids = self._tokenizer.encode(text, add_special_tokens=False)
            if len(ids) <= max_tokens:
                return text
            return self._tokenizer.decode(ids[:max_tokens], skip_special_tokens=True).strip()

    def chat_count(self, messages: Sequence[Mapping[str, str]]) -> int:
        with self._lock:
            token_ids = self._tokenizer.apply_chat_template(
                list(messages),
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            )
            return len(token_ids)


@dataclass(frozen=True, slots=True)
class SourceEntry:
    text: str
    category: str
    source_ref: str
    order: int

    def to_atom(self, *, sample_id: str, source: str) -> dict[str, Any]:
        return {
            "category": self.category,
            "text": self.text,
            "sample_id": sample_id,
            "source": source,
            "metadata": {"source_ref_hash": stable_hash(self.source_ref)[:16]},
        }


@dataclass(frozen=True, slots=True)
class SampleSpec:
    backend: str
    benchmark: str
    setting_key: str
    sample_index: int
    sample_id: str
    user_id: str
    task_id: str
    task_type: str
    attack_input: AttackInput
    source_row: dict[str, Any]
    metric_record: dict[str, Any]
    gold_atoms: tuple[dict[str, Any], ...]
    fixed_atoms: tuple[dict[str, Any], ...]
    recovered_atoms: tuple[dict[str, Any], ...]
    base_behavior: Any
    base_score: float
    memory_atoms: tuple[dict[str, Any], ...] = ()

    @property
    def task_group(self) -> tuple[str, str, str]:
        row = self.source_row
        metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
        domain = str(
            row.get("task_domain")
            or row.get("dialogue_domain")
            or row.get("primary_tool_category")
            or row.get("tool_category")
            or metadata.get("topic_query")
            or row.get("locomo_category")
            or ""
        ).strip().lower()
        if self.benchmark == "PersonaLens":
            action = str(row.get("task_template_id") or row.get("tool_action_category") or "")
        elif self.benchmark == "ETAPP":
            action = str(row.get("source_task_id") or row.get("task_id") or row.get("primary_tool_name") or "")
        elif self.benchmark == "LoCoMo":
            action = str(row.get("question_family") or row.get("question_template") or "")
        else:
            action = str(metadata.get("pref_type") or row.get("primary_tool_name") or "")
        action = action.strip().lower()
        return self.task_type, domain, action


def prepare_exp1_task_surface(
    benchmark: str,
    attack_input: AttackInput,
    row: Mapping[str, Any],
) -> tuple[AttackInput, dict[str, Any]]:
    """Build the public task shown to every Exp1 condition.

    PersonaMem-v2 stores the correct option first. Exp1 deterministically shuffles
    the options per benchmark row so choice accuracy cannot be obtained from
    position alone.
    """

    prepared_row = clone_json(dict(row))
    if benchmark_display(benchmark) != "PersonaMem-v2":
        return attack_input, prepared_row
    gold = prepared_row.get("gold") if isinstance(prepared_row.get("gold"), Mapping) else {}
    options = [str(item).strip() for item in gold.get("answer_options", []) if str(item).strip()]
    correct = str(gold.get("correct_answer") or "").strip()
    if not options or not correct:
        return attack_input, prepared_row
    shuffle_key = str(prepared_row.get("sample_key") or prepared_row.get("task_id") or attack_input.task_id)
    indexed = list(enumerate(options))
    indexed.sort(key=lambda item: stable_hash(["exp1_personamem_choice_v2", shuffle_key, item[0], item[1]]))
    shuffled = [option for _index, option in indexed]
    correct_index = next(index for index, option in enumerate(shuffled) if option == correct)
    task_input = prepared_row.get("task_input") if isinstance(prepared_row.get("task_input"), Mapping) else {}
    user_query = str(task_input.get("user_query") or attack_input.task_prompt.split("\nOptions:", 1)[0]).strip()
    labels = [chr(ord("A") + index) for index in range(len(shuffled))]
    prompt = "\n".join(
        [
            user_query,
            "Choose exactly one option. Return only its letter (for example, A).",
            "Options:",
            *[f"{label}. {option}" for label, option in zip(labels, shuffled)],
        ]
    )
    payload = attack_input.to_dict()
    payload["task_prompt"] = prompt
    prepared_row["_exp1_personamem_choice"] = {
        "shuffle_version": "deterministic_choice_shuffle_v2",
        "options": shuffled,
        "correct_index": correct_index,
        "correct_label": labels[correct_index],
        "source_order_hidden": True,
    }
    return AttackInput.from_dict(payload), prepared_row


def metric_atoms(metric_record: Mapping[str, Any]) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    metrics = metric_record.get("metrics") if isinstance(metric_record.get("metrics"), Mapping) else {}
    umr = metrics.get("UMR-F1") if isinstance(metrics.get("UMR-F1"), Mapping) else {}
    gold_parse = umr.get("gold_parse") if isinstance(umr.get("gold_parse"), Mapping) else {}
    recovered_parse = umr.get("recovered_parse") if isinstance(umr.get("recovered_parse"), Mapping) else {}
    gold = tuple(dict(item) for item in gold_parse.get("atoms", []) if isinstance(item, Mapping))
    recovered = tuple(dict(item) for item in recovered_parse.get("atoms", []) if isinstance(item, Mapping))
    fixed: list[dict[str, Any]] = []
    for item in umr.get("scope_excluded_gold_atoms", []) or []:
        atom = item.get("atom") if isinstance(item, Mapping) else None
        if isinstance(atom, Mapping):
            fixed.append(dict(atom))
    return gold, tuple(fixed), recovered


def atom_memory_items(atoms: Sequence[Mapping[str, Any]]) -> tuple[MemoryItem, ...]:
    out: list[MemoryItem] = []
    seen: set[tuple[str, str]] = set()
    for atom in atoms:
        category = str(atom.get("category") or "facts")
        text = str(atom.get("text") or atom.get("typed_text") or "").strip()
        if not text:
            continue
        key = (category, normalize_atom_text(text))
        if key in seen:
            continue
        seen.add(key)
        out.append(MemoryItem(text=text, category=category, source="a200_runtime_state"))
    return tuple(out)


def source_memory_items(entries: Sequence[SourceEntry]) -> tuple[MemoryItem, ...]:
    return tuple(
        MemoryItem(
            text=entry.text,
            category=entry.category,
            source="pre_task_user_information",
            metadata={
                "source_ref": entry.source_ref,
                "source_ref_hash": stable_hash(entry.source_ref)[:16],
            },
        )
        for entry in entries
        if entry.text.strip()
    )


def match_memory_items_to_context_budget(
    *,
    sample: SampleSpec,
    items: Sequence[MemoryItem],
    target_tokens: int,
    token_counter: QwenTokenCounter,
) -> tuple[MemoryItem, ...]:
    if not items or target_tokens <= 0:
        return ()

    def rendered_tokens(candidate_items: Sequence[MemoryItem]) -> int:
        context = runtime_context_from_items(
            backend=sample.backend,
            sample=sample,
            items=candidate_items,
            adapter_mode="swap_length_probe",
        )
        return token_counter.count(context.prompt_context)

    if rendered_tokens(items) <= target_tokens:
        return tuple(items)

    selected: list[MemoryItem] = []
    for item in items:
        if rendered_tokens([*selected, item]) <= target_tokens:
            selected.append(item)
            continue
        item_token_count = token_counter.count(item.text)
        low = 1
        high = max(1, item_token_count)
        best: MemoryItem | None = None
        while low <= high:
            middle = (low + high) // 2
            shortened = MemoryItem(
                text=token_counter.truncate(item.text, middle),
                category=item.category,
                source=item.source,
                metadata=dict(item.metadata),
            )
            if shortened.text and rendered_tokens([*selected, shortened]) <= target_tokens:
                best = shortened
                low = middle + 1
            else:
                high = middle - 1
        if best is not None:
            selected.append(best)
        break

    if selected:
        return tuple(selected)
    first = items[0]
    return (
        MemoryItem(
            text=token_counter.truncate(first.text, 1),
            category=first.category,
            source=first.source,
            metadata=dict(first.metadata),
        ),
    )


def runtime_context_from_items(
    *,
    backend: str,
    sample: SampleSpec,
    items: Sequence[MemoryItem],
    adapter_mode: str,
) -> RuntimeMemoryContext:
    adapter = build_backend_adapter(backend)
    prompt_context = adapter.render_items(items, query=sample.attack_input.task_prompt)
    model: dict[str, Any] = {
        "facts": [],
        "preferences": [],
        "constraints": [],
        "relations": [],
        "tool_state": [],
        "raw_text": "\n".join(item.text for item in items),
        "metadata": {"source": adapter_mode, "schema_version": SCHEMA_VERSION},
    }
    for item in items:
        category = item.category if item.category in model else "facts"
        model[category].append(item.text)
    return RuntimeMemoryContext(
        backend=backend,
        user_id=sample.user_id,
        task_id=sample.task_id,
        query=sample.attack_input.task_prompt,
        adapter_mode=adapter_mode,
        retrieved_items=tuple(items),
        prompt_context=prompt_context,
        gold_user_model=model,
        metadata={"exp12_condition": adapter_mode},
    )


def scoped_state_context(sample: SampleSpec, *, removed_atom_id: str | None = None) -> RuntimeMemoryContext:
    kept = [atom for atom in sample.gold_atoms if str(atom.get("atom_id") or "") != str(removed_atom_id or "")]
    atoms = [*kept, *sample.fixed_atoms]
    return runtime_context_from_items(
        backend=sample.backend,
        sample=sample,
        items=atom_memory_items(atoms),
        adapter_mode=("runtime_state" if removed_atom_id is None else "runtime_state_single_fact_removed"),
    )


def empty_state_context(sample: SampleSpec, *, mode: str) -> RuntimeMemoryContext:
    return runtime_context_from_items(backend=sample.backend, sample=sample, items=(), adapter_mode=mode)


def _flatten_behavior(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _normalized(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _flatten_behavior(value).lower()).strip()


def _token_f1(left: Any, right: Any) -> float:
    left_tokens = _TOKEN_RE.findall(_normalized(left))
    right_tokens = _TOKEN_RE.findall(_normalized(right))
    if not left_tokens or not right_tokens:
        return 0.0
    left_counts = Counter(left_tokens)
    right_counts = Counter(right_tokens)
    overlap = sum((left_counts & right_counts).values())
    if overlap <= 0:
        return 0.0
    precision = overlap / sum(left_counts.values())
    recall = overlap / sum(right_counts.values())
    return 2.0 * precision * recall / (precision + recall)


def _score_personamem(candidate: Any, row: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    gold = row.get("gold") if isinstance(row.get("gold"), Mapping) else {}
    choice = row.get("_exp1_personamem_choice") if isinstance(row.get("_exp1_personamem_choice"), Mapping) else {}
    options = [str(item) for item in (choice.get("options") or gold.get("answer_options", [])) if str(item).strip()]
    correct = str(gold.get("correct_answer") or "").strip()
    correct_index = choice.get("correct_index")
    if correct_index is None and correct and options:
        correct_index = next((index for index, option in enumerate(options) if _normalized(option) == _normalized(correct)), None)
    if correct_index is None or not options:
        return 0.0, {"status": "missing_choice_gold"}
    candidate_text = _flatten_behavior(candidate)
    letter_match = re.match(
        r"^\s*(?:(?:option|choice|answer)\s*[:=]?\s*)?([A-D])(?:\s*[\.)\]:,-]|\s*$)",
        candidate_text,
        flags=re.IGNORECASE,
    )
    if letter_match:
        predicted_index = ord(letter_match.group(1).lower()) - ord("a")
    else:
        scores = [_token_f1(candidate_text, option) for option in options]
        predicted_index = max(range(len(scores)), key=scores.__getitem__)
    return float(predicted_index == int(correct_index)), {
        "status": "ok",
        "predicted_option": predicted_index,
        "correct_option": int(correct_index),
        "scorer": "persona_choice_accuracy_shuffled_v2",
    }


def _value_present(candidate_text: str, value: str) -> bool:
    candidate_norm = _normalized(candidate_text)
    value_norm = _normalized(value)
    if not value_norm:
        return False
    if value_norm in candidate_norm:
        return True
    value_tokens = set(_TOKEN_RE.findall(value_norm))
    candidate_tokens = set(_TOKEN_RE.findall(candidate_norm))
    return bool(value_tokens) and len(value_tokens & candidate_tokens) / len(value_tokens) >= 0.8


def _score_personalens(candidate: Any, row: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    gold = row.get("gold") if isinstance(row.get("gold"), Mapping) else {}
    affinities = [item for item in gold.get("expected_affinities", []) if isinstance(item, Mapping)]
    text = _flatten_behavior(candidate).strip()
    expected_values = [
        str(value)
        for affinity in affinities
        for value in (affinity.get("values") if isinstance(affinity.get("values"), Sequence) and not isinstance(affinity.get("values"), (str, bytes, bytearray)) else [])
        if str(value).strip()
    ]
    matched_values = [value for value in expected_values if _value_present(text, value)]
    matched = len(matched_values)
    coverage = matched / len(expected_values) if expected_values else 0.0
    clarification_request = bool(
        re.search(
            r"\b(?:what|which)\s+(?:time|date|sound|day|option)|"
            r"\b(?:please|could you|can you)\s+(?:tell|provide|choose|specify|clarify|confirm|share)|"
            r"\bneed (?:your|a|the)\b|\bwhat would you like\b",
            text.lower(),
        )
    )
    substantive_result = bool(
        re.search(
            r"\b(?:i (?:set|scheduled|booked|selected|found|recommend)|"
            r"i will (?:search|look|use|apply)|here (?:are|is)|based on your preferences)",
            text.lower(),
        )
    )
    clarification = clarification_request and not substantive_result
    completion = float(bool(text) and not clarification)
    return 0.4 * completion + 0.6 * coverage, {
        "status": "ok" if expected_values else "missing_expected_affinities",
        "task_completion": completion,
        "clarification_request": clarification,
        "affinity_coverage": coverage,
        "matched_affinity_values": matched,
        "matched_affinity_value_list": matched_values,
        "expected_affinity_values": len(expected_values),
        "scorer": "personalens_completion_affinity_coverage_v3",
    }


def _locomo_normalize(value: Any) -> list[str]:
    text = _flatten_behavior(value).lower().replace(",", " ")
    text = re.sub(r"\b(?:a|an|the|and)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [_PORTER_STEMMER.stem(token) for token in text.split() if token]


def _locomo_f1(candidate: Any, target: Any) -> float:
    predicted = _locomo_normalize(candidate)
    gold = _locomo_normalize(target)
    if not predicted or not gold:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(gold)).values())
    if overlap <= 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 2.0 * precision * recall / (precision + recall)


def _score_locomo(candidate: Any, row: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    target = row.get("gold_answer")
    if target in (None, ""):
        gold = row.get("gold") if isinstance(row.get("gold"), Mapping) else {}
        target = gold.get("answer")
    if target in (None, ""):
        return 0.0, {"status": "missing_gold_answer", "scorer": "locomo_official_category_f1_v2"}
    try:
        category = int(row.get("locomo_category") or row.get("category") or 0)
    except (TypeError, ValueError):
        category = 0
    candidate_text = _flatten_behavior(candidate)
    target_text = _flatten_behavior(target)
    if category == 5:
        lowered = candidate_text.lower()
        score = float("no information available" in lowered or "not mentioned" in lowered)
    elif category == 3:
        score = _locomo_f1(candidate_text, target_text.split(";", 1)[0].strip())
    elif category == 1:
        predictions = [item.strip() for item in candidate_text.split(",") if item.strip()]
        answers = [item.strip() for item in target_text.split(",") if item.strip()]
        score = sum(max((_locomo_f1(prediction, answer) for prediction in predictions), default=0.0) for answer in answers) / max(len(answers), 1)
    else:
        score = _locomo_f1(candidate_text, target_text)
    return float(score), {
        "status": "ok",
        "locomo_category": category,
        "scorer": "locomo_official_category_f1_v2",
    }


def _etapp_actions(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        sequence = value.get("tool_sequence") or value.get("action_sequence")
        if isinstance(sequence, Sequence) and not isinstance(sequence, (str, bytes, bytearray)):
            return [dict(item) for item in sequence if isinstance(item, Mapping)]
        return [dict(value)] if value.get("tool_name") or value.get("name") else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def _etapp_target_actions(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    signature = row.get("action_signature")
    if isinstance(signature, str):
        try:
            signature = json.loads(signature)
        except json.JSONDecodeError:
            signature = None
    actions = _etapp_actions(signature)
    if actions:
        return actions
    return _etapp_actions(row.get("action_sequence"))


def _etapp_profile_numeric_ranges(row: Mapping[str, Any], field_name: str) -> list[tuple[float, float]]:
    if str(field_name).lower() not in {"volume", "volume_level"}:
        return []
    index = _load_etapp_profile_index()
    profile = None
    for key in (row.get("profile_id"), row.get("user_id"), row.get("source_user_id")):
        if key and str(key) in index:
            profile = index[str(key)]
            break
    if not isinstance(profile, Mapping):
        return []
    text = json.dumps(profile, ensure_ascii=False, sort_keys=True)
    ranges = []
    for lower, upper in re.findall(r"(\d+(?:\.\d+)?)\s*%?\s*(?:to|through|[-~])\s*(\d+(?:\.\d+)?)\s*%", text, flags=re.IGNORECASE):
        ranges.append((float(lower), float(upper)))
    return ranges


def _score_etapp_task_behavior(candidate: Any, row: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    targets = _etapp_target_actions(row)
    candidates = _etapp_actions(candidate)
    target_units = len(targets)
    field_units = 0
    target_fields: list[dict[str, Any]] = []
    for target in targets:
        fields = target.get("key_decision_fields")
        if not isinstance(fields, Mapping) or not fields:
            fields = target.get("normalized_args")
        fields = dict(fields) if isinstance(fields, Mapping) else {}
        target_fields.append(fields)
        field_units += len(fields)
    total_units = target_units + field_units
    if total_units <= 0:
        return 0.0, {
            "status": "missing_decision_units",
            "scorer": "etapp_ordered_decision_units_v3",
        }

    matched_tools = 0
    matched_fields = 0
    candidate_cursor = 0
    matched_pairs: list[tuple[int, int]] = []
    for target_index, target in enumerate(targets):
        target_tool = _normalized(target.get("tool_name") or target.get("name"))
        if not target_tool:
            continue
        matched_index = None
        for candidate_index in range(candidate_cursor, len(candidates)):
            candidate_tool = _normalized(candidates[candidate_index].get("tool_name") or candidates[candidate_index].get("name"))
            if candidate_tool == target_tool:
                matched_index = candidate_index
                break
        if matched_index is None:
            continue
        matched_tools += 1
        candidate_cursor = matched_index + 1
        matched_pairs.append((target_index, matched_index))
        candidate_args = candidates[matched_index].get("normalized_args") or candidates[matched_index].get("arguments") or {}
        candidate_args = dict(candidate_args) if isinstance(candidate_args, Mapping) else {}
        for key, target_value in target_fields[target_index].items():
            candidate_value = candidate_args.get(str(key))
            matched = _etapp_field_value_matches(candidate_value, target_value)
            if not matched:
                try:
                    numeric_candidate = float(candidate_value)
                except (TypeError, ValueError):
                    numeric_candidate = None
                matched = numeric_candidate is not None and any(
                    lower <= numeric_candidate <= upper
                    for lower, upper in _etapp_profile_numeric_ranges(row, str(key))
                )
            if matched:
                matched_fields += 1

    score = (matched_tools + matched_fields) / total_units
    return score, {
        "status": "ok",
        "scorer": "etapp_ordered_decision_units_v3",
        "tool_units": target_units,
        "field_units": field_units,
        "matched_tool_units": matched_tools,
        "matched_field_units": matched_fields,
        "matched_ordered_pairs": matched_pairs,
    }


def score_task_behavior(benchmark: str, candidate: Any, row: Mapping[str, Any], task_type: str) -> tuple[float, dict[str, Any]]:
    benchmark = benchmark_display(benchmark)
    if benchmark == "PersonaMem-v2":
        return _score_personamem(candidate, row)
    if benchmark == "PersonaLens":
        return _score_personalens(candidate, row)
    if benchmark == "ETAPP":
        return _score_etapp_task_behavior(candidate, row)
    if benchmark == "LoCoMo":
        return _score_locomo(candidate, row)
    target = row.get("gold_answer")
    return score_behavior(candidate, target, task_type), {"status": "fallback", "scorer": "eval2_score_behavior"}


def task_decision_signature(
    benchmark: str,
    behavior: Any,
    score_audit: Mapping[str, Any],
) -> Any:
    """Return the task decision, excluding harmless wording differences where possible."""

    benchmark = benchmark_display(benchmark)
    if benchmark == "PersonaMem-v2":
        return {"selected_option": score_audit.get("predicted_option")}
    if benchmark == "PersonaLens":
        return {
            "task_completed": bool(score_audit.get("task_completion")),
            "preference_values": sorted(str(item) for item in score_audit.get("matched_affinity_value_list", []) or []),
        }
    if benchmark == "ETAPP":
        return [
            {
                "tool_name": str(item.get("tool_name") or item.get("name") or "").strip().lower(),
                "normalized_args": clone_json(item.get("normalized_args") or item.get("arguments") or {}),
            }
            for item in _etapp_actions(behavior)
        ]
    if benchmark == "LoCoMo":
        return {"answer_tokens": _locomo_normalize(behavior)}
    return {"normalized_behavior": _normalized(behavior)}


def atom_recovery_scores(gold_atoms: Sequence[Mapping[str, Any]], recovered_atoms: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    recovered = [
        MetricAtom(
            category=str(atom.get("category") or "facts"),
            text=str(atom.get("text") or atom.get("typed_text") or ""),
            sample_id=str(atom.get("sample_id") or ""),
            atom_id=str(atom.get("atom_id") or ""),
            atom_type=str(atom.get("atom_type") or "semantic"),
            semantic_group=(None if atom.get("semantic_group") in (None, "") else str(atom.get("semantic_group"))),
            source="recovered_s",
            metadata=dict(atom.get("metadata", {})) if isinstance(atom.get("metadata"), Mapping) else {},
        )
        for atom in recovered_atoms
    ]
    scores: dict[str, float] = {}
    for atom in gold_atoms:
        gold = MetricAtom(
            category=str(atom.get("category") or "facts"),
            text=str(atom.get("text") or atom.get("typed_text") or ""),
            sample_id=str(atom.get("sample_id") or ""),
            atom_id=str(atom.get("atom_id") or ""),
            atom_type=str(atom.get("atom_type") or "semantic"),
            semantic_group=(None if atom.get("semantic_group") in (None, "") else str(atom.get("semantic_group"))),
            source="gold_s",
            metadata=dict(atom.get("metadata", {})) if isinstance(atom.get("metadata"), Mapping) else {},
        )
        scores[gold.atom_id] = max((semantic_atom_score(gold, candidate) for candidate in recovered), default=0.0)
    return scores


def _infer_category(path: str, text: str) -> str:
    haystack = f"{path} {text}".lower()
    for category, terms in _SOURCE_CATEGORY_TERMS.items():
        if any(term in haystack for term in terms):
            return category
    return "facts"


def _flatten_profile(value: Any, *, source_ref: str, path: tuple[str, ...] = (), start_order: int = 0) -> list[SourceEntry]:
    out: list[SourceEntry] = []

    def visit(item: Any, current: tuple[str, ...]) -> None:
        if item in (None, "", [], {}):
            return
        if isinstance(item, Mapping):
            for key in sorted(item, key=str):
                normalized_key = str(key).lower()
                if normalized_key in {"gold", "gold_answer", "correct_answer", "answer_options", "action_sequence", "heldout"}:
                    continue
                visit(item[key], (*current, str(key)))
            return
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for index, child in enumerate(item):
                visit(child, (*current, str(index)))
            return
        text = " ".join(str(item).split())
        if not text:
            return
        path_text = ".".join(current) or "value"
        out.append(
            SourceEntry(
                text=f"{path_text}={text}",
                category=_infer_category(path_text, text),
                source_ref=f"{source_ref}#{path_text}",
                order=start_order + len(out),
            )
        )

    visit(value, path)
    return out


class SourceRepository:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self._json_cache: dict[str, Any] = {}
        self._history_cache: dict[str, tuple[SourceEntry, ...]] = {}
        self._locomo_index: dict[str, Mapping[str, Any]] | None = None
        self._lock = threading.Lock()

    def _read_json(self, path: Path) -> Any:
        key = path.resolve().as_posix()
        with self._lock:
            if key not in self._json_cache:
                self._json_cache[key] = json.loads(path.read_text(encoding="utf-8"))
            return self._json_cache[key]

    def history(self, benchmark: str, row: Mapping[str, Any]) -> tuple[SourceEntry, ...]:
        benchmark = benchmark_display(benchmark)
        cache_key = stable_hash({
            "benchmark": benchmark,
            "user": row.get("user_id") or row.get("source_user_id"),
            "profile": row.get("profile_id"),
            "pre_history_ref": row.get("pre_history_ref"),
            "sample": row.get("sample_id"),
        })
        with self._lock:
            cached = self._history_cache.get(cache_key)
        if cached is not None:
            return cached
        if benchmark == "PersonaMem-v2":
            entries = self._personamem_history(row)
        elif benchmark == "PersonaLens":
            entries = self._personalens_history(row)
        elif benchmark == "ETAPP":
            entries = self._etapp_history(row)
        elif benchmark == "LoCoMo":
            entries = self._locomo_history(row)
        else:
            entries = []
        result = tuple(_dedupe_source_entries(entries))
        with self._lock:
            self._history_cache[cache_key] = result
        return result

    def _personamem_history(self, row: Mapping[str, Any]) -> list[SourceEntry]:
        ref = row.get("pre_history_ref")
        path = Path(str(ref)) if ref else None
        if path is None or not path.exists():
            return []
        payload = self._read_json(path)
        history = payload.get("chat_history", []) if isinstance(payload, Mapping) else []
        entries: list[SourceEntry] = []
        for index, item in enumerate(history if isinstance(history, Sequence) else []):
            if not isinstance(item, Mapping):
                continue
            role = str(item.get("role") or "message")
            text = " ".join(str(item.get("content") or "").split())
            if index == 0 and role == "system" and "{" in str(item.get("content") or ""):
                raw = str(item.get("content") or "")
                try:
                    profile, _end = json.JSONDecoder().raw_decode(raw[raw.index("{") :])
                except (ValueError, json.JSONDecodeError):
                    profile = None
                if isinstance(profile, Mapping):
                    entries.extend(_flatten_profile(profile, source_ref=f"{path}#persona", start_order=0))
                    continue
            if text:
                entries.append(SourceEntry(text=f"{role}: {text}", category="facts", source_ref=f"{path}#turn={index}", order=index))
        return entries

    def _personalens_history(self, row: Mapping[str, Any]) -> list[SourceEntry]:
        ref = row.get("pre_history_ref") if isinstance(row.get("pre_history_ref"), Mapping) else {}
        path = Path(str(ref.get("profile_source") or ""))
        if not path.exists():
            return []
        profile = self._read_json(path)
        return _flatten_profile(profile, source_ref=path.as_posix())

    def _etapp_history(self, row: Mapping[str, Any]) -> list[SourceEntry]:
        index = _load_etapp_profile_index()
        profile = None
        for key in (row.get("profile_id"), row.get("user_id"), row.get("source_user_id")):
            if key and str(key) in index:
                profile = index[str(key)]
                break
        if not isinstance(profile, Mapping):
            return []
        return _flatten_profile(profile, source_ref=f"etapp_profile:{row.get('profile_id') or row.get('user_id')}")

    def _locomo_history(self, row: Mapping[str, Any]) -> list[SourceEntry]:
        if self._locomo_index is None:
            path = self.project_root / "data/benchmarks/LoCoMo/locomo10.json"
            payload = self._read_json(path)
            self._locomo_index = {
                str(item.get("sample_id")): item
                for item in payload
                if isinstance(item, Mapping) and item.get("sample_id")
            }
        sample_id = str(row.get("sample_id") or "")
        payload = self._locomo_index.get(sample_id, {})
        conversation = payload.get("conversation") if isinstance(payload, Mapping) else {}
        if not isinstance(conversation, Mapping):
            return []
        entries: list[SourceEntry] = []
        session_keys = sorted(
            (key for key in conversation if re.fullmatch(r"session_\d+", str(key))),
            key=lambda key: int(str(key).split("_")[-1]),
        )
        order = 0
        for session_key in session_keys:
            timestamp = str(conversation.get(f"{session_key}_date_time") or "")
            turns = conversation.get(session_key)
            if not isinstance(turns, Sequence):
                continue
            for turn in turns:
                if not isinstance(turn, Mapping):
                    continue
                speaker = str(turn.get("speaker") or "speaker")
                text = " ".join(str(turn.get("text") or "").split())
                dia_id = str(turn.get("dia_id") or order)
                if text:
                    entries.append(
                        SourceEntry(
                            text=f"{timestamp}: {speaker}: {text}" if timestamp else f"{speaker}: {text}",
                            category="facts",
                            source_ref=f"locomo:{sample_id}:{dia_id}",
                            order=order,
                        )
                    )
                    order += 1
        return entries


def _dedupe_source_entries(entries: Sequence[SourceEntry]) -> list[SourceEntry]:
    out: list[SourceEntry] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        key = (entry.category, normalize_atom_text(entry.text))
        if not key[1] or key in seen:
            continue
        seen.add(key)
        out.append(entry)
    return out


def _bm25_tokens(text: str) -> list[str]:
    expanded = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(text or "")).replace("_", " ")
    return [
        token
        for token in _TOKEN_RE.findall(expanded.lower())
        if token not in {"the", "and", "for", "with", "that", "this", "user", "assistant"}
    ]


def bm25_rank(entries: Sequence[SourceEntry], query: str) -> list[SourceEntry]:
    if not entries:
        return []
    documents = [_bm25_tokens(entry.text) for entry in entries]
    query_tokens = _bm25_tokens(query)
    doc_freq: Counter[str] = Counter()
    for tokens in documents:
        doc_freq.update(set(tokens))
    avg_len = sum(len(tokens) for tokens in documents) / max(1, len(documents))
    k1 = 1.5
    b = 0.75
    scored: list[tuple[float, int, SourceEntry]] = []
    for entry, tokens in zip(entries, documents):
        counts = Counter(tokens)
        score = 0.0
        for term in query_tokens:
            tf = counts.get(term, 0)
            if tf <= 0:
                continue
            idf = math.log(1.0 + (len(documents) - doc_freq.get(term, 0) + 0.5) / (doc_freq.get(term, 0) + 0.5))
            denom = tf + k1 * (1.0 - b + b * len(tokens) / max(avg_len, 1.0))
            score += idf * (tf * (k1 + 1.0)) / denom
        scored.append((score, -entry.order, entry))
    return [entry for _score, _order, entry in sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)]


def locomo_relevant_dialogue(
    entries: Sequence[SourceEntry],
    query: str,
    *,
    anchor_count: int = 10,
    neighbor_radius: int = 2,
) -> list[SourceEntry]:
    """Retrieve query-matched LoCoMo turns together with nearby dialogue.

    LoCoMo answers often depend on a reply two turns after the utterance that
    shares the query terms. Expanding public-query BM25 anchors preserves that
    local exchange without consulting evaluator evidence or the gold answer.
    """

    ranked = bm25_rank(entries, query)
    by_order = {entry.order: entry for entry in entries}
    selected: list[SourceEntry] = []
    seen_orders: set[int] = set()
    for anchor in ranked[: max(1, int(anchor_count))]:
        for order in range(anchor.order - neighbor_radius, anchor.order + neighbor_radius + 1):
            entry = by_order.get(order)
            if entry is None or entry.order in seen_orders:
                continue
            seen_orders.add(entry.order)
            selected.append(entry)
    return selected


def _personamem_runtime_state_items(sample: SampleSpec) -> tuple[MemoryItem, ...]:
    """Build the current task-relevant PersonaMem-v2 state from its memory seed."""

    metadata = sample.source_row.get("metadata") if isinstance(sample.source_row.get("metadata"), Mapping) else {}
    preference = " ".join(str(metadata.get("preference") or "").split())
    previous = " ".join(str(metadata.get("prev_pref") or "").split())
    preference_type = str(metadata.get("pref_type") or "").strip().lower()
    items: list[MemoryItem] = []
    if preference_type == "ask_to_forget":
        former = previous or preference
        if former:
            items.append(
                MemoryItem(
                    text=f"The user asked the assistant not to retain or use this former preference: {former}",
                    category="constraints",
                    source="personamem_official_memory_seed",
                )
            )
    elif preference:
        label = "Sensitive user information" if preference_type == "sensitive_info" else "Current task-relevant user preference"
        items.append(
            MemoryItem(
                text=f"{label}: {preference}",
                category="preferences",
                source="personamem_official_memory_seed",
            )
        )
    return tuple(items)


def public_retrieval_query(sample: SampleSpec) -> str:
    """Return only information available to the victim's normal retrieval step."""

    tool_parts: list[str] = []
    for tool in sample.attack_input.visible_tools:
        if not isinstance(tool, Mapping):
            continue
        tool_parts.extend([str(tool.get("name") or ""), str(tool.get("description") or "")])
        parameters = tool.get("parameters") if isinstance(tool.get("parameters"), Mapping) else {}
        properties = parameters.get("properties") if isinstance(parameters.get("properties"), Mapping) else {}
        for name, spec in properties.items():
            tool_parts.append(str(name))
            if isinstance(spec, Mapping):
                tool_parts.append(str(spec.get("description") or ""))
    return "\n".join([sample.attack_input.task_prompt, *[part for part in tool_parts if part.strip()]])


def exp1_runtime_state_items(
    sample: SampleSpec,
    source_repository: SourceRepository,
    *,
    owner: SampleSpec | None = None,
) -> tuple[MemoryItem, ...]:
    """Build S(owner, sample.task) without evaluator-only answers or behavior."""

    source_owner = owner or sample
    if sample.benchmark == "PersonaMem-v2":
        existing = _personamem_runtime_state_items(source_owner)
        if existing:
            return existing
        history = source_repository.history(source_owner.benchmark, source_owner.source_row)
        ranked = bm25_rank(history, public_retrieval_query(sample))
        return source_memory_items(ranked[:8])
    history = source_repository.history(source_owner.benchmark, source_owner.source_row)
    ranked = bm25_rank(history, public_retrieval_query(sample))
    if sample.benchmark == "LoCoMo":
        selected = locomo_relevant_dialogue(history, public_retrieval_query(sample))
        return source_memory_items(selected)
    if sample.benchmark == "PersonaLens":
        domain = str(sample.source_row.get("task_domain") or "").strip().lower().replace("_", " ")
        direct = [
            entry
            for entry in history
            if entry.text.lower().startswith(f"affinities.{domain}.")
        ]
        selected = direct if direct else ranked[:12]
    else:
        selected = ranked[:20]
    return source_memory_items(selected)


def exp1_runtime_state_context(
    sample: SampleSpec,
    source_repository: SourceRepository,
    *,
    owner: SampleSpec | None = None,
    mode: str = "task_conditioned_runtime_state_v2",
) -> RuntimeMemoryContext:
    return runtime_context_from_items(
        backend=sample.backend,
        sample=sample,
        items=exp1_runtime_state_items(sample, source_repository, owner=owner),
        adapter_mode=mode,
    )


def select_entries_to_budget(entries: Sequence[SourceEntry], *, token_counter: QwenTokenCounter, budget_tokens: int) -> tuple[SourceEntry, ...]:
    selected: list[SourceEntry] = []
    used = 0
    for entry in entries:
        entry_tokens = token_counter.count(entry.text)
        if entry_tokens <= 0:
            continue
        if used + entry_tokens <= budget_tokens:
            selected.append(entry)
            used += entry_tokens
            continue
        remaining = budget_tokens - used
        if remaining > 0:
            text = token_counter.truncate(entry.text, remaining)
            if text:
                selected.append(SourceEntry(text=text, category=entry.category, source_ref=entry.source_ref, order=entry.order))
        break
    return tuple(selected)


def relevant_history(sample: SampleSpec, history: Sequence[SourceEntry], token_counter: QwenTokenCounter, state_tokens: int) -> tuple[SourceEntry, ...]:
    ranked = bm25_rank(history, sample.attack_input.task_prompt)
    return select_entries_to_budget(ranked, token_counter=token_counter, budget_tokens=max(1, state_tokens))


def full_memory_entries(sample: SampleSpec, history: Sequence[SourceEntry], token_counter: QwenTokenCounter) -> tuple[SourceEntry, ...]:
    task_tokens = token_counter.count(sample.attack_input.task_prompt)
    tool_tokens = token_counter.count(sample.attack_input.visible_tools)
    # Reserve space for the system prompt, JSON wrapper, and 768 output tokens.
    budget = max(256, 32768 - task_tokens - tool_tokens - 4096)
    selected = select_entries_to_budget(
        sorted(history, key=lambda entry: (entry.order, entry.source_ref)),
        token_counter=token_counter,
        budget_tokens=budget,
    )
    maximum_input_tokens = 31500
    for _ in range(4):
        context = runtime_context_from_items(
            backend=sample.backend,
            sample=sample,
            items=source_memory_items(selected),
            adapter_mode="full_memory_context_fit_probe",
        )
        prompt_tokens = victim_input_token_count(sample, context, token_counter)
        if prompt_tokens <= maximum_input_tokens:
            return selected
        content_tokens = sum(token_counter.count(entry.text) for entry in selected)
        reduction = prompt_tokens - maximum_input_tokens + 128
        selected = select_entries_to_budget(
            selected,
            token_counter=token_counter,
            budget_tokens=max(0, content_tokens - reduction),
        )
    return selected


def victim_input_token_count(
    sample: SampleSpec,
    context: RuntimeMemoryContext,
    token_counter: QwenTokenCounter,
) -> int:
    messages = [
        {"role": "system", "content": _system_prompt(sample.attack_input)},
        {
            "role": "user",
            "content": _victim_prompt(
                sample=sample.attack_input,
                runtime_context=context,
                source_row=sample.source_row,
                include_memory_section=True,
            ),
        },
    ]
    return token_counter.chat_count(messages)


def history_text(entries: Sequence[SourceEntry]) -> str:
    return "\n".join(entry.text for entry in entries if entry.text.strip())


def source_atoms(entries: Sequence[SourceEntry], *, sample_id: str, source: str) -> tuple[dict[str, Any], ...]:
    payload = {"atoms": [entry.to_atom(sample_id=sample_id, source=source) for entry in entries]}
    return tuple(atom.to_dict() for atom in atoms_from_user_model(payload, sample_id=sample_id, source=source).atoms)


def text_source_atoms(text: str, *, sample_id: str, source: str, category: str = "facts") -> tuple[dict[str, Any], ...]:
    if not text.strip():
        return ()
    payload = {"atoms": [{"category": category, "text": text, "sample_id": sample_id, "source": source}]}
    return tuple(atom.to_dict() for atom in atoms_from_user_model(payload, sample_id=sample_id, source=source).atoms)


def match_source_to_state(source_atom: Mapping[str, Any], state_atoms: Sequence[Mapping[str, Any]]) -> tuple[str | None, float]:
    source = MetricAtom(
        category=str(source_atom.get("category") or "facts"),
        text=str(source_atom.get("text") or ""),
        sample_id=str(source_atom.get("sample_id") or ""),
        atom_id=str(source_atom.get("atom_id") or ""),
        source="source",
    )
    best_id = None
    best_score = 0.0
    for item in state_atoms:
        target = MetricAtom(
            category=str(item.get("category") or "facts"),
            text=str(item.get("text") or item.get("typed_text") or ""),
            sample_id=str(item.get("sample_id") or ""),
            atom_id=str(item.get("atom_id") or ""),
            source="gold_s",
        )
        score = semantic_atom_score(target, source)
        if score > best_score:
            best_id = target.atom_id
            best_score = score
    return best_id, best_score


class EndpointPool:
    def __init__(
        self,
        base_urls: Sequence[str],
        *,
        model: str = MODEL_NAME,
        max_tokens: int = 768,
        timeout_s: float = 120.0,
        max_inflight_per_endpoint: int = 8,
    ) -> None:
        if not base_urls:
            raise ValueError("At least one vLLM endpoint is required.")
        self._clients: list[QwenVLLMClient] = []
        self._semaphores: list[threading.Semaphore] = []
        self._next = 0
        self._lock = threading.Lock()
        for base_url in base_urls:
            config = QwenVLLMConfig(
                base_url=str(base_url).rstrip("/"),
                model=model,
                temperature=0.0,
                top_p=1.0,
                seed=20260615,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                enable_thinking=False,
                require_live_endpoint=True,
                strict_model_check=True,
            )
            client = QwenVLLMClient(config)
            client.healthcheck()
            self._clients.append(client)
            self._semaphores.append(threading.Semaphore(max_inflight_per_endpoint))

    def _checkout(self) -> tuple[int, QwenVLLMClient, threading.Semaphore]:
        with self._lock:
            index = self._next % len(self._clients)
            self._next += 1
        return index, self._clients[index], self._semaphores[index]

    def run(
        self,
        *,
        sample: SampleSpec,
        context: RuntimeMemoryContext,
        include_memory_section: bool,
        retries: int = 2,
    ) -> dict[str, Any]:
        error: Exception | None = None
        for attempt in range(retries + 1):
            endpoint_index, client, semaphore = self._checkout()
            try:
                with semaphore:
                    result = run_victim_once(
                        llm=client,
                        sample=sample.attack_input,
                        runtime_context=context,
                        source_row=sample.source_row,
                        include_memory_section=include_memory_section,
                    )
                return {**result, "endpoint_index": endpoint_index, "retry_count": attempt}
            except Exception as exc:
                error = exc
                if attempt < retries:
                    time.sleep(min(4.0, 0.5 * (2**attempt)))
        assert error is not None
        raise error


def summary_prompt(sample: SampleSpec, history: Sequence[SourceEntry], *, token_counter: QwenTokenCounter, input_budget: int = 12000) -> tuple[list[dict[str, str]], int]:
    ranked = bm25_rank(history, sample.attack_input.task_prompt)
    selected = select_entries_to_budget(ranked, token_counter=token_counter, budget_tokens=input_budget)
    history_block = history_text(selected)
    messages = [
        {
            "role": "system",
            "content": (
                "Summarize only stable user facts, preferences, constraints, and relations from the provided pre-task history. "
                "Do not infer a task answer, copy answer options, or invent facts. Return plain text without hidden reasoning."
            ),
        },
        {
            "role": "user",
            "content": f"USER TASK:\n{sample.attack_input.task_prompt}\n\nPRE-TASK HISTORY:\n{history_block}",
        },
    ]
    return messages, token_counter.count(history_block)


def generate_history_summary(
    pool: EndpointPool,
    *,
    sample: SampleSpec,
    history: Sequence[SourceEntry],
    token_counter: QwenTokenCounter,
    output_budget: int,
    retries: int = 2,
) -> dict[str, Any]:
    messages, input_tokens = summary_prompt(sample, history, token_counter=token_counter)
    error: Exception | None = None
    for attempt in range(retries + 1):
        endpoint_index, client, semaphore = pool._checkout()
        config = client.config
        summary_client = QwenVLLMClient(
            QwenVLLMConfig(
                base_url=config.base_url,
                model=config.model,
                temperature=0.0,
                top_p=1.0,
                seed=config.seed,
                max_tokens=max(16, min(768, int(math.ceil(output_budget * 1.05)))),
                timeout_s=config.timeout_s,
                enable_thinking=False,
                require_live_endpoint=True,
                strict_model_check=True,
            )
        )
        try:
            with semaphore:
                response = summary_client.chat(messages, response_format_json=False)
            summary = token_counter.truncate(response.text.strip(), max(1, int(math.ceil(output_budget * 1.05))))
            return {
                "summary": summary,
                "endpoint_index": endpoint_index,
                "retry_count": attempt,
                "input_history_tokens": input_tokens,
                "llm_usage": {
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "latency_s": response.latency_s,
                    "model": response.model,
                    **clone_json(response.metadata),
                },
            }
        except Exception as exc:
            error = exc
            if attempt < retries:
                time.sleep(min(4.0, 0.5 * (2**attempt)))
    assert error is not None
    raise error


def choose_swap_donors(
    samples: Sequence[SampleSpec],
    token_counts: Mapping[str, int],
    *,
    candidate_samples: Sequence[SampleSpec] | None = None,
) -> dict[str, SampleSpec]:
    candidate_pool = candidate_samples if candidate_samples is not None else samples
    query_tokens = {
        candidate.sample_id: set(_bm25_tokens(public_retrieval_query(candidate)))
        for candidate in candidate_pool
    }
    donors: dict[str, SampleSpec] = {}
    for sample in samples:
        candidates = [
            candidate
            for candidate in candidate_pool
            if candidate.user_id != sample.user_id
            and candidate.benchmark == sample.benchmark
            and candidate.task_type == sample.task_type
            and candidate.task_group[1] == sample.task_group[1]
            and (
                sample.benchmark == "PersonaMem-v2"
                or candidate.task_group[2] == sample.task_group[2]
            )
        ]
        if not candidates:
            continue
        target_tokens = int(token_counts.get(sample.sample_id, 0))
        donors[sample.sample_id] = min(
            candidates,
            key=lambda candidate: (
                abs(int(token_counts.get(candidate.sample_id, 0)) - target_tokens),
                abs(len(candidate.gold_atoms) - len(sample.gold_atoms)),
                -len(query_tokens.get(sample.sample_id, set()) & query_tokens.get(candidate.sample_id, set())),
                stable_hash(candidate.sample_id),
            ),
        )
    return donors


def record_key(*parts: Any) -> str:
    return stable_hash([SCHEMA_VERSION, *parts])[:24]


def read_jsonl(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and len(rows) >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: Mapping[str, Any], lock: threading.Lock | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    if lock is None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
        return
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(payload)


def load_completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "ok" and row.get("record_key"):
                keys.add(str(row["record_key"]))
    return keys


def source_entry_tokens(entries: Sequence[SourceEntry], counter: QwenTokenCounter) -> int:
    return counter.count(history_text(entries))


def mean_ci95(values: Iterable[float]) -> tuple[float | None, float | None]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return None, None
    mean = sum(clean) / len(clean)
    if len(clean) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in clean) / len(clean)
    return mean, 1.96 * math.sqrt(variance) / math.sqrt(len(clean))
