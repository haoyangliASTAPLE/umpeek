from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from umpeek.attack_baselines.adapter import AttackAdapter
from umpeek.attack_baselines.adapters.common import dedupe_preserve_order, estimate_token_count, visible_assistant_text
from umpeek.attack_baselines.io import read_json, read_jsonl
from umpeek.attack_baselines.schema import AttackBaselineSpec, AttackInput, AttackPrediction, blank_predicted_user_model
from umpeek.exp1_whitebox.schema import stable_json


_FIELD_HINT_RE = re.compile(r"\('([^']+)'\)")
_WORD_RE = re.compile(r"[a-z0-9]+")
_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%?\s*(?:~|to|-)\s*(\d+(?:\.\d+)?)")
_CANONICAL_REPLACEMENTS = {
    "favourite": "favorite",
    "prefers": "prefer",
    "likes": "like",
    "dislikes": "dislike",
    "can't": "cannot",
    "can not": "cannot",
    "tool state": "tool_state",
}
_ETAPP_BACKEND_COMPATIBLE = frozenset({"mem0", "graphiti", "langmem"})


@dataclass(frozen=True, slots=True)
class PublicInstruction:
    task_id: str
    query: str
    source_query: str
    available_tools: tuple[str, ...]
    keypoints: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PublicProfile:
    profile_name: str
    path: Path
    sections: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RankedSection:
    profile_name: str
    profile_path: str
    section_name: str
    score: float
    payload_size: int
    matched_values: tuple[str, ...]
    matched_anchors: tuple[str, ...]
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EtappPublicPool:
    instructions: tuple[PublicInstruction, ...]
    profiles: tuple[PublicProfile, ...]


def build_task_conditioned_public_inversion_spec() -> AttackBaselineSpec:
    return AttackBaselineSpec(
        baseline="task_conditioned_public_inversion",
        short_name="TCPI",
        paper_title="In-repo research prototype: task-conditioned public inversion",
        paper_url="https://example.invalid/umpeek/tcpi",
        code_name="in_repo_task_conditioned_public_inversion",
        code_url="https://example.invalid/umpeek/tcpi",
        open_source=True,
        adaptation_target=(
            "Recover the task-conditioned runtime user slice by inverting visible ETAPP actions "
            "against the public instruction/profile pool."
        ),
        implementation_status="success",
        notes=(
            "ETAPP-only round-one research adapter.",
            "Uses public benchmark artifacts plus visible action signatures only.",
            "Evaluation keypoints are deliberately excluded from attack-time inputs.",
            "Does not query the replay victim; attack cost stays at zero queries.",
        ),
        metadata={
            "research_round": "R001",
            "applicable_benchmarks": ["ETAPP"],
            "threat_model": "black_box_visible_observation_plus_public_candidate_pool",
        },
    )


class TaskConditionedPublicInversionAdapter(AttackAdapter):
    def __init__(self, spec: AttackBaselineSpec, *, project_root: Path | str | None = None) -> None:
        super().__init__(spec)
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[4]
        )
        self._etapp_pool = _load_etapp_public_pool(str(self.project_root))

    def run(
        self,
        sample: AttackInput,
        victim_client: Any,
        budget: int | Mapping[str, Any],
    ) -> AttackPrediction:
        del victim_client, budget
        if sample.benchmark != "ETAPP":
            return AttackPrediction(
                baseline=self.spec.baseline,
                sample_id=sample.sample_id,
                predicted_user_model=blank_predicted_user_model(),
                status="not_applicable_no_external_content",
                notes="Round-one public-pool inversion is only implemented for ETAPP.",
                metadata={
                    "adapter_version": "tcpi_etapp_v1",
                    "supported_benchmarks": ["ETAPP"],
                },
            )

        assistant_text = visible_assistant_text(sample.visible_messages)
        action_signature = _parse_action_signature(assistant_text)
        observed_tools = _collect_observed_tools(sample, action_signature)
        observed_fields = _collect_observed_fields(sample, action_signature)
        observed_values = _collect_observed_values(observed_fields)

        instruction = _match_instruction(
            sample=sample,
            observed_tools=observed_tools,
            pool=self._etapp_pool,
        )
        anchors = _extract_anchor_fields(instruction.keypoints if instruction is not None else ())
        prompt_keywords = _prompt_keywords(sample.task_prompt)
        rankings = _rank_public_sections(
            pool=self._etapp_pool,
            observed_tools=observed_tools,
            observed_values=observed_values,
            anchors=anchors,
            prompt_keywords=prompt_keywords,
        )
        selected = rankings[0] if rankings else None
        if selected is None or selected.score <= 0.0:
            return AttackPrediction(
                baseline=self.spec.baseline,
                sample_id=sample.sample_id,
                predicted_user_model=blank_predicted_user_model(),
                status="invalid_input",
                notes="Visible ETAPP action signature did not constrain any public profile section.",
                metadata={
                    "adapter_version": "tcpi_etapp_v1",
                    "observed_tools": list(observed_tools),
                    "observed_fields": dict(observed_fields),
                    "public_pool_profile_count": len(self._etapp_pool.profiles),
                    "public_pool_instruction_count": len(self._etapp_pool.instructions),
                },
            )

        tool_hints = _build_tool_hints(sample, action_signature)
        user_related_fields = dict(observed_fields)
        predicted_user_model = blank_predicted_user_model()
        predicted_user_model["preferences"] = [
            _pseudo_json({selected.section_name: selected.payload}),
        ]
        if instruction is not None and instruction.keypoints:
            predicted_user_model["preferences"].append(_pseudo_json(list(instruction.keypoints)))
        predicted_user_model["tool_state"] = _build_tool_state_items(
            tool_hints=tool_hints,
            user_related_fields=user_related_fields,
            observed_tools=observed_tools,
        )
        predicted_user_model["raw_text"] = _render_raw_text(
            task_prompt=sample.task_prompt,
            selected=selected,
            keypoints=(instruction.keypoints if instruction is not None else ()),
            tool_state_items=predicted_user_model["tool_state"],
        )
        predicted_user_model["confidence"] = _estimate_confidence(selected, rankings[1] if len(rankings) > 1 else None)

        metadata = {
            "adapter_version": "tcpi_etapp_v1",
            "query_strategy": "public_pool_inversion_no_extra_queries",
            "selected_profile_name": selected.profile_name,
            "selected_profile_path": selected.profile_path,
            "selected_section": selected.section_name,
            "selected_section_score": round(selected.score, 6),
            "matched_values": list(selected.matched_values),
            "matched_anchors": list(selected.matched_anchors),
            "candidate_rankings": [
                {
                    "profile_name": item.profile_name,
                    "section_name": item.section_name,
                    "score": round(item.score, 6),
                    "matched_values": list(item.matched_values),
                    "matched_anchors": list(item.matched_anchors),
                }
                for item in rankings[:5]
            ],
            "instruction_query": instruction.query if instruction is not None else "",
            "instruction_keypoints": list(instruction.keypoints) if instruction is not None else [],
            "observed_tools": list(observed_tools),
            "observed_fields": dict(observed_fields),
            "public_pool_profile_count": len(self._etapp_pool.profiles),
            "public_pool_instruction_count": len(self._etapp_pool.instructions),
            "prompt_tokens": estimate_token_count(sample.task_prompt + "\n" + assistant_text),
            "completion_tokens": estimate_token_count(predicted_user_model["raw_text"]),
            "estimated_usd": 0.0,
            "model_calls": 0,
            "cost_metadata": {
                "extra_queries": 0,
                "pool_profiles_scored": len(self._etapp_pool.profiles),
                "pool_sections_scored": len(rankings),
            },
        }
        return AttackPrediction(
            baseline=self.spec.baseline,
            sample_id=sample.sample_id,
            predicted_user_model=predicted_user_model,
            source_refs=tuple(
                dedupe_preserve_order(
                    [
                        "visible_message:assistant",
                        f"public_profile:{selected.profile_path}#{selected.section_name}",
                        (
                            f"public_instruction:{instruction.task_id}"
                            if instruction is not None and instruction.task_id
                            else ""
                        ),
                    ]
                )
            ),
            notes="Task-conditioned public candidate inversion over ETAPP public profiles.",
            metadata=metadata,
        )


@lru_cache(maxsize=4)
def _load_etapp_public_pool(project_root: str) -> EtappPublicPool:
    root = Path(project_root)
    instruction_rows = read_jsonl(root / "data" / "interim" / "benchmarks" / "ETAPP" / "expanded_v1" / "instructions.jsonl")
    instructions = tuple(
        PublicInstruction(
            task_id=str(row.get("source_task_id") or row.get("task_id") or ""),
            query=str(row.get("query") or ""),
            source_query=str(row.get("source_instruction_query") or row.get("query") or ""),
            available_tools=tuple(str(item) for item in row.get("available_tools", []) if str(item).strip()),
            # ETAPP keypoints are evaluator references, not attacker-visible context.
            keypoints=(),
        )
        for row in instruction_rows
    )

    profile_dir = root / "data" / "benchmarks" / "ETAPP" / "profile" / "concrete_profile"
    profiles = []
    for path in sorted(profile_dir.glob("profile_*.json")):
        payload = read_json(path)
        sections = {
            str(section_name): value
            for section_name, value in payload.items()
            if isinstance(section_name, str) and value not in (None, "", [], {})
        }
        profiles.append(
            PublicProfile(
                profile_name=path.stem.removeprefix("profile_").replace("_", " "),
                path=path,
                sections=sections,
            )
        )
    return EtappPublicPool(instructions=instructions, profiles=tuple(profiles))


def _parse_action_signature(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    if not stripped.startswith("{"):
        return {}
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _collect_observed_tools(sample: AttackInput, action_signature: Mapping[str, Any]) -> tuple[str, ...]:
    names: list[str] = []
    for tool in sample.visible_tools:
        name = str(tool.get("name") or "").strip()
        if name:
            names.append(name)
    for item in sample.visible_tool_results:
        name = str(item.get("tool_name") or "").strip()
        if name:
            names.append(name)
    tool_sequence = action_signature.get("tool_sequence")
    if isinstance(tool_sequence, Sequence) and not isinstance(tool_sequence, (str, bytes, bytearray)):
        for step in tool_sequence:
            if not isinstance(step, Mapping):
                continue
            name = str(step.get("tool_name") or "").strip()
            if name:
                names.append(name)
    return tuple(dedupe_preserve_order(names))


def _collect_observed_fields(sample: AttackInput, action_signature: Mapping[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    if isinstance(action_signature.get("key_decision_fields"), Mapping):
        for key, value in action_signature["key_decision_fields"].items():
            fields[str(key)] = value
    tool_sequence = action_signature.get("tool_sequence")
    if isinstance(tool_sequence, Sequence) and not isinstance(tool_sequence, (str, bytes, bytearray)):
        for step in tool_sequence:
            if not isinstance(step, Mapping):
                continue
            for key_name in ("key_decision_fields", "normalized_args"):
                if not isinstance(step.get(key_name), Mapping):
                    continue
                for key, value in step[key_name].items():
                    fields[str(key)] = value
    for item in sample.visible_tool_results:
        args = item.get("normalized_args")
        if not isinstance(args, Mapping):
            continue
        for key, value in args.items():
            fields[str(key)] = value
    return fields


def _collect_observed_values(fields: Mapping[str, Any]) -> tuple[Any, ...]:
    values: list[Any] = []
    for value in fields.values():
        values.extend(_flatten_scalars(value))
    return tuple(values)


def _flatten_scalars(value: Any) -> list[Any]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, Mapping):
        rows: list[Any] = []
        for item in value.values():
            rows.extend(_flatten_scalars(item))
        return rows
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        rows: list[Any] = []
        for item in value:
            rows.extend(_flatten_scalars(item))
        return rows
    return [value]


def _match_instruction(
    *,
    sample: AttackInput,
    observed_tools: Sequence[str],
    pool: EtappPublicPool,
) -> PublicInstruction | None:
    prompt = _canonical_text(sample.task_prompt)
    prompt_tokens = set(_tokenize(prompt))
    observed_tool_set = {_canonical_text(tool) for tool in observed_tools}

    best: tuple[float, PublicInstruction | None] = (-1.0, None)
    for instruction in pool.instructions:
        query = _canonical_text(instruction.query)
        source_query = _canonical_text(instruction.source_query)
        query_tokens = set(_tokenize(query))
        source_tokens = set(_tokenize(source_query))
        tool_set = {_canonical_text(tool) for tool in instruction.available_tools}

        score = 0.0
        if prompt and prompt == query:
            score += 12.0
        if prompt and prompt == source_query:
            score += 9.0
        score += 4.0 * _jaccard(prompt_tokens, query_tokens)
        score += 2.0 * _jaccard(prompt_tokens, source_tokens)
        score += 2.5 * _jaccard(observed_tool_set, tool_set)
        if observed_tool_set and observed_tool_set == tool_set:
            score += 1.5
        if score > best[0]:
            best = (score, instruction)
    return best[1]


def _extract_anchor_fields(keypoints: Sequence[str]) -> tuple[str, ...]:
    anchors: list[str] = []
    for keypoint in keypoints:
        for match in _FIELD_HINT_RE.finditer(keypoint):
            anchors.append(match.group(1))
    return tuple(dedupe_preserve_order(anchors))


def _prompt_keywords(prompt: str) -> tuple[str, ...]:
    tokens = []
    for token in _tokenize(_canonical_text(prompt)):
        if len(token) >= 4 and token not in {"please", "usual", "preferences", "like", "with", "fits", "that", "this"}:
            tokens.append(token)
    return tuple(dedupe_preserve_order(tokens))


def _rank_public_sections(
    *,
    pool: EtappPublicPool,
    observed_tools: Sequence[str],
    observed_values: Sequence[Any],
    anchors: Sequence[str],
    prompt_keywords: Sequence[str],
) -> list[RankedSection]:
    rankings: list[RankedSection] = []
    for profile in pool.profiles:
        for section_name, payload in profile.sections.items():
            score, matched_values, matched_anchors, payload_size = _score_section(
                section_name=section_name,
                payload=payload,
                observed_tools=observed_tools,
                observed_values=observed_values,
                anchors=anchors,
                prompt_keywords=prompt_keywords,
            )
            rankings.append(
                RankedSection(
                    profile_name=profile.profile_name,
                    profile_path=str(profile.path),
                    section_name=section_name,
                    score=score,
                    payload_size=payload_size,
                    matched_values=matched_values,
                    matched_anchors=matched_anchors,
                    payload=dict(payload) if isinstance(payload, Mapping) else {"value": payload},
                )
            )
    rankings.sort(
        key=lambda item: (
            -item.score,
            -len(item.matched_values),
            -len(item.matched_anchors),
            item.payload_size,
            item.profile_name,
            item.section_name,
        )
    )
    return rankings


def _score_section(
    *,
    section_name: str,
    payload: Any,
    observed_tools: Sequence[str],
    observed_values: Sequence[Any],
    anchors: Sequence[str],
    prompt_keywords: Sequence[str],
) -> tuple[float, tuple[str, ...], tuple[str, ...], int]:
    section_blob = stable_json({section_name: payload})
    section_text = _canonical_text(section_blob)
    section_name_text = _canonical_text(section_name)

    score = 0.0
    matched_values: list[str] = []
    matched_anchors: list[str] = []

    for anchor in anchors:
        anchor_text = _canonical_text(anchor)
        if anchor_text and anchor_text in section_text:
            score += 3.0
            matched_anchors.append(anchor)

    for value in observed_values:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            numeric_value = float(value)
            if _matches_numeric_range(section_blob, numeric_value):
                score += 5.0
                matched_values.append(str(value))
            continue
        value_text = _canonical_text(value)
        if not value_text:
            continue
        if value_text in section_text:
            score += 8.0 if len(value_text) >= 8 else 5.0
            matched_values.append(str(value))
            continue
        value_tokens = [token for token in _tokenize(value_text) if len(token) >= 4]
        token_hits = sum(1 for token in value_tokens if token in section_text)
        if token_hits:
            score += min(3.0, 0.75 * token_hits)
            matched_values.append(str(value))

    for tool_name in observed_tools:
        tool_tokens = [token for token in _tokenize(_canonical_text(tool_name)) if len(token) >= 4]
        if section_name_text in tool_tokens:
            score += 1.5
        elif any(token in section_name_text or section_name_text in token for token in tool_tokens):
            score += 1.0

    keyword_hits = sum(1 for token in prompt_keywords if token in section_text)
    if keyword_hits:
        score += min(1.0, 0.25 * keyword_hits)

    return (
        round(score, 6),
        tuple(dedupe_preserve_order(matched_values)),
        tuple(dedupe_preserve_order(matched_anchors)),
        len(section_blob),
    )


def _matches_numeric_range(text: str, value: float) -> bool:
    for left_text, right_text in _RANGE_RE.findall(text):
        left = float(left_text)
        right = float(right_text)
        if left <= value <= right or right <= value <= left:
            return True
    return False


def _build_tool_hints(sample: AttackInput, action_signature: Mapping[str, Any]) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    tool_sequence = action_signature.get("tool_sequence")
    if isinstance(tool_sequence, Sequence) and not isinstance(tool_sequence, (str, bytes, bytearray)):
        for step in tool_sequence:
            if not isinstance(step, Mapping):
                continue
            user_related_args: dict[str, Any] = {}
            for key_name in ("key_decision_fields", "normalized_args"):
                if isinstance(step.get(key_name), Mapping):
                    user_related_args.update({str(key): value for key, value in step[key_name].items()})
            hints.append(
                {
                    "tool_name": str(step.get("tool_name") or ""),
                    "user_related_args": user_related_args,
                }
            )
    elif sample.visible_tool_results:
        for item in sample.visible_tool_results:
            hints.append(
                {
                    "tool_name": str(item.get("tool_name") or ""),
                    "user_related_args": dict(item.get("normalized_args") or {}),
                }
            )
    return hints


def _build_tool_state_items(
    *,
    tool_hints: Sequence[Mapping[str, Any]],
    user_related_fields: Mapping[str, Any],
    observed_tools: Sequence[str],
) -> list[str]:
    items: list[str] = []
    if tool_hints or user_related_fields:
        items.append(_pseudo_json({"tool_hints": list(tool_hints), "user_related_fields": dict(user_related_fields)}))
    if user_related_fields:
        items.append(_pseudo_json(dict(user_related_fields)))
    for tool_name in observed_tools:
        if tool_name:
            items.append(f"tool_name={tool_name}")
    for key, value in user_related_fields.items():
        items.append(f"{key}={value}")
    return dedupe_preserve_order(items)


def _render_raw_text(
    *,
    task_prompt: str,
    selected: RankedSection,
    keypoints: Sequence[str],
    tool_state_items: Sequence[str],
) -> str:
    lines = [
        "Task:",
        f"- {task_prompt}",
        "Preferences:",
        f"- {_pseudo_json({selected.section_name: selected.payload})}",
    ]
    if keypoints:
        lines.append(f"- {_pseudo_json(list(keypoints))}")
    if tool_state_items:
        lines.append("Tool state:")
        lines.extend(f"- {item}" for item in tool_state_items)
    return "\n".join(lines)


def _estimate_confidence(best: RankedSection, second: RankedSection | None) -> float:
    margin = best.score - (second.score if second is not None else 0.0)
    confidence = 0.35 + min(0.4, 0.04 * best.score) + min(0.2, 0.05 * max(margin, 0.0))
    return round(min(0.99, max(0.0, confidence)), 3)


def _pseudo_json(value: Any) -> str:
    return stable_json(value).replace('"', "'")


def _canonical_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    for source, target in _CANONICAL_REPLACEMENTS.items():
        text = text.replace(source, target)
    text = re.sub(r"[`'\"“”‘’]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([=:/|,;])\s*", r"\1", text)
    return text.strip(" .;,")


def _tokenize(text: str) -> list[str]:
    return [match.group(0) for match in _WORD_RE.finditer(text or "")]


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
