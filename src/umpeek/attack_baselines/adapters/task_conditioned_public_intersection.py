from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from umpeek.attack_baselines.adapters.common import dedupe_preserve_order, estimate_token_count, visible_assistant_text
from umpeek.attack_baselines.schema import AttackBaselineSpec, AttackInput, AttackPrediction, blank_predicted_user_model
from umpeek.exp1_whitebox.schema import stable_json

from .task_conditioned_public_inversion import (
    RankedSection,
    TaskConditionedPublicInversionAdapter,
    _build_tool_hints,
    _build_tool_state_items,
    _collect_observed_fields,
    _collect_observed_tools,
    _collect_observed_values,
    _estimate_confidence,
    _extract_anchor_fields,
    _match_instruction,
    _parse_action_signature,
    _prompt_keywords,
    _pseudo_json,
    _rank_public_sections,
)


def build_task_conditioned_public_intersection_spec() -> AttackBaselineSpec:
    return AttackBaselineSpec(
        baseline="task_conditioned_public_intersection",
        short_name="TCPI-INT",
        paper_title="In-repo research prototype: ambiguity-aware public intersection",
        paper_url="https://example.invalid/umpeek/tcpi-intersection",
        code_name="in_repo_task_conditioned_public_intersection",
        code_url="https://example.invalid/umpeek/tcpi-intersection",
        open_source=True,
        adaptation_target=(
            "Resolve ETAPP public-candidate ambiguity by returning the minimal intersection slice shared by "
            "exactly tied task-consistent candidates."
        ),
        implementation_status="success",
        notes=(
            "ETAPP-only round-two research adapter.",
            "Uses the same visible inputs as TCPI but replaces hard top-1 selection with exact-top ambiguity intersection.",
            "Evaluation keypoints are deliberately excluded by the parent adapter.",
            "Does not query the replay victim; attack cost stays at zero queries.",
        ),
        metadata={
            "research_round": "R002",
            "applicable_benchmarks": ["ETAPP"],
            "threat_model": "black_box_visible_observation_plus_public_candidate_pool",
            "parent_method": "task_conditioned_public_inversion",
        },
    )


class TaskConditionedPublicIntersectionAdapter(TaskConditionedPublicInversionAdapter):
    def __init__(self, spec: AttackBaselineSpec, *, project_root: Path | str | None = None) -> None:
        super().__init__(spec, project_root=project_root)

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
                notes="Round-two public intersection is only implemented for ETAPP.",
                metadata={
                    "adapter_version": "tcpi_intersection_v1",
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
                    "adapter_version": "tcpi_intersection_v1",
                    "observed_tools": list(observed_tools),
                    "observed_fields": dict(observed_fields),
                    "public_pool_profile_count": len(self._etapp_pool.profiles),
                    "public_pool_instruction_count": len(self._etapp_pool.instructions),
                },
            )

        ambiguity_group = _select_exact_top_ambiguity_group(rankings)
        resolved_section, ambiguity_resolution = _resolve_ambiguous_section(selected, ambiguity_group)
        runner_up = _first_non_group_candidate(rankings, ambiguity_group)

        tool_hints = _build_tool_hints(sample, action_signature)
        user_related_fields = dict(observed_fields)
        predicted_user_model = blank_predicted_user_model()
        predicted_user_model["preferences"] = []
        if resolved_section is not None and resolved_section.payload not in (None, {}, []):
            predicted_user_model["preferences"].append(
                _pseudo_json({resolved_section.section_name: resolved_section.payload})
            )
        if instruction is not None and instruction.keypoints:
            predicted_user_model["preferences"].append(_pseudo_json(list(instruction.keypoints)))
        predicted_user_model["tool_state"] = _build_tool_state_items(
            tool_hints=tool_hints,
            user_related_fields=user_related_fields,
            observed_tools=observed_tools,
        )
        predicted_user_model["raw_text"] = _render_resolved_raw_text(
            task_prompt=sample.task_prompt,
            resolved_section=resolved_section,
            keypoints=(instruction.keypoints if instruction is not None else ()),
            tool_state_items=predicted_user_model["tool_state"],
        )
        predicted_user_model["confidence"] = _estimate_confidence(selected, runner_up)

        metadata = {
            "adapter_version": "tcpi_intersection_v1",
            "query_strategy": "public_intersection_no_extra_queries",
            "selected_profile_name": selected.profile_name,
            "selected_profile_path": selected.profile_path,
            "selected_section": selected.section_name,
            "selected_section_score": round(selected.score, 6),
            "matched_values": list(selected.matched_values),
            "matched_anchors": list(selected.matched_anchors),
            "ambiguity_group_size": len(ambiguity_group),
            "ambiguity_resolution": ambiguity_resolution,
            "ambiguity_group_profiles": [item.profile_name for item in ambiguity_group],
            "resolved_payload_size": (
                len(stable_json({resolved_section.section_name: resolved_section.payload}))
                if resolved_section is not None and resolved_section.payload not in (None, {}, [])
                else 0
            ),
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
        source_refs = ["visible_message:assistant"]
        source_refs.extend(
            f"public_profile:{item.profile_path}#{item.section_name}"
            for item in ambiguity_group
        )
        if instruction is not None and instruction.task_id:
            source_refs.append(f"public_instruction:{instruction.task_id}")
        return AttackPrediction(
            baseline=self.spec.baseline,
            sample_id=sample.sample_id,
            predicted_user_model=predicted_user_model,
            source_refs=tuple(dedupe_preserve_order(source_refs)),
            notes="Task-conditioned public intersection over exactly tied ETAPP candidates.",
            metadata=metadata,
        )


def _select_exact_top_ambiguity_group(rankings: Sequence[RankedSection]) -> tuple[RankedSection, ...]:
    if not rankings:
        return ()
    top = rankings[0]
    return tuple(
        item
        for item in rankings
        if item.section_name == top.section_name
        and item.score == top.score
        and item.matched_values == top.matched_values
        and item.matched_anchors == top.matched_anchors
    )


def _resolve_ambiguous_section(
    selected: RankedSection,
    ambiguity_group: Sequence[RankedSection],
) -> tuple[RankedSection | None, str]:
    if len(ambiguity_group) <= 1:
        return selected, "top1_unique"
    payload = _intersect_payloads([item.payload for item in ambiguity_group])
    if payload in (None, {}, []):
        return None, "intersection_empty"
    return (
        RankedSection(
            profile_name=selected.profile_name,
            profile_path=selected.profile_path,
            section_name=selected.section_name,
            score=selected.score,
            payload_size=len(stable_json({selected.section_name: payload})),
            matched_values=selected.matched_values,
            matched_anchors=selected.matched_anchors,
            payload=payload,
        ),
        "intersection_slice",
    )


def _intersect_payloads(payloads: Sequence[Any]) -> Any | None:
    if not payloads:
        return None
    first = payloads[0]
    if all(isinstance(item, Mapping) for item in payloads):
        common_keys = set(first)
        for payload in payloads[1:]:
            common_keys &= set(payload)
        output: dict[str, Any] = {}
        for key in sorted(common_keys):
            value = _intersect_payloads([payload[key] for payload in payloads])
            if value not in (None, {}, []):
                output[str(key)] = value
        return output or None
    if all(isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)) for item in payloads):
        common_serialized = {stable_json(item) for item in first}
        for payload in payloads[1:]:
            common_serialized &= {stable_json(item) for item in payload}
        if not common_serialized:
            return None
        return [item for item in first if stable_json(item) in common_serialized]
    first_serialized = stable_json(first)
    if all(stable_json(item) == first_serialized for item in payloads[1:]):
        return first
    return None


def _first_non_group_candidate(
    rankings: Sequence[RankedSection],
    ambiguity_group: Sequence[RankedSection],
) -> RankedSection | None:
    group_keys = {
        (item.profile_name, item.section_name, item.score, item.matched_values, item.matched_anchors)
        for item in ambiguity_group
    }
    for candidate in rankings:
        key = (
            candidate.profile_name,
            candidate.section_name,
            candidate.score,
            candidate.matched_values,
            candidate.matched_anchors,
        )
        if key not in group_keys:
            return candidate
    return None


def _render_resolved_raw_text(
    *,
    task_prompt: str,
    resolved_section: RankedSection | None,
    keypoints: Sequence[str],
    tool_state_items: Sequence[str],
) -> str:
    lines = [
        "Task:",
        f"- {task_prompt}",
    ]
    if resolved_section is not None and resolved_section.payload not in (None, {}, []):
        lines.extend(
            [
                "Preferences:",
                f"- {_pseudo_json({resolved_section.section_name: resolved_section.payload})}",
            ]
        )
    elif keypoints:
        lines.append("Preferences:")
    if keypoints:
        lines.append(f"- {_pseudo_json(list(keypoints))}")
    if tool_state_items:
        lines.append("Tool state:")
        lines.extend(f"- {item}" for item in tool_state_items)
    return "\n".join(lines)
