from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from umpeek.attack_baselines.adapter import AttackAdapter
from umpeek.attack_baselines.adapters.common import dedupe_preserve_order, estimate_token_count, visible_assistant_text
from umpeek.attack_baselines.schema import AttackBaselineSpec, AttackInput, AttackPrediction, blank_predicted_user_model


def build_observable_behavior_projection_spec() -> AttackBaselineSpec:
    return AttackBaselineSpec(
        baseline="observable_behavior_projection",
        short_name="OBPI",
        paper_title="In-repo research prototype: Observable Behavioral Projection Inversion",
        paper_url="https://example.invalid/umpeek/obpi",
        code_name="in_repo_observable_behavior_projection",
        code_url="https://example.invalid/umpeek/obpi",
        open_source=True,
        adaptation_target=(
            "Recover the causally active tool-state slice exposed by a personalized agent's "
            "normal observable action."
        ),
        implementation_status="success",
        notes=(
            "Round-one minimal mechanism test on tool/action tasks.",
            "Reads only the normal visible response and visible tool calls supplied in AttackInput.",
            "Does not read profile pools, evaluator keypoints, gold labels, runtime traces, or replay state.",
            "Does not send an additional victim query.",
        ),
        metadata={
            "research_round": "R001",
            "full_method_name": "Observable Behavioral Projection Inversion",
            "applicable_benchmarks": ["ETAPP"],
            "threat_model": "single_normal_interaction_visible_action_only",
        },
    )


class ObservableBehaviorProjectionAdapter(AttackAdapter):
    def run(
        self,
        sample: AttackInput,
        victim_client: Any,
        budget: int | Mapping[str, Any],
    ) -> AttackPrediction:
        del victim_client, budget
        if sample.benchmark != "ETAPP" or not (sample.visible_tool_results or sample.visible_tools):
            return AttackPrediction(
                baseline=self.spec.baseline,
                sample_id=sample.sample_id,
                predicted_user_model=blank_predicted_user_model(),
                status="not_applicable_no_external_content",
                notes="Observable behavioral projection requires a benchmark-visible action channel.",
                metadata={"adapter_version": "obpi_v1", "supported_benchmarks": ["ETAPP"]},
            )

        assistant_text = visible_assistant_text(sample.visible_messages)
        action_signature = _parse_visible_action(assistant_text)
        intent = str(action_signature.get("intent") or "").strip()
        tools = _observed_tools(sample, action_signature)
        fields = _observed_decision_fields(sample, action_signature)
        if not tools and not fields and not intent:
            return AttackPrediction(
                baseline=self.spec.baseline,
                sample_id=sample.sample_id,
                predicted_user_model=blank_predicted_user_model(),
                status="invalid_input",
                notes="The visible action channel did not expose a usable action projection.",
                metadata={"adapter_version": "obpi_v1"},
            )

        predicted = blank_predicted_user_model()
        if intent:
            predicted["facts"] = [f"intent={intent}"]
        predicted["tool_state"] = [
            *[f"tool_name={tool_name}" for tool_name in tools],
            *[f"{key}={value}" for key, value in fields.items()],
        ]
        predicted["raw_text"] = _render_projection(intent=intent, tools=tools, fields=fields)
        predicted["confidence"] = 0.85 if fields else 0.5
        return AttackPrediction(
            baseline=self.spec.baseline,
            sample_id=sample.sample_id,
            predicted_user_model=predicted,
            source_refs=tuple(
                [
                    *(["visible_message:assistant"] if assistant_text else []),
                    *[f"visible_tool_result:{index}" for index, _ in enumerate(sample.visible_tool_results)],
                ]
            ),
            notes="Observable Behavioral Projection Inversion: recover only state committed by the visible action.",
            metadata={
                "adapter_version": "obpi_v1",
                "method_module": "observable_behavior_projection",
                "observed_intent": intent,
                "observed_tools": list(tools),
                "observed_fields": dict(fields),
                "prompt_tokens": estimate_token_count(sample.task_prompt + "\n" + assistant_text),
                "completion_tokens": estimate_token_count(predicted["raw_text"]),
                "model_calls": 0,
                "adaptive_rounds": 0,
                "cost_metadata": {"extra_queries": 0, "natural_observations": 1},
            },
        )


def _parse_visible_action(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    if not stripped.startswith("{"):
        return {}
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _observed_tools(sample: AttackInput, signature: Mapping[str, Any]) -> tuple[str, ...]:
    names: list[str] = []
    for result in sample.visible_tool_results:
        if result.get("tool_name"):
            names.append(str(result["tool_name"]))
    sequence = signature.get("tool_sequence")
    if isinstance(sequence, Sequence) and not isinstance(sequence, (str, bytes, bytearray)):
        for step in sequence:
            if isinstance(step, Mapping) and step.get("tool_name"):
                names.append(str(step["tool_name"]))
    return tuple(dedupe_preserve_order(names))


def _observed_decision_fields(sample: AttackInput, signature: Mapping[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    signature_fields = signature.get("key_decision_fields")
    if isinstance(signature_fields, Mapping):
        fields.update({str(key): value for key, value in signature_fields.items()})
    for result in sample.visible_tool_results:
        args = result.get("normalized_args")
        if isinstance(args, Mapping):
            fields.update({str(key): value for key, value in args.items()})
    return fields


def _render_projection(*, intent: str, tools: Sequence[str], fields: Mapping[str, Any]) -> str:
    lines: list[str] = []
    if intent:
        lines.extend(["Facts:", f"- intent={intent}"])
    if tools or fields:
        lines.append("Tool state:")
        lines.extend(f"- tool_name={tool_name}" for tool_name in tools)
        lines.extend(f"- {key}={value}" for key, value in fields.items())
    return "\n".join(lines)
