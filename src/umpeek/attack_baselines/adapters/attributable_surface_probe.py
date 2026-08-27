from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from umpeek.attack_baselines.adapter import AttackAdapter
from umpeek.attack_baselines.adapters.common import estimate_token_count
from umpeek.attack_baselines.adapters.observable_behavior_projection import (
    ObservableBehaviorProjectionAdapter,
    build_observable_behavior_projection_spec,
)
from umpeek.attack_baselines.adapters.task_conditioned_public_intersection import (
    TaskConditionedPublicIntersectionAdapter,
    build_task_conditioned_public_intersection_spec,
)
from umpeek.attack_baselines.adapters.task_conditioned_public_inversion import _match_instruction
from umpeek.attack_baselines.attributable_surface_probe import (
    MatchedIntentAttributableSurfaceOptimizer,
    public_representation_contract,
)
from umpeek.attack_baselines.schema import (
    AttackBaselineSpec,
    AttackInput,
    AttackPrediction,
    blank_predicted_user_model,
)
from umpeek.attack_baselines.victim import VictimTurn


def build_attributable_surface_probe_spec() -> AttackBaselineSpec:
    return AttackBaselineSpec(
        baseline="matched_intent_attributable_surface_probe",
        short_name="MI-AOSP",
        paper_title="In-repo research prototype: Matched-Intent Attributable Observation-Surface Probe",
        paper_url="https://example.invalid/umpeek/mi-aosp",
        code_name="in_repo_matched_intent_attributable_surface_probe",
        code_url="https://example.invalid/umpeek/mi-aosp",
        open_source=True,
        adaptation_target=(
            "Observe only public action arguments that have a deassociated state-carrier "
            "certificate under the unchanged original task intent."
        ),
        implementation_status="success",
        notes=(
            "R003 first mechanism test: expands only when a state-attributable surface is certified.",
            "The current ETAPP quick interface saturates its certifiable surface, so the method abstains from adding contexts.",
            "Uses one ordinary target query; evaluator-only original state remains private.",
        ),
        metadata={
            "research_round": "R003",
            "full_method_name": "Matched-Intent Attributable Observation-Surface Probe",
            "chinese_method_name": "配对意图可归因观测面探针",
            "applicable_benchmarks": ["ETAPP"],
        },
    )


def build_minimal_commitment_probe_spec() -> AttackBaselineSpec:
    return AttackBaselineSpec(
        baseline="intent_preserving_minimal_sufficient_commitment_probe",
        short_name="IP-MSCP",
        paper_title="In-repo research prototype: Intent-Preserving Minimal Sufficient Commitment Probe",
        paper_url="https://example.invalid/umpeek/ip-mscp",
        code_name="in_repo_intent_preserving_minimal_sufficient_commitment_probe",
        code_url="https://example.invalid/umpeek/ip-mscp",
        open_source=True,
        adaptation_target=(
            "Complete every already-certified state-carrying argument slot within the unchanged "
            "ordinary task intent while refusing any new scenario or unsupported action."
        ),
        implementation_status="success",
        notes=(
            "R003 second hypothesis: observation expansion is limited to filling existing attributable slots.",
            "Uses one normal target query and a public state representation contract.",
            "Does not introduce contrast contexts or optional alternative actions.",
        ),
        metadata={
            "research_round": "R003",
            "full_method_name": "Intent-Preserving Minimal Sufficient Commitment Probe",
            "chinese_method_name": "同意图最小充分承诺探针",
            "applicable_benchmarks": ["ETAPP"],
        },
    )


def build_provenance_preserving_probe_spec() -> AttackBaselineSpec:
    return AttackBaselineSpec(
        baseline="state_anchored_provenance_preserving_probe",
        short_name="SA-PPP",
        paper_title="In-repo research prototype: State-Anchored Provenance-Preserving Probe",
        paper_url="https://example.invalid/umpeek/sa-ppp",
        code_name="in_repo_state_anchored_provenance_preserving_probe",
        code_url="https://example.invalid/umpeek/sa-ppp",
        open_source=True,
        adaptation_target=(
            "Retain visible consultations of publicly certified user-bound resources only "
            "when the existing personalized state makes them relevant to the unchanged task."
        ),
        implementation_status="success",
        notes=(
            "R003 third hypothesis: state relevance may be exposed by resource-read provenance as well as argument values.",
            "Uses one normal target query and admits only certified parameter carriers or account-bound read tools.",
            "Does not add contexts, recommendations, optional alternatives, or evaluation labels.",
        ),
        metadata={
            "research_round": "R003",
            "full_method_name": "State-Anchored Provenance-Preserving Probe",
            "chinese_method_name": "状态锚定资源保真探针",
            "applicable_benchmarks": ["ETAPP"],
        },
    )


def build_dependency_closed_probe_spec() -> AttackBaselineSpec:
    return AttackBaselineSpec(
        baseline="same_goal_resource_dependency_closed_probe",
        short_name="SG-RDCP",
        paper_title="In-repo research prototype: Same-Goal Resource-Dependency-Closed Probe",
        paper_url="https://example.invalid/umpeek/sg-rdcp",
        code_name="in_repo_same_goal_resource_dependency_closed_probe",
        code_url="https://example.invalid/umpeek/sg-rdcp",
        open_source=True,
        adaptation_target=(
            "Make an unchanged ordinary task conditionally depend on existing user-bound "
            "resources in the same scope, so additional visible reads remain state-grounded."
        ),
        implementation_status="success",
        notes=(
            "R003 fourth hypothesis: a same-goal conditional dependency activates relevant existing resources without invented scenarios.",
            "Resource certificates derive only from public no-argument account-resource tool semantics.",
            "Uses one normal query; no evaluation state, trace, keypoint, or action label is consumed by the adapter.",
        ),
        metadata={
            "research_round": "R003",
            "full_method_name": "Same-Goal Resource-Dependency-Closed Probe",
            "chinese_method_name": "同目标资源依赖闭包探针",
            "applicable_benchmarks": ["ETAPP"],
        },
    )


def build_representation_invariant_probe_spec() -> AttackBaselineSpec:
    return AttackBaselineSpec(
        baseline="state_anchored_representation_invariant_inversion_probe",
        short_name="SA-RIIP",
        paper_title="In-repo research prototype: State-Anchored Representation-Invariant Inversion Probe",
        paper_url="https://example.invalid/umpeek/sa-riip",
        code_name="in_repo_state_anchored_representation_invariant_inversion_probe",
        code_url="https://example.invalid/umpeek/sa-riip",
        open_source=True,
        adaptation_target=(
            "Use observable state-bearing actions to certify activation, then recover only "
            "the invariant active-state atom declared by the public backend representation."
        ),
        implementation_status="success",
        notes=(
            "R003 fifth hypothesis: attributable visible action certifies a public backend activation invariant.",
            "The optimized query remains the provenance-preserving ordinary task from H3.",
            "Task intent and activation atoms come from public task/backend contracts; no labels or evaluator state are read.",
        ),
        metadata={
            "research_round": "R003",
            "full_method_name": "State-Anchored Representation-Invariant Inversion Probe",
            "chinese_method_name": "状态锚定表征不变量反演探针",
            "applicable_benchmarks": ["ETAPP"],
        },
    )


class AttributableSurfaceProbeAdapter(AttackAdapter):
    def __init__(
        self,
        spec: AttackBaselineSpec,
        *,
        project_root: Path | str | None = None,
        decode_mode: str = "representation_aligned",
        query_mode: str = "identity",
        optimizer: MatchedIntentAttributableSurfaceOptimizer | None = None,
    ) -> None:
        super().__init__(spec)
        if decode_mode not in {
            "projection_only",
            "representation_aligned",
            "state_anchor_invariant_aligned",
        }:
            raise ValueError(f"Unsupported attributable-surface decode mode: {decode_mode}")
        if query_mode not in {
            "identity",
            "minimal_sufficient_commitment",
            "admission_matched_minimal_commitment",
            "provenance_preserving",
            "dependency_closed",
        }:
            raise ValueError(f"Unsupported attributable-surface query mode: {query_mode}")
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[4]
        )
        self.decode_mode = decode_mode
        self.query_mode = query_mode
        self.optimizer = optimizer or MatchedIntentAttributableSurfaceOptimizer()
        self._projection_decoder = ObservableBehaviorProjectionAdapter(
            build_observable_behavior_projection_spec()
        )
        self._semantic_decoder = TaskConditionedPublicIntersectionAdapter(
            build_task_conditioned_public_intersection_spec(),
            project_root=self.project_root,
        )

    def run(
        self,
        sample: AttackInput,
        victim_client: Any,
        budget: int | Mapping[str, Any],
    ) -> AttackPrediction:
        del budget
        if sample.benchmark != "ETAPP":
            return self._not_applicable(
                sample,
                "R003 is preregistered for ETAPP visible-action tasks.",
                status="not_applicable_no_external_content",
            )
        optimization = self.optimizer.optimize(
            sample,
            project_root=self.project_root,
            query_mode=self.query_mode,
        )
        if optimization is None:
            return self._not_applicable(
                sample,
                "No public state-carrying action argument exists under this task interface.",
            )
        program = optimization.selected
        representation = public_representation_contract(sample.backend)
        common_metadata = {
            "adapter_version": "mi_aosp_v1",
            "method_module": "matched_intent_attributable_observation_surface",
            "decode_mode": self.decode_mode,
            "query_mode": self.query_mode,
            "probe_grammar_version": self.optimizer.GRAMMAR_VERSION,
            **optimization.diagnostics(),
            "public_representation_contract": {
                "state_family": representation.state_family,
                "decoder_family": representation.decoder_family,
                "public_code_basis": representation.public_code_basis,
                "active_state_invariant_category": representation.active_state_invariant_category,
            },
            "probe_prompt": program.prompt,
            "target_query_count": 0,
            "adaptive_rounds": 0,
        }
        if not bool(getattr(victim_client, "supports_constructed_prompts", False)):
            return AttackPrediction(
                baseline=self.spec.baseline,
                sample_id=sample.sample_id,
                predicted_user_model=blank_predicted_user_model(),
                status="blocked_non_executable_probe_victim",
                error_type="constructed_prompt_not_executable",
                notes="The available evaluation endpoint cannot execute a normal task prompt.",
                metadata=common_metadata,
            )
        observation = victim_client.interact(
            [
                VictimTurn(
                    prompt=program.prompt,
                    metadata={
                        "probe_program_id": program.program_id,
                        "probe_scope": "whole_original_state",
                        "observable_mode": "matched_intent_attributable_arguments",
                    },
                )
            ]
        )
        raw_sequence = list(observation.visible_tool_results or observation.visible_tool_calls)
        admitted_sequence, rejected_arguments = _admit_attributable_arguments(
            raw_sequence,
            program.attributable_argument_fields,
            program.attributable_tool_choices,
        )
        observed_attributable_fields = tuple(
            sorted(
                {
                    f"{row['tool_name']}.{key}"
                    for row in admitted_sequence
                    for key in row.get("normalized_args", {})
                }
            )
        )
        observed_attributable_tools = tuple(
            sorted(
                {
                    str(row["tool_name"])
                    for row in admitted_sequence
                    if str(row.get("tool_name") or "") in program.attributable_tool_choices
                }
            )
        )
        observed_input = AttackInput.from_dict(
            {
                "backend": sample.backend,
                "benchmark": sample.benchmark,
                "sample_id": sample.sample_id,
                "task_prompt": sample.task_prompt,
                "user_id": sample.user_id,
                "task_id": sample.task_id,
                "visible_messages": [
                    {"role": "user", "content": sample.task_prompt},
                    {"role": "assistant", "content": observation.response_text},
                ],
                "visible_tools": sample.visible_tools,
                "visible_tool_results": admitted_sequence,
                "public_context": sample.public_context,
                "metadata": sample.metadata,
            }
        )
        projection = self._projection_decoder.run(observed_input, None, {"max_queries": 0})
        semantic = self._semantic_decoder.run(observed_input, None, {"max_queries": 0})
        use_semantic = (
            self.decode_mode in {"representation_aligned", "state_anchor_invariant_aligned"}
            and representation.decoder_family == "unambiguous_public_semantic_section"
            and semantic.status == "success"
            and int(semantic.metadata.get("ambiguity_group_size", 0) or 0) == 1
        )
        predicted = (
            semantic.predicted_user_model if use_semantic else projection.predicted_user_model
        )
        recovered_public_intent = ""
        recovered_public_invariant = ""
        if self.decode_mode == "state_anchor_invariant_aligned":
            predicted, recovered_public_intent, recovered_public_invariant = (
                _augment_with_public_representation_invariants(
                    predicted,
                    sample=sample,
                    observed_input=observed_input,
                    observed_attributable_fields=observed_attributable_fields,
                    observed_attributable_tools=observed_attributable_tools,
                    representation=representation,
                    public_pool=self._semantic_decoder._etapp_pool,
                )
            )
        return AttackPrediction(
            baseline=self.spec.baseline,
            sample_id=sample.sample_id,
            predicted_user_model=predicted,
            source_refs=("constructed_normal_task:matched_intent_visible_action",),
            notes=(
                "Matched-Intent Attributable Observation-Surface Probe: decode only "
                "state-carrier-certified arguments observed under the original intent."
            ),
            metadata={
                **common_metadata,
                "target_query_count": 1,
                "adaptive_rounds": 1,
                "decoder_selected": (
                    (
                        "public_semantic_section+public_activation_invariant"
                        if use_semantic and self.decode_mode == "state_anchor_invariant_aligned"
                        else "observable_argument_projection+public_activation_invariant"
                        if self.decode_mode == "state_anchor_invariant_aligned"
                        else "public_semantic_section"
                        if use_semantic
                        else "observable_argument_projection"
                    )
                ),
                "recovered_public_task_intent": recovered_public_intent,
                "recovered_public_activation_invariant": recovered_public_invariant,
                "semantic_decoder_status": semantic.status,
                "observed_visible_tool_calls": raw_sequence,
                "admitted_visible_tool_calls": admitted_sequence,
                "rejected_unattributed_arguments": rejected_arguments,
                "observed_attributable_argument_fields": list(observed_attributable_fields),
                "observed_attributable_tool_choices": list(observed_attributable_tools),
                "realized_attributable_surface_coverage": round(
                    len(observed_attributable_fields) / program.attributable_capacity, 6
                )
                if program.attributable_capacity
                else 0.0,
                "realized_attributable_resource_coverage": round(
                    len(observed_attributable_tools) / program.attributable_choice_capacity, 6
                )
                if program.attributable_choice_capacity
                else None,
                "observed_sequence_length": len(raw_sequence),
                "admitted_action_count": len(admitted_sequence),
                "victim_runtime_mode": observation.metadata.get("runtime_mode", ""),
                "victim_planner_mode": observation.metadata.get("planner_mode", ""),
                "victim_planner_model": observation.metadata.get("planner_model", ""),
                "prompt_tokens": program.prompt_token_count,
                "completion_tokens": estimate_token_count(observation.response_text),
                "model_calls": 1,
                "cost_metadata": {"target_queries": 1, "design_cost_separate": True},
            },
        )

    def _not_applicable(
        self,
        sample: AttackInput,
        notes: str,
        *,
        status: str = "not_applicable_no_certifiable_component",
    ) -> AttackPrediction:
        return AttackPrediction(
            baseline=self.spec.baseline,
            sample_id=sample.sample_id,
            predicted_user_model=blank_predicted_user_model(),
            status=status,
            notes=notes,
            metadata={"adapter_version": "mi_aosp_v1", "target_query_count": 0},
        )


def _admit_attributable_arguments(
    sequence: Sequence[Mapping[str, Any]],
    admitted_fields: Sequence[str],
    admitted_tool_choices: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], list[str]]:
    admitted = set(admitted_fields)
    admitted_tools = set(admitted_tool_choices)
    accepted_rows: list[dict[str, Any]] = []
    rejected: list[str] = []
    for action in sequence:
        tool_name = str(action.get("tool_name") or action.get("name") or "")
        raw_args = action.get("normalized_args")
        if not isinstance(raw_args, Mapping):
            raw_args = action.get("arguments") if isinstance(action.get("arguments"), Mapping) else {}
        kept_args: dict[str, Any] = {}
        for key, value in raw_args.items():
            field_id = f"{tool_name}.{key}"
            if field_id in admitted:
                kept_args[str(key)] = value
            else:
                rejected.append(field_id)
        if kept_args or tool_name in admitted_tools:
            accepted_rows.append({"tool_name": tool_name, "normalized_args": kept_args})
    return accepted_rows, sorted(set(rejected))


def _augment_with_public_representation_invariants(
    predicted: Mapping[str, Any],
    *,
    sample: AttackInput,
    observed_input: AttackInput,
    observed_attributable_fields: Sequence[str],
    observed_attributable_tools: Sequence[str],
    representation: Any,
    public_pool: Any,
) -> tuple[dict[str, Any], str, str]:
    augmented: dict[str, Any] = {
        key: (list(value) if isinstance(value, list) else value)
        for key, value in predicted.items()
    }
    for category in ("facts", "preferences", "constraints", "relations", "tool_state"):
        augmented.setdefault(category, [])
    instruction = _match_instruction(
        sample=sample,
        observed_tools=[
            str(row.get("tool_name") or "") for row in observed_input.visible_tool_results
        ],
        pool=public_pool,
    )
    intent_atom = f"intent={instruction.task_id}" if instruction is not None and instruction.task_id else ""
    if intent_atom and intent_atom not in augmented["facts"]:
        augmented["facts"].append(intent_atom)
    invariant_atom = ""
    has_state_anchor = bool(observed_attributable_fields or observed_attributable_tools)
    category = str(representation.active_state_invariant_category or "")
    candidate_atom = str(representation.active_state_invariant_atom or "")
    if has_state_anchor and category in augmented and candidate_atom:
        invariant_atom = candidate_atom
        if candidate_atom not in augmented[category]:
            augmented[category].append(candidate_atom)
    lines = []
    for category in ("facts", "preferences", "constraints", "relations", "tool_state"):
        values = augmented.get(category, [])
        if values:
            lines.append(category.capitalize() + ":")
            lines.extend(f"- {value}" for value in values)
    augmented["raw_text"] = "\n".join(lines)
    return augmented, intent_atom, invariant_atom
