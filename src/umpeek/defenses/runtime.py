from __future__ import annotations

from collections import Counter
import os
from dataclasses import replace
from typing import Any, Mapping, Sequence

from umpeek.attack_baselines.schema import AttackInput
from umpeek.attack_baselines.victim import VictimClient, VictimObservation, VictimTurn
from umpeek.exp1_whitebox.schema import clone_json
from umpeek.real_agent.llm import QwenVLLMClient

from .config import benchmark_config, configured_defense_name, load_defense_config
from .privacy_checker import PrivacyCheckerDefense
from .schema import DefenseContext
from .stateful_counterfactual import StatefulCounterfactualExposureControl
from .theory_of_mind import TheoryOfMindDefense


class TheoryOfMindInitialQueryGate:
    """Run the ToM decision before the first victim response is generated."""

    def __init__(self, defense: TheoryOfMindDefense) -> None:
        self._defense = defense
        self._audit: Any | None = None
        self._blocked_observation: VictimObservation | None = None

    def __call__(self, sample: AttackInput) -> VictimObservation | None:
        blocked, audit = self._defense.review(sample.task_prompt)
        self._audit = audit
        if blocked:
            self._blocked_observation = self._defense.blocked_observation(audit)
        return self._blocked_observation

    def protect(self, observation: VictimObservation) -> VictimObservation:
        if self._audit is None:
            raise RuntimeError("The initial ToM gate was not evaluated.")
        if self._blocked_observation is not None:
            return self._blocked_observation
        return self._defense.annotate_allowed(observation, self._audit)


class DefendedVictimClient:
    supports_constructed_prompts = True

    def __init__(
        self,
        *,
        victim: VictimClient,
        defense: PrivacyCheckerDefense | TheoryOfMindDefense | StatefulCounterfactualExposureControl,
        max_queries: int,
        initial_query: str,
        initial_observation: VictimObservation,
    ) -> None:
        self._victim = victim
        self._defense = defense
        self._max_queries = int(max_queries)
        self._query_count = 0
        self._budget_exhausted = False
        self._defense_actions: Counter[str] = Counter()
        self._defense_totals: Counter[str] = Counter()
        self._latest_ledger: dict[str, Any] = {}
        self._defense.prime(initial_query, initial_observation)
        self._record_defense_audit(initial_observation)

    @property
    def query_count(self) -> int:
        return self._query_count

    @property
    def budget_exhausted(self) -> bool:
        return self._budget_exhausted or bool(getattr(self._victim, "budget_exhausted", False))

    @property
    def defense_audit_summary(self) -> dict[str, Any]:
        return {
            "defense": str(getattr(self._defense, "name", "adaptive_defense")),
            "audited_visible_turn_count": sum(self._defense_actions.values()),
            "action_counts": dict(sorted(self._defense_actions.items())),
            "model_calls": int(self._defense_totals["model_calls"]),
            "checks": int(self._defense_totals["checks"]),
            "revisions": int(self._defense_totals["revisions"]),
            "blocked_turn_count": int(self._defense_totals["blocked"]),
            "candidate_evidence_count": int(self._defense_totals["candidate_evidence_count"]),
            "released_evidence_count": int(self._defense_totals["released_evidence_count"]),
            "risky_evidence_count": int(self._defense_totals["risky_evidence_count"]),
            **clone_json(self._latest_ledger),
        }

    def interact(self, turns: Sequence[VictimTurn]) -> VictimObservation:
        if self._query_count >= self._max_queries:
            self._budget_exhausted = True
            return VictimObservation(
                response_text="",
                visible_tool_calls=[],
                visible_tool_results=[],
                finish_reason="budget_exhausted",
                metadata={"query_count": self._query_count, "adaptive_defense": True},
            )
        if not turns:
            raise ValueError("DefendedVictimClient requires at least one turn.")
        self._query_count += 1
        query = str(turns[-1].prompt)
        if isinstance(self._defense, TheoryOfMindDefense):
            blocked, audit = self._defense.review(query)
            if blocked:
                result = self._defense.blocked_observation(audit)
                self._record_defense_audit(result)
                return result
            observation = self._victim.interact(turns)
            result = self._defense.annotate_allowed(observation, audit)
            self._record_defense_audit(result)
            return result
        observation = self._victim.interact(turns)
        if isinstance(self._defense, StatefulCounterfactualExposureControl):
            counterfactual_fn = getattr(self._victim, "interact_without_personalization", None)
            counterfactual = (
                counterfactual_fn(turns)
                if self._defense.requires_counterfactual and callable(counterfactual_fn)
                else None
            )
            result = self._defense.protect(query, observation, counterfactual)
        else:
            result = self._defense.protect(query, observation)
        self._record_defense_audit(result)
        return result

    def _record_defense_audit(self, observation: VictimObservation) -> None:
        audit = observation.metadata.get("adaptive_defense")
        if not isinstance(audit, Mapping):
            return
        action = str(audit.get("action") or "unknown")
        self._defense_actions[action] += 1
        for key in (
            "model_calls",
            "checks",
            "revisions",
            "candidate_evidence_count",
            "released_evidence_count",
            "risky_evidence_count",
        ):
            self._defense_totals[key] += int(audit.get(key, 0) or 0)
        self._defense_totals["blocked"] += int(bool(audit.get("blocked")))
        for source, target in (
            ("ledger_entry_count", "ledger_entry_count"),
            ("ledger_max_score", "ledger_max_score"),
            ("intervention_count", "intervention_count"),
            ("exposure_threshold", "exposure_threshold"),
            ("use_cross_request_state", "use_cross_request_state"),
            ("use_counterfactual_comparison", "use_counterfactual_comparison"),
        ):
            if source in audit:
                self._latest_ledger[target] = audit[source]


def apply_configured_defense_to_payload(
    payload: Mapping[str, Any],
    *,
    llm: Any | None = None,
    config: Mapping[str, Any] | None = None,
    initial_gate: TheoryOfMindInitialQueryGate | None = None,
) -> dict[str, Any]:
    name = configured_defense_name()
    if name == "none":
        return dict(payload)
    attack_input = payload["attack_input"]
    if not isinstance(attack_input, AttackInput):
        raise TypeError("real-agent defense payload requires an AttackInput")
    context = DefenseContext.from_attack_input(attack_input)
    defense = _build_defense(name=name, context=context, llm=llm, config=config)
    seed = payload["seed_observation"]
    if not isinstance(seed, VictimObservation):
        raise TypeError("real-agent defense payload requires a VictimObservation")

    if initial_gate is not None:
        if not isinstance(defense, TheoryOfMindDefense):
            raise TypeError("An initial query gate is only valid for the Theory-of-Mind defense.")
        protected = initial_gate.protect(seed)
    elif isinstance(defense, TheoryOfMindDefense):
        blocked, audit = defense.review(context.task_prompt)
        protected = defense.blocked_observation(audit) if blocked else defense.annotate_allowed(seed, audit)
    elif isinstance(defense, StatefulCounterfactualExposureControl):
        counterfactual_seed = payload.get("counterfactual_seed_observation")
        if counterfactual_seed is not None and not isinstance(counterfactual_seed, VictimObservation):
            raise TypeError("counterfactual_seed_observation must be a VictimObservation")
        protected = defense.protect(context.task_prompt, seed, counterfactual_seed)
    else:
        protected = defense.protect(context.task_prompt, seed)

    patched_attack = _attack_input_with_observation(attack_input, protected, defense_name=name)
    behavior = _visible_behavior(protected)
    replay = payload["replay_context"]
    replay = replace(
        replay,
        original_behavior=clone_json(behavior),
        target_behavior=clone_json(behavior),
        metadata={
            **clone_json(replay.metadata),
            "adaptive_defense": name,
            "defended_visible_behavior": True,
        },
    )
    out = dict(payload)
    out.update(
        {
            "attack_input": patched_attack,
            "seed_observation": protected,
            "replay_context": replay,
            "original_behavior": clone_json(behavior),
            "metadata": {
                **clone_json(payload.get("metadata", {})),
                "adaptive_defense": name,
                "defended_visible_behavior": True,
            },
        }
    )
    out.pop("counterfactual_seed_observation", None)
    return out


def configured_initial_query_gate(
    attack_input: AttackInput,
    *,
    llm: Any | None = None,
    config: Mapping[str, Any] | None = None,
) -> TheoryOfMindInitialQueryGate | None:
    """Return a pre-response gate only for the defense that requires one."""

    name = configured_defense_name()
    if name != "theory_of_mind":
        return None
    context = DefenseContext.from_attack_input(attack_input)
    defense = _build_defense(name=name, context=context, llm=llm, config=config)
    if not isinstance(defense, TheoryOfMindDefense):
        raise TypeError("Theory-of-Mind configuration did not build the expected defense.")
    return TheoryOfMindInitialQueryGate(defense)


def wrap_configured_victim_client(
    victim: VictimClient,
    *,
    attack_input: AttackInput,
    seed_observation: VictimObservation,
    max_queries: int,
    llm: Any | None = None,
    config: Mapping[str, Any] | None = None,
) -> VictimClient:
    name = configured_defense_name()
    if name == "none":
        return victim
    context = DefenseContext.from_attack_input(attack_input)
    defense = _build_defense(name=name, context=context, llm=llm, config=config)
    return DefendedVictimClient(
        victim=victim,
        defense=defense,
        max_queries=max_queries,
        initial_query=context.task_prompt,
        initial_observation=seed_observation,
    )


def _build_defense(
    *,
    name: str,
    context: DefenseContext,
    llm: Any | None,
    config: Mapping[str, Any] | None,
) -> PrivacyCheckerDefense | TheoryOfMindDefense | StatefulCounterfactualExposureControl:
    resolved = dict(config or load_defense_config())
    client = llm or QwenVLLMClient()
    if name == "privacy_checker":
        section = dict(resolved.get("privacy_checker") or {})
        return PrivacyCheckerDefense(
            llm=client,
            context=context,
            max_revisions=int(section.get("max_revisions", 2)),
            fallback_response=str(section.get("fallback_response") or "I can help without sharing those details."),
        )
    if name == "theory_of_mind":
        section = dict(resolved.get("theory_of_mind") or {})
        per_benchmark = benchmark_config(resolved, context.benchmark)
        combinations = per_benchmark.get("sensitive_combinations") or section.get("sensitive_combinations") or []
        return TheoryOfMindDefense(
            llm=client,
            context=context,
            sensitive_combinations=combinations,
            block_message=str(section.get("block_message") or ""),
        )
    if name == "stateful_counterfactual":
        section = dict(resolved.get("stateful_counterfactual") or {})
        return StatefulCounterfactualExposureControl(
            llm=client,
            context=context,
            exposure_threshold=_env_float(
                "UMPEEK_STATEFUL_EXPOSURE_THRESHOLD",
                section.get("exposure_threshold", 1.0),
            ),
            first_evidence_weight=float(section.get("first_evidence_weight", 0.45)),
            repeated_context_weight=float(section.get("repeated_context_weight", 0.65)),
            same_context_weight=float(section.get("same_context_weight", 0.10)),
            direct_expression_weight=float(section.get("direct_expression_weight", 1.0)),
            low_alternative_weight=float(section.get("low_alternative_weight", 0.35)),
            use_cross_request_state=_env_bool(
                "UMPEEK_STATEFUL_USE_CROSS_REQUEST_STATE",
                section.get("use_cross_request_state", True),
            ),
            use_counterfactual_comparison=_env_bool(
                "UMPEEK_STATEFUL_USE_COUNTERFACTUAL_COMPARISON",
                section.get("use_counterfactual_comparison", True),
            ),
            neutral_response=str(
                section.get("neutral_response")
                or "I need a little more information to complete that request."
            ),
        )
    raise ValueError(f"Unsupported adaptive defense: {name}")


def _env_float(name: str, default: Any) -> float:
    raw = str(os.environ.get(name) or "").strip()
    return float(raw) if raw else float(default)


def _env_bool(name: str, default: Any) -> bool:
    raw = str(os.environ.get(name) or "").strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _attack_input_with_observation(
    sample: AttackInput,
    observation: VictimObservation,
    *,
    defense_name: str,
) -> AttackInput:
    payload = sample.to_dict()
    messages = [dict(item) for item in payload.get("visible_messages", []) if str(item.get("role")) != "assistant"]
    if observation.response_text.strip():
        messages.append({"role": "assistant", "content": observation.response_text})
    payload["visible_messages"] = messages
    payload["visible_tool_calls"] = clone_json(observation.visible_tool_calls)
    payload["visible_tool_results"] = clone_json(observation.visible_tool_results)
    payload["metadata"] = {
        **dict(payload.get("metadata", {})),
        "adaptive_defense": defense_name,
        "defended_visible_behavior": True,
    }
    return AttackInput.from_dict(payload)


def _visible_behavior(observation: VictimObservation) -> Any:
    if observation.visible_tool_results:
        return clone_json(observation.visible_tool_results)
    if observation.visible_tool_calls:
        return clone_json(observation.visible_tool_calls)
    return str(observation.response_text)
