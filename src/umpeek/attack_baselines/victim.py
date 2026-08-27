from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from umpeek.exp1_whitebox.schema import clone_json

from .schema import EXPERIMENT_VERSION, assert_public_attack_payload


@dataclass(frozen=True, slots=True)
class VictimTurn:
    prompt: str
    tool_name: str | None = None
    tool_input: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    experiment_version: str = EXPERIMENT_VERSION

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("VictimTurn.prompt must be non-empty.")
        assert_public_attack_payload(self.tool_input)
        assert_public_attack_payload(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_version": self.experiment_version,
            "prompt": self.prompt,
            "tool_name": self.tool_name,
            "tool_input": clone_json(self.tool_input),
            "metadata": clone_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class VictimObservation:
    response_text: str = ""
    visible_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    visible_tool_results: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"
    metadata: dict[str, Any] = field(default_factory=dict)
    experiment_version: str = EXPERIMENT_VERSION

    def __post_init__(self) -> None:
        assert_public_attack_payload(self.visible_tool_calls)
        assert_public_attack_payload(self.visible_tool_results)
        assert_public_attack_payload(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_version": self.experiment_version,
            "response_text": self.response_text,
            "visible_tool_calls": clone_json(self.visible_tool_calls),
            "visible_tool_results": clone_json(self.visible_tool_results),
            "finish_reason": self.finish_reason,
            "metadata": clone_json(self.metadata),
        }


class VictimClient(Protocol):
    def interact(self, turns: Sequence[VictimTurn]) -> VictimObservation:
        ...


class NoOpVictimClient:
    supports_constructed_prompts = False

    def interact(self, turns: Sequence[VictimTurn]) -> VictimObservation:
        return VictimObservation(
            response_text="",
            metadata={"turn_count": len(turns), "runtime_mode": "scaffold_noop"},
        )


class ReplayVictimClient:
    supports_constructed_prompts = False

    def __init__(
        self,
        *,
        seed_observation: VictimObservation,
        max_queries: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._seed_observation = seed_observation
        self._max_queries = max_queries
        self._metadata = dict(metadata or {})
        self._query_count = 0
        self._budget_exhausted = False

    @property
    def query_count(self) -> int:
        return self._query_count

    @property
    def budget_exhausted(self) -> bool:
        return self._budget_exhausted

    def interact(self, turns: Sequence[VictimTurn]) -> VictimObservation:
        if self._max_queries is not None and self._query_count >= self._max_queries:
            self._budget_exhausted = True
            return VictimObservation(
                response_text="",
                finish_reason="budget_exhausted",
                metadata={
                    **self._metadata,
                    "query_count": self._query_count,
                    "runtime_mode": "replay_budget_exhausted",
                },
            )

        self._query_count += 1
        last_prompt = turns[-1].prompt if turns else ""
        return VictimObservation(
            response_text=self._seed_observation.response_text,
            visible_tool_calls=clone_json(self._seed_observation.visible_tool_calls),
            visible_tool_results=clone_json(self._seed_observation.visible_tool_results),
            finish_reason=self._seed_observation.finish_reason,
            metadata={
                **clone_json(self._seed_observation.metadata),
                **self._metadata,
                "last_prompt": last_prompt,
                "query_count": self._query_count,
                "runtime_mode": "replay_single_observation",
            },
        )


class ExecutableProbeVictimClient:
    """Evaluation-side contract for a victim that actually executes a new benign probe."""

    supports_constructed_prompts = True

    def __init__(
        self,
        *,
        executor: Callable[[VictimTurn], VictimObservation],
        max_queries: int | None = 1,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._executor = executor
        self._max_queries = max_queries
        self._metadata = dict(metadata or {})
        self._query_count = 0
        self._budget_exhausted = False

    @property
    def query_count(self) -> int:
        return self._query_count

    @property
    def budget_exhausted(self) -> bool:
        return self._budget_exhausted

    def interact(self, turns: Sequence[VictimTurn]) -> VictimObservation:
        if not turns:
            raise ValueError("ExecutableProbeVictimClient requires one non-empty probe turn.")
        if self._max_queries is not None and self._query_count >= self._max_queries:
            self._budget_exhausted = True
            return VictimObservation(
                response_text="",
                finish_reason="budget_exhausted",
                metadata={
                    **self._metadata,
                    "query_count": self._query_count,
                    "runtime_mode": "executable_probe_budget_exhausted",
                },
            )

        self._query_count += 1
        observation = self._executor(turns[-1])
        return VictimObservation(
            response_text=observation.response_text,
            visible_tool_calls=clone_json(observation.visible_tool_calls),
            visible_tool_results=clone_json(observation.visible_tool_results),
            finish_reason=observation.finish_reason,
            metadata={
                **clone_json(observation.metadata),
                **self._metadata,
                "query_count": self._query_count,
                "runtime_mode": "executable_constructed_probe",
            },
        )
