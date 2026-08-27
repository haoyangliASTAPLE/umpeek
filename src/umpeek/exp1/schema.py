from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ConditionName = Literal["no_memory", "candidate", "delete", "swap"]
TaskType = Literal["choice", "action", "open"]
CoverageStatus = Literal["available", "missing", "unsupported"]


@dataclass(frozen=True, slots=True)
class UserRecord:
    user_id: str
    profile: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TaskRecord:
    user_id: str
    task_id: str
    benchmark: str
    task_type: TaskType
    prompt: str
    gold_label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CandidatePayload:
    candidate_id: str
    source_user_id: str
    condition: ConditionName
    user_id: str | None = None
    task_id: str | None = None
    payload_text: str = ""
    payload_json: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    source_refs: tuple[str, ...] = ()
    coverage_status: CoverageStatus = "available"
    fragments: tuple[str, ...] = ()
    user_attributes: dict[str, Any] = field(default_factory=dict)
    non_user_context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_available(self) -> bool:
        return self.coverage_status == "available"


@dataclass(frozen=True, slots=True)
class ModelOutput:
    response_text: str = ""
    predicted_label: str | None = None
    gold_rank: int | None = None
    action_signature: str | None = None
    correctness: float | None = None
    judge_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvalRecord:
    backend: str
    benchmark: str
    run_id: str
    condition: ConditionName
    candidate_id: str
    user_id: str
    task_id: str
    task_type: TaskType
    payload_user_id: str | None
    payload_present: bool
    task: TaskRecord
    payload: CandidatePayload | None
    output: ModelOutput
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def pair_key(self) -> tuple[str, str]:
        return (self.user_id, self.task_id)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        output_metadata = dict(self.output.metadata)
        record_metadata = dict(self.metadata)
        payload.update(
            {
                "sample_id": record_metadata.get("sample_id"),
                "score": (
                    self.output.correctness
                    if self.output.correctness is not None
                    else self.output.judge_score
                ),
                "timeout_status": bool(
                    record_metadata.get("timeout_status")
                    or output_metadata.get("timeout_status")
                ),
                "llm_model": (
                    output_metadata.get("llm_model")
                    or output_metadata.get("model")
                    or output_metadata.get("response_model")
                ),
                "eval_mode": (
                    output_metadata.get("evaluation_mode")
                    or output_metadata.get("eval_mode")
                    or record_metadata.get("eval_mode")
                ),
                "sample_scope": (
                    record_metadata.get("sample_scope")
                    or record_metadata.get("run_scope")
                ),
            }
        )
        return payload


@dataclass(frozen=True, slots=True)
class MetricSummary:
    backend: str
    benchmark: str
    candidate: str
    task_score: float
    no_memory_score: float
    task_score_gain: float
    directional_ps_rate: float
    delete_drop: float
    swap_effect: float
    directional_swap_effect: float | None
    user_coverage: float
    task_coverage: float
    n_eval: int
    verdict: str

    @property
    def coverage(self) -> str:
        return (
            f"users={self.user_coverage:.2f};"
            f"tasks={self.task_coverage:.2f};"
            f"n_eval={self.n_eval}"
        )

    def to_row(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "benchmark": self.benchmark,
            "candidate": self.candidate,
            "task_score": round(self.task_score, 6),
            "no_memory_score": round(self.no_memory_score, 6),
            "task_score_gain": round(self.task_score_gain, 6),
            "directional_ps_rate": round(self.directional_ps_rate, 6),
            "delete_drop": round(self.delete_drop, 6),
            "swap_effect": round(self.swap_effect, 6),
            "directional_swap_effect": (
                round(self.directional_swap_effect, 6)
                if self.directional_swap_effect is not None
                else ""
            ),
            "user_coverage": round(self.user_coverage, 6),
            "task_coverage": round(self.task_coverage, 6),
            "n_eval": self.n_eval,
            "coverage": self.coverage,
            "verdict": self.verdict,
        }
