from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from umpeek.exp1_whitebox.schema import clone_json


EXPERIMENT_VERSION = "user_model_extraction_baselines_v1"
REQUIRED_PREDICTED_USER_MODEL_KEYS = (
    "facts",
    "preferences",
    "constraints",
    "relations",
    "tool_state",
    "raw_text",
    "confidence",
)
FORBIDDEN_PRIVATE_STATE_KEYS = frozenset(
    {
        "answer_hint",
        "gold",
        "gold_answer",
        "gold_label",
        "gold_s",
        "judge_feedback",
        "judge_score",
        "runtime_trace",
        "runtime_user_model",
        "s_json",
        "s_text",
        "task_targets",
    }
)
_ALLOWED_STATUS_VALUES = {
    "success",
    "missing_whitebox_source",
    "not_implemented",
    "not_applicable_no_external_content",
    "not_applicable_no_certifiable_component",
    "blocked_non_executable_probe_victim",
    "blocked_paid_dependency",
    "budget_exhausted",
    "invalid_input",
    "runtime_error",
    "skipped",
}

AttackStatus = Literal[
    "success",
    "missing_whitebox_source",
    "not_implemented",
    "not_applicable_no_external_content",
    "not_applicable_no_certifiable_component",
    "blocked_non_executable_probe_victim",
    "blocked_paid_dependency",
    "budget_exhausted",
    "invalid_input",
    "runtime_error",
    "skipped",
]


def blank_predicted_user_model() -> dict[str, Any]:
    return {
        "facts": [],
        "preferences": [],
        "constraints": [],
        "relations": [],
        "tool_state": {},
        "raw_text": "",
        "confidence": 0.0,
    }


def _scan_forbidden_keys(value: Any, path: tuple[str, ...] = ()) -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in FORBIDDEN_PRIVATE_STATE_KEYS:
                hits.append(".".join((*path, key_text)))
            hits.extend(_scan_forbidden_keys(item, (*path, key_text)))
        return hits
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            hits.extend(_scan_forbidden_keys(item, (*path, str(index))))
    return hits


def assert_public_attack_payload(value: Any) -> None:
    hits = _scan_forbidden_keys(value)
    if hits:
        raise ValueError(f"Attack inputs must remain black-box visible only; forbidden keys found at {hits}")


def _validate_status(status: str) -> None:
    if status not in _ALLOWED_STATUS_VALUES:
        raise ValueError(f"Unsupported attack status: {status}")


def _validate_predicted_user_model(payload: Mapping[str, Any]) -> None:
    missing = [key for key in REQUIRED_PREDICTED_USER_MODEL_KEYS if key not in payload]
    if missing:
        raise ValueError(f"predicted_user_model missing required keys: {missing}")
    assert_public_attack_payload(payload)


@dataclass(frozen=True, slots=True)
class AttackBaselineSpec:
    baseline: str
    short_name: str
    paper_title: str
    paper_url: str
    code_name: str
    code_url: str
    open_source: bool
    adaptation_target: str
    implementation_status: AttackStatus = "not_implemented"
    notes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.baseline:
            raise ValueError("AttackBaselineSpec.baseline must be non-empty.")
        if not self.short_name:
            raise ValueError("AttackBaselineSpec.short_name must be non-empty.")
        if not self.paper_title or not self.paper_url:
            raise ValueError("AttackBaselineSpec requires paper_title and paper_url.")
        if not self.code_name:
            raise ValueError("AttackBaselineSpec.code_name must be non-empty.")
        if not self.adaptation_target:
            raise ValueError("AttackBaselineSpec.adaptation_target must be non-empty.")
        _validate_status(self.implementation_status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_version": EXPERIMENT_VERSION,
            "baseline": self.baseline,
            "short_name": self.short_name,
            "paper_title": self.paper_title,
            "paper_url": self.paper_url,
            "code_name": self.code_name,
            "code_url": self.code_url,
            "open_source": bool(self.open_source),
            "adaptation_target": self.adaptation_target,
            "implementation_status": self.implementation_status,
            "notes": list(self.notes),
            "official_source": {
                "paper": self.paper_title,
                "paper_url": self.paper_url,
                "code": self.code_name,
                "code_url": self.code_url,
                "open_source": bool(self.open_source),
            },
            "metadata": clone_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class AttackInput:
    backend: str
    benchmark: str
    sample_id: str
    task_prompt: str
    user_id: str | None = None
    task_id: str | None = None
    visible_messages: list[dict[str, Any]] = field(default_factory=list)
    visible_tools: list[dict[str, Any]] = field(default_factory=list)
    visible_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    visible_tool_results: list[dict[str, Any]] = field(default_factory=list)
    public_context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    experiment_version: str = EXPERIMENT_VERSION

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("AttackInput.backend must be non-empty.")
        if not self.benchmark:
            raise ValueError("AttackInput.benchmark must be non-empty.")
        if not self.sample_id:
            raise ValueError("AttackInput.sample_id must be non-empty.")
        if not self.task_prompt:
            raise ValueError("AttackInput.task_prompt must be non-empty.")
        assert_public_attack_payload(self.visible_messages)
        assert_public_attack_payload(self.visible_tools)
        assert_public_attack_payload(self.visible_tool_calls)
        assert_public_attack_payload(self.visible_tool_results)
        assert_public_attack_payload(self.public_context)
        assert_public_attack_payload(self.metadata)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AttackInput":
        allowed_keys = {
            "experiment_version",
            "backend",
            "benchmark",
            "sample_id",
            "task_prompt",
            "user_id",
            "task_id",
            "visible_messages",
            "visible_tools",
            "visible_tool_calls",
            "visible_tool_results",
            "public_context",
            "metadata",
        }
        unknown = sorted(set(payload) - allowed_keys)
        if unknown:
            raise ValueError(f"AttackInput received unsupported keys: {unknown}")
        assert_public_attack_payload(payload)
        return cls(
            backend=str(payload["backend"]),
            benchmark=str(payload["benchmark"]),
            sample_id=str(payload["sample_id"]),
            task_prompt=str(payload["task_prompt"]),
            user_id=(None if payload.get("user_id") is None else str(payload.get("user_id"))),
            task_id=(None if payload.get("task_id") is None else str(payload.get("task_id"))),
            visible_messages=[dict(item) for item in payload.get("visible_messages", [])],
            visible_tools=[dict(item) for item in payload.get("visible_tools", [])],
            visible_tool_calls=[dict(item) for item in payload.get("visible_tool_calls", [])],
            visible_tool_results=[dict(item) for item in payload.get("visible_tool_results", [])],
            public_context=dict(payload.get("public_context", {})),
            metadata=dict(payload.get("metadata", {})),
            experiment_version=str(payload.get("experiment_version") or EXPERIMENT_VERSION),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_version": self.experiment_version,
            "backend": self.backend,
            "benchmark": self.benchmark,
            "sample_id": self.sample_id,
            "task_prompt": self.task_prompt,
            "user_id": self.user_id,
            "task_id": self.task_id,
            "visible_messages": clone_json(self.visible_messages),
            "visible_tools": clone_json(self.visible_tools),
            "visible_tool_calls": clone_json(self.visible_tool_calls),
            "visible_tool_results": clone_json(self.visible_tool_results),
            "public_context": clone_json(self.public_context),
            "metadata": clone_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class AttackPrediction:
    baseline: str
    sample_id: str
    predicted_user_model: dict[str, Any] = field(default_factory=blank_predicted_user_model)
    source_refs: tuple[str, ...] = ()
    notes: str = ""
    status: AttackStatus = "success"
    error_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    experiment_version: str = EXPERIMENT_VERSION

    def __post_init__(self) -> None:
        if not self.baseline:
            raise ValueError("AttackPrediction.baseline must be non-empty.")
        if not self.sample_id:
            raise ValueError("AttackPrediction.sample_id must be non-empty.")
        _validate_status(self.status)
        _validate_predicted_user_model(self.predicted_user_model)
        assert_public_attack_payload(self.metadata)

    @classmethod
    def not_implemented(
        cls,
        *,
        baseline: str,
        sample_id: str,
        reason: str = "stub_adapter",
    ) -> "AttackPrediction":
        return cls(
            baseline=baseline,
            sample_id=sample_id,
            status="not_implemented",
            error_type=reason,
            notes="A026 scaffold only; concrete baseline adapter is intentionally deferred.",
            metadata={"fallback_reason": "scaffold_stub"},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_version": self.experiment_version,
            "baseline": self.baseline,
            "sample_id": self.sample_id,
            "predicted_user_model": clone_json(self.predicted_user_model),
            "source_refs": list(self.source_refs),
            "notes": self.notes,
            "status": self.status,
            "error_type": self.error_type,
            "metadata": clone_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class AttackCost:
    query_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_clock_s: float = 0.0
    estimated_usd: float = 0.0
    model_calls: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.query_count < 0:
            raise ValueError("AttackCost.query_count must be non-negative.")
        if self.prompt_tokens < 0 or self.completion_tokens < 0:
            raise ValueError("AttackCost token counts must be non-negative.")
        if self.wall_clock_s < 0 or self.estimated_usd < 0:
            raise ValueError("AttackCost wall_clock_s and estimated_usd must be non-negative.")
        if self.model_calls < 0:
            raise ValueError("AttackCost.model_calls must be non-negative.")

    @property
    def total_tokens(self) -> int:
        return int(self.prompt_tokens + self.completion_tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_count": int(self.query_count),
            "prompt_tokens": int(self.prompt_tokens),
            "completion_tokens": int(self.completion_tokens),
            "total_tokens": self.total_tokens,
            "wall_clock_s": float(self.wall_clock_s),
            "estimated_usd": float(self.estimated_usd),
            "model_calls": int(self.model_calls),
            "metadata": clone_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class AttackRunRecord:
    backend: str
    benchmark: str
    sample_id: str
    baseline: str
    status: AttackStatus
    prediction_path: str
    cost: AttackCost
    error_type: str | None = None
    run_id: str = ""
    user_id: str | None = None
    task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    experiment_version: str = EXPERIMENT_VERSION

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("AttackRunRecord.backend must be non-empty.")
        if not self.benchmark:
            raise ValueError("AttackRunRecord.benchmark must be non-empty.")
        if not self.sample_id:
            raise ValueError("AttackRunRecord.sample_id must be non-empty.")
        if not self.baseline:
            raise ValueError("AttackRunRecord.baseline must be non-empty.")
        _validate_status(self.status)
        assert_public_attack_payload(self.metadata)

    @classmethod
    def from_prediction(
        cls,
        *,
        backend: str,
        benchmark: str,
        sample_id: str,
        baseline: str,
        prediction_path: str,
        prediction: AttackPrediction,
        cost: AttackCost,
        run_id: str = "",
        user_id: str | None = None,
        task_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "AttackRunRecord":
        return cls(
            backend=backend,
            benchmark=benchmark,
            sample_id=sample_id,
            baseline=baseline,
            status=prediction.status,
            prediction_path=str(prediction_path),
            cost=cost,
            error_type=prediction.error_type,
            run_id=run_id,
            user_id=user_id,
            task_id=task_id,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_version": self.experiment_version,
            "backend": self.backend,
            "benchmark": self.benchmark,
            "run_id": self.run_id,
            "sample_id": self.sample_id,
            "baseline": self.baseline,
            "status": self.status,
            "prediction_path": self.prediction_path,
            "cost": self.cost.to_dict(),
            "error_type": self.error_type,
            "user_id": self.user_id,
            "task_id": self.task_id,
            "metadata": clone_json(self.metadata),
        }


def validate_attack_input(payload: AttackInput | Mapping[str, Any]) -> None:
    data = payload.to_dict() if isinstance(payload, AttackInput) else dict(payload)
    required = {"backend", "benchmark", "sample_id", "task_prompt"}
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"AttackInput missing required keys: {missing}")
    AttackInput.from_dict(data)


def validate_attack_prediction(payload: AttackPrediction | Mapping[str, Any]) -> None:
    data = payload.to_dict() if isinstance(payload, AttackPrediction) else dict(payload)
    required = {
        "baseline",
        "sample_id",
        "predicted_user_model",
        "status",
        "error_type",
        "experiment_version",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"AttackPrediction missing required keys: {missing}")
    _validate_status(str(data["status"]))
    _validate_predicted_user_model(dict(data["predicted_user_model"]))


def validate_attack_run_record(payload: AttackRunRecord | Mapping[str, Any]) -> None:
    data = payload.to_dict() if isinstance(payload, AttackRunRecord) else dict(payload)
    required = {
        "backend",
        "benchmark",
        "sample_id",
        "baseline",
        "status",
        "prediction_path",
        "cost",
        "error_type",
        "experiment_version",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"AttackRunRecord missing required keys: {missing}")
    _validate_status(str(data["status"]))
    if not isinstance(data["cost"], Mapping):
        raise ValueError("AttackRunRecord.cost must serialize to a mapping.")
    cost_required = {
        "query_count",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "wall_clock_s",
        "estimated_usd",
        "model_calls",
    }
    missing_cost = sorted(cost_required - set(data["cost"]))
    if missing_cost:
        raise ValueError(f"AttackRunRecord.cost missing required keys: {missing_cost}")
