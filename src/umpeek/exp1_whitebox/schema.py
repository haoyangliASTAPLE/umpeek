from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal, Mapping, Sequence


EXPERIMENT_VERSION = "whitebox_runtime_user_model_v1"
TRACE_STATUS_VALUES = {"captured", "missing_hook", "empty_valid"}
CONDITION_VALUES = {"personalized", "no_memory", "delete", "swap"}
SOURCE_TYPE_VALUES = {
    "message",
    "retrieved_memory_context",
    "personalization_block",
    "tool_action_state",
    "graph_state_snapshot",
    "agent_state",
    "unknown_runtime_user_state",
}

TraceStatus = Literal["captured", "missing_hook", "empty_valid"]
ConditionName = Literal["personalized", "no_memory", "delete", "swap"]


def to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return to_serializable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(item) for item in value]
    return value


def stable_json(value: Any) -> str:
    return json.dumps(
        to_serializable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def clone_json(value: Any) -> Any:
    return json.loads(stable_json(value))


def estimate_token_count(text: str) -> int:
    return len([token for token in text.split() if token])


def render_message_content(message: Mapping[str, Any]) -> str:
    if "content" in message and message["content"] is not None:
        return str(message["content"])
    segments = message.get("segments", [])
    if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
        raise ValueError("Message segments must be a sequence of mappings.")
    return "".join(str(segment.get("text", "")) for segment in segments)


@dataclass(frozen=True, slots=True)
class RuntimeUserStateFragment:
    call_index: int
    call_role: str
    source_type: str
    source_ref: str
    content: Any
    span_path: str = ""
    text: str = ""
    start_char: int | None = None
    end_char: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_index": int(self.call_index),
            "call_role": self.call_role,
            "source_type": self.source_type,
            "source_ref": self.source_ref,
            "content": clone_json(self.content),
            "span_path": self.span_path,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": clone_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RuntimeTraceCall:
    call_index: int
    call_role: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    retrieved_memory_context: list[dict[str, Any]] = field(default_factory=list)
    personalization_blocks: list[dict[str, Any]] = field(default_factory=list)
    tool_action_state: dict[str, Any] = field(default_factory=dict)
    graph_state_snapshot: dict[str, Any] = field(default_factory=dict)
    agent_state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_index": int(self.call_index),
            "call_role": self.call_role,
            "messages": clone_json(self.messages),
            "retrieved_memory_context": clone_json(self.retrieved_memory_context),
            "personalization_blocks": clone_json(self.personalization_blocks),
            "tool_action_state": clone_json(self.tool_action_state),
            "graph_state_snapshot": clone_json(self.graph_state_snapshot),
            "agent_state": clone_json(self.agent_state),
            "metadata": clone_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RuntimeTrace:
    backend: str
    benchmark: str
    user_id: str
    task_id: str
    trace_status: TraceStatus
    calls: list[RuntimeTraceCall] = field(default_factory=list)
    non_user_context: dict[str, Any] = field(default_factory=dict)
    model_config: dict[str, Any] = field(default_factory=dict)
    coverage_flags: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    experiment_version: str = EXPERIMENT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_version": self.experiment_version,
            "backend": self.backend,
            "benchmark": self.benchmark,
            "user_id": self.user_id,
            "task_id": self.task_id,
            "trace_status": self.trace_status,
            "calls": [call.to_dict() for call in self.calls],
            "non_user_context": clone_json(self.non_user_context),
            "model_config": clone_json(self.model_config),
            "coverage_flags": clone_json(self.coverage_flags),
            "metadata": clone_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RuntimeUserModel:
    backend: str
    benchmark: str
    user_id: str
    task_id: str
    call_count: int
    S_text: str
    S_json: list[dict[str, Any]] | dict[str, Any]
    source_spans: list[RuntimeUserStateFragment] = field(default_factory=list)
    token_count: int = 0
    trace_status: TraceStatus = "missing_hook"
    coverage_flags: dict[str, Any] = field(default_factory=dict)
    trace_ref: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    experiment_version: str = EXPERIMENT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_version": self.experiment_version,
            "backend": self.backend,
            "benchmark": self.benchmark,
            "user_id": self.user_id,
            "task_id": self.task_id,
            "call_count": int(self.call_count),
            "S_text": self.S_text,
            "S_json": clone_json(self.S_json),
            "source_spans": [fragment.to_dict() for fragment in self.source_spans],
            "token_count": int(self.token_count),
            "trace_status": self.trace_status,
            "coverage_flags": clone_json(self.coverage_flags),
            "trace_ref": self.trace_ref,
            "metadata": clone_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ConditionedRunRecord:
    backend: str
    benchmark: str
    run_id: str
    user_id: str
    task_id: str
    condition: ConditionName
    output: dict[str, Any]
    score: float | None
    judge_metadata: dict[str, Any] = field(default_factory=dict)
    trace_ref: str = ""
    model_config: dict[str, Any] = field(default_factory=dict)
    call_count: int = 0
    trace_status: TraceStatus = "missing_hook"
    coverage_flags: dict[str, Any] = field(default_factory=dict)
    non_user_context_hash: str = ""
    timeout_status: bool = False
    retry_count: int = 0
    timeout_s: float | None = None
    valid: bool = True
    invalid_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    experiment_version: str = EXPERIMENT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_version": self.experiment_version,
            "backend": self.backend,
            "benchmark": self.benchmark,
            "run_id": self.run_id,
            "user_id": self.user_id,
            "task_id": self.task_id,
            "condition": self.condition,
            "output": clone_json(self.output),
            "score": self.score,
            "judge_metadata": clone_json(self.judge_metadata),
            "trace_ref": self.trace_ref,
            "model_config": clone_json(self.model_config),
            "call_count": int(self.call_count),
            "trace_status": self.trace_status,
            "coverage_flags": clone_json(self.coverage_flags),
            "non_user_context_hash": self.non_user_context_hash,
            "timeout_status": bool(self.timeout_status),
            "retry_count": int(self.retry_count),
            "timeout_s": self.timeout_s,
            "valid": bool(self.valid),
            "invalid_reason": self.invalid_reason,
            "metadata": clone_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CoverageSummary:
    backend: str
    benchmark: str
    run_id: str
    total_users: int
    total_tasks: int
    users_with_valid_S: int
    tasks_with_valid_S: int
    n_eval: int
    trace_status_counts: dict[str, int] = field(default_factory=dict)
    coverage_flags: dict[str, Any] = field(default_factory=dict)
    unknown_source_ratio: float = 0.0
    experiment_version: str = EXPERIMENT_VERSION

    def to_dict(self) -> dict[str, Any]:
        user_coverage = self.users_with_valid_S / self.total_users if self.total_users else 0.0
        task_coverage = self.tasks_with_valid_S / self.total_tasks if self.total_tasks else 0.0
        return {
            "experiment_version": self.experiment_version,
            "backend": self.backend,
            "benchmark": self.benchmark,
            "run_id": self.run_id,
            "total_users": int(self.total_users),
            "total_tasks": int(self.total_tasks),
            "users_with_valid_S": int(self.users_with_valid_S),
            "tasks_with_valid_S": int(self.tasks_with_valid_S),
            "n_eval": int(self.n_eval),
            "user_coverage": round(user_coverage, 6),
            "task_coverage": round(task_coverage, 6),
            "trace_status_counts": clone_json(self.trace_status_counts),
            "coverage_flags": clone_json(self.coverage_flags),
            "unknown_source_ratio": round(float(self.unknown_source_ratio), 6),
        }


@dataclass(frozen=True, slots=True)
class MetricSummary:
    backend: str
    benchmark: str
    run_id: str
    task_score_gain: float
    directional_ps_rate: float
    delete_drop: float
    swap_effect: float
    coverage: dict[str, Any]
    n_eval: int
    experiment_version: str = EXPERIMENT_VERSION

    def to_row(self) -> dict[str, Any]:
        return {
            "experiment_version": self.experiment_version,
            "backend": self.backend,
            "benchmark": self.benchmark,
            "run_id": self.run_id,
            "task_score_gain": round(float(self.task_score_gain), 6),
            "directional_ps_rate": round(float(self.directional_ps_rate), 6),
            "delete_drop": round(float(self.delete_drop), 6),
            "swap_effect": round(float(self.swap_effect), 6),
            "coverage": stable_json(self.coverage),
            "n_eval": int(self.n_eval),
        }


def _validate_fragment(fragment: RuntimeUserStateFragment | Mapping[str, Any]) -> None:
    payload = fragment.to_dict() if isinstance(fragment, RuntimeUserStateFragment) else dict(fragment)
    required_keys = {"call_index", "call_role", "source_type", "source_ref", "content"}
    missing_keys = sorted(required_keys - payload.keys())
    if missing_keys:
        raise ValueError(f"Runtime user state fragment missing required keys: {missing_keys}")
    if int(payload["call_index"]) < 0:
        raise ValueError("Runtime user state fragments require call_index >= 0.")
    if not str(payload["call_role"]):
        raise ValueError("Runtime user state fragments require a non-empty call_role.")
    if not str(payload["source_ref"]):
        raise ValueError("Runtime user state fragments require a non-empty source_ref.")
    source_type = str(payload["source_type"])
    if source_type not in SOURCE_TYPE_VALUES:
        raise ValueError(f"Unsupported runtime user state source_type: {source_type}")


def _validate_message(message: Mapping[str, Any]) -> None:
    if not str(message.get("role", "")):
        raise ValueError("Messages require a non-empty role.")
    render_message_content(message)
    segments = message.get("segments")
    if segments is None:
        return
    if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
        raise ValueError("Message segments must be a sequence.")
    for segment in segments:
        if not isinstance(segment, Mapping):
            raise ValueError("Message segments must be mappings.")
        if "text" not in segment:
            raise ValueError("Message segments require a text field.")


def _validate_call(call: RuntimeTraceCall | Mapping[str, Any]) -> None:
    payload = call.to_dict() if isinstance(call, RuntimeTraceCall) else dict(call)
    required_keys = {
        "call_index",
        "call_role",
        "messages",
        "retrieved_memory_context",
        "personalization_blocks",
        "tool_action_state",
        "graph_state_snapshot",
        "agent_state",
    }
    missing_keys = sorted(required_keys - payload.keys())
    if missing_keys:
        raise ValueError(f"Runtime trace call missing required keys: {missing_keys}")
    if int(payload["call_index"]) < 0:
        raise ValueError("Runtime trace calls require call_index >= 0.")
    if not str(payload["call_role"]):
        raise ValueError("Runtime trace calls require a non-empty call_role.")
    for field_name in ("messages", "retrieved_memory_context", "personalization_blocks"):
        if not isinstance(payload[field_name], list):
            raise ValueError(f"Runtime trace call field {field_name} must be a list.")
    for message in payload["messages"]:
        if not isinstance(message, Mapping):
            raise ValueError("Runtime trace messages must be mappings.")
        _validate_message(message)
    for field_name in ("tool_action_state", "graph_state_snapshot", "agent_state"):
        if not isinstance(payload[field_name], Mapping):
            raise ValueError(f"Runtime trace call field {field_name} must be a mapping.")


def validate_runtime_trace(trace: RuntimeTrace | Mapping[str, Any]) -> None:
    payload = trace.to_dict() if isinstance(trace, RuntimeTrace) else dict(trace)
    required_keys = {
        "experiment_version",
        "backend",
        "benchmark",
        "user_id",
        "task_id",
        "trace_status",
        "calls",
        "non_user_context",
        "model_config",
        "coverage_flags",
    }
    missing_keys = sorted(required_keys - payload.keys())
    if missing_keys:
        raise ValueError(f"Runtime trace missing required keys: {missing_keys}")
    if str(payload["experiment_version"]) != EXPERIMENT_VERSION:
        raise ValueError("Runtime trace experiment_version does not match whitebox v1.")
    trace_status = str(payload["trace_status"])
    if trace_status not in TRACE_STATUS_VALUES:
        raise ValueError(f"Unsupported runtime trace status: {trace_status}")
    if not isinstance(payload["calls"], list):
        raise ValueError("Runtime trace calls must be a list.")
    for call in payload["calls"]:
        _validate_call(call)
    if trace_status != "missing_hook":
        if not any(call.get("messages") for call in payload["calls"]):
            raise ValueError(
                "Runtime traces must capture final messages before at least one decision."
            )
    if trace_status == "missing_hook" and payload["calls"]:
        for call in payload["calls"]:
            if call.get("messages") or call.get("retrieved_memory_context") or call.get("personalization_blocks"):
                raise ValueError("missing_hook traces must not fabricate runtime call payloads.")
    for field_name in ("non_user_context", "model_config", "coverage_flags"):
        if not isinstance(payload[field_name], Mapping):
            raise ValueError(f"Runtime trace field {field_name} must be a mapping.")


def _collect_source_refs_from_value(value: Any, refs: set[str]) -> None:
    if isinstance(value, Mapping):
        source_ref = value.get("source_ref")
        if source_ref not in (None, ""):
            refs.add(str(source_ref))
        for item in value.values():
            _collect_source_refs_from_value(item, refs)
        return
    if isinstance(value, list):
        for item in value:
            _collect_source_refs_from_value(item, refs)


def collect_trace_source_refs(trace: RuntimeTrace | Mapping[str, Any]) -> set[str]:
    payload = trace.to_dict() if isinstance(trace, RuntimeTrace) else dict(trace)
    refs: set[str] = set()
    for call in payload.get("calls", []):
        _collect_source_refs_from_value(call, refs)
    return refs


def validate_runtime_user_model(
    runtime_user_model: RuntimeUserModel | Mapping[str, Any],
    *,
    trace: RuntimeTrace | Mapping[str, Any] | None = None,
) -> None:
    payload = runtime_user_model.to_dict() if isinstance(runtime_user_model, RuntimeUserModel) else dict(runtime_user_model)
    required_keys = {
        "experiment_version",
        "backend",
        "benchmark",
        "user_id",
        "task_id",
        "call_count",
        "S_text",
        "S_json",
        "source_spans",
        "token_count",
        "trace_status",
        "coverage_flags",
    }
    missing_keys = sorted(required_keys - payload.keys())
    if missing_keys:
        raise ValueError(f"Runtime user model missing required keys: {missing_keys}")
    if str(payload["experiment_version"]) != EXPERIMENT_VERSION:
        raise ValueError("Runtime user model experiment_version does not match whitebox v1.")
    trace_status = str(payload["trace_status"])
    if trace_status not in TRACE_STATUS_VALUES:
        raise ValueError(f"Unsupported runtime user model status: {trace_status}")
    if int(payload["call_count"]) < 0:
        raise ValueError("Runtime user model call_count must be non-negative.")
    if int(payload["token_count"]) < 0:
        raise ValueError("Runtime user model token_count must be non-negative.")
    if not isinstance(payload["source_spans"], list):
        raise ValueError("Runtime user model source_spans must be a list.")
    for fragment in payload["source_spans"]:
        _validate_fragment(fragment)
    if trace_status in {"missing_hook", "empty_valid"} and payload["source_spans"]:
        raise ValueError(f"{trace_status} runtime user models must not fabricate source spans.")
    if trace is not None:
        validate_runtime_trace(trace)
        trace_payload = trace.to_dict() if isinstance(trace, RuntimeTrace) else dict(trace)
        available_refs = collect_trace_source_refs(trace_payload)
        max_call_index = len(trace_payload.get("calls", [])) - 1
        for fragment in payload["source_spans"]:
            if str(fragment["source_ref"]) not in available_refs:
                raise ValueError(
                    "Runtime user model may only reference user state that appeared in the captured trace."
                )
            if int(fragment["call_index"]) > max_call_index:
                raise ValueError("Runtime user model call_index exceeds captured trace length.")


def validate_conditioned_run_record(record: ConditionedRunRecord | Mapping[str, Any]) -> None:
    payload = record.to_dict() if isinstance(record, ConditionedRunRecord) else dict(record)
    required_keys = {
        "experiment_version",
        "backend",
        "benchmark",
        "run_id",
        "user_id",
        "task_id",
        "condition",
        "output",
        "score",
        "judge_metadata",
        "trace_ref",
        "model_config",
        "call_count",
        "trace_status",
        "coverage_flags",
        "non_user_context_hash",
        "timeout_status",
        "retry_count",
        "valid",
    }
    missing_keys = sorted(required_keys - payload.keys())
    if missing_keys:
        raise ValueError(f"Conditioned run record missing required keys: {missing_keys}")
    if str(payload["experiment_version"]) != EXPERIMENT_VERSION:
        raise ValueError("Conditioned run record experiment_version does not match whitebox v1.")
    condition = str(payload["condition"])
    if condition not in CONDITION_VALUES:
        raise ValueError(f"Unsupported condition: {condition}")
    if str(payload["trace_status"]) not in TRACE_STATUS_VALUES:
        raise ValueError(f"Unsupported trace_status: {payload['trace_status']}")
    if int(payload["call_count"]) < 0:
        raise ValueError("Conditioned run record call_count must be non-negative.")
    if int(payload["retry_count"]) < 0:
        raise ValueError("Conditioned run record retry_count must be non-negative.")
    if not isinstance(payload["output"], Mapping):
        raise ValueError("Conditioned run record output must be a mapping.")
    if not isinstance(payload["judge_metadata"], Mapping):
        raise ValueError("Conditioned run record judge_metadata must be a mapping.")
    if not isinstance(payload["model_config"], Mapping):
        raise ValueError("Conditioned run record model_config must be a mapping.")
    if not isinstance(payload["coverage_flags"], Mapping):
        raise ValueError("Conditioned run record coverage_flags must be a mapping.")


def non_user_context_hash(payload: RuntimeTrace | Mapping[str, Any]) -> str:
    trace_payload = payload.to_dict() if isinstance(payload, RuntimeTrace) else dict(payload)
    return hashlib.sha256(
        stable_json(trace_payload.get("non_user_context", {})).encode("utf-8")
    ).hexdigest()


def trace_non_user_context_hash(payload: RuntimeTrace | Mapping[str, Any]) -> str:
    trace_payload = payload.to_dict() if isinstance(payload, RuntimeTrace) else dict(payload)

    def _strip_user_state(value: Any) -> Any:
        if isinstance(value, Mapping):
            if bool(value.get("user_related")):
                return None
            transformed: dict[str, Any] = {}
            for key, item in value.items():
                stripped_item = _strip_user_state(item)
                if stripped_item is None and key != "user_related":
                    continue
                transformed[str(key)] = stripped_item
            if "role" in value and isinstance(value.get("segments"), list):
                transformed_segments = transformed.get("segments", [])
                transformed["content"] = "".join(
                    str(segment.get("text", ""))
                    for segment in transformed_segments
                    if isinstance(segment, Mapping)
                )
            return transformed
        if isinstance(value, list):
            transformed_items = []
            for item in value:
                stripped_item = _strip_user_state(item)
                if stripped_item is None:
                    continue
                transformed_items.append(stripped_item)
            return transformed_items
        return clone_json(value)

    sanitized_calls = []
    for call in trace_payload.get("calls", []):
        sanitized_calls.append(
            {
                "call_index": int(call.get("call_index", 0)),
                "call_role": str(call.get("call_role", "")),
                "messages": _strip_user_state(call.get("messages", [])),
                "retrieved_memory_context": _strip_user_state(
                    call.get("retrieved_memory_context", [])
                ),
                "personalization_blocks": _strip_user_state(
                    call.get("personalization_blocks", [])
                ),
                "tool_action_state": _strip_user_state(call.get("tool_action_state", {})),
                "graph_state_snapshot": _strip_user_state(
                    call.get("graph_state_snapshot", {})
                ),
                "agent_state": _strip_user_state(call.get("agent_state", {})),
                "metadata": clone_json(call.get("metadata", {})),
            }
        )

    return hashlib.sha256(
        stable_json(
            {
                "non_user_context": trace_payload.get("non_user_context", {}),
                "calls": sanitized_calls,
            }
        ).encode("utf-8")
    ).hexdigest()