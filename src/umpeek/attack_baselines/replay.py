from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from umpeek.exp1_whitebox.schema import clone_json

from .schema import EXPERIMENT_VERSION, blank_predicted_user_model


ReplayScoreResult = float | int | Mapping[str, Any]
ReplayScorer = Callable[[Mapping[str, Any]], ReplayScoreResult]


@dataclass(frozen=True, slots=True)
class ReplayContext:
    backend: str
    benchmark: str
    sample_id: str
    gold_user_model: Any | None = None
    no_memory_user_model: Any | None = None
    behavior_target: Any | None = None
    runtime_trace: dict[str, Any] | None = None
    replay_adapter_status: str = "synthetic_scorer_placeholder"
    metadata: dict[str, Any] = field(default_factory=dict)
    experiment_version: str = EXPERIMENT_VERSION

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("ReplayContext.backend must be non-empty.")
        if not self.benchmark:
            raise ValueError("ReplayContext.benchmark must be non-empty.")
        if not self.sample_id:
            raise ValueError("ReplayContext.sample_id must be non-empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_version": self.experiment_version,
            "backend": self.backend,
            "benchmark": self.benchmark,
            "sample_id": self.sample_id,
            "gold_user_model": clone_json(self.gold_user_model),
            "no_memory_user_model": clone_json(self.no_memory_user_model),
            "behavior_target": clone_json(self.behavior_target),
            "runtime_trace": clone_json(self.runtime_trace),
            "replay_adapter_status": self.replay_adapter_status,
            "metadata": clone_json(self.metadata),
        }


def _context_payload(replay_context: ReplayContext | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(replay_context, ReplayContext):
        return replay_context.to_dict()
    payload = dict(replay_context)
    return {
        "experiment_version": str(payload.get("experiment_version") or EXPERIMENT_VERSION),
        "backend": str(payload.get("backend") or ""),
        "benchmark": str(payload.get("benchmark") or ""),
        "sample_id": str(payload.get("sample_id") or ""),
        "gold_user_model": clone_json(payload.get("gold_user_model")),
        "no_memory_user_model": clone_json(payload.get("no_memory_user_model")),
        "behavior_target": clone_json(payload.get("behavior_target")),
        "runtime_trace": clone_json(payload.get("runtime_trace")),
        "replay_adapter_status": str(
            payload.get("replay_adapter_status") or "synthetic_scorer_placeholder"
        ),
        "metadata": clone_json(payload.get("metadata", {})),
    }


def build_replay_payloads(
    predicted_user_model: Any,
    replay_context: ReplayContext | Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    context = _context_payload(replay_context)
    if not context["backend"] or not context["benchmark"] or not context["sample_id"]:
        raise ValueError("ReplayContext requires backend, benchmark, and sample_id.")
    target_user_model = (
        context["behavior_target"]
        if context["behavior_target"] not in (None, "")
        else context["gold_user_model"]
    )
    no_memory_user_model = context["no_memory_user_model"]
    if no_memory_user_model in (None, ""):
        no_memory_user_model = blank_predicted_user_model()

    base_payload = {
        "experiment_version": context["experiment_version"],
        "backend": context["backend"],
        "benchmark": context["benchmark"],
        "sample_id": context["sample_id"],
        "target_user_model": clone_json(target_user_model),
        "runtime_trace": clone_json(context["runtime_trace"]),
        "replay_adapter_status": context["replay_adapter_status"],
        "metadata": clone_json(context["metadata"]),
    }
    return {
        "original": {
            **base_payload,
            "condition": "original",
            "user_model": clone_json(context["gold_user_model"]),
        },
        "recovered": {
            **base_payload,
            "condition": "recovered",
            "user_model": clone_json(predicted_user_model),
        },
        "no_memory": {
            **base_payload,
            "condition": "no_memory",
            "user_model": clone_json(no_memory_user_model),
        },
    }


def invoke_replay_scorer(scorer: ReplayScorer, payload: Mapping[str, Any]) -> float:
    result = scorer(payload)
    if isinstance(result, Mapping):
        status = result.get("status")
        if status not in (None, "ok", "success"):
            raise ValueError(f"Replay scorer returned non-success status: {status}")
        if "score" not in result:
            raise ValueError("Replay scorer mapping results must include a score field.")
        return float(result["score"])
    return float(result)


def normalized_overlap_scorer(payload: Mapping[str, Any]) -> float:
    candidate = {
        str(item)
        for item in dict(payload.get("normalized_user_model", {})).get("all_items", [])
    }
    target = {
        str(item)
        for item in dict(payload.get("normalized_target_user_model", {})).get("all_items", [])
    }
    if not target:
        return 1.0 if not candidate else 0.0
    return len(candidate & target) / len(target)