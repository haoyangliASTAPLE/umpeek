from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Sequence

from .schema import ReplayEvaluationContext, clone_json


ReplayRunner = Callable[[Any, ReplayEvaluationContext], Any]


def _context_from_mapping(payload: Mapping[str, Any]) -> ReplayEvaluationContext:
    return ReplayEvaluationContext(
        backend=str(payload.get("backend") or "unknown"),
        benchmark=str(payload.get("benchmark") or "unknown"),
        sample_id=str(payload.get("sample_id") or "unknown"),
        task_type=str(payload.get("task_type") or payload.get("benchmark_type") or "open"),
        original_behavior=payload.get("original_behavior", payload.get("gold_behavior")),
        no_user_behavior=payload.get("no_user_behavior", payload.get("baseline_behavior")),
        target_behavior=payload.get("target_behavior"),
        sandbox=str(payload.get("sandbox") or "local_benchmark_sandbox"),
        metadata=dict(payload.get("metadata", {})),
    )


def _normalize_label(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _token_f1(candidate: Any, target: Any) -> float:
    candidate_tokens = {token for token in re.split(r"\W+", _normalize_label(candidate)) if token}
    target_tokens = {token for token in re.split(r"\W+", _normalize_label(target)) if token}
    if not candidate_tokens and not target_tokens:
        return 1.0
    if not candidate_tokens or not target_tokens:
        return 0.0
    overlap = len(candidate_tokens & target_tokens)
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(target_tokens)
    return 0.0 if precision == 0.0 or recall == 0.0 else 2 * precision * recall / (precision + recall)


def _as_action(value: Any) -> tuple[str, dict[str, str]]:
    if isinstance(value, Mapping):
        name = str(value.get("action_name") or value.get("tool_name") or value.get("name") or "")
        arguments = value.get("arguments") or value.get("args") or value.get("normalized_args") or {}
        if not isinstance(arguments, Mapping):
            arguments = {}
        return _normalize_label(name), {str(key): _normalize_label(item) for key, item in arguments.items()}
    text = _normalize_label(value)
    if "(" not in text:
        return text, {}
    name, raw_arguments = text.split("(", 1)
    raw_arguments = raw_arguments.rstrip(")")
    arguments: dict[str, str] = {}
    for segment in raw_arguments.split(","):
        if "=" not in segment:
            continue
        key, item = segment.split("=", 1)
        arguments[key.strip()] = item.strip()
    return name.strip(), arguments


def _action_sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, Mapping)):
        return [item for item in value if item not in (None, "", [], {})]
    return []


def _score_single_action(candidate: Any, target: Any) -> float:
    candidate_name, candidate_arguments = _as_action(candidate)
    target_name, target_arguments = _as_action(target)
    name_score = 1.0 if candidate_name and candidate_name == target_name else 0.0
    if not target_arguments:
        return name_score
    matched_arguments = sum(
        1
        for key, target_value in target_arguments.items()
        if candidate_arguments.get(key) == target_value
    )
    argument_recall = matched_arguments / len(target_arguments)
    if not candidate_arguments:
        argument_precision = 0.0
    else:
        argument_precision = matched_arguments / len(candidate_arguments)
    argument_f1 = 0.0 if argument_precision == 0.0 or argument_recall == 0.0 else (
        2 * argument_precision * argument_recall / (argument_precision + argument_recall)
    )
    return 0.5 * name_score + 0.5 * argument_f1


def score_action_behavior(candidate: Any, target: Any) -> float:
    candidate_sequence = _action_sequence(candidate)
    target_sequence = _action_sequence(target)
    if candidate_sequence and target_sequence:
        return max(
            (_score_single_action(candidate_item, target_item) for candidate_item in candidate_sequence for target_item in target_sequence),
            default=0.0,
        )
    if target_sequence:
        return max((_score_single_action(candidate, target_item) for target_item in target_sequence), default=0.0)
    if candidate_sequence:
        return max((_score_single_action(candidate_item, target) for candidate_item in candidate_sequence), default=0.0)
    return _score_single_action(candidate, target)


def score_choice_behavior(candidate: Any, target: Any) -> float:
    return 1.0 if _normalize_label(candidate) == _normalize_label(target) else 0.0


def score_ranking_behavior(candidate: Any, target: Any) -> float:
    if not isinstance(candidate, Sequence) or isinstance(candidate, (str, bytes, bytearray)):
        return score_choice_behavior(candidate, target)
    target_item = target[0] if isinstance(target, Sequence) and not isinstance(target, (str, bytes, bytearray)) and target else target
    normalized_target = _normalize_label(target_item)
    normalized_items = [_normalize_label(item) for item in candidate]
    if normalized_target not in normalized_items:
        return 0.0
    if len(normalized_items) == 1:
        return 1.0
    rank_index = normalized_items.index(normalized_target)
    return max(0.0, 1.0 - rank_index / (len(normalized_items) - 1))


def score_behavior(candidate: Any, target: Any, task_type: str) -> float:
    if task_type == "choice":
        return score_choice_behavior(candidate, target)
    if task_type == "ranking":
        return score_ranking_behavior(candidate, target)
    if task_type in {"tool", "action"}:
        return score_action_behavior(candidate, target)
    return _token_f1(candidate, target)


def _resolve_recovered_behavior(recovered_s: Any, context: ReplayEvaluationContext, replay_runner: ReplayRunner | None) -> Any:
    if replay_runner is not None:
        result = replay_runner(recovered_s, context)
        if isinstance(result, Mapping):
            status = result.get("status")
            if status not in (None, "ok", "success"):
                raise RuntimeError(f"replay_runner_status={status}")
            return result.get("behavior", result.get("output", result.get("score")))
        return result
    if "recovered_behavior" in context.metadata:
        return context.metadata["recovered_behavior"]
    if isinstance(recovered_s, Mapping) and "replayed_behavior" in recovered_s:
        return recovered_s["replayed_behavior"]
    raise RuntimeError("No replay runner or recovered_behavior metadata was provided.")


def evaluate_crs(
    recovered_s: Any,
    replay_context: ReplayEvaluationContext | Mapping[str, Any],
    replay_runner: ReplayRunner | None = None,
) -> dict[str, Any]:
    try:
        context = replay_context if isinstance(replay_context, ReplayEvaluationContext) else _context_from_mapping(replay_context)
    except Exception as exc:
        return {
            "crs": None,
            "crs_status": "blocked_invalid_replay_context",
            "error_type": exc.__class__.__name__,
            "replay_context": None,
        }

    try:
        target_behavior = context.target_behavior if context.target_behavior is not None else context.original_behavior
        recovered_behavior = _resolve_recovered_behavior(recovered_s, context, replay_runner)
        score = score_behavior(recovered_behavior, target_behavior, context.task_type)
        return {
            "crs": round(max(0.0, min(1.0, float(score))), 6),
            "crs_status": "ok",
            "replay_score_recovered": round(max(0.0, min(1.0, float(score))), 6),
            "replay_score_original": 1.0,
            "replay_score_no_user": (
                None
                if context.no_user_behavior is None
                else round(score_behavior(context.no_user_behavior, target_behavior, context.task_type), 6)
            ),
            "task_type": context.task_type,
            "recovered_behavior": clone_json(recovered_behavior),
            "target_behavior": clone_json(target_behavior),
            "error_type": None,
        }
    except Exception as exc:
        return {
            "crs": None,
            "crs_status": "replay_failed",
            "replay_score_recovered": None,
            "replay_score_original": None,
            "replay_score_no_user": None,
            "task_type": context.task_type,
            "error_type": exc.__class__.__name__,
        }
