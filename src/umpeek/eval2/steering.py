from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from .schema import clone_json


SteeringRunner = Callable[[Any, Mapping[str, Any]], Any]


def build_legal_steering_target(sample: Mapping[str, Any]) -> dict[str, Any]:
    if bool(sample.get("requires_external_tool") or sample.get("requires_real_account")):
        return {
            "dsg_status": "unsafe_external_tool_blocked",
            "target": None,
            "reason": "sample_requires_real_external_tool_or_account",
        }
    task_type = str(sample.get("task_type") or sample.get("benchmark_type") or "open")
    if task_type in {"choice", "ranking"}:
        gold_target = str(sample.get("gold_label") or sample.get("gold_choice") or "")
        options = [str(option) for option in sample.get("options", [])]
        target = next((option for option in options if option != gold_target), None)
        if target is None:
            return {"dsg_status": "no_valid_target", "target": None, "reason": "no_non_gold_choice"}
        return {"dsg_status": "ok", "target": target, "target_type": task_type, "reason": "non_gold_choice"}
    if task_type in {"tool", "action"}:
        preferred = sample.get("preferred_action") or sample.get("gold_action")
        available_actions = list(sample.get("available_actions", []))
        target = next((action for action in available_actions if action != preferred), None)
        if target is None:
            return {"dsg_status": "no_valid_target", "target": None, "reason": "no_non_preferred_action"}
        return {"dsg_status": "ok", "target": clone_json(target), "target_type": task_type, "reason": "non_preferred_action"}
    target_preference = sample.get("target_preference") or sample.get("rubric_target_preference")
    if target_preference in (None, ""):
        return {"dsg_status": "no_valid_target", "target": None, "reason": "missing_rubric_target_preference"}
    return {"dsg_status": "ok", "target": clone_json(target_preference), "target_type": "open", "reason": "rubric_defined_target"}


def evaluate_dsg(
    recovered_s: Any,
    steering_target: Mapping[str, Any],
    steering_runner: SteeringRunner,
    *,
    baseline_score: float = 0.0,
) -> dict[str, Any]:
    status = str(steering_target.get("dsg_status") or steering_target.get("status") or "ok")
    if status != "ok":
        return {
            "dsg": None,
            "dsg_status": status,
            "steering_score": None,
            "baseline_score": round(float(baseline_score), 6),
            "reason": steering_target.get("reason"),
        }
    if bool(steering_target.get("requires_external_tool") or steering_target.get("requires_real_account")):
        return {
            "dsg": None,
            "dsg_status": "unsafe_external_tool_blocked",
            "steering_score": None,
            "baseline_score": round(float(baseline_score), 6),
            "reason": "target_requires_real_external_tool_or_account",
        }
    result = steering_runner(recovered_s, steering_target)
    if isinstance(result, Mapping):
        result_status = result.get("status")
        if result_status not in (None, "ok", "success"):
            return {
                "dsg": None,
                "dsg_status": str(result_status),
                "steering_score": None,
                "baseline_score": round(float(baseline_score), 6),
                "reason": result.get("reason"),
            }
        steering_score = float(result.get("score", result.get("steering_score", 0.0)))
    else:
        steering_score = float(result)
    return {
        "dsg": round(steering_score - float(baseline_score), 6),
        "dsg_status": "ok",
        "steering_score": round(steering_score, 6),
        "baseline_score": round(float(baseline_score), 6),
        "target": clone_json(steering_target.get("target")),
    }
