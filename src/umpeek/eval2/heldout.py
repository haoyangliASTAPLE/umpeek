from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from .replay import score_behavior
from .schema import HeldoutTask, clone_json


HeldoutPredictor = Callable[[Any, HeldoutTask], Any]
HeldoutScorer = Callable[[Any, Any, str], float]


def _coerce_task(payload: HeldoutTask | Mapping[str, Any]) -> HeldoutTask:
    if isinstance(payload, HeldoutTask):
        return payload
    return HeldoutTask(
        user_id=str(payload.get("user_id") or ""),
        task_id=str(payload.get("task_id") or ""),
        task_type=str(payload.get("task_type") or payload.get("benchmark_type") or "open"),
        prompt=str(payload.get("prompt") or ""),
        gold_behavior=payload.get("gold_behavior", payload.get("gold_label")),
        split=str(payload.get("split") or "candidate"),
        sort_key=str(payload.get("sort_key") or payload.get("session_id") or payload.get("timestamp") or payload.get("task_id") or ""),
        metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), Mapping) else {},
    )


def build_heldout_split(
    tasks: Sequence[HeldoutTask | Mapping[str, Any]],
    *,
    probe_task_ids: Sequence[str] = (),
    official_split_available: bool | None = None,
) -> dict[str, Any]:
    task_records = [_coerce_task(task) for task in tasks]
    probe_ids = {str(task_id) for task_id in probe_task_ids}
    if len(task_records) < 2:
        return {
            "hbps_status": "insufficient_heldout",
            "heldout_tasks": [],
            "probe_task_ids": sorted(probe_ids),
            "split_strategy": "insufficient_tasks",
        }

    has_official = official_split_available if official_split_available is not None else any(
        task.split in {"heldout", "test", "official_test"} for task in task_records
    )
    if not has_official:
        return {
            "hbps_status": "insufficient_heldout",
            "heldout_tasks": [],
            "train_tasks": [task.to_dict() for task in task_records],
            "probe_task_ids": sorted(probe_ids),
            "split_strategy": "precomputed_or_official_heldout_required",
            "reason": "Implicit tail-split fallback is disabled for current evaluations.",
        }
    heldout_tasks = [task for task in task_records if task.split in {"heldout", "test", "official_test"}]
    train_tasks = [task for task in task_records if task not in heldout_tasks]
    split_strategy = "official_or_precomputed_heldout_split"

    heldout_ids = {task.task_id for task in heldout_tasks}
    overlap = sorted(heldout_ids & probe_ids)
    if overlap:
        raise ValueError(f"Held-out tasks overlap attack probe tasks: {overlap}")
    if not heldout_tasks:
        return {
            "hbps_status": "insufficient_heldout",
            "heldout_tasks": [],
            "probe_task_ids": sorted(probe_ids),
            "split_strategy": split_strategy,
        }

    return {
        "hbps_status": "ok",
        "heldout_tasks": [task.to_dict() for task in heldout_tasks],
        "train_tasks": [task.to_dict() for task in train_tasks],
        "probe_task_ids": sorted(probe_ids),
        "split_strategy": split_strategy,
    }


def evaluate_hbps(
    recovered_s: Any,
    heldout_tasks: Sequence[HeldoutTask | Mapping[str, Any]],
    predictor: HeldoutPredictor,
    *,
    scorer: HeldoutScorer | None = None,
) -> dict[str, Any]:
    task_records = [_coerce_task(task) for task in heldout_tasks]
    if not task_records:
        return {"hbps": None, "hbps_status": "insufficient_heldout", "n_heldout": 0, "task_scores": []}
    resolved_scorer = scorer or score_behavior
    task_scores: list[dict[str, Any]] = []
    for task in task_records:
        prediction = predictor(recovered_s, task)
        score = resolved_scorer(prediction, task.gold_behavior, task.task_type)
        task_scores.append(
            {
                "task_id": task.task_id,
                "user_id": task.user_id,
                "task_type": task.task_type,
                "score": round(float(score), 6),
                "prediction": clone_json(prediction),
            }
        )
    average_score = sum(item["score"] for item in task_scores) / len(task_scores)
    return {
        "hbps": round(average_score, 6),
        "hbps_status": "ok",
        "n_heldout": len(task_scores),
        "task_scores": task_scores,
    }
