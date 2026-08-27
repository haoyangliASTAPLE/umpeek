from __future__ import annotations

from typing import Any, Mapping, Sequence

from .schema import AttackCostRound, clone_json


def coerce_cost_round(payload: AttackCostRound | Mapping[str, Any]) -> AttackCostRound:
    if isinstance(payload, AttackCostRound):
        return payload
    cost = payload.get("cost", payload) if isinstance(payload.get("cost"), Mapping) else payload
    return AttackCostRound(
        round_index=int(cost.get("round_index", cost.get("round", 1))),
        num_queries=int(cost.get("num_queries", cost.get("query_count", 0)) or 0),
        num_effective_queries=int(
            cost.get("num_effective_queries", cost.get("effective_query_count", cost.get("query_count", 0))) or 0
        ),
        prompt_tokens=int(cost.get("prompt_tokens", 0) or 0),
        completion_tokens=int(cost.get("completion_tokens", 0) or 0),
        wall_time_sec=float(cost.get("wall_time_sec", cost.get("wall_clock_s", 0.0)) or 0.0),
        num_retries=int(cost.get("num_retries", cost.get("retries", 0)) or 0),
        num_timeouts=int(cost.get("num_timeouts", cost.get("timeouts", 0)) or 0),
        num_invalid_outputs=int(cost.get("num_invalid_outputs", cost.get("invalid_outputs", 0)) or 0),
        reconstruction=cost.get("reconstruction", cost.get("latest_reconstruction")),
        metadata=dict(cost.get("metadata", {})) if isinstance(cost.get("metadata", {}), Mapping) else {},
    )


def cumulative_costs(rounds: Sequence[AttackCostRound | Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered_rounds = sorted((coerce_cost_round(item) for item in rounds), key=lambda item: item.round_index)
    totals = {
        "num_queries": 0,
        "num_effective_queries": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "wall_time_sec": 0.0,
        "num_retries": 0,
        "num_timeouts": 0,
        "num_invalid_outputs": 0,
    }
    cumulative_rows: list[dict[str, Any]] = []
    for round_record in ordered_rounds:
        totals["num_queries"] += round_record.num_queries
        totals["num_effective_queries"] += round_record.num_effective_queries
        totals["prompt_tokens"] += round_record.prompt_tokens
        totals["completion_tokens"] += round_record.completion_tokens
        totals["total_tokens"] += round_record.total_tokens
        totals["wall_time_sec"] += round_record.wall_time_sec
        totals["num_retries"] += round_record.num_retries
        totals["num_timeouts"] += round_record.num_timeouts
        totals["num_invalid_outputs"] += round_record.num_invalid_outputs
        cumulative_rows.append(
            {
                "round_index": round_record.round_index,
                **clone_json(totals),
                "latest_reconstruction": clone_json(round_record.reconstruction),
                "metadata": clone_json(round_record.metadata),
            }
        )
    return cumulative_rows


def summarize_attack_cost(rounds: Sequence[AttackCostRound | Mapping[str, Any]]) -> dict[str, Any]:
    cumulative_rows = cumulative_costs(rounds)
    if not cumulative_rows:
        return {
            "num_rounds": 0,
            "num_queries": 0,
            "num_effective_queries": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "wall_time_sec": 0.0,
            "num_retries": 0,
            "num_timeouts": 0,
            "num_invalid_outputs": 0,
            "cumulative_rounds": [],
        }
    final_row = cumulative_rows[-1]
    return {
        "num_rounds": len(cumulative_rows),
        "num_queries": int(final_row["num_queries"]),
        "num_effective_queries": int(final_row["num_effective_queries"]),
        "prompt_tokens": int(final_row["prompt_tokens"]),
        "completion_tokens": int(final_row["completion_tokens"]),
        "total_tokens": int(final_row["total_tokens"]),
        "wall_time_sec": round(float(final_row["wall_time_sec"]), 6),
        "num_retries": int(final_row["num_retries"]),
        "num_timeouts": int(final_row["num_timeouts"]),
        "num_invalid_outputs": int(final_row["num_invalid_outputs"]),
        "cumulative_rounds": cumulative_rows,
    }
