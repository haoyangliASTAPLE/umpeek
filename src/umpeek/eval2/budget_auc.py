from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from .costing import coerce_cost_round, cumulative_costs
from .schema import DEFAULT_BUDGET_GRID, clone_json


MetricEvaluator = Callable[[Any], Mapping[str, Any] | float | int | None]


def normalized_trapezoid(values: Sequence[float | int | None], q_grid: Sequence[int]) -> float | None:
    if len(values) != len(q_grid):
        raise ValueError("Budget-AUC values and q_grid must have the same length.")
    if any(value is None for value in values):
        return None
    if len(values) == 0:
        return None
    if len(values) == 1:
        return round(float(values[0]), 6)
    width = float(q_grid[-1] - q_grid[0])
    if width <= 0:
        raise ValueError("q_grid must be strictly increasing for normalized trapezoid.")
    area = 0.0
    for left_index in range(len(values) - 1):
        left_q = int(q_grid[left_index])
        right_q = int(q_grid[left_index + 1])
        if right_q <= left_q:
            raise ValueError("q_grid must be strictly increasing.")
        left_value = float(values[left_index])
        right_value = float(values[left_index + 1])
        area += (right_q - left_q) * (left_value + right_value) / 2.0
    return round(area / width, 6)


def _extract_metric_value(result: Mapping[str, Any] | float | int | None, metric_name: str) -> float | None:
    if result is None:
        return None
    if isinstance(result, Mapping):
        value = result.get(metric_name)
        if value is None and metric_name == "umr_f1":
            value = result.get("f1")
        return None if value is None else float(value)
    return float(result)


def build_budget_curve(
    trajectory: Sequence[Any],
    *,
    metric_name: str,
    budget_grid: Sequence[int] = DEFAULT_BUDGET_GRID,
    metric_evaluator: MetricEvaluator | None = None,
    final_metric: float | int | None = None,
    final_query_count: int | None = None,
    missing_before_final: float | None = 0.0,
) -> dict[str, Any]:
    q_grid = [int(budget_value) for budget_value in budget_grid]
    if q_grid != sorted(q_grid) or len(set(q_grid)) != len(q_grid):
        raise ValueError("All methods in one setting must share a strictly increasing budget_grid.")

    cumulative_rows = cumulative_costs([coerce_cost_round(item) for item in trajectory])
    has_intermediate = metric_evaluator is not None and any(row.get("latest_reconstruction") is not None for row in cumulative_rows)
    values: list[float | None] = []
    curve_missing_reason: str | None = None

    if has_intermediate:
        for budget_value in q_grid:
            eligible_rows = [row for row in cumulative_rows if int(row["num_effective_queries"]) <= budget_value and row.get("latest_reconstruction") is not None]
            if not eligible_rows:
                values.append(missing_before_final)
                continue
            latest_row = eligible_rows[-1]
            metric_value = _extract_metric_value(metric_evaluator(latest_row["latest_reconstruction"]), metric_name)
            if metric_value is None:
                curve_missing_reason = f"missing_{metric_name}_at_budget_{budget_value}"
            values.append(metric_value)
        curve_mode = "adaptive_prefix"
    else:
        actual_query_count = final_query_count
        if actual_query_count is None:
            actual_query_count = int(cumulative_rows[-1]["num_effective_queries"]) if cumulative_rows else 0
        for budget_value in q_grid:
            if final_metric is None:
                values.append(None)
                curve_missing_reason = "missing_final_metric"
            elif budget_value < int(actual_query_count):
                values.append(missing_before_final)
            else:
                values.append(float(final_metric))
        curve_mode = "step_final_only"

    auc = normalized_trapezoid(values, q_grid) if curve_missing_reason is None else None
    return {
        "metric_name": metric_name,
        "budget_grid": q_grid,
        "values": values,
        "budget_auc": auc,
        "curve_mode": curve_mode,
        "curve_missing_reason": curve_missing_reason,
        "cumulative_costs": clone_json(cumulative_rows),
    }


def evaluate_budget_auc(curves: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    reference_grid: list[int] | None = None
    for metric_name, curve in curves.items():
        grid = [int(value) for value in curve.get("budget_grid", [])]
        if reference_grid is None:
            reference_grid = grid
        elif grid != reference_grid:
            raise ValueError("Budget-AUC requires the same budget_grid for every method/metric setting.")
        output[f"budget_auc_{metric_name}"] = curve.get("budget_auc")
        output[f"budget_curve_{metric_name}"] = clone_json(curve)
    output["budget_grid"] = reference_grid or []
    return output
