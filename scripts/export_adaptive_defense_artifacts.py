#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs" / "adaptive_defense"
DEFAULT_EXTERNAL_ROOT = RUNS_ROOT / "A003_stratified128_external_defenses_prefix_current"
DEFAULT_STATEFUL_ROOT = RUNS_ROOT / "A004_stratified128_stateful_counterfactual_full_threshold_1p0_current"
DEFAULT_THRESHOLD_LOW_ROOT = RUNS_ROOT / "A005_stratified32_stateful_counterfactual_full_threshold_0p5_current"
DEFAULT_THRESHOLD_HIGH_ROOT = RUNS_ROOT / "A005_stratified32_stateful_counterfactual_full_threshold_1p5_current"
DEFAULT_NO_STATE_ROOT = (
    RUNS_ROOT
    / "A005_stratified32_stateful_counterfactual_without_cross_request_state_threshold_1p0_current"
)
DEFAULT_NO_COUNTERFACTUAL_ROOT = (
    RUNS_ROOT
    / "A005_stratified32_stateful_counterfactual_without_counterfactual_comparison_threshold_1p0_current"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "my" / "paper-facing" / "adaptive_defense"

BENCHMARK_ORDER = (
    "PersonaMem-v2",
    "PersonaLens",
    "ETAPP_150x32",
    "LoCoMo_10conv_1523QA_20speakers",
)
BENCHMARK_LABELS = {
    "PersonaMem-v2": "PersonaMem-v2",
    "PersonaLens": "PersonaLens",
    "ETAPP_150x32": "ETAPP",
    "LoCoMo_10conv_1523QA_20speakers": "LoCoMo",
}
MAIN_CONDITIONS = (
    "undefended",
    "privacy_checker",
    "theory_of_mind",
    "stateful_counterfactual",
)
TABLE_CONDITIONS = (
    "privacy_checker",
    "theory_of_mind",
    "stateful_counterfactual",
)
CONDITION_LABELS = {
    "undefended": "Undefended",
    "privacy_checker": "PrivacyChecker",
    "theory_of_mind": "Theory-of-Mind Defense",
    "stateful_counterfactual": "Stateful Counterfactual",
    "stateful_threshold_0p5": "Stateful threshold 0.5",
    "stateful_threshold_1p0": "Stateful threshold 1.0",
    "stateful_threshold_1p5": "Stateful threshold 1.5",
    "without_cross_request_state": "Without cross-request state",
    "without_counterfactual_comparison": "Without counterfactual comparison",
}
CONDITION_STYLES = {
    "undefended": {"color": "#777777", "marker": "o", "linestyle": "--"},
    "privacy_checker": {"color": "#0072B2", "marker": "D", "linestyle": "-"},
    "theory_of_mind": {"color": "#D55E00", "marker": "s", "linestyle": "-"},
    "stateful_counterfactual": {"color": "#009E73", "marker": "o", "linestyle": "-"},
}
THRESHOLD_CONDITIONS = (
    (0.5, "stateful_threshold_0p5"),
    (1.0, "stateful_threshold_1p0"),
    (1.5, "stateful_threshold_1p5"),
)
ABLATION_CONDITIONS = (
    "stateful_counterfactual",
    "without_cross_request_state",
    "without_counterfactual_comparison",
)
METRIC_PATHS = {
    "umr_f1": ("UMR-F1", "umr_f1"),
    "hbps": ("HBPS", "hbps"),
    "task_score": ("TaskScore", "task_score"),
}
SCALAR_FIELDS = (
    "source_run",
    "condition",
    "variant",
    "threshold",
    "budget",
    "backend",
    "benchmark",
    "sample_index",
    "sample_id_hash",
    "run_status",
    "umr_f1",
    "hbps",
    "task_score",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export final adaptive-defense paper artifacts.")
    parser.add_argument("--external-run-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--stateful-run-root", type=Path, default=DEFAULT_STATEFUL_ROOT)
    parser.add_argument("--threshold-low-run-root", type=Path, default=DEFAULT_THRESHOLD_LOW_ROOT)
    parser.add_argument("--threshold-high-run-root", type=Path, default=DEFAULT_THRESHOLD_HIGH_ROOT)
    parser.add_argument("--no-state-run-root", type=Path, default=DEFAULT_NO_STATE_ROOT)
    parser.add_argument("--no-counterfactual-run-root", type=Path, default=DEFAULT_NO_COUNTERFACTUAL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    return parser.parse_args()


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _nested(record: Mapping[str, Any], *path: str) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _hash(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def _relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
            if isinstance(value, Mapping):
                yield dict(value)


def _metric_paths(run_root: Path) -> list[Path]:
    paths = set((run_root / "settings").glob("*/*/metric_records.jsonl"))
    paths.update((run_root / "settings").glob("*/budget_*/*/metric_records.jsonl"))
    return sorted(paths)


def collect_scalar_records(
    run_root: Path,
    *,
    condition_override: str | None = None,
    variant: str,
    threshold: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    launch_path = run_root / "launch_manifest.json"
    if not launch_path.exists():
        raise FileNotFoundError(f"Missing launch manifest: {launch_path}")
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    files = _metric_paths(run_root)
    for path in files:
        prefix_layout = path.parents[1].parent == run_root / "settings"
        path_condition = path.parents[1].name if prefix_layout else path.parents[2].name
        path_budget = None if prefix_layout else int(path.parents[1].name.removeprefix("budget_"))
        for record in _iter_jsonl(path):
            budget = int(record.get("adaptive_budget", path_budget if path_budget is not None else 0))
            rows.append(
                {
                    "source_run": run_root.name,
                    "condition": condition_override or path_condition,
                    "variant": variant,
                    "threshold": threshold,
                    "budget": budget,
                    "backend": str(record.get("backend") or ""),
                    "benchmark": str(record.get("benchmark") or ""),
                    "sample_index": int(record.get("sample_index") or 0),
                    "sample_id_hash": _hash(record.get("sample_id") or ""),
                    "run_status": str(record.get("run_status") or ""),
                    "umr_f1": _number(_nested(record, "metrics", *METRIC_PATHS["umr_f1"])),
                    "hbps": _number(_nested(record, "metrics", *METRIC_PATHS["hbps"])),
                    "task_score": _number(_nested(record, "metrics", *METRIC_PATHS["task_score"])),
                }
            )
    expected = int(launch.get("planned_record_count", 0) or 0)
    missing_metric_counts = {
        metric: sum(row[metric] is None for row in rows) for metric in METRIC_PATHS
    }
    audit = {
        "run_root": _relative(run_root),
        "source_file_count": len(files),
        "expected_record_count": expected,
        "observed_record_count": len(rows),
        "failed_record_count": sum(row["run_status"] != "ok" for row in rows),
        "missing_metric_counts": missing_metric_counts,
        "complete": bool(files)
        and len(rows) == expected
        and all(row["run_status"] == "ok" for row in rows)
        and not any(missing_metric_counts.values()),
    }
    return rows, audit


def _sample_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row["backend"]), str(row["benchmark"]), str(row["sample_id_hash"])


def _final_sample_keys(rows: Sequence[Mapping[str, Any]], final_budget: int) -> set[tuple[str, str, str]]:
    return {_sample_key(row) for row in rows if int(row["budget"]) == final_budget}


def _condition_sample_keys(
    rows: Sequence[Mapping[str, Any]], condition: str, final_budget: int
) -> set[tuple[str, str, str]]:
    return {
        _sample_key(row)
        for row in rows
        if row["condition"] == condition and int(row["budget"]) == final_budget
    }


def _filter_keys(rows: Sequence[Mapping[str, Any]], keys: set[tuple[str, str, str]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if _sample_key(row) in keys]


def _cluster_values(
    rows: Sequence[Mapping[str, Any]], metric: str, *, cluster_fields: Sequence[str]
) -> list[float]:
    by_cluster: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = _number(row.get(metric))
        if value is not None:
            key = tuple(row[field] for field in cluster_fields)
            by_cluster[key].append(value)
    return [float(np.mean(values)) for _, values in sorted(by_cluster.items()) if values]


def _bootstrap_mean_ci(
    values: Sequence[float], *, replicates: int, seed: int
) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    if len(array) == 1 or replicates <= 0:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(replicates, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return mean, float(low), float(high)


def _add_metric_summary(
    item: dict[str, Any],
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    cluster_fields: Sequence[str],
    replicates: int,
    seed_key: str,
) -> None:
    values = _cluster_values(rows, metric, cluster_fields=cluster_fields)
    seed = int(_hash(f"{seed_key}|{metric}"), 16) % (2**32)
    mean, low, high = _bootstrap_mean_ci(values, replicates=replicates, seed=seed)
    item[metric] = mean
    item[f"{metric}_ci_low"] = low
    item[f"{metric}_ci_high"] = high
    item[f"{metric}_n"] = len(values)


def summarize_by_benchmark(
    rows: Sequence[Mapping[str, Any]], *, replicates: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("run_status") == "ok":
            groups[(str(row["condition"]), int(row["budget"]), str(row["benchmark"]))].append(row)
    output: list[dict[str, Any]] = []
    for (condition, budget, benchmark), group in sorted(groups.items()):
        item: dict[str, Any] = {
            "condition": condition,
            "budget": budget,
            "benchmark": benchmark,
            "N_records": len(group),
            "N_clusters": len({int(row["sample_index"]) for row in group}),
            "backend_count": len({str(row["backend"]) for row in group}),
        }
        for metric in METRIC_PATHS:
            _add_metric_summary(
                item,
                group,
                metric,
                cluster_fields=("sample_index",),
                replicates=replicates,
                seed_key=f"benchmark|{condition}|{budget}|{benchmark}",
            )
        output.append(item)
    return output


def summarize_ablations(
    rows: Sequence[Mapping[str, Any]], *, final_budget: int, replicates: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for condition in ABLATION_CONDITIONS:
        group = [
            row
            for row in rows
            if row["condition"] == condition and int(row["budget"]) == final_budget
        ]
        item: dict[str, Any] = {
            "condition": condition,
            "budget": final_budget,
            "N_records": len(group),
            "N_clusters": len({(row["benchmark"], row["sample_index"]) for row in group}),
            "backend_count": len({str(row["backend"]) for row in group}),
        }
        for metric in ("umr_f1", "hbps"):
            _add_metric_summary(
                item,
                group,
                metric,
                cluster_fields=("benchmark", "sample_index"),
                replicates=replicates,
                seed_key=f"ablation|{condition}|overall",
            )
        for benchmark in BENCHMARK_ORDER:
            benchmark_rows = [row for row in group if row["benchmark"] == benchmark]
            field = _task_field(benchmark)
            _add_metric_summary(
                item,
                benchmark_rows,
                "task_score",
                cluster_fields=("sample_index",),
                replicates=replicates,
                seed_key=f"ablation|{condition}|{benchmark}",
            )
            item[field] = item.pop("task_score")
            item[f"{field}_ci_low"] = item.pop("task_score_ci_low")
            item[f"{field}_ci_high"] = item.pop("task_score_ci_high")
            item[f"{field}_n"] = item.pop("task_score_n")
        output.append(item)
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _cell(value: Any, low: Any, high: Any) -> str:
    numbers = (_number(value), _number(low), _number(high))
    if any(item is None for item in numbers):
        return "n/a"
    return f"{numbers[0]:.3f} [{numbers[1]:.3f}, {numbers[2]:.3f}]"


def _main_table_latex(summary: Sequence[Mapping[str, Any]], final_budget: int) -> str:
    lookup = {(row["condition"], row["benchmark"], row["budget"]): row for row in summary}
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Benchmark & Defense & UMR-F1 $\\downarrow$ & HBPS $\\downarrow$ & TaskScore $\\uparrow$ \\\\",
        "\\midrule",
    ]
    for benchmark_index, benchmark in enumerate(BENCHMARK_ORDER):
        for condition_index, condition in enumerate(TABLE_CONDITIONS):
            row = lookup.get((condition, benchmark, final_budget), {})
            lines.append(
                " & ".join(
                    [
                        BENCHMARK_LABELS[benchmark] if condition_index == 0 else "",
                        CONDITION_LABELS[condition],
                        _cell(row.get("umr_f1"), row.get("umr_f1_ci_low"), row.get("umr_f1_ci_high")),
                        _cell(row.get("hbps"), row.get("hbps_ci_low"), row.get("hbps_ci_high")),
                        _cell(row.get("task_score"), row.get("task_score_ci_low"), row.get("task_score_ci_high")),
                    ]
                )
                + " \\\\"
            )
        if benchmark_index < len(BENCHMARK_ORDER) - 1:
            lines.append("\\addlinespace")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def _task_field(benchmark: str) -> str:
    return f"task_score__{BENCHMARK_LABELS[benchmark].lower().replace('-', '_')}"


def _ablation_table_latex(summary: Sequence[Mapping[str, Any]]) -> str:
    lookup = {str(row["condition"]): row for row in summary}
    row_labels = {
        "stateful_counterfactual": "Stateful Counterfactual",
        "without_cross_request_state": "\\shortstack[l]{Without cross-request\\\\state}",
        "without_counterfactual_comparison": "\\shortstack[l]{Without counterfactual\\\\comparison}",
    }
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Variant & UMR-F1 $\\downarrow$ & HBPS $\\downarrow$ & "
        "\\shortstack{PersonaMem-v2\\\\TaskScore $\\uparrow$} & "
        "\\shortstack{PersonaLens\\\\TaskScore $\\uparrow$} & "
        "\\shortstack{ETAPP\\\\TaskScore $\\uparrow$} & "
        "\\shortstack{LoCoMo\\\\TaskScore $\\uparrow$} \\\\",
        "\\midrule",
    ]
    for condition in ABLATION_CONDITIONS:
        row = lookup.get(condition, {})
        cells = [
            row_labels[condition],
            _cell(row.get("umr_f1"), row.get("umr_f1_ci_low"), row.get("umr_f1_ci_high")),
            _cell(row.get("hbps"), row.get("hbps_ci_low"), row.get("hbps_ci_high")),
        ]
        for benchmark in BENCHMARK_ORDER:
            field = _task_field(benchmark)
            cells.append(_cell(row.get(field), row.get(f"{field}_ci_low"), row.get(f"{field}_ci_high")))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def _error(value: float, low: float | None, high: float | None) -> np.ndarray | None:
    if low is None or high is None:
        return None
    return np.asarray([[max(0.0, value - low)], [max(0.0, high - value)]])


def _configure_plot_style() -> str:
    font_dir = Path.home() / ".local/share/fonts/TimesNewRoman"
    font_files = sorted(font_dir.glob("*.TTF"))
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)
    families = ("Times New Roman", "Tinos", "Nimbus Roman", "Liberation Serif", "DejaVu Serif")
    font_path = font_manager.findfont("DejaVu Serif")
    resolved_family = "DejaVu Serif"
    for family in families:
        try:
            candidate = font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        if candidate:
            font_path = candidate
            resolved_family = family
            break
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": list(families),
            "font.size": 16,
            "axes.labelsize": 17,
            "axes.titlesize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 1.0,
        }
    )
    return f"{resolved_family}: {font_path}"


def _plot(
    main_summary: Sequence[Mapping[str, Any]],
    tradeoff_summary: Sequence[Mapping[str, Any]],
    out_dir: Path,
    final_budget: int,
) -> str:
    font_path = _configure_plot_style()
    main_lookup = {
        (row["condition"], row["benchmark"], row["budget"]): row for row in main_summary
    }
    tradeoff_lookup = {
        (row["condition"], row["benchmark"], row["budget"]): row
        for row in tradeoff_summary
    }
    budgets = sorted({int(row["budget"]) for row in main_summary})
    fig, axes = plt.subplots(2, 4, figsize=(17.5, 9.5), sharey="row", constrained_layout=True)

    for column, benchmark in enumerate(BENCHMARK_ORDER):
        top = axes[0, column]
        bottom = axes[1, column]
        top.set_title(BENCHMARK_LABELS[benchmark], pad=10)

        threshold_rows: list[tuple[float, Mapping[str, Any]]] = []
        for threshold, condition in THRESHOLD_CONDITIONS:
            row = tradeoff_lookup.get((condition, benchmark, final_budget))
            if row and row.get("umr_f1") is not None and row.get("task_score") is not None:
                threshold_rows.append((threshold, row))
        if threshold_rows:
            xs = [float(row["umr_f1"]) for _, row in threshold_rows]
            ys = [float(row["task_score"]) for _, row in threshold_rows]
            top.plot(
                xs,
                ys,
                color=CONDITION_STYLES["stateful_counterfactual"]["color"],
                marker="o",
                linewidth=2.4,
                markersize=8,
                label="Stateful Counterfactual",
                zorder=4,
            )
            for threshold, row in threshold_rows:
                x = float(row["umr_f1"])
                y = float(row["task_score"])
                top.errorbar(
                    x,
                    y,
                    xerr=_error(x, _number(row.get("umr_f1_ci_low")), _number(row.get("umr_f1_ci_high"))),
                    yerr=_error(y, _number(row.get("task_score_ci_low")), _number(row.get("task_score_ci_high"))),
                    color=CONDITION_STYLES["stateful_counterfactual"]["color"],
                    marker="o",
                    markersize=8,
                    linewidth=1.7,
                    capsize=2.5,
                    zorder=4,
                )
                label_offsets = {
                    0.5: (-24, -20),
                    1.0: (8, -4),
                    1.5: (8, 14),
                }
                offset = label_offsets.get(threshold, (8, 8))
                top.annotate(
                    f"{threshold:g}",
                    (x, y),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=12,
                    color="#176B4B",
                    ha="right" if offset[0] < 0 else "left",
                    va="top" if offset[1] < 0 else "bottom",
                    arrowprops={"arrowstyle": "-", "color": "#176B4B", "linewidth": 0.7},
                )

        for condition in ("privacy_checker", "theory_of_mind", "undefended"):
            row = tradeoff_lookup.get((condition, benchmark, final_budget))
            if not row or row.get("umr_f1") is None or row.get("task_score") is None:
                continue
            style = CONDITION_STYLES[condition]
            x = float(row["umr_f1"])
            y = float(row["task_score"])
            top.errorbar(
                x,
                y,
                xerr=_error(x, _number(row.get("umr_f1_ci_low")), _number(row.get("umr_f1_ci_high"))),
                yerr=_error(y, _number(row.get("task_score_ci_low")), _number(row.get("task_score_ci_high"))),
                color=style["color"],
                marker=style["marker"],
                markersize=10 if condition == "undefended" else 8,
                markerfacecolor="none" if condition == "undefended" else style["color"],
                markeredgewidth=1.8 if condition == "undefended" else 1.0,
                linewidth=1.7,
                capsize=2.5,
                label=CONDITION_LABELS[condition],
                zorder=5 if condition == "undefended" else 3,
            )

        for condition in MAIN_CONDITIONS:
            style = CONDITION_STYLES[condition]
            curve = [main_lookup.get((condition, benchmark, budget)) for budget in budgets]
            curve = [row for row in curve if row and row.get("umr_f1") is not None]
            if not curve:
                continue
            xs = [int(row["budget"]) for row in curve]
            ys = [float(row["umr_f1"]) for row in curve]
            lows = [float(row.get("umr_f1_ci_low") if row.get("umr_f1_ci_low") is not None else value) for row, value in zip(curve, ys)]
            highs = [float(row.get("umr_f1_ci_high") if row.get("umr_f1_ci_high") is not None else value) for row, value in zip(curve, ys)]
            bottom.plot(
                xs,
                ys,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.2,
                markersize=6.5,
                zorder=2 if condition == "undefended" else 3,
                label=CONDITION_LABELS[condition],
            )
            bottom.fill_between(xs, lows, highs, color=style["color"], alpha=0.13, linewidth=0)

        top.set_xlim(-0.03, 1.03)
        top.set_ylim(-0.03, 1.03)
        top.set_xlabel("UMR-F1")
        bottom.set_ylim(-0.03, 1.03)
        bottom.set_xticks(budgets)
        bottom.set_xlabel("Follow-up budget")
        for axis in (top, bottom):
            axis.grid(axis="y", color="#D6D6D6", linewidth=0.75)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
    axes[0, 0].set_ylabel("TaskScore")
    axes[1, 0].set_ylabel("UMR-F1")
    axes[0, 0].text(-0.24, 1.09, "(a)", transform=axes[0, 0].transAxes, fontsize=18, fontweight="bold")
    axes[1, 0].text(-0.24, 1.09, "(b)", transform=axes[1, 0].transAxes, fontsize=18, fontweight="bold")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = {
        "Undefended": 0,
        "PrivacyChecker": 1,
        "Theory-of-Mind Defense": 2,
        "Stateful Counterfactual": 3,
    }
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    labels = sorted(unique, key=lambda label: order.get(label, 99))
    if labels:
        fig.legend(
            [unique[label] for label in labels],
            labels,
            loc="outside upper center",
            ncol=len(labels),
            frameon=False,
        )
    fig.savefig(
        out_dir / "figure_adaptive_defense.pdf",
        bbox_inches="tight",
        metadata={"Creator": "Matplotlib", "Producer": "Matplotlib", "CreationDate": None, "ModDate": None},
    )
    fig.savefig(out_dir / "figure_adaptive_defense.jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return font_path


def _validate_sources(source_audits: Sequence[Mapping[str, Any]]) -> None:
    incomplete = [audit["run_root"] for audit in source_audits if not audit.get("complete")]
    if incomplete:
        raise RuntimeError(f"Incomplete adaptive-defense sources: {incomplete}")


def _matched_sample_audit(
    external_rows: Sequence[Mapping[str, Any]],
    stateful_rows: Sequence[Mapping[str, Any]],
    threshold_low_rows: Sequence[Mapping[str, Any]],
    threshold_high_rows: Sequence[Mapping[str, Any]],
    no_state_rows: Sequence[Mapping[str, Any]],
    no_counterfactual_rows: Sequence[Mapping[str, Any]],
    *,
    final_budget: int,
) -> tuple[set[tuple[str, str, str]], dict[str, Any]]:
    matched_keys = _final_sample_keys(threshold_low_rows, final_budget)
    comparisons = {
        "threshold_1p5": _final_sample_keys(threshold_high_rows, final_budget),
        "without_cross_request_state": _final_sample_keys(no_state_rows, final_budget),
        "without_counterfactual_comparison": _final_sample_keys(no_counterfactual_rows, final_budget),
    }
    external_sets = {
        condition: _condition_sample_keys(external_rows, condition, final_budget)
        for condition in ("undefended", "privacy_checker", "theory_of_mind")
    }
    stateful_set = _final_sample_keys(stateful_rows, final_budget)
    equal_variant_sets = {name: values == matched_keys for name, values in comparisons.items()}
    external_contains = {name: matched_keys <= values for name, values in external_sets.items()}
    per_setting_counts: dict[str, int] = defaultdict(int)
    for backend, benchmark, _ in matched_keys:
        per_setting_counts[f"{backend}|{benchmark}"] += 1
    audit = {
        "schema_version": "adaptive_defense_matched_sample_audit_v1",
        "final_budget": final_budget,
        "matched_sample_count": len(matched_keys),
        "expected_matched_sample_count": 32 * 3 * 4,
        "per_backend_benchmark_sample_counts": dict(sorted(per_setting_counts.items())),
        "variant_sample_sets_equal": equal_variant_sets,
        "stateful_128_contains_matched_subset": matched_keys <= stateful_set,
        "external_128_contains_matched_subset": external_contains,
        "all_checks_pass": len(matched_keys) == 32 * 3 * 4
        and all(equal_variant_sets.values())
        and matched_keys <= stateful_set
        and all(external_contains.values()),
    }
    if not audit["all_checks_pass"]:
        raise RuntimeError(f"Adaptive-defense matched-sample audit failed: {audit}")
    return matched_keys, audit


def _artifact_audit(
    *,
    source_audits: Sequence[Mapping[str, Any]],
    matched_audit: Mapping[str, Any],
    main_summary: Sequence[Mapping[str, Any]],
    tradeoff_summary: Sequence[Mapping[str, Any]],
    ablation_summary: Sequence[Mapping[str, Any]],
    final_budget: int,
    font_path: str,
    out_dir: Path,
) -> dict[str, Any]:
    main_lookup = {(row["condition"], row["benchmark"], row["budget"]): row for row in main_summary}
    tradeoff_lookup = {(row["condition"], row["benchmark"], row["budget"]): row for row in tradeoff_summary}
    expected_main = {
        (condition, benchmark, budget)
        for condition in MAIN_CONDITIONS
        for benchmark in BENCHMARK_ORDER
        for budget in (0, 1, 2, 4, 8, 16)
    }
    expected_tradeoff = {
        (condition, benchmark, final_budget)
        for condition in (
            "undefended",
            "privacy_checker",
            "theory_of_mind",
            "stateful_threshold_0p5",
            "stateful_threshold_1p0",
            "stateful_threshold_1p5",
        )
        for benchmark in BENCHMARK_ORDER
    }
    expected_files = (
        "table_adaptive_defense_results.tex",
        "figure_adaptive_defense.pdf",
        "figure_adaptive_defense.jpg",
        "table_stateful_defense_ablations.tex",
        "captions.tex",
        "README.md",
    )
    missing_main = sorted(expected_main - set(main_lookup))
    missing_tradeoff = sorted(expected_tradeoff - set(tradeoff_lookup))
    missing_files = [name for name in expected_files if not (out_dir / name).exists()]
    metrics_present = all(
        row.get(metric) is not None
        for row in list(main_summary) + list(tradeoff_summary)
        for metric in ("umr_f1", "hbps", "task_score")
    ) and all(
        row.get(metric) is not None
        for row in ablation_summary
        for metric in ("umr_f1", "hbps", *(_task_field(benchmark) for benchmark in BENCHMARK_ORDER))
    )
    audit = {
        "schema_version": "adaptive_defense_artifact_audit_v2",
        "source_runs": list(source_audits),
        "matched_sample_audit": dict(matched_audit),
        "final_budget": final_budget,
        "main_result_sample_count_per_backend_benchmark": 128,
        "threshold_and_ablation_sample_count_per_backend_benchmark": 32,
        "missing_main_summary_cells": ["|".join(map(str, item)) for item in missing_main],
        "missing_tradeoff_summary_cells": ["|".join(map(str, item)) for item in missing_tradeoff],
        "ablation_row_count": len(ablation_summary),
        "expected_ablation_row_count": len(ABLATION_CONDITIONS),
        "all_required_metrics_present": metrics_present,
        "missing_final_files": missing_files,
        "font_family": "Times New Roman",
        "font_path": font_path,
        "confidence_interval": "95% matched cluster bootstrap with fixed per-cell seeds",
        "cluster_unit": "benchmark sample index; backend values averaged within each cluster",
        "contains_raw_prompts_or_private_behaviors": False,
        "excluded_from_paper_artifacts": [
            "ASR",
            "Best Baseline",
            "privacy-utility composite scores",
            "token and latency diagnostics",
            "raw prompts and trajectories",
        ],
    }
    audit["all_checks_pass"] = (
        all(item.get("complete") for item in source_audits)
        and bool(matched_audit.get("all_checks_pass"))
        and not missing_main
        and not missing_tradeoff
        and len(ablation_summary) == len(ABLATION_CONDITIONS)
        and metrics_present
        and not missing_files
    )
    return audit


def main() -> int:
    args = _arguments()
    sources = (
        (args.external_run_root, None, "external_defenses", None),
        (args.stateful_run_root, "stateful_counterfactual", "full", 1.0),
        (args.threshold_low_run_root, "stateful_threshold_0p5", "full", 0.5),
        (args.threshold_high_run_root, "stateful_threshold_1p5", "full", 1.5),
        (args.no_state_run_root, "without_cross_request_state", "without_cross_request_state", 1.0),
        (
            args.no_counterfactual_run_root,
            "without_counterfactual_comparison",
            "without_counterfactual_comparison",
            1.0,
        ),
    )
    collected: dict[str, list[dict[str, Any]]] = {}
    source_audits: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for run_root, condition, variant, threshold in sources:
        rows, audit = collect_scalar_records(
            run_root,
            condition_override=condition,
            variant=variant,
            threshold=threshold,
        )
        collected[variant if condition is None else condition] = rows
        source_audits.append(audit)
        all_rows.extend(rows)
    _validate_sources(source_audits)

    external_rows = collected["external_defenses"]
    stateful_rows = collected["stateful_counterfactual"]
    threshold_low_rows = collected["stateful_threshold_0p5"]
    threshold_high_rows = collected["stateful_threshold_1p5"]
    no_state_rows = collected["without_cross_request_state"]
    no_counterfactual_rows = collected["without_counterfactual_comparison"]
    budgets = sorted({int(row["budget"]) for row in external_rows + stateful_rows})
    if budgets != [0, 1, 2, 4, 8, 16]:
        raise RuntimeError(f"Unexpected follow-up budgets: {budgets}")
    final_budget = max(budgets)

    matched_keys, matched_audit = _matched_sample_audit(
        external_rows,
        stateful_rows,
        threshold_low_rows,
        threshold_high_rows,
        no_state_rows,
        no_counterfactual_rows,
        final_budget=final_budget,
    )

    main_rows = external_rows + stateful_rows
    main_summary = summarize_by_benchmark(main_rows, replicates=args.bootstrap_replicates)

    external_matched = _filter_keys(external_rows, matched_keys)
    stateful_matched = _filter_keys(stateful_rows, matched_keys)
    for row in stateful_matched:
        row["condition"] = "stateful_threshold_1p0"
    tradeoff_rows = [
        row
        for row in (
            external_matched
            + _filter_keys(threshold_low_rows, matched_keys)
            + stateful_matched
            + _filter_keys(threshold_high_rows, matched_keys)
        )
        if int(row["budget"]) == final_budget
    ]
    tradeoff_summary = summarize_by_benchmark(tradeoff_rows, replicates=args.bootstrap_replicates)

    full_ablation_rows = _filter_keys(stateful_rows, matched_keys)
    ablation_rows = full_ablation_rows + _filter_keys(no_state_rows, matched_keys) + _filter_keys(
        no_counterfactual_rows, matched_keys
    )
    ablation_summary = summarize_ablations(
        ablation_rows,
        final_budget=final_budget,
        replicates=args.bootstrap_replicates,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    intermediate = args.out_dir / "intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)
    _write_csv(intermediate / "adaptive_defense_scalar_records.csv", all_rows, SCALAR_FIELDS)
    _write_csv(
        intermediate / "adaptive_defense_main_summary.csv",
        main_summary,
        list(main_summary[0]) if main_summary else (),
    )
    _write_csv(
        intermediate / "adaptive_defense_tradeoff_summary.csv",
        tradeoff_summary,
        list(tradeoff_summary[0]) if tradeoff_summary else (),
    )
    _write_csv(
        intermediate / "adaptive_defense_ablation_summary.csv",
        ablation_summary,
        list(ablation_summary[0]) if ablation_summary else (),
    )
    (intermediate / "matched_sample_audit.json").write_text(
        json.dumps(matched_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    (args.out_dir / "table_adaptive_defense_results.tex").write_text(
        _main_table_latex(main_summary, final_budget), encoding="utf-8"
    )
    (args.out_dir / "table_stateful_defense_ablations.tex").write_text(
        _ablation_table_latex(ablation_summary), encoding="utf-8"
    )
    font_path = _plot(main_summary, tradeoff_summary, args.out_dir, final_budget)
    (args.out_dir / "captions.tex").write_text(
        "\\newcommand{\\AdaptiveDefenseTableCaption}{UMPeek recovery and normal-task quality under three inference-time defenses at an additional follow-up budget of 16. Results use 128 samples per backend and benchmark. Backend scores are averaged within matched sample clusters; brackets report 95\\% cluster-bootstrap confidence intervals. The undefended recovery and utility references appear in Tables~1 and~2.}\n"
        "\\newcommand{\\AdaptiveDefenseFigureCaption}{Privacy--utility operating points and recovery across additional follow-up budgets. Panel~(a) uses a matched stratified subset of 32 samples per backend and benchmark for the Stateful Counterfactual threshold sweep; fixed defense points and the undefended reference use the same subset. Panel~(b) uses 128 samples per backend and benchmark. Bands report 95\\% cluster-bootstrap confidence intervals.}\n"
        "\\newcommand{\\AdaptiveDefenseAblationCaption}{Mechanism ablations for Stateful Counterfactual Exposure Control on a matched stratified subset of 32 samples per backend and benchmark. UMR-F1 and HBPS are averaged across benchmarks after backend aggregation; TaskScore remains benchmark-specific. Brackets report 95\\% cluster-bootstrap confidence intervals.}\n",
        encoding="utf-8",
    )

    (args.out_dir / "README.md").write_text(
        "# Adaptive Defense Paper Artifacts\n\n"
        "Final artifacts:\n"
        "- `table_adaptive_defense_results.tex`: three-defense main results at budget 16, using 128 samples per backend and benchmark.\n"
        "- `figure_adaptive_defense.pdf` and `.jpg`: matched threshold trade-off and recovery-by-budget panels.\n"
        "- `table_stateful_defense_ablations.tex`: two mechanism ablations on the matched stratified-32 subset.\n"
        "- `captions.tex`: captions kept outside the figure and table bodies.\n\n"
        "Numeric summaries, hashed sample identifiers, source checks, and artifact audits are in `intermediate/`. "
        "The existing `smoke/` directory is an earlier smoke export and is not a final paper artifact.\n",
        encoding="utf-8",
    )

    audit = _artifact_audit(
        source_audits=source_audits,
        matched_audit=matched_audit,
        main_summary=main_summary,
        tradeoff_summary=tradeoff_summary,
        ablation_summary=ablation_summary,
        final_budget=final_budget,
        font_path=font_path,
        out_dir=args.out_dir,
    )
    (intermediate / "artifact_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not audit["all_checks_pass"]:
        raise RuntimeError("Adaptive-defense artifact audit failed; inspect intermediate/artifact_audit.json")
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
