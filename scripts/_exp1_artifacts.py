from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


BENCHMARK_ORDER = ("PersonaMem-v2", "PersonaLens", "ETAPP", "LoCoMo")
BACKEND_ORDER = ("Mem0", "Graphiti", "LangMem+LangGraph")
FIGURE_SOURCE_ORDER = ("S", "swap S", "delete S", "no memory")
TOKEN_SOURCE_ORDER = ("S", "M_u", "H_rel", "H_sum")
SOURCE_COLORS = {"S": "#1b6ca8", "M_u": "#b35c24", "H_rel": "#3b7f4c", "H_sum": "#8f5aa2"}
BACKEND_COLORS = {"Mem0": "#1b6ca8", "Graphiti": "#b35c24", "LangMem+LangGraph": "#3b7f4c"}
BACKEND_MARKERS = {"Mem0": "o", "Graphiti": "s", "LangMem+LangGraph": "D"}
SOURCE_DISPLAY = {
    "S": "Runtime state ($S$)",
    "M_u": "Full memory ($M_u$)",
    "H_rel": "Relevant history ($H_{rel}$)",
    "H_sum": "History summary ($H_{sum}$)",
}


def setup_style() -> str:
    from matplotlib import font_manager

    font_dir = Path.home() / ".local/share/fonts/TimesNewRoman"
    for font_path in sorted(font_dir.glob("*.TTF")):
        font_manager.fontManager.addfont(str(font_path))
    families = ("Times New Roman", "Tinos", "Nimbus Roman", "Liberation Serif", "DejaVu Serif")
    resolved = "DejaVu Serif"
    for family in families:
        try:
            path = font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        if path:
            resolved = family
            break
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": list(families),
            "font.size": 20,
            "axes.labelsize": 23,
            "axes.titlesize": 23,
            "xtick.labelsize": 19,
            "ytick.labelsize": 19,
            "legend.fontsize": 18,
            "figure.titlesize": 24,
            "axes.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return resolved


def clean_output(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    intermediate = path / "intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)
    return intermediate


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def tex_escape(value: Any) -> str:
    return (
        str(value)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def mean_ci(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray([float(value) for value in values], dtype=float)
    if len(array) == 0:
        return float("nan"), float("nan")
    mean = float(array.mean())
    ci = 0.0 if len(array) == 1 else 1.96 * float(array.std(ddof=1)) / math.sqrt(len(array))
    return mean, ci


def fmt_ci(mean: float, ci: float) -> str:
    return f"{mean:.3f} $\\pm$ {ci:.3f}"


def exp1_data(condition: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pivot = condition.pivot_table(
        index=["benchmark", "backend", "sample_id"],
        columns="condition",
        values=["task_score", "state_token_count"],
        aggfunc="first",
    )
    pivot.columns = [f"{left}::{right}" for left, right in pivot.columns]
    pivot = pivot.reset_index()
    signatures = condition.pivot_table(
        index=["benchmark", "backend", "sample_id"],
        columns="condition",
        values="decision_signature_hash",
        aggfunc="first",
    ).reset_index()
    signatures.columns = [column if isinstance(column, str) else column[-1] for column in signatures.columns]
    signatures = signatures.rename(
        columns={"S": "decision_signature::S", "swap_S": "decision_signature::swap_S"}
    )
    pivot = pivot.merge(
        signatures[
            [
                "benchmark",
                "backend",
                "sample_id",
                "decision_signature::S",
                "decision_signature::swap_S",
            ]
        ],
        on=["benchmark", "backend", "sample_id"],
        how="left",
        validate="one_to_one",
    )
    for condition_name in ("S", "no_memory", "delete_S", "swap_S", "M_u", "H_rel", "H_sum"):
        if f"task_score::{condition_name}" not in pivot:
            raise RuntimeError(f"Missing Experiment 1 condition: {condition_name}")
    pivot["Task Score Gain"] = pivot["task_score::S"] - pivot["task_score::no_memory"]
    pivot["Delete Drop"] = pivot["task_score::S"] - pivot["task_score::delete_S"]
    pivot["Swap Change"] = (
        pivot["decision_signature::S"] != pivot["decision_signature::swap_S"]
    ).astype(float)

    table_rows: list[dict[str, Any]] = []
    plot_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    for (benchmark, backend), group in pivot.groupby(["benchmark", "backend"], sort=False):
        row: dict[str, Any] = {"benchmark": benchmark, "backend": backend, "N": len(group)}
        for label, column in (
            ("TaskScore(S)", "task_score::S"),
            ("TaskScore(no memory)", "task_score::no_memory"),
            ("TaskScore(delete S)", "task_score::delete_S"),
            ("TaskScore(swap S)", "task_score::swap_S"),
            ("Task Score Gain", "Task Score Gain"),
            ("Delete Drop", "Delete Drop"),
            ("Swap Change", "Swap Change"),
        ):
            mu, ci = mean_ci(group[column].tolist())
            row[f"{label}_mean"] = mu
            row[f"{label}_ci95"] = ci
        table_rows.append(row)
        for source, column in (
            ("S", "task_score::S"),
            ("swap S", "task_score::swap_S"),
            ("delete S", "task_score::delete_S"),
            ("no memory", "task_score::no_memory"),
        ):
            mu, ci = mean_ci(group[column].tolist())
            plot_rows.append(
                {
                    "benchmark": benchmark,
                    "backend": backend,
                    "source": source,
                    "task_score_mean": mu,
                    "task_score_ci95": ci,
                    "N": len(group),
                }
            )
        for source in TOKEN_SOURCE_ORDER:
            gains = group[f"task_score::{source}"] - group["task_score::no_memory"]
            mean_gain, gain_ci = mean_ci(gains.tolist())
            mean_tokens = float(group[f"state_token_count::{source}"].mean())
            token_rows.append(
                {
                    "benchmark": benchmark,
                    "backend": backend,
                    "source": source,
                    "mean_state_tokens": mean_tokens,
                    "mean_task_score_gain": mean_gain,
                    "task_score_gain_ci95": gain_ci,
                    "score_gain_per_1k_tokens": mean_gain / max(mean_tokens / 1000.0, 1e-9),
                    "N": len(group),
                }
            )
    return pivot, pd.DataFrame(table_rows), pd.DataFrame(plot_rows), pd.DataFrame(token_rows)


def write_exp1_table(table: pd.DataFrame, out_dir: Path, intermediate: Path) -> None:
    table.to_csv(intermediate / "table1_personalization_numeric.csv", index=False)
    benchmark_rank = {value: index for index, value in enumerate(BENCHMARK_ORDER)}
    backend_rank = {value: index for index, value in enumerate(BACKEND_ORDER)}
    table = table.sort_values(
        by=["benchmark", "backend"],
        key=lambda column: column.map(benchmark_rank if column.name == "benchmark" else backend_rank),
    )
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{llrccccccc}",
        "\\toprule",
        "Benchmark & Backend & N & Score($S$) & No memory & Delete $S$ & Swap $S$ & Gain & Delete drop & Swap change "
        + "\\\\",
        "\\midrule",
    ]
    previous = None
    for _, row in table.iterrows():
        if previous is not None and row["benchmark"] != previous:
            lines.append("\\addlinespace")
        previous = row["benchmark"]
        cells = [
            tex_escape(row["benchmark"]),
            tex_escape(row["backend"]),
            str(int(row["N"])),
            fmt_ci(row["TaskScore(S)_mean"], row["TaskScore(S)_ci95"]),
            fmt_ci(row["TaskScore(no memory)_mean"], row["TaskScore(no memory)_ci95"]),
            fmt_ci(row["TaskScore(delete S)_mean"], row["TaskScore(delete S)_ci95"]),
            fmt_ci(row["TaskScore(swap S)_mean"], row["TaskScore(swap S)_ci95"]),
            "\\textbf{" + fmt_ci(row["Task Score Gain_mean"], row["Task Score Gain_ci95"]) + "}",
            "\\textbf{" + fmt_ci(row["Delete Drop_mean"], row["Delete Drop_ci95"]) + "}",
            "\\textbf{" + fmt_ci(row["Swap Change_mean"], row["Swap Change_ci95"]) + "}",
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    write_text(out_dir / "table1_personalization.tex", "\n".join(lines))


def draw_exp1_figures(plot: pd.DataFrame, token: pd.DataFrame, out_dir: Path, intermediate: Path) -> None:
    plot.to_csv(intermediate / "figure_a1_task_score_by_state_plot_data.csv", index=False)
    token.to_csv(intermediate / "figure_a2_score_gain_per_token_plot_data.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 11.0), sharey=True)
    x = np.arange(len(FIGURE_SOURCE_ORDER))
    width = 0.24
    offsets = {"Mem0": -width, "Graphiti": 0.0, "LangMem+LangGraph": width}
    for ax, benchmark in zip(axes.flatten(), BENCHMARK_ORDER):
        subset = plot[plot["benchmark"] == benchmark]
        for backend in BACKEND_ORDER:
            rows = subset[subset["backend"] == backend].set_index("source")
            if rows.empty:
                continue
            values = [float(rows.loc[source, "task_score_mean"]) for source in FIGURE_SOURCE_ORDER]
            errors = [float(rows.loc[source, "task_score_ci95"]) for source in FIGURE_SOURCE_ORDER]
            ax.bar(
                x + offsets[backend],
                values,
                width=width * 0.92,
                color=BACKEND_COLORS[backend],
                edgecolor="#202020",
                linewidth=0.7,
                yerr=errors,
                capsize=4,
            )
        ax.set_title(benchmark)
        ax.set_xticks(x)
        ax.set_xticklabels(FIGURE_SOURCE_ORDER, rotation=16, ha="right")
        ax.set_ylim(-0.03, 1.05)
        ax.grid(axis="y", color="#d7d7d7", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Task score")
    axes[1, 0].set_ylabel("Task score")
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor=BACKEND_COLORS[backend],
            markeredgecolor=BACKEND_COLORS[backend],
            markersize=10,
            label=backend,
        )
        for backend in BACKEND_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(
        out_dir / "figure_a1_task_score_by_state.pdf",
        bbox_inches="tight",
        metadata={"Creator": "Matplotlib", "Producer": "Matplotlib", "CreationDate": None, "ModDate": None},
    )
    fig.savefig(out_dir / "figure_a1_task_score_by_state.jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 11.0), sharey=True)
    for ax, benchmark in zip(axes.flatten(), BENCHMARK_ORDER):
        subset = token[token["benchmark"] == benchmark]
        for source in TOKEN_SOURCE_ORDER:
            for _, row in subset[subset["source"] == source].iterrows():
                ax.errorbar(
                    row["mean_state_tokens"],
                    row["mean_task_score_gain"],
                    yerr=row["task_score_gain_ci95"],
                    marker=BACKEND_MARKERS[row["backend"]],
                    color=SOURCE_COLORS[source],
                    markersize=9,
                    capsize=3,
                    linestyle="",
                )
        ax.set_xscale("log")
        ax.axhline(0.0, color="#555555", linewidth=1.0)
        ax.set_title(benchmark)
        ax.grid(color="#dedede", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_xlabel("User-information tokens (log scale)")
    axes[1, 1].set_xlabel("User-information tokens (log scale)")
    axes[0, 0].set_ylabel("Task Score Gain")
    axes[1, 0].set_ylabel("Task Score Gain")
    source_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=SOURCE_COLORS[source], markersize=9, label=SOURCE_DISPLAY[source])
        for source in TOKEN_SOURCE_ORDER
    ]
    backend_handles = [
        Line2D([0], [0], marker=BACKEND_MARKERS[backend], linestyle="", color="#333333", markersize=9, label=backend)
        for backend in BACKEND_ORDER
    ]
    fig.legend(
        handles=source_handles + backend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(
        out_dir / "figure_a2_score_gain_per_token.pdf",
        bbox_inches="tight",
        metadata={"Creator": "Matplotlib", "Producer": "Matplotlib", "CreationDate": None, "ModDate": None},
    )
    fig.savefig(out_dir / "figure_a2_score_gain_per_token.jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)
