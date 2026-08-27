from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "runs" / "exp4_umpeek_evidence" / "A001_current_mechanism_subset"
DEFAULT_OUT_DIR = PROJECT_ROOT / "my" / "paper-facing" / "exp4"

VARIANT_LABELS = {
    "full_umpeek": "Full UMPeek",
    "no_candidate_fields": "No candidate fields",
    "no_followup": "No follow-up",
    "no_visible_tool_evidence": "No visible tool evidence",
    "no_public_schema_tool_info": "No public schema/tool information",
    "no_strict_acceptance": "No strict acceptance",
    "random_followup": "Random follow-up",
    "output_only": "Output-only extraction",
    "schema_only": "Initial response plus public schema/tool information",
}
TABLE_VARIANTS = (
    "full_umpeek",
    "no_candidate_fields",
    "no_followup",
    "no_visible_tool_evidence",
    "no_public_schema_tool_info",
    "no_strict_acceptance",
    "random_followup",
    "output_only",
)
FIGURE_STEPS = (
    ("output_only", "initial response only"),
    ("schema_only", "+ public schema/tool information"),
    ("no_followup", "+ visible tool arguments"),
    ("no_strict_acceptance", "+ bounded follow-up behavior"),
    ("full_umpeek", "+ strict acceptance"),
)
BACKEND_ORDER = ("Mem0", "Graphiti", "LangMem+LangGraph")
BENCHMARK_ORDER = (
    "PersonaMem-v2",
    "PersonaLens",
    "ETAPP_150x32",
    "LoCoMo_10conv_1523QA_20speakers",
)
LEGACY_INTERMEDIATE_FILES = (
    "artifact_audit.json",
    "figure4a_evidence_added_by_step_plot_data.csv",
    "table_a4_umpeek_ablations_by_backend_benchmark.csv",
    "table_a4_umpeek_ablations_display.csv",
    "table_a4_umpeek_ablations_numeric.csv",
)


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(dict(json.loads(line)))
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def metric_value(record: Mapping[str, Any], metric: str) -> float | None:
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    if metric == "accepted_precision":
        value = parse_float(record.get("accepted_fact_precision"))
        if value is not None:
            return value
        umr = metrics.get("UMR-F1")
        return parse_float(umr.get("umr_precision") if isinstance(umr, Mapping) else None)
    if metric == "used_f1":
        cw = metrics.get("Causal-Weighted UMR-F1")
        return parse_float(cw.get("cw_umr_f1") if isinstance(cw, Mapping) else None)
    if metric == "dsg":
        dsg = metrics.get("DSG")
        return parse_float(dsg.get("dsg") if isinstance(dsg, Mapping) else None)
    if metric == "queries":
        return parse_float(record.get("actual_query_count"))
    return None


def mean_ci(values: Sequence[float | None]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return {"mean": None, "ci_low": None, "ci_high": None, "se": None, "n": 0}
    n = len(clean)
    mean = sum(clean) / n
    if n == 1:
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "se": 0.0, "n": n}
    var = sum((value - mean) ** 2 for value in clean) / (n - 1)
    se = math.sqrt(var / n)
    delta = 1.96 * se
    return {
        "mean": mean,
        "ci_low": max(0.0, mean - delta),
        "ci_high": min(1.0, mean + delta),
        "se": se,
        "n": n,
    }


def format_mean(value: Any, decimals: int = 3) -> str:
    parsed = parse_float(value)
    return "n/a" if parsed is None else f"{parsed:.{decimals}f}"


def format_ci(row: Mapping[str, Any], key: str, decimals: int = 3) -> str:
    mean = parse_float(row.get(key))
    if mean is None:
        return "n/a"
    lo = parse_float(row.get(f"{key}_ci_low"))
    hi = parse_float(row.get(f"{key}_ci_high"))
    if lo is None or hi is None:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
    )


def load_records(run_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    settings_root = run_root / "settings"
    if not settings_root.exists():
        return [], [str(settings_root)]
    for setting_dir in sorted(path for path in settings_root.iterdir() if path.is_dir()):
        path = setting_dir / "metric_records.jsonl"
        if path.exists():
            records.extend(read_jsonl(path))
        else:
            missing.append(str(path))
    return records, missing


def aggregate(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_variant: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_grid: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("run_status") != "ok":
            continue
        variant = str(record.get("exp4_variant") or "")
        backend = str(record.get("backend") or "")
        benchmark = str(record.get("benchmark") or "")
        by_variant[variant].append(record)
        by_grid[(variant, backend, benchmark)].append(record)

    numeric_rows: list[dict[str, Any]] = []
    for variant, rows in sorted(by_variant.items()):
        row: dict[str, Any] = {
            "variant": variant,
            "label": VARIANT_LABELS.get(variant, variant),
            "N": len(rows),
        }
        for metric, out_key in (
            ("accepted_precision", "accepted_fact_precision"),
            ("used_f1", "umr_f1_on_used_facts"),
            ("dsg", "dsg"),
            ("queries", "avg_queries"),
        ):
            stats = mean_ci([metric_value(record, metric) for record in rows])
            row[out_key] = stats["mean"]
            row[f"{out_key}_ci_low"] = stats["ci_low"]
            row[f"{out_key}_ci_high"] = stats["ci_high"]
            row[f"{out_key}_se"] = stats["se"]
            row[f"{out_key}_n"] = stats["n"]
        numeric_rows.append(row)

    grid_rows: list[dict[str, Any]] = []
    for variant in sorted(by_variant):
        for benchmark in BENCHMARK_ORDER:
            for backend in BACKEND_ORDER:
                rows = by_grid.get((variant, backend, benchmark), [])
                row = {
                    "variant": variant,
                    "label": VARIANT_LABELS.get(variant, variant),
                    "benchmark": benchmark,
                    "backend": backend,
                    "N": len(rows),
                }
                for metric, out_key in (
                    ("accepted_precision", "accepted_fact_precision"),
                    ("used_f1", "umr_f1_on_used_facts"),
                    ("dsg", "dsg"),
                    ("queries", "avg_queries"),
                ):
                    stats = mean_ci([metric_value(record, metric) for record in rows])
                    row[out_key] = stats["mean"]
                    row[f"{out_key}_ci_low"] = stats["ci_low"]
                    row[f"{out_key}_ci_high"] = stats["ci_high"]
                    row[f"{out_key}_n"] = stats["n"]
                grid_rows.append(row)
    return numeric_rows, grid_rows


def build_figure_data(numeric_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_variant = {str(row.get("variant")): row for row in numeric_rows}
    plot_rows: list[dict[str, Any]] = []
    for order, (variant, step) in enumerate(FIGURE_STEPS, start=1):
        row = by_variant.get(variant, {})
        plot_rows.append(
            {
                "step_order": order,
                "variant": variant,
                "step": step,
                "umr_f1_on_used_facts": row.get("umr_f1_on_used_facts"),
                "umr_f1_on_used_facts_ci_low": row.get("umr_f1_on_used_facts_ci_low"),
                "umr_f1_on_used_facts_ci_high": row.get("umr_f1_on_used_facts_ci_high"),
                "accepted_fact_precision": row.get("accepted_fact_precision"),
                "accepted_fact_precision_ci_low": row.get("accepted_fact_precision_ci_low"),
                "accepted_fact_precision_ci_high": row.get("accepted_fact_precision_ci_high"),
                "N": row.get("N", 0),
            }
        )
    return plot_rows


def write_figure(plot_rows: Sequence[Mapping[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
            "font.size": 17,
            "axes.labelsize": 19,
            "axes.titlesize": 19,
            "xtick.labelsize": 15,
            "ytick.labelsize": 16,
            "legend.fontsize": 15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    x = [int(row["step_order"]) for row in plot_rows]
    labels = [str(row["step"]) for row in plot_rows]

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.8), sharex=True, constrained_layout=True)
    specs = [
        (axes[0], "umr_f1_on_used_facts", "UMR-F1 on Used Facts", "#1f77b4"),
        (axes[1], "accepted_fact_precision", "Accepted Fact Precision", "#2ca02c"),
    ]
    for ax, key, ylabel, color in specs:
        values = [parse_float(row.get(key)) for row in plot_rows]
        lows = [parse_float(row.get(f"{key}_ci_low")) for row in plot_rows]
        highs = [parse_float(row.get(f"{key}_ci_high")) for row in plot_rows]
        y = [0.0 if value is None else value for value in values]
        yerr_low = [
            0.0 if value is None or low is None else max(0.0, value - low)
            for value, low in zip(values, lows)
        ]
        yerr_high = [
            0.0 if value is None or high is None else max(0.0, high - value)
            for value, high in zip(values, highs)
        ]
        ax.errorbar(
            x,
            y,
            yerr=[yerr_low, yerr_high],
            marker="o",
            markersize=8.5,
            linewidth=2.8,
            capsize=5,
            color=color,
        )
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 1.03)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set_xlabel("Evidence available to UMPeek")
    fig.savefig(
        out_dir / "figure4a_evidence_added_by_step.pdf",
        bbox_inches="tight",
        metadata={"Creator": "Matplotlib", "Producer": "Matplotlib", "CreationDate": None, "ModDate": None},
    )
    fig.savefig(out_dir / "figure4a_evidence_added_by_step.jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_latex_table(display_rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4.2pt}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Variant & Accepted precision & Used-F1 & DSG & Avg. queries & $N$ \\\\",
        "\\midrule",
    ]
    for row in display_rows:
        lines.append(
            " & ".join(
                [
                    latex_escape(str(row["Variant"])),
                    str(row["accepted_fact_precision_text"]),
                    str(row["umr_f1_on_used_facts_text"]),
                    str(row["dsg_text"]),
                    str(row["avg_queries_text"]),
                    str(row["N"]),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def build_display_rows(numeric_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_variant = {str(row.get("variant")): row for row in numeric_rows}
    display_rows: list[dict[str, Any]] = []
    for variant in TABLE_VARIANTS:
        source = by_variant.get(variant, {"variant": variant, "label": VARIANT_LABELS.get(variant, variant), "N": 0})
        display_rows.append(
            {
                "Variant": VARIANT_LABELS.get(variant, str(source.get("label") or variant)),
                "N": int(source.get("N") or 0),
                "accepted_fact_precision_text": format_ci(source, "accepted_fact_precision"),
                "umr_f1_on_used_facts_text": format_ci(source, "umr_f1_on_used_facts"),
                "dsg_text": format_ci(source, "dsg"),
                "avg_queries_text": format_mean(source.get("avg_queries"), decimals=2),
            }
        )
    return display_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Experiment 4 paper-facing artifacts.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    run_root = args.run_root if args.run_root.is_absolute() else PROJECT_ROOT / args.run_root
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    intermediate_dir = out_dir / "intermediate"
    out_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    for name in LEGACY_INTERMEDIATE_FILES:
        (out_dir / name).unlink(missing_ok=True)

    records, missing_files = load_records(run_root)
    numeric_rows, grid_rows = aggregate(records)
    figure_rows = build_figure_data(numeric_rows)
    display_rows = build_display_rows(numeric_rows)

    numeric_fields = ["variant", "label", "N"]
    for key in ("accepted_fact_precision", "umr_f1_on_used_facts", "dsg", "avg_queries"):
        numeric_fields.extend([key, f"{key}_ci_low", f"{key}_ci_high", f"{key}_se", f"{key}_n"])
    write_csv(intermediate_dir / "table_a4_umpeek_ablations_numeric.csv", numeric_rows, numeric_fields)
    write_csv(
        intermediate_dir / "table_a4_umpeek_ablations_display.csv",
        display_rows,
        [
            "Variant",
            "accepted_fact_precision_text",
            "umr_f1_on_used_facts_text",
            "dsg_text",
            "avg_queries_text",
            "N",
        ],
    )
    grid_fields = ["variant", "label", "benchmark", "backend", "N"]
    for key in ("accepted_fact_precision", "umr_f1_on_used_facts", "dsg", "avg_queries"):
        grid_fields.extend([key, f"{key}_ci_low", f"{key}_ci_high", f"{key}_n"])
    write_csv(
        intermediate_dir / "table_a4_umpeek_ablations_by_backend_benchmark.csv",
        grid_rows,
        grid_fields,
    )
    write_csv(
        intermediate_dir / "figure4a_evidence_added_by_step_plot_data.csv",
        figure_rows,
        [
            "step_order",
            "variant",
            "step",
            "umr_f1_on_used_facts",
            "umr_f1_on_used_facts_ci_low",
            "umr_f1_on_used_facts_ci_high",
            "accepted_fact_precision",
            "accepted_fact_precision_ci_low",
            "accepted_fact_precision_ci_high",
            "N",
        ],
    )
    write_figure(figure_rows, out_dir)
    (out_dir / "table_a4_umpeek_ablations.tex").write_text(build_latex_table(display_rows), encoding="utf-8")
    (out_dir / "captions.tex").write_text(
        "\n".join(
            [
                "% Figure 4a",
                "\\newcommand{\\ExpFourFigureCaption}{Evidence added by each UMPeek step. The upper panel reports UMR-F1 on Used Facts and the lower panel reports accepted fact precision. Error bars show 95\\% confidence intervals over the Exp4 mechanism subset.}",
                "",
                "% Table A4",
                "\\newcommand{\\ExpFourTableCaption}{UMPeek ablations on the current real-agent mechanism subset. Each row removes or changes one evidence source while keeping the benchmark split, backend setting, and metrics fixed.}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    audit = {
        "source_run_root": str(run_root.relative_to(PROJECT_ROOT)) if run_root.is_relative_to(PROJECT_ROOT) else str(run_root),
        "output_dir": str(out_dir.relative_to(PROJECT_ROOT)) if out_dir.is_relative_to(PROJECT_ROOT) else str(out_dir),
        "intermediate_dir": (
            str(intermediate_dir.relative_to(PROJECT_ROOT))
            if intermediate_dir.is_relative_to(PROJECT_ROOT)
            else str(intermediate_dir)
        ),
        "record_count": len(records),
        "missing_metric_record_files": missing_files,
        "observed_variants": sorted({str(row.get("variant")) for row in numeric_rows}),
        "missing_table_variants": [variant for variant in TABLE_VARIANTS if variant not in {str(row.get("variant")) for row in numeric_rows}],
        "missing_figure_steps": [variant for variant, _step in FIGURE_STEPS if variant not in {str(row.get("variant")) for row in numeric_rows}],
        "figure_has_embedded_caption": False,
        "raw_prompt_or_private_row_copied": False,
        "font_policy": "Times-compatible serif, large labels, embedded as PDF Type 42 where supported.",
    }
    (intermediate_dir / "artifact_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Experiment 4 Paper-Facing Artifacts",
                "",
                f"Source run: `{audit['source_run_root']}`.",
                "",
                "Final artifacts:",
                "- `figure4a_evidence_added_by_step.pdf` and `.jpg`: Figure 4a without embedded caption text.",
                "- `table_a4_umpeek_ablations.tex`: LaTeX table.",
                "- `captions.tex`: captions kept outside the figure files.",
                "",
                "Intermediate data and audits are stored in `intermediate/`.",
                "",
                "The export intentionally excludes raw prompts, private benchmark rows, recovered examples, and attack trajectories.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
