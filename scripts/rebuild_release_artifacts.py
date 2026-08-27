#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import export_adaptive_defense_artifacts as defense  # noqa: E402
import export_exp3_table2 as exp3  # noqa: E402
import export_exp4_paper_artifacts as exp4  # noqa: E402
from _exp1_artifacts import (  # noqa: E402
    clean_output,
    draw_exp1_figures,
    setup_style,
    write_exp1_table,
)


DEFAULT_INPUT_ROOT = PROJECT_ROOT / "results"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "build" / "released_artifacts"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild paper tables and figures from the aggregate CSV files included in the artifact."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise RuntimeError(f"Released aggregate is empty: {path}")
    return [_typed_row(row) for row in rows]


def _typed_row(row: Mapping[str, str]) -> dict[str, Any]:
    typed: dict[str, Any] = {}
    integer_fields = {
        "N",
        "N_records",
        "N_clusters",
        "backend_count",
        "budget",
        "step_order",
        "umr_f1_n",
        "hbps_n",
        "task_score_n",
    }
    for key, value in row.items():
        text = str(value).strip()
        if text == "":
            typed[key] = None
            continue
        if key in integer_fields or key.endswith("_n"):
            try:
                typed[key] = int(float(text))
                continue
            except ValueError:
                pass
        try:
            typed[key] = float(text)
        except ValueError:
            typed[key] = text
    return typed


def _copy_caption(input_dir: Path, output_dir: Path) -> None:
    source = input_dir / "captions.tex"
    if source.is_file():
        shutil.copyfile(source, output_dir / "captions.tex")


def _rebuild_exp1(input_root: Path, output_root: Path) -> list[Path]:
    source = input_root / "exp1"
    out = output_root / "exp1"
    intermediate = clean_output(out)
    setup_style()
    table = pd.read_csv(source / "intermediate/table1_personalization_numeric.csv")
    plot = pd.read_csv(source / "intermediate/figure_a1_task_score_by_state_plot_data.csv")
    token = pd.read_csv(source / "intermediate/figure_a2_score_gain_per_token_plot_data.csv")
    write_exp1_table(table, out, intermediate)
    draw_exp1_figures(plot, token, out, intermediate)
    _copy_caption(source, out)
    return [
        out / "table1_personalization.tex",
        out / "figure_a1_task_score_by_state.pdf",
        out / "figure_a1_task_score_by_state.jpg",
        out / "figure_a2_score_gain_per_token.pdf",
        out / "figure_a2_score_gain_per_token.jpg",
    ]


def _rebuild_exp3(input_root: Path, output_root: Path) -> list[Path]:
    source = input_root / "exp3/intermediate/table2_attack_results_numeric.csv"
    out = output_root / "exp3"
    out.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(source)
    display_rows: list[dict[str, Any]] = []
    for row in rows:
        display: dict[str, Any] = {"Method": row["method"], "N": row["N"]}
        for _label, key in exp3.TABLE_METRICS:
            display[f"{key}_text"] = exp3.format_mean_ci(
                exp3.parse_float(row.get(key)),
                exp3.parse_float(row.get(f"{key}_ci_low")),
                exp3.parse_float(row.get(f"{key}_ci_high")),
            )
        display_rows.append(display)
    ranks: dict[str, tuple[str | None, str | None]] = {}
    for _label, key in exp3.TABLE_METRICS:
        candidates = [
            (str(row["method"]), exp3.parse_float(row.get(key)))
            for row in rows
            if exp3.parse_float(row.get(key)) is not None
        ]
        candidates.sort(key=lambda item: float(item[1]), reverse=(key != "queries"))
        ranks[key] = (
            candidates[0][0] if candidates else None,
            candidates[1][0] if len(candidates) > 1 else None,
        )
    target = out / "table2_attack_results.tex"
    target.write_text(exp3.build_latex(display_rows, ranks), encoding="utf-8")
    return [target]


def _rebuild_exp4(input_root: Path, output_root: Path) -> list[Path]:
    source = input_root / "exp4"
    out = output_root / "exp4"
    out.mkdir(parents=True, exist_ok=True)
    numeric = _read_csv(source / "intermediate/table_a4_umpeek_ablations_numeric.csv")
    figure = _read_csv(source / "intermediate/figure4a_evidence_added_by_step_plot_data.csv")
    display = exp4.build_display_rows(numeric)
    (out / "table_a4_umpeek_ablations.tex").write_text(
        exp4.build_latex_table(display), encoding="utf-8"
    )
    exp4.write_figure(figure, out)
    _copy_caption(source, out)
    return [
        out / "table_a4_umpeek_ablations.tex",
        out / "figure4a_evidence_added_by_step.pdf",
        out / "figure4a_evidence_added_by_step.jpg",
    ]


def _rebuild_defense(input_root: Path, output_root: Path) -> list[Path]:
    source = input_root / "adaptive_defense"
    out = output_root / "adaptive_defense"
    out.mkdir(parents=True, exist_ok=True)
    main = _read_csv(source / "intermediate/adaptive_defense_main_summary.csv")
    tradeoff = _read_csv(source / "intermediate/adaptive_defense_tradeoff_summary.csv")
    ablation = _read_csv(source / "intermediate/adaptive_defense_ablation_summary.csv")
    final_budget = max(int(row["budget"]) for row in main)
    (out / "table_adaptive_defense_results.tex").write_text(
        defense._main_table_latex(main, final_budget), encoding="utf-8"
    )
    (out / "table_stateful_defense_ablations.tex").write_text(
        defense._ablation_table_latex(ablation), encoding="utf-8"
    )
    defense._plot(main, tradeoff, out, final_budget)
    _copy_caption(source, out)
    return [
        out / "table_adaptive_defense_results.tex",
        out / "table_stateful_defense_ablations.tex",
        out / "figure_adaptive_defense.pdf",
        out / "figure_adaptive_defense.jpg",
    ]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = arguments()
    input_root = _resolve(args.input_root)
    output_root = _resolve(args.output_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    outputs = [
        *_rebuild_exp1(input_root, output_root),
        *_rebuild_exp3(input_root, output_root),
        *_rebuild_exp4(input_root, output_root),
        *_rebuild_defense(input_root, output_root),
    ]
    missing = [str(path) for path in outputs if not path.is_file() or path.stat().st_size == 0]
    audit = {
        "status": "ok" if not missing else "failed",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "outputs": {
            str(path.relative_to(output_root)): {"bytes": path.stat().st_size, "sha256": _digest(path)}
            for path in outputs
            if path.is_file()
        },
        "missing_outputs": missing,
        "input_scope": "released aggregate CSV files; no private rows or model calls",
    }
    (output_root / "rebuild_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    if missing:
        raise RuntimeError(f"Artifact rebuild is incomplete: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
