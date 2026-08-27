#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data/release_samples"


def jsonl_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            forbidden = {"gold", "gold_user_model", "heldout_tasks", "private_memory", "user_id"}
            overlap = forbidden.intersection(row)
            if overlap:
                raise RuntimeError(f"{path} contains forbidden release fields: {sorted(overlap)}")
            count += 1
    return count


def csv_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> int:
    comparison_source_path = ROOT / "configs" / "comparison_sources.json"
    comparison_sources = json.loads(comparison_source_path.read_text(encoding="utf-8"))
    expected_comparisons = {"ADAM", "LLM-PBE", "PLeak", "IPI", "Imprompter", "AttrInf", "PIE"}
    if comparison_sources.get("implementations_included") is not False:
        raise RuntimeError("Comparison source manifest must state that implementations are not included.")
    if set(comparison_sources.get("methods", {})) != expected_comparisons:
        raise RuntimeError("Comparison source manifest does not list the seven reported methods.")
    for method, source in comparison_sources["methods"].items():
        if not source.get("paper_url") and not source.get("code_url"):
            raise RuntimeError(f"{method} has no official source URL.")

    forbidden_comparison_files = (
        "src/umpeek/attack_baselines/adapters/adam.py",
        "src/umpeek/attack_baselines/adapters/llm_pbe.py",
        "src/umpeek/attack_baselines/adapters/pleak.py",
        "src/umpeek/attack_baselines/adapters/ipi.py",
        "src/umpeek/attack_baselines/adapters/imprompter.py",
        "src/umpeek/attack_baselines/adapters/attrinf.py",
        "src/umpeek/attack_baselines/adapters/pie.py",
        "src/umpeek/attack_baselines/adapters/_profile_inference.py",
        "configs/comparison_adapters.json",
    )
    unexpected = [path for path in forbidden_comparison_files if (ROOT / path).exists()]
    if unexpected:
        raise RuntimeError(f"Comparison adaptations must not be distributed: {unexpected}")

    expected_jsonl = {
        "public_requests.jsonl": 8,
        "visible_outputs.jsonl": 12,
        "main_evaluation_metrics.jsonl": 96,
    }
    counts = {name: jsonl_count(SAMPLES / name) for name in expected_jsonl}
    for name, expected in expected_jsonl.items():
        if counts[name] != expected:
            raise RuntimeError(f"{name}: expected {expected} rows, found {counts[name]}")
    adaptive_count = csv_count(SAMPLES / "adaptive_defense_metrics.csv")
    exp1_count = csv_count(SAMPLES / "exp1_condition_examples.csv")
    if exp1_count != 24:
        raise RuntimeError(f"exp1_condition_examples.csv: expected 24 rows, found {exp1_count}")
    aggregate_inputs = {
        "results/exp1/intermediate/table1_personalization_numeric.csv": 12,
        "results/exp1/intermediate/figure_a1_task_score_by_state_plot_data.csv": 48,
        "results/exp1/intermediate/figure_a2_score_gain_per_token_plot_data.csv": 48,
        "results/exp3/intermediate/table2_attack_results_numeric.csv": 8,
        "results/exp3/intermediate/table2_attack_results_by_backend_benchmark.csv": 96,
        "results/exp4/intermediate/table_a4_umpeek_ablations_numeric.csv": 9,
        "results/exp4/intermediate/table_a4_umpeek_ablations_by_backend_benchmark.csv": 108,
        "results/exp4/intermediate/figure4a_evidence_added_by_step_plot_data.csv": 5,
        "results/adaptive_defense/intermediate/adaptive_defense_main_summary.csv": 96,
        "results/adaptive_defense/intermediate/adaptive_defense_tradeoff_summary.csv": 24,
        "results/adaptive_defense/intermediate/adaptive_defense_ablation_summary.csv": 3,
    }
    aggregate_counts = {name: csv_count(ROOT / name) for name in aggregate_inputs}
    bad_aggregates = {
        name: {"expected": expected, "found": aggregate_counts[name]}
        for name, expected in aggregate_inputs.items()
        if aggregate_counts[name] != expected
    }
    if bad_aggregates:
        raise RuntimeError(f"Released aggregate count mismatch: {bad_aggregates}")
    required = (
        ROOT / "results/exp1/table1_personalization.tex",
        ROOT / "results/exp1/figure_a1_task_score_by_state.pdf",
        ROOT / "results/exp1/figure_a2_score_gain_per_token.pdf",
        ROOT / "results/exp3/table2_attack_results.tex",
        ROOT / "results/exp4/table_a4_umpeek_ablations.tex",
        ROOT / "results/exp4/figure4a_evidence_added_by_step.pdf",
        ROOT / "results/adaptive_defense/table_adaptive_defense_results.tex",
        ROOT / "results/adaptive_defense/table_stateful_defense_ablations.tex",
        ROOT / "results/adaptive_defense/figure_adaptive_defense.pdf",
    )
    missing = [path.relative_to(ROOT).as_posix() for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing paper artifacts: {missing}")
    summary = {
        "status": "ok",
        "public_requests": counts["public_requests.jsonl"],
        "visible_outputs": counts["visible_outputs.jsonl"],
        "main_metric_examples": counts["main_evaluation_metrics.jsonl"],
        "adaptive_metric_examples": adaptive_count,
        "exp1_derived_record_examples": exp1_count,
        "released_aggregate_files": len(aggregate_inputs),
        "paper_artifacts": len(required),
        "comparison_official_sources": len(expected_comparisons),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
