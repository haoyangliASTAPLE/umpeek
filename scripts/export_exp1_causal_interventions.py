#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from _exp1_artifacts import (
    clean_output,
    draw_exp1_figures,
    exp1_data,
    setup_style,
    write_exp1_table,
    write_text,
)
from umpeek.exp12_true_interventions import BENCHMARK_ORDER, BACKEND_ORDER, SCHEMA_VERSION, read_jsonl, stable_hash


DEFAULT_RUN_ROOT = PROJECT_ROOT / "runs/exp1_exp2_current_completion/A204_exp1_causal_repair"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "my/paper-facing/exp1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export repaired Experiment 1 artifacts only.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def validate(run_root: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Experiment 1 schema mismatch")
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Experiment 1 run is not complete: {manifest.get('status')}")
    if int(manifest.get("expected_atom_intervention_records", -1)) != 0:
        raise RuntimeError("The Exp1-only exporter requires --skip-atom-interventions")
    raw = read_jsonl(run_root / "records/condition_records.jsonl")
    for original in raw:
        if original.get("status") != "ok":
            continue
        record = dict(original)
        expected_hash = str(record.pop("integrity_hash", ""))
        if not expected_hash or stable_hash(record) != expected_hash:
            raise RuntimeError(f"condition record integrity failed: {record.get('record_key')}")
    condition = pd.DataFrame(raw)
    condition = condition[condition["status"] == "ok"].drop_duplicates("record_key", keep="last")
    if len(condition) != int(manifest["expected_condition_records"]):
        raise RuntimeError(f"condition count mismatch: {len(condition)} != {manifest['expected_condition_records']}")
    expected_conditions = {"S", "no_memory", "delete_S", "swap_S", "M_u", "H_rel", "H_sum"}
    if set(condition["condition"]) != expected_conditions:
        raise RuntimeError(f"condition coverage mismatch: {sorted(condition['condition'].unique())}")
    if not condition["call_executed"].astype(bool).all():
        raise RuntimeError("all seven conditions must be live Qwen calls")
    if condition["decision_signature_hash"].isna().any():
        raise RuntimeError("decision signatures are required for swap-change rate")
    forbidden = ("proxy", "category_prior", "soft_behavior_support", "A200_real_agent_target_behavior")
    public_text = json.dumps(
        condition.drop(columns=[column for column in ("private_behavior", "private_decision_signature") if column in condition]).to_dict(orient="records"),
        ensure_ascii=False,
    )
    hits = [term for term in forbidden if term in public_text]
    if hits:
        raise RuntimeError(f"stale or proxy labels found: {hits}")
    return manifest, condition


def directional_audit(pivot: pd.DataFrame) -> dict[str, Any]:
    by_setting: list[dict[str, Any]] = []
    for (benchmark, backend), group in pivot.groupby(["benchmark", "backend"], sort=False):
        gain = float((group["task_score::S"] - group["task_score::no_memory"]).mean())
        delete_drop = float((group["task_score::S"] - group["task_score::delete_S"]).mean())
        swap_change = float(group["Swap Change"].mean())
        by_setting.append(
            {
                "benchmark": benchmark,
                "backend": backend,
                "task_score_gain": gain,
                "delete_drop": delete_drop,
                "swap_behavior_change_rate": swap_change,
                "expected_direction": bool(gain > 0 and delete_drop > 0 and swap_change > 0),
            }
        )
    return {
        "settings": by_setting,
        "settings_in_expected_direction": sum(int(row["expected_direction"]) for row in by_setting),
        "setting_count": len(by_setting),
        "note": "Direction is audited, not enforced; the exporter never alters measured scores.",
    }


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    out_root = args.out_root.resolve()
    manifest, condition = validate(run_root)
    font = setup_style()
    intermediate = clean_output(out_root)
    pivot, table, plot, token = exp1_data(condition)
    pivot.to_csv(intermediate / "condition_records_wide.csv", index=False)
    write_exp1_table(table, out_root, intermediate)
    draw_exp1_figures(plot, token, out_root, intermediate)
    audit: dict[str, Any] = {
        "status": "pass",
        "schema_version": SCHEMA_VERSION,
        "source_run": run_root.relative_to(PROJECT_ROOT).as_posix(),
        "condition_records": len(condition),
        "base_samples": int(condition[condition["condition"] == "S"]["sample_id"].nunique()),
        "benchmarks": list(BENCHMARK_ORDER),
        "backends": list(BACKEND_ORDER),
        "all_conditions_live_qwen": True,
        "proxy_conditions_present": False,
        "personamem_options_shuffled": True,
        "swap_metric": "paired_task_decision_change_rate",
        "font_resolved": font,
        "font_requirement": "Times New Roman",
        "directional_audit": directional_audit(pivot),
        "run_state_relevance_audit": manifest.get("state_relevance_audit", {}),
    }
    write_text(intermediate / "artifact_audit.json", json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    write_text(
        out_root / "captions.tex",
        "\\newcommand{\\ExpOneTableOneCaption}{Personalization under the retrieved, swapped, removed, and absent runtime user-state conditions. Every condition uses an independent Qwen3-14B call on the same task.}\n"
        "\\newcommand{\\ExpOneFigureAOneCaption}{Task score under paired runtime-state interventions. Bars show means and whiskers show 95\\% confidence intervals.}\n"
        "\\newcommand{\\ExpOneFigureATwoCaption}{Task Score Gain over no memory against the number of injected user-information tokens.}\n",
    )
    write_text(
        out_root / "README.md",
        "# Experiment 1 Paper-Facing Artifacts\n\n"
        f"Source: `{run_root.relative_to(PROJECT_ROOT)}`.\n\n"
        "All seven conditions are independent Qwen3-14B non-thinking calls. PersonaMem-v2 option order is deterministic and shuffled. Swap Change is a paired task-decision change rate, not a score difference.\n\n"
        "Export code: `scripts/export_exp1_causal_interventions.py`. Run code: `scripts/run_exp1_interventions.py`.\n",
    )
    write_text(run_root / "artifact_audit_exp1.json", json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
