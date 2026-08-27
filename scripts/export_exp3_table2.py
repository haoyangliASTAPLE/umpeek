from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "runs" / "exp2_full_comparison" / "A200_frozen_qwen3_paper_subset_current"
DEFAULT_OUT_DIR = PROJECT_ROOT / "my" / "paper-facing" / "exp3"
LEGACY_INTERMEDIATE_FILES = (
    "table2_attack_results_numeric.csv",
    "table2_attack_results_display.csv",
    "table2_attack_results_by_backend_benchmark.csv",
    "table2_completeness_audit.json",
)

METHOD_ORDER = ["UMPeek", "ADAM", "LLM-PBE", "PLeak", "IPI", "Imprompter", "AttrInf", "PIE"]
METHOD_LABELS = {
    "UMPeek_final": "UMPeek",
    "umpeek_final": "UMPeek",
    "ADAM": "ADAM",
    "adam": "ADAM",
    "LLM-PBE": "LLM-PBE",
    "llm_pbe": "LLM-PBE",
    "PLeak": "PLeak",
    "pleak": "PLeak",
    "IPI": "IPI",
    "ipi": "IPI",
    "Imprompter": "Imprompter",
    "imprompter": "Imprompter",
    "AttrInf": "AttrInf",
    "attrinf": "AttrInf",
    "PIE": "PIE",
    "pie": "PIE",
}
BACKENDS = ["Mem0", "Graphiti", "LangMem+LangGraph"]
BENCHMARKS = ["PersonaMem-v2", "PersonaLens", "ETAPP_150x32", "LoCoMo_10conv_1523QA_20speakers"]

SUMMARY_METRICS = [
    ("UMR-F1", "umr_f1_mean", "umr_f1_se", "umr_f1"),
    ("UMR-F1 on Used Facts", "cw_umr_f1_mean", "cw_umr_f1_se", "used_f1"),
    ("HBPS", "hbps_mean", "hbps_se", "hbps"),
    ("DSG", "dsg_mean", "dsg_se", "dsg"),
    ("ASR@tau", "asr_mean", "", "asr"),
]
TABLE_METRICS = [
    ("UMR-F1", "umr_f1"),
    ("UMR-F1 on Used Facts", "used_f1"),
    ("HBPS", "hbps"),
    ("DSG", "dsg"),
    ("ASR@tau", "asr"),
    ("Average queries", "queries"),
]

QUERY_RE = re.compile(rb'"actual_query_count"\s*:\s*([0-9]+(?:\.[0-9]+)?)')


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def parse_int(value: Any) -> int:
    x = parse_float(value)
    return int(x) if x is not None else 0


def mean_se_from_sum(count: int, total: float, sumsq: float) -> tuple[float | None, float | None]:
    if count <= 0:
        return None, None
    mean = total / count
    if count == 1:
        return mean, 0.0
    variance = max(0.0, (sumsq - (total * total / count)) / (count - 1))
    return mean, math.sqrt(variance / count)


def scan_query_stats(path: Path, limit: int | None = None) -> dict[str, float | int | None]:
    count = 0
    total = 0.0
    sumsq = 0.0
    tail = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            data = tail + chunk
            for match in QUERY_RE.finditer(data):
                value = float(match.group(1))
                count += 1
                total += value
                sumsq += value * value
                if limit is not None and count >= limit:
                    avg, se = mean_se_from_sum(count, total, sumsq)
                    return {"count": count, "mean": avg, "se": se, "sum": total, "sumsq": sumsq}
            tail = data[-64:]
    avg, se = mean_se_from_sum(count, total, sumsq)
    return {"count": count, "mean": avg, "se": se, "sum": total, "sumsq": sumsq}


def combine_setting_summaries(settings: list[Mapping[str, Any]], metric_key: str) -> dict[str, float | int | None]:
    pieces = []
    for setting in settings:
        mean_value = parse_float(setting.get(metric_key))
        if mean_value is None:
            continue
        n = parse_int(setting.get(f"{metric_key}_n")) or parse_int(setting.get("valid_denominator"))
        if n <= 0:
            continue
        se = parse_float(setting.get(f"{metric_key}_se")) or 0.0
        pieces.append((n, mean_value, se))
    total_n = sum(n for n, _m, _se in pieces)
    if total_n <= 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0, "se": None}
    overall = sum(n * m for n, m, _se in pieces) / total_n
    if total_n == 1:
        return {"mean": overall, "ci_low": overall, "ci_high": overall, "n": total_n, "se": 0.0}

    ss = 0.0
    for n, m, se in pieces:
        within_var = (se * se) * n
        ss += max(0, n - 1) * within_var
        ss += n * ((m - overall) ** 2)
    pooled_var = max(0.0, ss / max(total_n - 1, 1))
    overall_se = math.sqrt(pooled_var / total_n)
    delta = 1.96 * overall_se
    return {
        "mean": overall,
        "ci_low": max(0.0, overall - delta),
        "ci_high": min(1.0, overall + delta),
        "n": total_n,
        "se": overall_se,
    }


def combine_query_summaries(settings: list[Mapping[str, Any]]) -> dict[str, float | int | None]:
    count = 0
    total = 0.0
    sumsq = 0.0
    for setting in settings:
        n = parse_int(setting.get("queries_n"))
        q_sum = parse_float(setting.get("queries_sum"))
        q_sumsq = parse_float(setting.get("queries_sumsq"))
        if n <= 0 or q_sum is None or q_sumsq is None:
            continue
        count += n
        total += q_sum
        sumsq += q_sumsq
    avg, se = mean_se_from_sum(count, total, sumsq)
    if avg is None:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0, "se": None}
    delta = 1.96 * (se or 0.0)
    return {"mean": avg, "ci_low": max(0.0, avg - delta), "ci_high": avg + delta, "n": count, "se": se}


def ci_for_setting(mean_value: float | None, se_value: float | None) -> tuple[float | None, float | None]:
    if mean_value is None:
        return None, None
    se_value = se_value or 0.0
    delta = 1.96 * se_value
    return max(0.0, mean_value - delta), min(1.0, mean_value + delta)


def format_mean_ci(value: float | None, low: float | None, high: float | None, decimals: int = 3) -> str:
    if value is None:
        return "n/a"
    if low is None or high is None:
        return f"{value:.{decimals}f}"
    return f"{value:.{decimals}f} [{low:.{decimals}f}, {high:.{decimals}f}]"


def write_csv(path: Path, rows: list[Mapping[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
    )


def build_latex(display_rows: list[Mapping[str, Any]], ranks: Mapping[str, tuple[str | None, str | None]]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Method & UMR-F1 & Used-F1 & HBPS & DSG & ASR@$\\tau$ & Avg. queries \\\\",
        "\\midrule",
    ]
    for row in display_rows:
        cells = [latex_escape(str(row["Method"]))]
        for _label, key in TABLE_METRICS:
            cell = str(row[f"{key}_text"])
            best, second = ranks.get(key, (None, None))
            if row["Method"] == best:
                cell = f"\\textbf{{{cell}}}"
            elif row["Method"] == second:
                cell = f"\\underline{{{cell}}}"
            cells.append(cell)
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def read_method_summary(path: Path) -> dict[str, Any] | None:
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    row = rows[0]
    method = METHOD_LABELS.get(row.get("method", ""), row.get("method", ""))
    out: dict[str, Any] = {
        "method": method,
        "backend": row.get("backend", ""),
        "benchmark": row.get("benchmark", ""),
        "N": parse_int(row.get("valid_denominator")),
        "method_status": row.get("method_status", ""),
        "valid_metric_rate": parse_float(row.get("valid_metric_rate")),
        "source_summary": str(path.relative_to(PROJECT_ROOT)),
    }
    for _label, mean_col, se_col, key in SUMMARY_METRICS:
        out[key] = parse_float(row.get(mean_col))
        out[f"{key}_se"] = parse_float(row.get(se_col)) if se_col else None
        out[f"{key}_n"] = out["N"] if out[key] is not None else 0
    out["tokens"] = parse_float(row.get("cost_total_tokens_mean"))
    out["tokens_se"] = parse_float(row.get("cost_total_tokens_se"))
    return out


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Experiment 3 from full metric records.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    run_root = args.run_root if args.run_root.is_absolute() else PROJECT_ROOT / args.run_root
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    settings_root = run_root / "settings"
    intermediate_dir = out_dir / "intermediate"
    out_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    for name in LEGACY_INTERMEDIATE_FILES:
        (out_dir / name).unlink(missing_ok=True)
    if not settings_root.is_dir():
        raise FileNotFoundError(
            f"Missing full-run settings at {settings_root}. "
            "Use scripts/rebuild_release_artifacts.py to rebuild from released aggregate CSV files."
        )
    setting_dirs = sorted(p for p in settings_root.iterdir() if p.is_dir())

    setting_rows: list[dict[str, Any]] = []
    missing_summary_files: list[str] = []
    missing_metric_files: list[str] = []
    malformed_tail_files: list[str] = []
    query_scan_counts: dict[str, int] = {}

    for setting_dir in setting_dirs:
        summary_path = setting_dir / "method_summary.csv"
        metric_path = setting_dir / "metric_records.jsonl"
        if not summary_path.exists():
            missing_summary_files.append(str(summary_path.relative_to(PROJECT_ROOT)))
            continue
        row = read_method_summary(summary_path)
        if row is None:
            continue
        if metric_path.exists():
            q = scan_query_stats(metric_path, limit=parse_int(row.get("N")))
            row["queries"] = q["mean"]
            row["queries_se"] = q["se"]
            row["queries_n"] = q["count"]
            row["queries_sum"] = q["sum"]
            row["queries_sumsq"] = q["sumsq"]
            query_scan_counts[setting_dir.name] = int(q["count"] or 0)
        else:
            missing_metric_files.append(str(metric_path.relative_to(PROJECT_ROOT)))
            row["queries"] = None
            row["queries_se"] = None
            row["queries_n"] = 0
            row["queries_sum"] = 0.0
            row["queries_sumsq"] = 0.0
        if (metric_path.with_suffix(metric_path.suffix + ".corrupt_tail")).exists():
            malformed_tail_files.append(str(metric_path.with_suffix(metric_path.suffix + ".corrupt_tail").relative_to(PROJECT_ROOT)))
        setting_rows.append(row)

    expected = {(m, b, bm) for m in METHOD_ORDER for b in BACKENDS for bm in BENCHMARKS}
    observed = {(r["method"], r["backend"], r["benchmark"]) for r in setting_rows}
    missing_settings = sorted(expected - observed)
    unexpected_settings = sorted(observed - expected)

    by_method: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in setting_rows:
        by_method[row["method"]].append(row)

    numeric_rows: list[dict[str, Any]] = []
    display_rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        rows = by_method.get(method, [])
        out: dict[str, Any] = {"method": method, "N": sum(parse_int(r.get("N")) for r in rows)}
        disp: dict[str, Any] = {"Method": method, "N": out["N"]}
        for _label, key in TABLE_METRICS:
            stats = combine_query_summaries(rows) if key == "queries" else combine_setting_summaries(rows, key)
            out[key] = stats["mean"]
            out[f"{key}_ci_low"] = stats["ci_low"]
            out[f"{key}_ci_high"] = stats["ci_high"]
            out[f"{key}_n"] = stats["n"]
            disp[f"{key}_text"] = format_mean_ci(stats["mean"], stats["ci_low"], stats["ci_high"])
        numeric_rows.append(out)
        display_rows.append(disp)

    appendix_rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        for benchmark in BENCHMARKS:
            for backend in BACKENDS:
                source = next(
                    (
                        r
                        for r in setting_rows
                        if r["method"] == method and r["benchmark"] == benchmark and r["backend"] == backend
                    ),
                    None,
                )
                out = {"method": method, "benchmark": benchmark, "backend": backend}
                if source is None:
                    out["N"] = 0
                    for _label, key in TABLE_METRICS:
                        out[key] = None
                        out[f"{key}_ci_low"] = None
                        out[f"{key}_ci_high"] = None
                        out[f"{key}_n"] = 0
                else:
                    out["N"] = source["N"]
                    for _label, key in TABLE_METRICS:
                        val = parse_float(source.get(key))
                        se = parse_float(source.get(f"{key}_se"))
                        lo, hi = ci_for_setting(val, se)
                        out[key] = val
                        out[f"{key}_ci_low"] = lo
                        out[f"{key}_ci_high"] = hi
                        out[f"{key}_n"] = source.get(f"{key}_n") or source.get("N")
                appendix_rows.append(out)

    ranks: dict[str, tuple[str | None, str | None]] = {}
    for _label, key in TABLE_METRICS:
        candidates = [(row["method"], parse_float(row.get(key))) for row in numeric_rows]
        candidates = [(m, v) for m, v in candidates if v is not None]
        candidates.sort(key=lambda item: item[1], reverse=(key != "queries"))
        ranks[key] = (
            candidates[0][0] if candidates else None,
            candidates[1][0] if len(candidates) > 1 else None,
        )

    numeric_fields = ["method", "N"]
    display_fields = ["Method", "N"]
    appendix_fields = ["method", "benchmark", "backend", "N"]
    for _label, key in TABLE_METRICS:
        numeric_fields.extend([key, f"{key}_ci_low", f"{key}_ci_high", f"{key}_n"])
        display_fields.append(f"{key}_text")
        appendix_fields.extend([key, f"{key}_ci_low", f"{key}_ci_high", f"{key}_n"])
    write_csv(intermediate_dir / "table2_attack_results_numeric.csv", numeric_rows, numeric_fields)
    write_csv(intermediate_dir / "table2_attack_results_display.csv", display_rows, display_fields)
    write_csv(
        intermediate_dir / "table2_attack_results_by_backend_benchmark.csv",
        appendix_rows,
        appendix_fields,
    )
    (out_dir / "table2_attack_results.tex").write_text(build_latex(display_rows, ranks), encoding="utf-8")

    audit = {
        "source_run_root": str(run_root.relative_to(PROJECT_ROOT)) if run_root.is_relative_to(PROJECT_ROOT) else str(run_root),
        "source_settings_root": str(settings_root.relative_to(PROJECT_ROOT)) if settings_root.is_relative_to(PROJECT_ROOT) else str(settings_root),
        "output_dir": str(out_dir.relative_to(PROJECT_ROOT)) if out_dir.is_relative_to(PROJECT_ROOT) else str(out_dir),
        "intermediate_dir": str(intermediate_dir.relative_to(PROJECT_ROOT)) if intermediate_dir.is_relative_to(PROJECT_ROOT) else str(intermediate_dir),
        "expected_settings": len(expected),
        "observed_settings": len(observed),
        "total_setting_summaries": len(setting_rows),
        "total_metric_records_counted_for_queries": sum(query_scan_counts.values()),
        "missing_settings": ["|".join(item) for item in missing_settings],
        "unexpected_settings": ["|".join(item) for item in unexpected_settings],
        "missing_summary_files": missing_summary_files,
        "missing_metric_files": missing_metric_files,
        "corrupt_tail_files_seen": malformed_tail_files,
        "query_scan_counts": query_scan_counts,
        "ci_method": (
            "Metric CIs combine per-setting means and SEs from method_summary.csv. "
            "Average-query CIs use direct scans of actual_query_count in metric_records.jsonl. "
            "Raw-record bootstrap was intentionally avoided because the JSONL files contain large nested audit payloads."
        ),
        "known_issue_avoided": (
            "The existing A063 aggregate deduplicates runs by backend and benchmark, so it keeps only one method per setting. "
            "This exporter groups by method, backend, and benchmark."
        ),
    }
    (intermediate_dir / "table2_completeness_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    notes = [
        "# Experiment 3 Table 2 Export",
        "",
        f"Source: `{settings_root}`.",
        "",
        "Final artifact:",
        "- `table2_attack_results.tex`: LaTeX table with best and second-best marking.",
        "",
        "Intermediate data and audits are stored in `intermediate/`.",
        "",
        "Figure 3 was not exported because this pass requested Table 2 only.",
    ]
    (out_dir / "README.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
