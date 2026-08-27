#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from umpeek.eval2.runner import run_full_setting  # noqa: E402


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "runs"
    / "exp2_full_comparison"
    / "A200_real_agent_qwen3_full_current"
    / "manifest"
    / "full_matrix_manifest.json"
)
BACKENDS = ("Mem0", "Graphiti", "LangMem+LangGraph")
BENCHMARKS = (
    "PersonaMem-v2",
    "PersonaLens",
    "ETAPP_150x32",
    "LoCoMo_10conv_1523QA_20speakers",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the external adaptive defenses against frozen UMPeek.")
    parser.add_argument(
        "--defense",
        required=True,
        choices=("privacy_checker", "theory_of_mind", "stateful_counterfactual"),
    )
    parser.add_argument("--backend", action="append", choices=BACKENDS)
    parser.add_argument("--benchmark", action="append", choices=BENCHMARKS)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    os.environ["UMPEEK_EVAL2_DEFENSE"] = args.defense
    os.environ.setdefault("UMPEEK_EVAL2_REAL_AGENT_MODE", "1")
    os.environ.setdefault("UMPEEK_REAL_AGENT_ENABLE_THINKING", "0")
    os.environ.setdefault("UMPEEK_REAL_AGENT_MODEL", "Qwen/Qwen3-14B")
    out_root = args.out_root or (
        PROJECT_ROOT / "runs" / "adaptive_defense" / f"quick_{args.defense}_current"
    )
    rows: list[dict[str, object]] = []
    for benchmark in args.benchmark or BENCHMARKS:
        for backend in args.backend or BACKENDS:
            setting = "__".join(
                part.lower().replace("+", "_").replace("-", "_")
                for part in (backend, benchmark)
            )
            result = run_full_setting(
                project_root=PROJECT_ROOT,
                manifest_path=args.manifest,
                backend=backend,
                benchmark=benchmark,
                methods=("UMPeek_final",),
                out_dir=out_root / "settings" / setting,
                dry_run=args.dry_run,
                limit=args.limit,
                resume=not args.no_resume,
                force_smoke=False,
                max_retries=0,
            )
            rows.append(
                {
                    "defense": args.defense,
                    "backend": backend,
                    "benchmark": benchmark,
                    "dry_run": bool(args.dry_run),
                    "record_count": result.get("record_count", result.get("planned_total_records", 0)),
                    "status_counts": result.get("status_counts", {}),
                }
            )
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "launcher_status.json").write_text(
        json.dumps({"status": "complete", "settings": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
