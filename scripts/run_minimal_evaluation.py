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


BACKENDS = ("Mem0", "Graphiti", "LangMem+LangGraph")
BENCHMARKS = (
    "PersonaMem-v2",
    "PersonaLens",
    "ETAPP_150x32",
    "LoCoMo_10conv_1523QA_20speakers",
)
METHODS = ("UMPeek_final",)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one compact UMPeek evaluation setting.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backend", choices=BACKENDS, default="Mem0")
    parser.add_argument("--benchmark", choices=BENCHMARKS, default="PersonaMem-v2")
    parser.add_argument("--method", choices=METHODS, default="UMPeek_final")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--budgets", default="1,2,4")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "runs/minimal_evaluation")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    budgets = tuple(int(value) for value in args.budgets.split(",") if value.strip())
    if args.limit <= 0 or not budgets:
        raise ValueError("--limit and --budgets must be positive and non-empty.")
    defaults = {
        "UMPEEK_EVAL2_REAL_AGENT_MODE": "1",
        "UMPEEK_REAL_AGENT_MODEL": "Qwen/Qwen3-14B",
        "UMPEEK_REAL_AGENT_VLLM_BASE_URL": "http://127.0.0.1:8010/v1",
        "UMPEEK_REAL_AGENT_REQUIRE_LIVE_ENDPOINT": "1",
        "UMPEEK_REAL_AGENT_ENABLE_THINKING": "0",
        "UMPEEK_REAL_AGENT_STRICT_MODEL_CHECK": "1",
        "UMPEEK_EVAL2_GENERATE_MISSING_VISIBLE": "0",
        "UMPEEK_EVAL2_DISABLE_GENERATION_CACHE": "1",
        "UMPEEK_EVAL2_LATENT_GOLD_MODE": "profile",
        "UMPEEK_EVAL2_PAPER_FACING_ONLY": "1",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
    result = run_full_setting(
        project_root=PROJECT_ROOT,
        manifest_path=args.manifest,
        backend=args.backend,
        benchmark=args.benchmark,
        methods=(args.method,),
        out_dir=args.out_dir,
        dry_run=args.dry_run,
        limit=args.limit,
        resume=True,
        force_smoke=False,
        max_retries=0,
        budget_grid_override=budgets,
        shared_budget_prefix=True,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
