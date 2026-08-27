#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from umpeek.exp1.etapp_expanded import build_expanded_etapp_artifacts  # noqa: E402
from umpeek.exp1.locomo import build_locomo_task_rows, resolve_locomo_data_path  # noqa: E402
from umpeek.exp1_whitebox.personamem_v2_dataset import (  # noqa: E402
    SamplingConfig,
    materialize_personamemv2_dataset,
)
from umpeek.exp1_whitebox.personalens_subset import build_personalens_whitebox_subset  # noqa: E402


BENCHMARKS = ("PersonaMem-v2", "PersonaLens", "ETAPP", "LoCoMo")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the four official benchmark downloads into the canonical rows used by UMPeek."
    )
    parser.add_argument(
        "--benchmarks",
        default=",".join(BENCHMARKS),
        help="Comma-separated subset of PersonaMem-v2, PersonaLens, ETAPP, and LoCoMo.",
    )
    parser.add_argument("--personamem-csv", type=Path)
    parser.add_argument(
        "--personalens-root",
        type=Path,
        default=PROJECT_ROOT / "data/benchmarks/PersonaLens",
    )
    parser.add_argument(
        "--locomo-path",
        type=Path,
        default=PROJECT_ROOT / "data/benchmarks/LoCoMo",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create canonical rows but do not create role-separated splits or the evaluation manifest.",
    )
    return parser.parse_args()


def _path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else PROJECT_ROOT / path


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in materialized:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    return len(materialized)


def _prepare_personamem(csv_path: Path | None) -> dict[str, Any]:
    result = materialize_personamemv2_dataset(
        project_root=PROJECT_ROOT,
        config=SamplingConfig(),
        benchmark_csv=csv_path,
    )
    return {
        "benchmark": "PersonaMem-v2",
        "rows": len(result.task_records),
        "output": str(result.output_paths["task_records"].relative_to(PROJECT_ROOT)),
    }


def _prepare_personalens(dataset_root: Path) -> dict[str, Any]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"PersonaLens is missing at {dataset_root}. Download the official dataset as described in docs/DATA.md."
        )
    result = build_personalens_whitebox_subset(
        PROJECT_ROOT,
        dataset_root=dataset_root,
        output_dir=PROJECT_ROOT / "data/interim/exp1_whitebox/PersonaLens",
    )
    return {
        "benchmark": "PersonaLens",
        "rows": result.total_candidates,
        "output": str(result.task_records_path.relative_to(PROJECT_ROOT)),
    }


def _prepare_etapp() -> dict[str, Any]:
    source = PROJECT_ROOT / "data/benchmarks/ETAPP"
    if not source.is_dir():
        raise FileNotFoundError(
            f"ETAPP is missing at {source}. Clone or download the official repository as described in docs/DATA.md."
        )
    result = build_expanded_etapp_artifacts(project_root=PROJECT_ROOT)
    return {
        "benchmark": "ETAPP",
        "rows": result.example_count,
        "output": str(result.examples_path.relative_to(PROJECT_ROOT)),
    }


def _prepare_locomo(data_path: Path) -> dict[str, Any]:
    resolved = resolve_locomo_data_path(project_root=PROJECT_ROOT, data_path=data_path)
    rows = build_locomo_task_rows(project_root=PROJECT_ROOT, data_path=resolved)
    output_root = PROJECT_ROOT / "data/benchmarks/LoCoMo"
    output = output_root / "task_rows.jsonl"
    count = _write_jsonl(output, rows)
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    speakers = {
        str(conversation.get(key))
        for sample in raw
        for conversation in [dict(sample.get("conversation", {}))]
        for key in ("speaker_a", "speaker_b")
        if conversation.get(key)
    }
    manifest = {
        "source": str(resolved),
        "conversation_count": len(raw),
        "qa_count": count,
        "speaker_count": len(speakers),
        "task_rows": str(output.relative_to(PROJECT_ROOT)),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"benchmark": "LoCoMo", "rows": count, "output": str(output.relative_to(PROJECT_ROOT))}


def main() -> int:
    args = arguments()
    requested = tuple(item.strip() for item in args.benchmarks.split(",") if item.strip())
    unknown = sorted(set(requested).difference(BENCHMARKS))
    if not requested or unknown:
        raise ValueError(f"Unknown or empty benchmark selection: {unknown or requested}")
    reports: list[dict[str, Any]] = []
    if "PersonaMem-v2" in requested:
        reports.append(_prepare_personamem(_path(args.personamem_csv)))
    if "PersonaLens" in requested:
        reports.append(_prepare_personalens(_path(args.personalens_root)))
    if "ETAPP" in requested:
        reports.append(_prepare_etapp())
    if "LoCoMo" in requested:
        reports.append(_prepare_locomo(_path(args.locomo_path)))
    if not args.prepare_only:
        if set(requested) != set(BENCHMARKS):
            raise ValueError("Role-separated splits require all four canonical benchmark sources. Use --prepare-only for a subset.")
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts/a200_materialize_strong_query_splits.py")],
            cwd=PROJECT_ROOT,
            check=True,
        )
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts/build_minimal_manifest.py")],
            cwd=PROJECT_ROOT,
            check=True,
        )
    print(json.dumps({"status": "ok", "benchmarks": reports}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
