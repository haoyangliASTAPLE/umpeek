from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from umpeek.exp1.io import to_serializable

from .schema import EXPERIMENT_VERSION


LEGACY_FORBIDDEN_FILENAMES = (
    "candidate_metrics.csv",
    "candidate_metrics.json",
)


def ensure_run_layout(project_root: Path, scope: str, run_id: str) -> Path:
    run_dir = project_root / "runs" / "attack_baselines" / scope / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def validate_output_filenames(paths: Sequence[str | Path]) -> None:
    basenames = {Path(path).name for path in paths}
    legacy_hits = sorted(name for name in basenames if name in LEGACY_FORBIDDEN_FILENAMES)
    if legacy_hits:
        raise ValueError(f"Legacy candidate output files are not allowed: {legacy_hits}")
    if any("candidate_metrics" in name for name in basenames):
        raise ValueError("Legacy candidate metrics outputs are not allowed for attack baselines.")


def _normalize_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(to_serializable(payload))
    existing = data.get("experiment_version")
    if existing is not None and existing != EXPERIMENT_VERSION:
        raise ValueError(
            f"Unexpected experiment_version {existing!r}; expected {EXPERIMENT_VERSION!r}."
        )
    data["experiment_version"] = EXPERIMENT_VERSION
    return data


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_mapping(payload)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(normalized, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    return path


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            normalized = _normalize_mapping(row)
            handle.write(json.dumps(normalized, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return path


def append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            normalized = _normalize_mapping(row)
            handle.write(json.dumps(normalized, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return path


def read_jsonl(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(dict(json.loads(text)))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized_rows = [_normalize_mapping(row) for row in rows]
    if not normalized_rows:
        raise ValueError("write_csv requires at least one row.")
    fieldnames = list(normalized_rows[0].keys())
    known_fields = set(fieldnames)
    for row in normalized_rows[1:]:
        for field in row:
            if field not in known_fields:
                fieldnames.append(field)
                known_fields.add(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)
    return path


def write_markdown(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
