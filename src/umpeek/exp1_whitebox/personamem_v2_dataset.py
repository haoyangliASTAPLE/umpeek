from __future__ import annotations

import ast
import csv
import json
import math
import random
import shutil
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError

from .io import write_json, write_jsonl
from .schema import EXPERIMENT_VERSION


PERSONAMEM_V2_BENCHMARK = "PersonaMemv2"
PERSONAMEM_V2_DATASET_REPO = "bowen-upenn/PersonaMem-v2"
PERSONAMEM_V2_GITHUB_URL = "https://github.com/bowen-upenn/PersonaMem-v2"
PERSONAMEM_V2_PAPER_URL = "https://arxiv.org/abs/2512.06688"
PERSONAMEM_V2_TEXT_BENCHMARK_URL = (
    "https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2/resolve/main/"
    "benchmark/text/benchmark.csv?download=true"
)
PERSONAMEM_V2_RESOLVE_ROOT = (
    "https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2/resolve/main/"
)
PERSONAMEM_V2_EXTERNAL_RELATIVE_ROOT = Path("data/external/PersonaMemv2")
PERSONAMEM_V2_INTERIM_RELATIVE_ROOT = Path("data/interim/exp1_whitebox/PersonaMemv2")
DEFAULT_CONFIG_PATH = Path("configs/exp1_whitebox/personamemv2_representative_subset.json")
DEFAULT_HISTORY_BINS = ("short", "medium", "long")
DEFAULT_EVIDENCE_BINS = ("explicit", "implicit", "weak", "ambiguous")
NOT_APPLICABLE_TOOL_CATEGORY = "not_applicable"

_DIRECT_PREFERENCE_MARKERS = (
    "i like",
    "i prefer",
    "i love",
    "i enjoy",
    "i hate",
    "i dislike",
    "my favorite",
    "i always",
    "i usually",
    "i never",
    "i want",
    "i need",
)
_AMBIGUOUS_MARKERS = (
    "not sure",
    "maybe",
    "whatever",
    "anything is fine",
    "either is fine",
    "depends",
    "kind of",
    "sort of",
    "sometimes",
)


@dataclass(slots=True)
class SamplingConfig:
    seed: int = 20260515
    full_include_threshold: int = 300
    target_sample_count: int = 240
    min_target_count: int = 150
    max_target_count: int = 300
    max_tasks_per_user: int = 4
    min_distinct_users: int = 12
    task_type_min_ratio: float = 0.2
    task_type_max_floor: int = 20
    stratum_min_ratio: float = 0.08
    stratum_max_floor: int = 20
    source_timeout_s: float = 60.0
    source_retry_limit: int = 2

    @classmethod
    def from_path(cls, path: Path) -> SamplingConfig:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            seed=int(payload.get("seed", cls.seed)),
            full_include_threshold=int(
                payload.get("full_include_threshold", cls.full_include_threshold)
            ),
            target_sample_count=int(payload.get("target_sample_count", cls.target_sample_count)),
            min_target_count=int(payload.get("min_target_count", cls.min_target_count)),
            max_target_count=int(payload.get("max_target_count", cls.max_target_count)),
            max_tasks_per_user=int(payload.get("max_tasks_per_user", cls.max_tasks_per_user)),
            min_distinct_users=int(payload.get("min_distinct_users", cls.min_distinct_users)),
            task_type_min_ratio=float(payload.get("task_type_min_ratio", cls.task_type_min_ratio)),
            task_type_max_floor=int(payload.get("task_type_max_floor", cls.task_type_max_floor)),
            stratum_min_ratio=float(payload.get("stratum_min_ratio", cls.stratum_min_ratio)),
            stratum_max_floor=int(payload.get("stratum_max_floor", cls.stratum_max_floor)),
            source_timeout_s=float(payload.get("source_timeout_s", cls.source_timeout_s)),
            source_retry_limit=int(payload.get("source_retry_limit", cls.source_retry_limit)),
        )

    def target_size_for(self, total_count: int) -> int:
        if total_count <= self.full_include_threshold:
            return total_count
        desired = max(self.min_target_count, self.target_sample_count)
        desired = min(desired, self.max_target_count)
        return min(total_count, desired)

    def task_type_quota(self, available_count: int) -> int:
        if available_count <= 0:
            return 0
        proportional = math.ceil(available_count * self.task_type_min_ratio)
        return min(self.task_type_max_floor, max(1, proportional))

    def stratum_quota(self, target_count: int) -> int:
        if target_count <= 0:
            return 0
        proportional = math.ceil(target_count * self.stratum_min_ratio)
        return min(self.stratum_max_floor, max(1, proportional))


@dataclass(slots=True)
class PersonaMemTaskRecord:
    benchmark: str
    user_id: str
    task_id: str
    pre_history_ref: str
    task_input: dict[str, Any]
    gold: dict[str, Any]
    task_type: str
    personalization_evidence: list[dict[str, Any]]
    tool_schema: dict[str, Any]
    scoring_method: dict[str, Any]
    history_length: int | None = None
    history_length_bin: str = "unknown"
    personalization_evidence_strength: str = "ambiguous"
    tool_category: str = NOT_APPLICABLE_TOOL_CATEGORY
    metadata: dict[str, Any] = field(default_factory=dict)
    experiment_version: str = EXPERIMENT_VERSION

    def sample_key(self) -> str:
        return f"{self.user_id}::{self.task_id}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sample_key"] = self.sample_key()
        return payload


@dataclass(frozen=True, slots=True)
class DatasetBuildResult:
    task_records: list[PersonaMemTaskRecord]
    representative_subset: list[PersonaMemTaskRecord]
    sampling_report: dict[str, Any]
    output_paths: dict[str, Path]


def load_sampling_config(config_path: Path | None = None) -> SamplingConfig:
    resolved = config_path or DEFAULT_CONFIG_PATH
    return SamplingConfig.from_path(resolved) if resolved.is_file() else SamplingConfig()


def materialize_personamemv2_dataset(
    *,
    project_root: Path,
    config: SamplingConfig | None = None,
    config_path: Path | None = None,
    external_root: Path | None = None,
    output_root: Path | None = None,
    benchmark_csv: Path | None = None,
) -> DatasetBuildResult:
    resolved_config = config or load_sampling_config(config_path)
    resolved_external_root = external_root or (project_root / PERSONAMEM_V2_EXTERNAL_RELATIVE_ROOT)
    resolved_output_root = output_root or (project_root / PERSONAMEM_V2_INTERIM_RELATIVE_ROOT)
    task_records, normalization_summary = build_personamemv2_task_records(
        project_root=project_root,
        config=resolved_config,
        external_root=resolved_external_root,
        benchmark_csv=benchmark_csv,
    )
    representative_subset, sampling_report = select_representative_subset(
        task_records,
        config=resolved_config,
        normalization_summary=normalization_summary,
    )
    output_paths = {
        "task_records": resolved_output_root / "task_records.jsonl",
        "representative_subset": resolved_output_root / "representative_subset.jsonl",
        "sampling_report": resolved_output_root / "sampling_report.json",
    }
    write_jsonl(output_paths["task_records"], [record.to_dict() for record in task_records])
    write_jsonl(
        output_paths["representative_subset"],
        [record.to_dict() for record in representative_subset],
    )
    write_json(output_paths["sampling_report"], sampling_report)
    return DatasetBuildResult(
        task_records=task_records,
        representative_subset=representative_subset,
        sampling_report=sampling_report,
        output_paths=output_paths,
    )


def build_personamemv2_task_records(
    *,
    project_root: Path,
    config: SamplingConfig,
    external_root: Path,
    benchmark_csv: Path | None = None,
) -> tuple[list[PersonaMemTaskRecord], dict[str, Any]]:
    rows, benchmark_csv_path, source_summary = load_personamemv2_rows(
        project_root=project_root,
        external_root=external_root,
        benchmark_csv=benchmark_csv,
        config=config,
    )

    history_length_cache: dict[str, tuple[int | None, str, str]] = {}
    task_records: list[PersonaMemTaskRecord] = []
    skipped_rows: list[dict[str, Any]] = []
    history_failures: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows):
        user_id = str(row.get("persona_id", "")).strip()
        if not user_id:
            skipped_rows.append({"row_index": row_index, "reason": "missing_persona_id"})
            continue
        user_query = _user_query_text(row)
        options = _answer_options(row)
        correct_answer = str(row.get("correct_answer", "")).strip()
        if not user_query or len(options) < 2 or not correct_answer:
            skipped_rows.append(
                {
                    "row_index": row_index,
                    "reason": "missing_query_or_choices",
                    "user_id": user_id,
                }
            )
            continue
        if correct_answer not in options:
            options = [correct_answer, *[option for option in options if option != correct_answer]]

        task_id = f"pmv2_{row_index:05d}"
        history_relative_path = str(row.get("chat_history_32k_link", "")).strip()
        history_length: int | None = None
        history_source = "missing"
        pre_history_ref = ""
        if history_relative_path:
            if history_relative_path not in history_length_cache:
                try:
                    history_path = ensure_personamemv2_asset(
                        project_root=project_root,
                        external_root=external_root,
                        relative_path=history_relative_path,
                        config=config,
                    )
                    history_length_cache[history_relative_path] = (
                        _count_history_messages(_load_json(history_path)),
                        str(history_path),
                        "downloaded_or_cached",
                    )
                except RuntimeError as exc:
                    history_length_cache[history_relative_path] = (None, "", "missing_asset")
                    history_failures.append(
                        {
                            "row_index": row_index,
                            "user_id": user_id,
                            "relative_path": history_relative_path,
                            "reason": str(exc),
                        }
                    )
            history_length, pre_history_ref, history_source = history_length_cache[history_relative_path]

        evidence, evidence_strength = _build_personalization_evidence(
            row=row,
            row_index=row_index,
            benchmark_csv_path=benchmark_csv_path,
        )
        task_records.append(
            PersonaMemTaskRecord(
                benchmark=PERSONAMEM_V2_BENCHMARK,
                user_id=user_id,
                task_id=task_id,
                pre_history_ref=pre_history_ref,
                task_input={
                    "user_query": user_query,
                    "answer_options": options,
                    "topic_query": str(row.get("topic_query", "")).strip(),
                    "conversation_scenario": str(row.get("conversation_scenario", "")).strip(),
                    "topic_preference": str(row.get("topic_preference", "")).strip(),
                },
                gold={
                    "correct_answer": correct_answer,
                    "answer_options": options,
                    "incorrect_answers": [option for option in options if option != correct_answer],
                },
                task_type="choice",
                personalization_evidence=evidence,
                tool_schema={
                    "type": "no_tool_benchmark",
                    "tools": [],
                    "tool_categories": [],
                },
                scoring_method={
                    "name": "multiple_choice_exact_match",
                    "score_range": [0.0, 1.0],
                    "prediction_field": "predicted_answer",
                    "gold_field": "gold.correct_answer",
                    "valid_labels": options,
                    "notes": "Predicted answer must exactly match the gold answer text.",
                },
                history_length=history_length,
                personalization_evidence_strength=evidence_strength,
                metadata={
                    "source_csv": str(benchmark_csv_path),
                    "source_row_index": row_index,
                    "history_length_source": history_source,
                    "history_relative_path": history_relative_path,
                    "raw_persona_file": str(row.get("raw_persona_file", "")).strip(),
                    "pref_type": str(row.get("pref_type", "")).strip(),
                    "preference": str(row.get("preference", "")).strip(),
                    "prev_pref": str(row.get("prev_pref", "")).strip(),
                    "short_persona": _short_persona_text(row),
                    "topic_query": str(row.get("topic_query", "")).strip(),
                    "conversation_scenario": str(row.get("conversation_scenario", "")).strip(),
                    "official_source": {
                        "dataset_repo": PERSONAMEM_V2_DATASET_REPO,
                        "github": PERSONAMEM_V2_GITHUB_URL,
                        "paper": PERSONAMEM_V2_PAPER_URL,
                    },
                },
            )
        )

    history_thresholds = _history_bin_thresholds(
        [record.history_length for record in task_records if record.history_length is not None]
    )
    for record in task_records:
        record.history_length_bin = _history_length_bin(record.history_length, history_thresholds)

    normalization_summary = {
        "experiment_version": EXPERIMENT_VERSION,
        "benchmark": PERSONAMEM_V2_BENCHMARK,
        "source_summary": source_summary,
        "benchmark_csv": str(benchmark_csv_path),
        "task_record_count": len(task_records),
        "skipped_rows": skipped_rows,
        "history_failures": history_failures,
        "history_thresholds": history_thresholds,
        "distinct_users": len({record.user_id for record in task_records}),
    }
    return task_records, normalization_summary


def select_representative_subset(
    task_records: Sequence[PersonaMemTaskRecord],
    *,
    config: SamplingConfig,
    normalization_summary: Mapping[str, Any] | None = None,
) -> tuple[list[PersonaMemTaskRecord], dict[str, Any]]:
    total_count = len(task_records)
    target_count = config.target_size_for(total_count)
    selection_mode = "full_include" if target_count >= total_count else "stratified_sample"
    available_counts = _distribution_by_axes(task_records)
    quotas = _build_axis_quotas(task_records, target_count=target_count, config=config)

    if selection_mode == "full_include":
        selected = list(task_records)
        selection_reasons = {record.sample_key(): ["full_include_threshold"] for record in selected}
    else:
        selected, selection_reasons = _greedy_select(task_records, quotas=quotas, target_count=target_count, config=config)

    selected.sort(key=lambda record: (record.user_id, record.task_id))
    selected_counts = _distribution_by_axes(selected)
    gaps = _sampling_gaps(available_counts=available_counts, selected_counts=selected_counts, quotas=quotas)
    subset_preview = [
        {
            **record.to_dict(),
            "selection_reasons": selection_reasons.get(record.sample_key(), []),
        }
        for record in selected[:25]
    ]

    report = {
        "experiment_version": EXPERIMENT_VERSION,
        "benchmark": PERSONAMEM_V2_BENCHMARK,
        "selection_mode": selection_mode,
        "seed": config.seed,
        "total_task_records": total_count,
        "selected_task_records": len(selected),
        "distinct_users_total": len({record.user_id for record in task_records}),
        "distinct_users_selected": len({record.user_id for record in selected}),
        "config": asdict(config),
        "official_source": {
            "dataset_repo": PERSONAMEM_V2_DATASET_REPO,
            "github": PERSONAMEM_V2_GITHUB_URL,
            "paper": PERSONAMEM_V2_PAPER_URL,
        },
        "available_distributions": available_counts,
        "selected_distributions": selected_counts,
        "required_quotas": quotas,
        "gaps": gaps,
        "selection_preview": subset_preview,
        "normalization_summary": dict(normalization_summary or {}),
    }
    return selected, report


def load_personamemv2_rows(
    *,
    project_root: Path,
    external_root: Path,
    benchmark_csv: Path | None,
    config: SamplingConfig,
) -> tuple[list[dict[str, Any]], Path, dict[str, Any]]:
    external_root.mkdir(parents=True, exist_ok=True)
    benchmark_path = benchmark_csv or (external_root / "benchmark_text.csv")
    source_mode = "provided_path" if benchmark_csv else "external_cache"
    if not benchmark_path.exists():
        local_cache = _find_local_benchmark_cache(project_root)
        if local_cache is not None:
            benchmark_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local_cache, benchmark_path)
            source_mode = "copied_from_local_cache"
        else:
            _download_to_path(
                PERSONAMEM_V2_TEXT_BENCHMARK_URL,
                benchmark_path,
                timeout_s=config.source_timeout_s,
                retry_limit=config.source_retry_limit,
            )
            source_mode = "downloaded_from_official_hf"
    with benchmark_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows, benchmark_path, {
        "source_mode": source_mode,
        "external_root": str(external_root),
        "row_count": len(rows),
    }


def ensure_personamemv2_asset(
    *,
    project_root: Path,
    external_root: Path,
    relative_path: str,
    config: SamplingConfig,
) -> Path:
    destination = external_root / "source_data" / relative_path
    if destination.exists():
        return destination
    cached_source = _find_local_asset_cache(project_root=project_root, relative_path=relative_path)
    if cached_source is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached_source, destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    quoted_path = urllib.parse.quote(relative_path)
    url = f"{PERSONAMEM_V2_RESOLVE_ROOT}{quoted_path}?download=true"
    _download_to_path(
        url,
        destination,
        timeout_s=config.source_timeout_s,
        retry_limit=config.source_retry_limit,
    )
    return destination


def _download_to_path(url: str, path: Path, *, timeout_s: float, retry_limit: int) -> None:
    last_error: Exception | None = None
    for _ in range(retry_limit + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as response:
                path.write_bytes(response.read())
            return
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def _find_local_benchmark_cache(project_root: Path) -> Path | None:
    candidates = sorted(project_root.glob("runs/exp1/*/PersonaMemv2/_cache/benchmark_text.csv"))
    return candidates[0] if candidates else None


def _find_local_asset_cache(*, project_root: Path, relative_path: str) -> Path | None:
    candidate_patterns = (
        f"runs/exp1/*/PersonaMemv2/**/source_data/{relative_path}",
    )
    for pattern in candidate_patterns:
        matches = sorted(project_root.glob(pattern))
        if matches:
            return matches[0]
    return None


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_eval_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(stripped)
            except (ValueError, SyntaxError, json.JSONDecodeError, TypeError):
                continue
            if isinstance(parsed, dict):
                return parsed
    return {}


def _safe_eval_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(stripped)
            except (ValueError, SyntaxError, json.JSONDecodeError, TypeError):
                continue
            if isinstance(parsed, list):
                return parsed
    return []


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" or "text" in item:
                parts.append(str(item.get("text", "")))
        return " ".join(part for part in parts if part).strip()
    if isinstance(content, dict):
        if "content" in content:
            return _content_to_text(content["content"])
        if "text" in content:
            return str(content.get("text", "")).strip()
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _messages_to_text(messages: Sequence[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", message.get("speaker", "unknown"))).strip() or "unknown"
        content = _content_to_text(message.get("content", message.get("text", "")))
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _snippet_messages(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = _safe_eval_list(row.get("related_conversation_snippet", []))
    return [item for item in values if isinstance(item, dict)]


def _user_query_text(row: Mapping[str, Any]) -> str:
    user_query = row.get("user_query", "")
    if isinstance(user_query, dict):
        return str(user_query.get("content", user_query.get("text", ""))).strip()
    parsed = _safe_eval_dict(user_query)
    if parsed:
        return str(parsed.get("content", parsed.get("text", ""))).strip()
    return str(user_query).strip()


def _short_persona_text(row: Mapping[str, Any]) -> str:
    parsed = _safe_eval_dict(row.get("short_persona", ""))
    persona = parsed.get("persona") if parsed else None
    if persona:
        return str(persona).strip()
    return str(row.get("short_persona", "")).strip()


def _answer_options(row: Mapping[str, Any]) -> list[str]:
    correct = str(row.get("correct_answer", "")).strip()
    incorrect = [str(value).strip() for value in _safe_eval_list(row.get("incorrect_answers", []))]
    options = [correct, *[value for value in incorrect if value]]
    deduped: list[str] = []
    for option in options:
        if option and option not in deduped:
            deduped.append(option)
    return deduped


def _build_personalization_evidence(
    *,
    row: Mapping[str, Any],
    row_index: int,
    benchmark_csv_path: Path,
) -> tuple[list[dict[str, Any]], str]:
    evidence: list[dict[str, Any]] = []
    preference = str(row.get("preference", "")).strip()
    prev_pref = str(row.get("prev_pref", "")).strip()
    short_persona = _short_persona_text(row)
    snippet_messages = _snippet_messages(row)
    snippet_text = _messages_to_text(snippet_messages)
    source_base = f"{benchmark_csv_path}#row={row_index}"

    if short_persona:
        evidence.append(
            {
                "source_type": "short_persona",
                "source_ref": f"{source_base}:short_persona",
                "text": short_persona,
            }
        )
    if preference:
        evidence.append(
            {
                "source_type": "preference_field",
                "source_ref": f"{source_base}:preference",
                "text": preference,
            }
        )
    if prev_pref:
        evidence.append(
            {
                "source_type": "previous_preference_field",
                "source_ref": f"{source_base}:prev_pref",
                "text": prev_pref,
            }
        )
    if snippet_text:
        evidence.append(
            {
                "source_type": "conversation_snippet",
                "source_ref": f"{source_base}:related_conversation_snippet",
                "text": snippet_text,
                "message_count": len(snippet_messages),
            }
        )

    combined = "\n".join(item["text"] for item in evidence if item.get("text"))
    evidence_strength = _classify_personalization_strength(
        combined_text=combined,
        snippet_count=len(snippet_messages),
        has_profile=bool(short_persona or preference or prev_pref),
    )
    return evidence, evidence_strength


def _classify_personalization_strength(
    *,
    combined_text: str,
    snippet_count: int,
    has_profile: bool,
) -> str:
    lowered = combined_text.lower()
    has_direct = any(marker in lowered for marker in _DIRECT_PREFERENCE_MARKERS)
    has_ambiguous = any(marker in lowered for marker in _AMBIGUOUS_MARKERS)
    token_count = len([token for token in lowered.split() if token])
    if has_ambiguous and not has_direct:
        return "ambiguous"
    if has_direct and snippet_count > 0:
        return "explicit"
    if snippet_count >= 1 and has_profile:
        return "implicit"
    if has_profile and snippet_count == 0:
        return "weak"
    if token_count < 5:
        return "ambiguous"
    if snippet_count > 0:
        return "implicit"
    return "weak"


def _count_history_messages(payload: Any) -> int:
    if isinstance(payload, dict):
        if (
            ("role" in payload or "speaker" in payload)
            and any(key in payload for key in ("content", "text", "utterance", "message"))
        ):
            return 1
        for key in ("messages", "conversation", "chat_history", "history", "dialogue", "dialog"):
            if key in payload:
                count = _count_history_messages(payload[key])
                if count:
                    return count
        return sum(_count_history_messages(value) for value in payload.values())
    if isinstance(payload, list):
        return sum(_count_history_messages(item) for item in payload)
    return 0


def _history_bin_thresholds(values: Sequence[int | None]) -> dict[str, int | None]:
    usable = sorted(value for value in values if value is not None)
    if not usable:
        return {"short_max": None, "medium_max": None}
    low_index = max(0, math.floor((len(usable) - 1) / 3))
    high_index = max(0, math.floor(((len(usable) - 1) * 2) / 3))
    return {
        "short_max": usable[low_index],
        "medium_max": usable[high_index],
    }


def _history_length_bin(history_length: int | None, thresholds: Mapping[str, int | None]) -> str:
    short_max = thresholds.get("short_max")
    medium_max = thresholds.get("medium_max")
    if history_length is None or short_max is None or medium_max is None:
        return "unknown"
    if history_length <= short_max:
        return "short"
    if history_length <= medium_max:
        return "medium"
    return "long"


def _distribution_by_axes(records: Sequence[PersonaMemTaskRecord]) -> dict[str, dict[str, int]]:
    axes = {
        "task_type": {},
        "history_length_bin": {},
        "personalization_evidence_strength": {},
        "tool_category": {},
    }
    for record in records:
        axes["task_type"][record.task_type] = axes["task_type"].get(record.task_type, 0) + 1
        axes["history_length_bin"][record.history_length_bin] = (
            axes["history_length_bin"].get(record.history_length_bin, 0) + 1
        )
        axes["personalization_evidence_strength"][record.personalization_evidence_strength] = (
            axes["personalization_evidence_strength"].get(record.personalization_evidence_strength, 0)
            + 1
        )
        axes["tool_category"][record.tool_category] = axes["tool_category"].get(record.tool_category, 0) + 1
    axes["distinct_users"] = {"count": len({record.user_id for record in records})}
    return axes


def _build_axis_quotas(
    task_records: Sequence[PersonaMemTaskRecord],
    *,
    target_count: int,
    config: SamplingConfig,
) -> dict[str, dict[str, int]]:
    quotas: dict[str, dict[str, int]] = {
        "task_type": {},
        "history_length_bin": {},
        "personalization_evidence_strength": {},
        "tool_category": {},
    }
    distributions = _distribution_by_axes(task_records)
    for value, count in distributions["task_type"].items():
        quotas["task_type"][value] = min(target_count, config.task_type_quota(count))

    history_floor = config.stratum_quota(target_count)
    for value in DEFAULT_HISTORY_BINS:
        count = distributions["history_length_bin"].get(value, 0)
        if count > 0:
            quotas["history_length_bin"][value] = min(count, history_floor)

    evidence_floor = config.stratum_quota(target_count)
    for value in DEFAULT_EVIDENCE_BINS:
        count = distributions["personalization_evidence_strength"].get(value, 0)
        if count > 0:
            quotas["personalization_evidence_strength"][value] = min(count, evidence_floor)

    for value, count in distributions["tool_category"].items():
        if value == NOT_APPLICABLE_TOOL_CATEGORY:
            continue
        quotas["tool_category"][value] = min(count, config.stratum_quota(target_count))
    return quotas


def _greedy_select(
    task_records: Sequence[PersonaMemTaskRecord],
    *,
    quotas: Mapping[str, Mapping[str, int]],
    target_count: int,
    config: SamplingConfig,
) -> tuple[list[PersonaMemTaskRecord], dict[str, list[str]]]:
    rng = random.Random(config.seed)
    pool = list(task_records)
    rng.shuffle(pool)
    selected: list[PersonaMemTaskRecord] = []
    selected_keys: set[str] = set()
    selection_reasons: dict[str, list[str]] = {}
    user_counts: dict[str, int] = {}
    available_counts = _distribution_by_axes(pool)
    selected_counts = {
        "task_type": {},
        "history_length_bin": {},
        "personalization_evidence_strength": {},
        "tool_category": {},
    }
    minimum_users = min(
        len({record.user_id for record in pool}),
        max(config.min_distinct_users, math.ceil(target_count / max(1, config.max_tasks_per_user))),
    )

    value_getters = {
        "task_type": lambda record: record.task_type,
        "history_length_bin": lambda record: record.history_length_bin,
        "personalization_evidence_strength": lambda record: record.personalization_evidence_strength,
        "tool_category": lambda record: record.tool_category,
    }

    while len(selected) < target_count and len(selected_keys) < len(pool):
        best_record: PersonaMemTaskRecord | None = None
        best_score: tuple[float, str, str] | None = None
        best_reasons: list[str] = []

        for record in pool:
            sample_key = record.sample_key()
            if sample_key in selected_keys:
                continue
            reasons: list[str] = []
            score = 0.0
            for axis, getter in value_getters.items():
                value = getter(record)
                target_for_value = quotas.get(axis, {}).get(value, 0)
                current_for_value = selected_counts[axis].get(value, 0)
                if target_for_value > current_for_value:
                    reasons.append(f"quota:{axis}:{value}")
                    score += 100.0
                available_for_value = available_counts.get(axis, {}).get(value, 0)
                if available_for_value > 0:
                    score += 5.0 / available_for_value

            if user_counts.get(record.user_id, 0) == 0:
                reasons.append("new_user")
                score += 40.0
            if len(user_counts) < minimum_users and user_counts.get(record.user_id, 0) == 0:
                score += 30.0
            score -= 15.0 * max(0, user_counts.get(record.user_id, 0) - (config.max_tasks_per_user - 1))
            score += max(0, config.max_tasks_per_user - user_counts.get(record.user_id, 0))

            current_score = (score, record.user_id, record.task_id)
            if best_score is None or current_score > best_score:
                best_record = record
                best_score = current_score
                best_reasons = reasons

        if best_record is None:
            break

        selected.append(best_record)
        sample_key = best_record.sample_key()
        selected_keys.add(sample_key)
        selection_reasons[sample_key] = best_reasons or ["diversity_fill"]
        user_counts[best_record.user_id] = user_counts.get(best_record.user_id, 0) + 1
        selected_counts["task_type"][best_record.task_type] = (
            selected_counts["task_type"].get(best_record.task_type, 0) + 1
        )
        selected_counts["history_length_bin"][best_record.history_length_bin] = (
            selected_counts["history_length_bin"].get(best_record.history_length_bin, 0) + 1
        )
        selected_counts["personalization_evidence_strength"][best_record.personalization_evidence_strength] = (
            selected_counts["personalization_evidence_strength"].get(
                best_record.personalization_evidence_strength,
                0,
            )
            + 1
        )
        selected_counts["tool_category"][best_record.tool_category] = (
            selected_counts["tool_category"].get(best_record.tool_category, 0) + 1
        )

    return selected[:target_count], selection_reasons


def _sampling_gaps(
    *,
    available_counts: Mapping[str, Mapping[str, int]],
    selected_counts: Mapping[str, Mapping[str, int]],
    quotas: Mapping[str, Mapping[str, int]],
) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    for axis, values in quotas.items():
        for value, target in values.items():
            available = available_counts.get(axis, {}).get(value, 0)
            selected = selected_counts.get(axis, {}).get(value, 0)
            if selected >= target:
                continue
            gaps.append(
                {
                    "axis": axis,
                    "value": value,
                    "target": target,
                    "available": available,
                    "selected": selected,
                    "reason": "insufficient_available_records" if available < target else "selection_underfilled",
                }
            )

    for value in DEFAULT_HISTORY_BINS:
        if available_counts.get("history_length_bin", {}).get(value, 0) == 0:
            gaps.append(
                {
                    "axis": "history_length_bin",
                    "value": value,
                    "target": 1,
                    "available": 0,
                    "selected": 0,
                    "reason": "benchmark_missing_history_bin",
                }
            )
    for value in DEFAULT_EVIDENCE_BINS:
        if available_counts.get("personalization_evidence_strength", {}).get(value, 0) == 0:
            gaps.append(
                {
                    "axis": "personalization_evidence_strength",
                    "value": value,
                    "target": 1,
                    "available": 0,
                    "selected": 0,
                    "reason": "benchmark_missing_evidence_bin",
                }
            )
    return gaps
