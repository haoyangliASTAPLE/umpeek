from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .schema import TaskRecord, UserRecord


ETAPP_BENCHMARK = "ETAPP"
ETAPP_BENCHMARK_SPLIT = "public_function_call_traces"
ETAPP_RELATIVE_ROOT = Path(".vendor/ETAPP")
ETAPP_SMOKE_QUERY_LIMIT = 5
ETAPP_SMOKE_USER_LIMIT = 2

_QUERY_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_USER_TURN_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)
_PROFILE_RE = re.compile(
    r"#### \*\*User Profile:\*\*\s*\n(.*?)\n\n#### \*\*specific and detailed preferences of user:\*\*",
    re.S,
)
_PREFERENCES_RE = re.compile(
    r"#### \*\*specific and detailed preferences of user:\*\*\s*\n(.*?)\n\n#### \*\*User Status:\*\*",
    re.S,
)
_STATUS_RE = re.compile(r"#### \*\*User Status:\*\*\s*\n(.*?)\n\n\n\n# Tools", re.S)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _approximate_token_count(text: str) -> int:
    return len(_QUERY_TOKEN_RE.findall(text))
_DECISION_KEYS = (
    "location",
    "city",
    "origin_city",
    "destination_city",
    "departure_date",
    "start_time",
    "end_time",
    "music_name",
    "volume_level",
)


@dataclass(frozen=True, slots=True)
class EtappInstruction:
    instruction_id: int
    query: str
    keypoint_personal: tuple[str, ...]
    keypoint_proactive: tuple[str, ...]
    available_tools: tuple[str, ...]
    location: str

    @property
    def task_id(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", self.query.lower()).strip("_")
        return f"etapp_{self.instruction_id:02d}_{slug[:32].rstrip('_')}"


@dataclass(frozen=True, slots=True)
class EtappTraceRecord:
    query: str
    user_name: str
    profile: dict[str, Any]
    detailed_preferences: list[dict[str, Any]]
    status: dict[str, Any]
    call_name: str
    call_args: dict[str, Any]
    tool_response_count: int


@dataclass(frozen=True, slots=True)
class EtappExample:
    instruction: EtappInstruction
    user_name: str
    profile: dict[str, Any]
    detailed_preferences: list[dict[str, Any]]
    status: dict[str, Any]
    action_sequence: tuple[dict[str, Any], ...]

    @property
    def user_id(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", self.user_name.lower()).strip("_")
        return f"etapp_{slug}"

    @property
    def action_signature(self) -> str:
        return canonicalize_etapp_action_sequence(
            task_id=self.instruction.task_id,
            action_sequence=self.action_sequence,
        )


def canonicalize_etapp_action_sequence(
    *,
    task_id: str,
    action_sequence: Iterable[Mapping[str, Any]],
    proactive: bool | None = None,
) -> str:
    components = [
        _sequence_component(dict(item))
        for item in action_sequence
        if str(item.get("tool_name", "")).strip()
    ]
    effective_proactive = proactive if proactive is not None else len(components) > 1
    payload = {
        "intent": task_id,
        "proactive": bool(effective_proactive),
        "tool_sequence": components,
        "key_decision_fields": components[-1]["key_decision_fields"] if components else {},
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def build_etapp_fallback_action_signature(
    *,
    task_id: str,
    available_tools: Iterable[str],
    used_tools: Iterable[str] = (),
) -> str:
    used = {str(tool_name).strip() for tool_name in used_tools if str(tool_name).strip()}
    fallback_tool = next(
        (tool_name for tool_name in available_tools if str(tool_name).strip() not in used),
        "respond_without_tool",
    )
    return canonicalize_etapp_action_sequence(
        task_id=task_id,
        action_sequence=(
            {
                "tool_name": str(fallback_tool),
                "normalized_args": {},
            },
        ),
        proactive=False,
    )


@dataclass(frozen=True, slots=True)
class EtappSmokeDataset:
    users: list[UserRecord]
    tasks: list[TaskRecord]
    examples_by_user_task: dict[tuple[str, str], EtappExample]
    selected_queries: tuple[str, ...]
    selected_user_names: tuple[str, ...]
    swap_user_map: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EtappFullDataset:
    users: list[UserRecord]
    tasks: list[TaskRecord]
    examples_by_user_task: dict[tuple[str, str], EtappExample]
    user_ids_by_task: dict[str, tuple[str, ...]]
    swap_user_by_user_task: dict[tuple[str, str], str]
    metadata: dict[str, Any] = field(default_factory=dict)


def build_etapp_full_dataset(
    *,
    project_root: Path,
    benchmark: str = ETAPP_BENCHMARK,
    data_root: Path | None = None,
    variant: str | None = None,
) -> EtappFullDataset:
    etapp_root, instructions, grouped_examples, dataset_metadata = _load_etapp_dataset_inputs(
        project_root=project_root,
        data_root=data_root,
        variant=variant,
    )
    instruction_order = {instruction.query: index for index, instruction in enumerate(instructions)}

    ordered_keys = sorted(
        grouped_examples,
        key=lambda item: (
            instruction_order.get(item[0], len(instructions)),
            item[1].lower(),
        ),
    )
    users_by_id: dict[str, UserRecord] = {}
    tasks: list[TaskRecord] = []
    examples_by_user_task: dict[tuple[str, str], EtappExample] = {}
    user_ids_by_task: dict[str, list[str]] = defaultdict(list)

    for key in ordered_keys:
        example = grouped_examples[key]
        users_by_id.setdefault(example.user_id, _build_user_record(example, benchmark))
        task = _build_task_record(example, benchmark)
        tasks.append(task)
        user_task_key = (task.user_id, task.task_id)
        examples_by_user_task[user_task_key] = example
        user_ids_by_task.setdefault(task.task_id, []).append(task.user_id)

    normalized_user_ids_by_task = {
        task_id: tuple(sorted({user_id for user_id in user_ids}))
        for task_id, user_ids in user_ids_by_task.items()
    }
    swap_user_by_user_task: dict[tuple[str, str], str] = {}
    for task_id, user_ids in normalized_user_ids_by_task.items():
        per_task_swap = _build_swap_user_map(user_ids)
        for user_id, swap_user_id in per_task_swap.items():
            swap_user_by_user_task[(user_id, task_id)] = swap_user_id

    return EtappFullDataset(
        users=list(users_by_id.values()),
        tasks=tasks,
        examples_by_user_task=examples_by_user_task,
        user_ids_by_task=normalized_user_ids_by_task,
        swap_user_by_user_task=swap_user_by_user_task,
        metadata={
            "benchmark": benchmark,
            "evaluation": "offline_action_sequence_exact_match",
            "data_root": str(dataset_metadata.get("data_root") or etapp_root),
            "n_users": len(users_by_id),
            "n_tasks": len(tasks),
            "n_examples": len(examples_by_user_task),
            "supported_query_count": len({query for query, _ in grouped_examples}),
            "instruction_count": len(instructions),
            "split_label": str(dataset_metadata.get("benchmark_split", ETAPP_BENCHMARK_SPLIT)),
            **{
                key: value
                for key, value in dataset_metadata.items()
                if key not in {"benchmark_split", "data_root"}
            },
        },
    )


def build_etapp_smoke_dataset(
    *,
    project_root: Path,
    benchmark: str = ETAPP_BENCHMARK,
    data_root: Path | None = None,
    max_queries: int = ETAPP_SMOKE_QUERY_LIMIT,
    max_users: int = ETAPP_SMOKE_USER_LIMIT,
    variant: str | None = None,
    task_ids: Sequence[str] | None = None,
) -> EtappSmokeDataset:
    etapp_root, instructions, grouped_examples, dataset_metadata = _load_etapp_dataset_inputs(
        project_root=project_root,
        data_root=data_root,
        variant=variant,
    )
    normalized_task_ids = tuple(
        dict.fromkeys(str(task_id).strip() for task_id in (task_ids or ()) if str(task_id).strip())
    )
    selected_queries = _select_queries(
        instructions,
        grouped_examples,
        max_queries=max_queries,
        task_ids=normalized_task_ids or None,
    )
    selected_users = _select_users(selected_queries, grouped_examples, max_users=max_users)
    query_to_task_id = {instruction.query: instruction.task_id for instruction in instructions}

    users: list[UserRecord] = []
    tasks: list[TaskRecord] = []
    examples_by_user_task: dict[tuple[str, str], EtappExample] = {}

    for user_name in selected_users:
        anchor_example = grouped_examples[(selected_queries[0], user_name)]
        users.append(_build_user_record(anchor_example, benchmark))
        for query in selected_queries:
            example = grouped_examples[(query, user_name)]
            task = _build_task_record(example, benchmark)
            tasks.append(task)
            examples_by_user_task[(example.user_id, task.task_id)] = example

    return EtappSmokeDataset(
        users=users,
        tasks=tasks,
        examples_by_user_task=examples_by_user_task,
        selected_queries=tuple(selected_queries),
        selected_user_names=tuple(selected_users),
        swap_user_map=_build_swap_user_map(
            grouped_examples[(selected_queries[0], user_name)].user_id for user_name in selected_users
        ),
        metadata={
            "benchmark": benchmark,
            "evaluation": "offline_action_sequence_exact_match",
            "data_root": str(dataset_metadata.get("data_root") or etapp_root),
            "selected_queries": list(selected_queries),
            "selected_task_ids": [query_to_task_id[query] for query in selected_queries if query in query_to_task_id],
            "selected_user_names": list(selected_users),
            "n_users": len(users),
            "n_tasks": len(tasks),
            "n_examples": len(examples_by_user_task),
            "supported_query_count": len({query for query, _ in grouped_examples}),
            "split_label": str(dataset_metadata.get("benchmark_split", ETAPP_BENCHMARK_SPLIT)),
            **{
                key: value
                for key, value in dataset_metadata.items()
                if key not in {"benchmark_split", "data_root"}
            },
        },
    )


def build_etapp_task_rows_from_local(
    data_root: Path,
    *,
    benchmark: str = ETAPP_BENCHMARK,
    variant: str | None = None,
) -> list[dict[str, Any]]:
    dataset = build_etapp_full_dataset(
        project_root=data_root,
        benchmark=benchmark,
        data_root=data_root,
        variant=variant,
    )
    rows: list[dict[str, Any]] = []
    ordered_tasks = sorted(
        dataset.tasks,
        key=lambda task: (
            task.task_id,
            task.user_id,
        ),
    )
    for row_index, task in enumerate(ordered_tasks):
        swap_user_id = dataset.swap_user_by_user_task.get((task.user_id, task.task_id))
        rows.append(
            {
                "__row_index": row_index,
                "benchmark": benchmark,
                "benchmark_split": task.metadata.get("benchmark_split", ETAPP_BENCHMARK_SPLIT),
                "user_id": task.user_id,
                "task_id": task.task_id,
                "task_type": task.task_type,
                "query": task.prompt,
                "dialogue_domain": task.metadata.get("dialogue_domain", "general_assistant"),
                "tool_name": task.metadata.get("tool_name", "respond_without_tool"),
                "primary_tool_name": task.metadata.get("tool_name", "respond_without_tool"),
                "action_type": task.metadata.get("action_type", "single_tool"),
                "task_oriented_intent": task.metadata.get("action_type", "single_tool"),
                "proactivity_required": bool(task.metadata.get("proactivity_required", False)),
                "personalization_constraint_type": task.metadata.get(
                    "personalization_constraint_type",
                    "general_profile",
                ),
                "history_length_estimate": int(task.metadata.get("history_length_estimate", 0) or 0),
                "history_length_bin": task.metadata.get("history_length_bin", "short"),
                "personalization_strength_score": float(
                    task.metadata.get("personalization_strength_score", 0.0) or 0.0
                ),
                "personalization_strength_bin": task.metadata.get(
                    "personalization_strength_bin",
                    "low",
                ),
                "difficulty": task.metadata.get("difficulty", "easy"),
                "answer_judge_label_bin": "exact_action_sequence",
                "swap_user_id": swap_user_id,
                "swap_status": "available" if swap_user_id else "swap_unavailable",
            }
        )
    return rows


def _resolve_etapp_root(*, project_root: Path, data_root: Path | None) -> Path:
    root = data_root or (project_root / ETAPP_RELATIVE_ROOT)
    if not root.exists():
        raise FileNotFoundError(
            f"ETAPP data root not found at {root}. Clone the public repo under .vendor/ETAPP first."
        )
    return root


def _load_etapp_dataset_inputs(
    *,
    project_root: Path,
    data_root: Path | None,
    variant: str | None,
) -> tuple[Path, list[EtappInstruction], dict[tuple[str, str], EtappExample], dict[str, Any]]:
    if variant is None:
        etapp_root = _resolve_etapp_root(project_root=project_root, data_root=data_root)
        instructions = _load_instructions(etapp_root)
        grouped_examples = _load_grouped_examples(etapp_root, instructions)
        return etapp_root, instructions, grouped_examples, {
            "data_root": str(etapp_root),
            "benchmark_split": ETAPP_BENCHMARK_SPLIT,
        }
    return _load_materialized_etapp_variant(
        project_root=project_root,
        variant=variant,
        data_root=data_root,
    )


def _load_materialized_etapp_variant(
    *,
    project_root: Path,
    variant: str,
    data_root: Path | None,
) -> tuple[Path, list[EtappInstruction], dict[tuple[str, str], EtappExample], dict[str, Any]]:
    from .etapp_expanded import read_expanded_etapp_variant_rows

    payload = read_expanded_etapp_variant_rows(
        project_root=project_root,
        variant=variant,
        data_root=data_root,
    )
    instructions = [
        EtappInstruction(
            instruction_id=int(row["instruction_id"]),
            query=str(row["query"]),
            keypoint_personal=tuple(str(item) for item in row.get("keypoint_personal", [])),
            keypoint_proactive=tuple(str(item) for item in row.get("keypoint_proactive", [])),
            available_tools=tuple(str(item) for item in row.get("available_tools", [])),
            location=str(row.get("location", "")),
        )
        for row in sorted(payload["instructions"], key=lambda item: int(item["instruction_id"]))
    ]
    instruction_by_id = {int(row["instruction_id"]): row for row in payload["instructions"]}
    instruction_object_by_id = {instruction.instruction_id: instruction for instruction in instructions}
    profile_by_id = {str(row["profile_id"]): row for row in payload["profiles"]}

    grouped_examples: dict[tuple[str, str], EtappExample] = {}
    for row in payload["examples"]:
        instruction_id = int(row["instruction_id"])
        profile_row = profile_by_id[str(row["profile_id"])]
        instruction = instruction_object_by_id[instruction_id]
        grouped_examples[(instruction.query, str(profile_row["user_name"]))] = EtappExample(
            instruction=instruction,
            user_name=str(profile_row["user_name"]),
            profile=dict(profile_row.get("basic_profile", {})),
            detailed_preferences=[dict(item) for item in profile_row.get("detailed_preferences", [])],
            status=dict(row.get("status", {})),
            action_sequence=tuple(dict(item) for item in row.get("action_sequence", [])),
        )

    manifest = dict(payload["manifest"])
    return Path(payload["root"]), instructions, grouped_examples, {
        "data_root": str(payload["root"]),
        "benchmark_split": str(manifest.get("benchmark_split", f"local_{variant}")),
        "variant": variant,
        "variant_kind": str(manifest.get("kind", "local_expanded_benchmark")),
        "original_instruction_count": int(manifest.get("original_instruction_count", 0) or 0),
        "original_profile_count": int(manifest.get("original_profile_count", 0) or 0),
        "original_example_count": int(manifest.get("original_example_count", 0) or 0),
        "trace_backed_instruction_count": int(manifest.get("trace_backed_instruction_count", 0) or 0),
        "trace_backed_profile_count": int(manifest.get("trace_backed_profile_count", 0) or 0),
        "trace_backed_pair_count": int(manifest.get("trace_backed_pair_count", 0) or 0),
        "pair_grid_count": int(manifest.get("pair_grid_count", 0) or 0),
        "actual_example_count": int(manifest.get("actual_example_count", 0) or 0),
        "instruction_variant_count": len({
            str(row.get("augmentation_rule", "")) for row in payload["instructions"]
        }),
        "profile_variant_count": len({
            str(row.get("augmentation_rule", "")) for row in payload["profiles"]
        }),
        "manifest_summary": {
            "instruction_count": len(instruction_by_id),
            "profile_count": len(profile_by_id),
            "quick_subset_count": len(payload["quick_subset"]),
        },
    }


def _load_instructions(etapp_root: Path) -> list[EtappInstruction]:
    instruction_path = etapp_root / "data" / "instruction" / "instruction.json"
    raw_items = json.loads(instruction_path.read_text(encoding="utf-8"))
    return [
        EtappInstruction(
            instruction_id=index,
            query=item["query"],
            keypoint_personal=tuple(item.get("keypoint for personal", [])),
            keypoint_proactive=tuple(item.get("keypoint for proactive", [])),
            available_tools=tuple(item.get("available_tools_name", [])),
            location=str(item.get("location", "")),
        )
        for index, item in enumerate(raw_items)
    ]


def _load_grouped_examples(
    etapp_root: Path,
    instructions: list[EtappInstruction],
) -> dict[tuple[str, str], EtappExample]:
    instruction_by_query = {instruction.query: instruction for instruction in instructions}
    training_path = etapp_root / "data" / "training_data" / "training_data_qwen_format_fc.json"
    raw_items = json.loads(training_path.read_text(encoding="utf-8"))

    grouped_records: dict[tuple[str, str], list[EtappTraceRecord]] = defaultdict(list)
    for raw_item in raw_items:
        record = _parse_trace_record(raw_item)
        if record is None or record.query not in instruction_by_query:
            continue
        grouped_records[(record.query, record.user_name)].append(record)

    grouped_examples: dict[tuple[str, str], EtappExample] = {}
    for key, records in grouped_records.items():
        sequence = _dedupe_action_sequence(records)
        if not sequence:
            continue
        first = records[0]
        grouped_examples[key] = EtappExample(
            instruction=instruction_by_query[first.query],
            user_name=first.user_name,
            profile=first.profile,
            detailed_preferences=first.detailed_preferences,
            status=first.status,
            action_sequence=sequence,
        )
    return grouped_examples


def _parse_trace_record(raw_item: dict[str, Any]) -> EtappTraceRecord | None:
    input_text = str(raw_item.get("input", ""))
    output_text = str(raw_item.get("output", ""))
    query = _first_real_user_turn(input_text)
    if query is None:
        return None

    profile_match = _PROFILE_RE.search(input_text)
    preferences_match = _PREFERENCES_RE.search(input_text)
    status_match = _STATUS_RE.search(input_text)
    tool_call_match = _TOOL_CALL_RE.search(output_text)
    if not (profile_match and preferences_match and status_match and tool_call_match):
        return None

    profile = ast.literal_eval(profile_match.group(1).strip())
    preferences = ast.literal_eval(preferences_match.group(1).strip())
    status = ast.literal_eval(status_match.group(1).strip())
    call = ast.literal_eval(tool_call_match.group(1).strip())
    basic_info = profile.get("DemographicData", {}).get("BasicInformation", {})
    user_name = str(basic_info.get("Name", "")).strip()
    if not user_name:
        return None

    return EtappTraceRecord(
        query=query,
        user_name=user_name,
        profile=profile,
        detailed_preferences=list(preferences),
        status=status,
        call_name=str(call.get("name", "")).strip(),
        call_args=_parse_arguments(call.get("arguments", {})),
        tool_response_count=sum(
            1
            for user_turn in _USER_TURN_RE.findall(input_text)
            if user_turn.strip().startswith("<tool_response>")
        ),
    )


def _first_real_user_turn(input_text: str) -> str | None:
    for user_turn in _USER_TURN_RE.findall(input_text):
        text = user_turn.strip()
        if text.startswith("<tool_response>"):
            continue
        return text
    return None


def _parse_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return _normalize_value(value)
    if not isinstance(value, str):
        return {"value": value}

    stripped = value.strip()
    if not stripped:
        return {}
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(stripped)
            if isinstance(parsed, dict):
                return _normalize_value(parsed)
            return {"value": _normalize_value(parsed)}
        except Exception:
            continue
    return {"raw": stripped}


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _dedupe_action_sequence(records: list[EtappTraceRecord]) -> tuple[dict[str, Any], ...]:
    sequence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in sorted(
        records,
        key=lambda item: (
            item.tool_response_count,
            item.call_name,
            json.dumps(item.call_args, sort_keys=True, ensure_ascii=False),
        ),
    ):
        payload = {
            "tool_name": record.call_name,
            "normalized_args": record.call_args,
        }
        marker = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        if marker in seen:
            continue
        seen.add(marker)
        sequence.append(payload)
    return tuple(sequence)


def _sequence_component(item: dict[str, Any]) -> dict[str, Any]:
    normalized_args = _normalize_value(item.get("normalized_args", {}))
    return {
        "tool_name": str(item.get("tool_name", "")),
        "normalized_args": normalized_args,
        "key_decision_fields": {
            key: normalized_args[key]
            for key in _DECISION_KEYS
            if isinstance(normalized_args, dict) and key in normalized_args
        },
    }


def _select_queries(
    instructions: list[EtappInstruction],
    grouped_examples: dict[tuple[str, str], EtappExample],
    *,
    max_queries: int,
    task_ids: Sequence[str] | None = None,
) -> list[str]:
    supported_queries = {query for query, _ in grouped_examples}
    if task_ids:
        query_by_task_id = {
            instruction.task_id: instruction.query
            for instruction in instructions
            if instruction.query in supported_queries
        }
        missing = [task_id for task_id in task_ids if task_id not in query_by_task_id]
        if missing:
            raise ValueError(f"ETAPP smoke requested unsupported task ids: {missing}.")
        return [query_by_task_id[task_id] for task_id in task_ids]
    selected = [instruction.query for instruction in instructions if instruction.query in supported_queries]
    if len(selected) < max_queries:
        raise ValueError(
            f"ETAPP smoke requires at least {max_queries} supported queries, found {len(selected)}."
        )
    return selected[:max_queries]


def _select_users(
    selected_queries: list[str],
    grouped_examples: dict[tuple[str, str], EtappExample],
    *,
    max_users: int,
) -> list[str]:
    per_query_users = [
        {user_name for query, user_name in grouped_examples if query == selected_query}
        for selected_query in selected_queries
    ]
    common_users = sorted(set.intersection(*per_query_users)) if per_query_users else []
    if len(common_users) < max_users:
        raise ValueError(
            f"ETAPP smoke requires at least {max_users} users that appear in every selected query, found {len(common_users)}."
        )
    return common_users[:max_users]


def _primary_tool_name(example: EtappExample) -> str:
    if example.action_sequence:
        return str(example.action_sequence[0].get("tool_name", "")).strip() or "respond_without_tool"
    if example.instruction.available_tools:
        return str(example.instruction.available_tools[0]).strip() or "respond_without_tool"
    return "respond_without_tool"


def _action_type(example: EtappExample) -> str:
    return "single_tool" if len(example.action_sequence) <= 1 else "multi_tool"


def _proactivity_required(example: EtappExample) -> bool:
    return bool(example.instruction.keypoint_proactive) or len(example.action_sequence) > 1


def _personalization_constraint_type(example: EtappExample) -> str:
    text = " ".join(example.instruction.keypoint_personal).lower()
    keyword_groups = (
        ("travel_preference", ("travel", "flight", "accommodation", "attraction", "trip")),
        ("dining_preference", ("restaurant", "cuisine", "dietary", "food", "meal")),
        ("wellness_preference", ("workout", "health", "mood", "wellness", "exercise")),
        ("smart_home_preference", ("temperature", "humidity", "lighting", "bathtub", "water", "home")),
        ("music_preference", ("music", "artist", "genre", "volume", "listening")),
        ("schedule_constraint", ("calendar", "schedule", "alarm", "time slot", "today")),
        ("location_context", ("location", "city", "nearby", "weather")),
        ("communication_priority", ("email", "contact", "priority", "mail")),
    )
    for label, keywords in keyword_groups:
        if any(keyword in text for keyword in keywords):
            return label
    return "general_profile"


def _history_length_estimate(example: EtappExample) -> int:
    basic_info = example.profile.get("DemographicData", {}).get("BasicInformation", {})
    profile_text = (
        f"{example.user_name} {basic_info.get('Location', '')} {basic_info.get('Occupation', '')} "
        f"{basic_info.get('Character', '')}"
    )
    status_text = json.dumps(example.status, sort_keys=True, ensure_ascii=False)
    preference_text = _summarize_preference_blob(example.detailed_preferences)
    return _approximate_token_count(
        "\n".join(
            [
                profile_text,
                preference_text,
                status_text,
                example.instruction.query,
                " ".join(example.instruction.keypoint_personal),
            ]
        )
    )


def _history_length_bin(example: EtappExample) -> str:
    estimate = _history_length_estimate(example)
    if estimate < 220:
        return "short"
    if estimate < 340:
        return "medium"
    return "long"


def _personalization_strength_score(example: EtappExample) -> float:
    preference_item_count = len(_flatten_items(example.detailed_preferences))
    raw_score = len(example.instruction.keypoint_personal) + min(3, preference_item_count / 4.0)
    return round(min(raw_score, 6.0), 3)


def _personalization_strength_bin(score: float) -> str:
    if score >= 4.0:
        return "high"
    if score >= 2.0:
        return "medium"
    return "low"


def _difficulty(example: EtappExample) -> str:
    difficulty_score = len(example.action_sequence)
    if _proactivity_required(example):
        difficulty_score += 1
    if len(example.instruction.available_tools) >= 5:
        difficulty_score += 1
    if _personalization_strength_bin(_personalization_strength_score(example)) == "high":
        difficulty_score += 1
    if difficulty_score <= 2:
        return "easy"
    if difficulty_score <= 5:
        return "medium"
    return "hard"


def _dialogue_domain(example: EtappExample) -> str:
    tool_text = " ".join(example.instruction.available_tools).lower()
    query_text = example.instruction.query.lower()
    combined = f"{tool_text} {query_text}"
    domain_keywords = (
        ("travel", ("flight", "accommodation", "attraction", "travel", "trip")),
        ("dining", ("restaurant", "cuisine", "meal", "lunch", "dinner")),
        ("fitness", ("workout", "health", "mood", "exercise", "walk")),
        ("smart_home", ("home", "light", "temperature", "humidity", "bathtub", "water")),
        ("music", ("music", "song", "artist", "playlist")),
        ("weather", ("weather",)),
        ("calendar", ("calendar", "schedule", "event", "alarm")),
        ("communication", ("email", "mail")),
        ("shopping", ("shopping", "product")),
        ("news", ("news",)),
    )
    for domain, keywords in domain_keywords:
        if any(keyword in combined for keyword in keywords):
            return domain
    return "general_assistant"


def _build_user_record(example: EtappExample, benchmark: str) -> UserRecord:
    basic_info = example.profile.get("DemographicData", {}).get("BasicInformation", {})
    return UserRecord(
        user_id=example.user_id,
        profile={
            "basic_profile": example.profile,
            "detailed_preferences": example.detailed_preferences,
        },
        metadata={
            "benchmark": benchmark,
            "etapp_user_name": example.user_name,
            "home_location": basic_info.get("Location"),
        },
    )


def _build_task_record(example: EtappExample, benchmark: str) -> TaskRecord:
    personalization_strength_score = _personalization_strength_score(example)
    return TaskRecord(
        user_id=example.user_id,
        task_id=example.instruction.task_id,
        benchmark=benchmark,
        task_type="action",
        prompt=example.instruction.query,
        gold_label=example.action_signature,
        metadata={
            "available_tools": list(example.instruction.available_tools),
            "keypoint_personal": list(example.instruction.keypoint_personal),
            "keypoint_proactive": list(example.instruction.keypoint_proactive),
            "action_sequence": list(example.action_sequence),
            "fallback_prediction": _fallback_action_signature(example),
            "search_terms": _search_terms(example),
            "evaluation": "offline_action_sequence_exact_match",
            "status": dict(example.status),
            "user_name": example.user_name,
            "benchmark_split": ETAPP_BENCHMARK_SPLIT,
            "tool_name": _primary_tool_name(example),
            "action_type": _action_type(example),
            "proactivity_required": _proactivity_required(example),
            "personalization_constraint_type": _personalization_constraint_type(example),
            "history_length_estimate": _history_length_estimate(example),
            "history_length_bin": _history_length_bin(example),
            "personalization_strength_score": personalization_strength_score,
            "personalization_strength_bin": _personalization_strength_bin(personalization_strength_score),
            "difficulty": _difficulty(example),
            "dialogue_domain": _dialogue_domain(example),
        },
    )


def _fallback_action_signature(example: EtappExample) -> str:
    return build_etapp_fallback_action_signature(
        task_id=example.instruction.task_id,
        available_tools=example.instruction.available_tools,
        used_tools=(item["tool_name"] for item in example.action_sequence),
    )


def _search_terms(example: EtappExample) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for source in (example.instruction.query, " ".join(example.instruction.available_tools)):
        for token in _QUERY_TOKEN_RE.findall(source.lower()):
            if len(token) <= 2 or token in seen:
                continue
            seen.add(token)
            terms.append(token)
    for action in example.action_sequence:
        for value in action.get("normalized_args", {}).values():
            if not isinstance(value, str):
                continue
            for token in _QUERY_TOKEN_RE.findall(value.lower()):
                if len(token) <= 2 or token in seen:
                    continue
                seen.add(token)
                terms.append(token)
    return terms[:10]


def _summarize_preference_blob(detailed_preferences: list[dict[str, Any]]) -> str:
    fragments: list[str] = []
    for preference_group in detailed_preferences:
        for path, value in _flatten_items(preference_group):
            if not isinstance(value, str):
                continue
            fragments.append(f"{path}: {value}")
            if len(fragments) >= 6:
                return "Preference summary: " + "; ".join(fragments)
    if not fragments:
        return "Preference summary: no detailed ETAPP preferences provided."
    return "Preference summary: " + "; ".join(fragments)


def _flatten_items(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.extend(_flatten_items(value[key], child_prefix))
        return items
    if isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            items.extend(_flatten_items(child, child_prefix))
        return items
    items.append((prefix, value))
    return items


def _build_swap_user_map(user_ids: Any) -> dict[str, str]:
    ordered = list(user_ids)
    if len(ordered) < 2:
        return {}
    if len(ordered) == 2:
        return {ordered[0]: ordered[1], ordered[1]: ordered[0]}

    swap_map: dict[str, str] = {}
    for index in range(0, len(ordered) - 1, 2):
        left = ordered[index]
        right = ordered[index + 1]
        swap_map[left] = right
        swap_map[right] = left
    if len(ordered) % 2 == 1:
        swap_map[ordered[-1]] = ordered[0]
    return swap_map
