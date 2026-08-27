from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .io import read_json, read_jsonl, write_json, write_jsonl


EXPANDED_ETAPP_VARIANT = "expanded_v1"
EXPANDED_ETAPP_BENCHMARK = "ETAPP"
EXPANDED_ETAPP_SOURCE_ROOT = Path("data/benchmarks/ETAPP")
EXPANDED_ETAPP_OUTPUT_ROOT = Path("data/interim/benchmarks/ETAPP")
EXPANDED_ETAPP_BENCHMARK_SPLIT = "local_expanded_v1"
EXPANDED_ETAPP_TARGET_INSTRUCTION_COUNT = 150
EXPANDED_ETAPP_TARGET_PROFILE_COUNT = 32
EXPANDED_ETAPP_TARGET_PAIR_COUNT = (
    EXPANDED_ETAPP_TARGET_INSTRUCTION_COUNT * EXPANDED_ETAPP_TARGET_PROFILE_COUNT
)
EXPANDED_ETAPP_QUICK_SUBSET_PROFILE_STEP = 11
_SLUG_RE = re.compile(r"[^a-z0-9]+")

_INSTRUCTION_VARIANT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "rule": "official_identity",
        "is_augmented": False,
        "mode": "identity",
    },
    {
        "rule": "preferences_prefixed",
        "is_augmented": True,
        "mode": "prefix",
        "text": "Please handle this in a way that fits my usual preferences: ",
    },
    {
        "rule": "routine_prefixed",
        "is_augmented": True,
        "mode": "prefix",
        "text": "Keeping my normal routine in mind, please help with this: ",
    },
    {
        "rule": "proactive_prefixed",
        "is_augmented": True,
        "mode": "prefix",
        "text": "Please be proactive where it helps while handling this: ",
    },
    {
        "rule": "preferences_suffixed",
        "is_augmented": True,
        "mode": "suffix",
        "text": "Please keep my usual preferences in mind.",
    },
    {
        "rule": "constraints_suffixed",
        "is_augmented": True,
        "mode": "suffix",
        "text": "Please keep my regular preferences and schedule constraints in mind.",
    },
)

_PROFILE_VARIANT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "rule": "official_identity",
        "is_augmented": False,
        "name_suffix": "",
        "character_suffix": "",
        "interest_suffix": "",
        "hobby_suffix": "",
    },
    {
        "rule": "biography_refresh_harbor",
        "is_augmented": True,
        "name_suffix": "Harbor",
        "character_suffix": "Keeps a very similar routine but plans errands slightly earlier in the day.",
        "interest_suffix": "Also enjoys quiet sketching sessions on slower weekends.",
        "hobby_suffix": "Adds quiet sketching and short neighborhood walks.",
    },
    {
        "rule": "biography_refresh_grove",
        "is_augmented": True,
        "name_suffix": "Grove",
        "character_suffix": "Keeps the same preferences but prefers a calmer pace after work.",
        "interest_suffix": "Also likes low-key reading breaks between errands.",
        "hobby_suffix": "Adds low-key reading breaks and calm evening routines.",
    },
    {
        "rule": "biography_refresh_summit",
        "is_augmented": True,
        "name_suffix": "Summit",
        "character_suffix": "Keeps the same preferences while planning a bit more proactively for the next day.",
        "interest_suffix": "Also likes organizing the next day before winding down.",
        "hobby_suffix": "Adds next-day planning and quiet home organization.",
    },
)


@dataclass(frozen=True, slots=True)
class ExpandedEtappArtifacts:
    variant: str
    output_root: Path
    instructions_path: Path
    profiles_path: Path
    examples_path: Path
    manifest_path: Path
    quick_subset_path: Path
    instruction_count: int
    profile_count: int
    example_count: int
    quick_subset_count: int


def resolve_expanded_etapp_variant_root(
    *,
    project_root: Path,
    variant: str = EXPANDED_ETAPP_VARIANT,
    data_root: Path | None = None,
) -> Path:
    return data_root or (project_root / EXPANDED_ETAPP_OUTPUT_ROOT / variant)


def read_expanded_etapp_variant_rows(
    *,
    project_root: Path,
    variant: str = EXPANDED_ETAPP_VARIANT,
    data_root: Path | None = None,
) -> dict[str, Any]:
    root = resolve_expanded_etapp_variant_root(
        project_root=project_root,
        variant=variant,
        data_root=data_root,
    )
    instructions_path = root / "instructions.jsonl"
    profiles_path = root / "profiles.jsonl"
    examples_path = root / "examples.jsonl"
    manifest_path = root / "manifest.json"
    quick_subset_path = root / "quick_subset.jsonl"

    required_paths = (
        instructions_path,
        profiles_path,
        examples_path,
        manifest_path,
        quick_subset_path,
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Expanded ETAPP variant '{variant}' is incomplete under {root}: missing {missing}."
        )

    manifest = read_json(manifest_path)
    return {
        "root": root,
        "instructions": read_jsonl(instructions_path),
        "profiles": read_jsonl(profiles_path),
        "examples": read_jsonl(examples_path),
        "quick_subset": read_jsonl(quick_subset_path),
        "manifest": manifest,
    }


def build_expanded_etapp_artifacts(
    *,
    project_root: Path,
    variant: str = EXPANDED_ETAPP_VARIANT,
    output_root: Path | None = None,
) -> ExpandedEtappArtifacts:
    from .etapp import build_etapp_full_dataset

    resolved_output_root = resolve_expanded_etapp_variant_root(
        project_root=project_root,
        variant=variant,
        data_root=output_root,
    )
    source_root = project_root / EXPANDED_ETAPP_SOURCE_ROOT
    if not source_root.exists():
        raise FileNotFoundError(
            f"ETAPP benchmark source root not found at {source_root}."
        )

    source_dataset = build_etapp_full_dataset(
        project_root=project_root,
        benchmark=EXPANDED_ETAPP_BENCHMARK,
        data_root=source_root,
    )
    source_instructions = _extract_source_instructions(source_dataset)
    source_profiles = _extract_source_profiles(source_dataset)
    source_pairs = {
        (user_id, task_id): example
        for (user_id, task_id), example in source_dataset.examples_by_user_task.items()
    }
    source_task_metadata = {
        (task.user_id, task.task_id): dict(task.metadata)
        for task in source_dataset.tasks
    }
    _validate_source_grid(
        source_instructions=source_instructions,
        source_profiles=source_profiles,
        source_pairs=source_pairs,
    )

    instructions = _build_instruction_rows(source_instructions)
    profiles = _build_profile_rows(source_profiles)
    examples = _build_example_rows(
        instructions=instructions,
        profiles=profiles,
        source_pairs=source_pairs,
        source_task_metadata=source_task_metadata,
    )
    quick_subset, quick_subset_stats = _build_quick_subset(
        instructions=instructions,
        profiles=profiles,
        examples=examples,
    )

    instructions_path = resolved_output_root / "instructions.jsonl"
    profiles_path = resolved_output_root / "profiles.jsonl"
    examples_path = resolved_output_root / "examples.jsonl"
    quick_subset_path = resolved_output_root / "quick_subset.jsonl"
    manifest_path = resolved_output_root / "manifest.json"

    write_jsonl(instructions_path, instructions)
    write_jsonl(profiles_path, profiles)
    write_jsonl(examples_path, examples)
    write_jsonl(quick_subset_path, quick_subset)

    manifest = _build_manifest(
        project_root=project_root,
        variant=variant,
        output_root=resolved_output_root,
        source_root=source_root,
        source_dataset=source_dataset,
        instructions=instructions,
        profiles=profiles,
        examples=examples,
        quick_subset=quick_subset,
        quick_subset_stats=quick_subset_stats,
    )
    write_json(manifest_path, manifest)

    return ExpandedEtappArtifacts(
        variant=variant,
        output_root=resolved_output_root,
        instructions_path=instructions_path,
        profiles_path=profiles_path,
        examples_path=examples_path,
        manifest_path=manifest_path,
        quick_subset_path=quick_subset_path,
        instruction_count=len(instructions),
        profile_count=len(profiles),
        example_count=len(examples),
        quick_subset_count=len(quick_subset),
    )


def _extract_source_instructions(source_dataset: Any) -> list[dict[str, Any]]:
    first_example_by_task: dict[str, Any] = {}
    for example in sorted(
        source_dataset.examples_by_user_task.values(),
        key=lambda item: (item.instruction.instruction_id, item.user_name.lower()),
    ):
        first_example_by_task.setdefault(example.instruction.task_id, example)

    rows = [
        {
            "source_task_id": example.instruction.task_id,
            "source_instruction_id": int(example.instruction.instruction_id),
            "query": example.instruction.query,
            "available_tools": list(example.instruction.available_tools),
            "location": example.instruction.location,
            "keypoint_personal": list(example.instruction.keypoint_personal),
            "keypoint_proactive": list(example.instruction.keypoint_proactive),
        }
        for example in sorted(
            first_example_by_task.values(),
            key=lambda item: (item.instruction.instruction_id, item.user_name.lower()),
        )
    ]
    if not rows:
        raise ValueError("No trace-backed ETAPP instructions were found in the local benchmark package.")
    return rows


def _extract_source_profiles(source_dataset: Any) -> list[dict[str, Any]]:
    first_example_by_user: dict[str, Any] = {}
    for example in sorted(
        source_dataset.examples_by_user_task.values(),
        key=lambda item: (item.user_id, item.instruction.instruction_id),
    ):
        first_example_by_user.setdefault(example.user_id, example)

    profiles: list[dict[str, Any]] = []
    for user in source_dataset.users:
        example = first_example_by_user.get(user.user_id)
        if example is None:
            continue
        profiles.append(
            {
                "source_user_id": user.user_id,
                "source_user_name": example.user_name,
                "basic_profile": deepcopy(user.profile.get("basic_profile", {})),
                "detailed_preferences": deepcopy(user.profile.get("detailed_preferences", [])),
                "home_location": user.metadata.get("home_location"),
            }
        )

    if not profiles:
        raise ValueError("No trace-backed ETAPP profiles were found in the local benchmark package.")
    return profiles


def _validate_source_grid(
    *,
    source_instructions: list[dict[str, Any]],
    source_profiles: list[dict[str, Any]],
    source_pairs: Mapping[tuple[str, str], Any],
) -> None:
    expected_instruction_variants = (
        EXPANDED_ETAPP_TARGET_INSTRUCTION_COUNT // len(source_instructions)
        if source_instructions
        else 0
    )
    expected_profile_variants = (
        EXPANDED_ETAPP_TARGET_PROFILE_COUNT // len(source_profiles)
        if source_profiles
        else 0
    )
    if expected_instruction_variants != len(_INSTRUCTION_VARIANT_SPECS):
        raise ValueError(
            "Expanded ETAPP instruction target no longer matches the configured variant rules."
        )
    if expected_profile_variants != len(_PROFILE_VARIANT_SPECS):
        raise ValueError(
            "Expanded ETAPP profile target no longer matches the configured variant rules."
        )

    missing_pairs = []
    for instruction in source_instructions:
        for profile in source_profiles:
            pair_key = (profile["source_user_id"], instruction["source_task_id"])
            if pair_key not in source_pairs:
                missing_pairs.append(pair_key)
    if missing_pairs:
        raise ValueError(
            f"Trace-backed ETAPP source grid is incomplete; missing pairs such as {missing_pairs[:5]}."
        )


def _build_instruction_rows(source_instructions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    for source_index, source_row in enumerate(source_instructions):
        for variant_index, spec in enumerate(_INSTRUCTION_VARIANT_SPECS):
            instruction_id = len(rows)
            query = _apply_instruction_variant(str(source_row["query"]), spec)
            if query in seen_queries:
                raise ValueError(f"Expanded ETAPP query collision detected for: {query}")
            seen_queries.add(query)
            task_id = _task_id_for_instruction(instruction_id, query)
            rows.append(
                {
                    "instruction_id": instruction_id,
                    "task_id": task_id,
                    "query": query,
                    "available_tools": list(source_row["available_tools"]),
                    "location": str(source_row["location"]),
                    "keypoint_personal": list(source_row["keypoint_personal"]),
                    "keypoint_proactive": list(source_row["keypoint_proactive"]),
                    "benchmark_split": EXPANDED_ETAPP_BENCHMARK_SPLIT,
                    "source_task_id": str(source_row["source_task_id"]),
                    "source_instruction_id": int(source_row["source_instruction_id"]),
                    "source_instruction_query": str(source_row["query"]),
                    "source_instruction_position": source_index,
                    "variant_index": variant_index,
                    "is_augmented": bool(spec["is_augmented"]),
                    "augmentation_rule": str(spec["rule"]),
                    "augmentation_source": {
                        "source_task_id": str(source_row["source_task_id"]),
                        "source_instruction_query": str(source_row["query"]),
                        "rule": str(spec["rule"]),
                    },
                }
            )
    if len(rows) != EXPANDED_ETAPP_TARGET_INSTRUCTION_COUNT:
        raise ValueError(
            f"Expanded ETAPP generated {len(rows)} instructions, expected {EXPANDED_ETAPP_TARGET_INSTRUCTION_COUNT}."
        )
    return rows


def _build_profile_rows(source_profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_user_ids: set[str] = set()
    for source_index, source_row in enumerate(source_profiles):
        for variant_index, spec in enumerate(_PROFILE_VARIANT_SPECS):
            basic_profile = deepcopy(source_row["basic_profile"])
            detailed_preferences = deepcopy(source_row["detailed_preferences"])
            user_name = _build_profile_user_name(
                source_name=str(source_row["source_user_name"]),
                spec=spec,
            )
            modified_fields = _apply_profile_variant(
                basic_profile=basic_profile,
                source_name=str(source_row["source_user_name"]),
                user_name=user_name,
                spec=spec,
            )
            user_id = _user_id_for_name(user_name)
            if user_id in seen_user_ids:
                raise ValueError(f"Expanded ETAPP user_id collision detected for: {user_id}")
            seen_user_ids.add(user_id)
            profile_id = f"{user_id}__v{variant_index}"
            rows.append(
                {
                    "profile_id": profile_id,
                    "user_id": user_id,
                    "user_name": user_name,
                    "basic_profile": basic_profile,
                    "detailed_preferences": detailed_preferences,
                    "benchmark_split": EXPANDED_ETAPP_BENCHMARK_SPLIT,
                    "source_user_id": str(source_row["source_user_id"]),
                    "source_user_name": str(source_row["source_user_name"]),
                    "source_profile_position": source_index,
                    "variant_index": variant_index,
                    "is_augmented": bool(spec["is_augmented"]),
                    "augmentation_rule": str(spec["rule"]),
                    "augmentation_source": {
                        "source_user_id": str(source_row["source_user_id"]),
                        "source_user_name": str(source_row["source_user_name"]),
                        "rule": str(spec["rule"]),
                    },
                    "modified_profile_paths": modified_fields,
                    "preserved_action_fields": [
                        "detailed_preferences",
                        "DemographicData.BasicInformation.Location",
                    ],
                }
            )
    if len(rows) != EXPANDED_ETAPP_TARGET_PROFILE_COUNT:
        raise ValueError(
            f"Expanded ETAPP generated {len(rows)} profiles, expected {EXPANDED_ETAPP_TARGET_PROFILE_COUNT}."
        )
    return rows


def _build_example_rows(
    *,
    instructions: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    source_pairs: Mapping[tuple[str, str], Any],
    source_task_metadata: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for instruction in instructions:
        for profile in profiles:
            source_pair_key = (profile["source_user_id"], instruction["source_task_id"])
            source_example = source_pairs[source_pair_key]
            source_metadata = dict(source_task_metadata[source_pair_key])
            action_sequence = [deepcopy(item) for item in source_example.action_sequence]
            action_signature = _canonicalize_action_sequence(
                task_id=str(instruction["task_id"]),
                action_sequence=action_sequence,
                proactive=bool(source_metadata.get("proactivity_required", False)),
            )
            example_id = f"{instruction['task_id']}::{profile['user_id']}"
            rows.append(
                {
                    "example_id": example_id,
                    "instruction_id": int(instruction["instruction_id"]),
                    "profile_id": str(profile["profile_id"]),
                    "task_id": str(instruction["task_id"]),
                    "user_id": str(profile["user_id"]),
                    "query": str(instruction["query"]),
                    "status": deepcopy(source_example.status),
                    "available_tools": list(instruction["available_tools"]),
                    "action_sequence": action_sequence,
                    "action_signature": action_signature,
                    "primary_tool_name": str(source_metadata.get("tool_name", "respond_without_tool")),
                    "primary_tool_category": _primary_tool_category(source_metadata),
                    "action_type": str(source_metadata.get("action_type", "unknown")),
                    "dialogue_domain": str(source_metadata.get("dialogue_domain", "general_assistant")),
                    "personalization_constraint_type": str(
                        source_metadata.get("personalization_constraint_type", "general_profile")
                    ),
                    "history_length_bin": str(source_metadata.get("history_length_bin", "unknown")),
                    "personalization_strength_bin": str(
                        source_metadata.get("personalization_strength_bin", "low")
                    ),
                    "difficulty": str(source_metadata.get("difficulty", "easy")),
                    "benchmark": EXPANDED_ETAPP_BENCHMARK,
                    "benchmark_split": EXPANDED_ETAPP_BENCHMARK_SPLIT,
                    "source_task_id": str(instruction["source_task_id"]),
                    "source_user_id": str(profile["source_user_id"]),
                    "source_pair_id": f"{instruction['source_task_id']}::{profile['source_user_id']}",
                    "is_augmented": bool(instruction["is_augmented"] or profile["is_augmented"]),
                    "augmentation_source": {
                        "instruction_rule": str(instruction["augmentation_rule"]),
                        "profile_rule": str(profile["augmentation_rule"]),
                        "source_task_id": str(instruction["source_task_id"]),
                        "source_user_id": str(profile["source_user_id"]),
                    },
                }
            )
    if len(rows) != EXPANDED_ETAPP_TARGET_PAIR_COUNT:
        raise ValueError(
            f"Expanded ETAPP generated {len(rows)} examples, expected {EXPANDED_ETAPP_TARGET_PAIR_COUNT}."
        )
    return rows


def _build_quick_subset(
    *,
    instructions: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    examples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    example_by_pair = {
        (int(row["instruction_id"]), str(row["profile_id"])): row
        for row in examples
    }
    selected_examples: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    for instruction in instructions:
        profile_index = (
            int(instruction["instruction_id"]) * EXPANDED_ETAPP_QUICK_SUBSET_PROFILE_STEP + 7
        ) % len(profiles)
        profile = profiles[profile_index]
        row = example_by_pair[(int(instruction["instruction_id"]), str(profile["profile_id"]))]
        selected_examples.append(row)
        selected_ids.add(str(row["example_id"]))

    full_tool_categories = sorted({str(row["primary_tool_category"]) for row in examples})
    full_action_types = sorted({str(row["action_type"]) for row in examples})
    selected_tool_categories = {str(row["primary_tool_category"]) for row in selected_examples}
    selected_action_types = {str(row["action_type"]) for row in selected_examples}

    for category in full_tool_categories:
        if category in selected_tool_categories:
            continue
        candidate = next(row for row in examples if str(row["primary_tool_category"]) == category)
        if str(candidate["example_id"]) not in selected_ids:
            selected_examples.append(candidate)
            selected_ids.add(str(candidate["example_id"]))
            selected_tool_categories.add(category)

    for action_type in full_action_types:
        if action_type in selected_action_types:
            continue
        candidate = next(row for row in examples if str(row["action_type"]) == action_type)
        if str(candidate["example_id"]) not in selected_ids:
            selected_examples.append(candidate)
            selected_ids.add(str(candidate["example_id"]))
            selected_action_types.add(action_type)

    quick_subset = [
        {
            "subset_rank": index,
            "example_id": str(row["example_id"]),
            "instruction_id": int(row["instruction_id"]),
            "profile_id": str(row["profile_id"]),
            "task_id": str(row["task_id"]),
            "user_id": str(row["user_id"]),
            "primary_tool_category": str(row["primary_tool_category"]),
            "action_type": str(row["action_type"]),
            "selection_rule": "instruction_complete_with_coprime_profile_rotation",
            "selection_metadata": {
                "source_task_id": str(row["source_task_id"]),
                "source_user_id": str(row["source_user_id"]),
            },
        }
        for index, row in enumerate(selected_examples)
    ]
    stats = {
        "quick_subset_count": len(quick_subset),
        "instruction_coverage": len({row["instruction_id"] for row in quick_subset}),
        "profile_coverage": len({row["profile_id"] for row in quick_subset}),
        "tool_categories": sorted({row["primary_tool_category"] for row in quick_subset}),
        "action_types": sorted({row["action_type"] for row in quick_subset}),
        "selection_rule": "instruction_complete_with_coprime_profile_rotation",
        "profile_rotation_step": EXPANDED_ETAPP_QUICK_SUBSET_PROFILE_STEP,
    }
    return quick_subset, stats


def _build_manifest(
    *,
    project_root: Path,
    variant: str,
    output_root: Path,
    source_root: Path,
    source_dataset: Any,
    instructions: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    quick_subset: list[dict[str, Any]],
    quick_subset_stats: Mapping[str, Any],
) -> dict[str, Any]:
    source_dataset_manifest = read_json(source_root / "dataset_manifest.json")
    source_data_manifest = read_json(source_root / "data_manifest.json")
    official_profile_count = len(list((source_root / "profile" / "concrete_profile").glob("profile_*.json")))
    trace_backed_instruction_count = len({str(row["source_task_id"]) for row in instructions})
    trace_backed_profile_count = len({str(row["source_user_id"]) for row in profiles})
    primary_tool_categories = sorted({str(row["primary_tool_category"]) for row in examples})
    action_types = sorted({str(row["action_type"]) for row in examples})
    instruction_rule_counts = _count_values(row["augmentation_rule"] for row in instructions)
    profile_rule_counts = _count_values(row["augmentation_rule"] for row in profiles)
    example_rule_counts = _count_values(
        f"{row['augmentation_source']['instruction_rule']}+{row['augmentation_source']['profile_rule']}"
        for row in examples
    )
    path_hashes = {
        "instructions_jsonl": _sha256_file(output_root / "instructions.jsonl"),
        "profiles_jsonl": _sha256_file(output_root / "profiles.jsonl"),
        "examples_jsonl": _sha256_file(output_root / "examples.jsonl"),
        "quick_subset_jsonl": _sha256_file(output_root / "quick_subset.jsonl"),
    }
    return {
        "benchmark": EXPANDED_ETAPP_BENCHMARK,
        "variant": variant,
        "kind": "local_expanded_benchmark",
        "benchmark_split": EXPANDED_ETAPP_BENCHMARK_SPLIT,
        "source_repo": source_data_manifest.get("source_repo"),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "official_data_free": bool(source_dataset_manifest.get("official_data_free", True)),
        "augmentation_enabled": True,
        "original_instruction_count": int(source_data_manifest.get("instruction_count", 0)),
        "original_profile_count": official_profile_count,
        "original_example_count": int(source_data_manifest.get("training_example_count", 0)),
        "trace_backed_instruction_count": trace_backed_instruction_count,
        "trace_backed_profile_count": trace_backed_profile_count,
        "trace_backed_pair_count": len(source_dataset.examples_by_user_task),
        "instruction_count": len(instructions),
        "profile_count": len(profiles),
        "target_pair_count": EXPANDED_ETAPP_TARGET_PAIR_COUNT,
        "pair_grid_count": len(instructions) * len(profiles),
        "actual_example_count": len(examples),
        "quick_subset_count": len(quick_subset),
        "source_counts_explanation": (
            "The official ETAPP package ships 50 instruction specs, 16 concrete profile files, and 841 raw training rows, "
            "but the public function-call trace split materialized locally into 25 unique query intents across 8 trace-backed users (200 complete query-user pairs). "
            "The 841 raw rows exceed the 200 pair grid because each multi-tool trajectory contributes multiple trace rows before action-sequence deduplication."
        ),
        "trace_backed_source": {
            "instruction_queries": trace_backed_instruction_count,
            "users": trace_backed_profile_count,
            "pair_grid_complete": True,
        },
        "augmentation_summary": {
            "instruction_rules": instruction_rule_counts,
            "profile_rules": profile_rule_counts,
            "example_rule_pairs": example_rule_counts,
            "instruction_variant_count_per_trace_backed_source": len(_INSTRUCTION_VARIANT_SPECS),
            "profile_variant_count_per_trace_backed_source": len(_PROFILE_VARIANT_SPECS),
        },
        "coverage": {
            "primary_tool_categories": primary_tool_categories,
            "action_types": action_types,
            "source_task_count": len({row["source_task_id"] for row in examples}),
            "source_user_count": len({row["source_user_id"] for row in examples}),
        },
        "quick_subset": dict(quick_subset_stats),
        "source_hashes": {
            "instruction_sha256": source_data_manifest.get("instruction_sha256"),
            "training_data_sha256": source_data_manifest.get("training_data_sha256"),
        },
        "output_hashes": path_hashes,
        "local_loader_expectation": {
            "variant_argument": variant,
            "default_behavior_unchanged": True,
        },
        "generated_with": {
            "builder": "umpeek.exp1.etapp_expanded.build_expanded_etapp_artifacts",
            "project_root": str(project_root),
        },
    }


def _apply_instruction_variant(query: str, spec: Mapping[str, Any]) -> str:
    stripped = query.strip()
    mode = str(spec.get("mode", "identity"))
    if mode == "identity":
        return stripped
    if mode == "prefix":
        return f"{spec['text']}{stripped}"
    if mode == "suffix":
        return f"{stripped} {spec['text']}"
    raise ValueError(f"Unsupported instruction variant mode: {mode}")


def _build_profile_user_name(*, source_name: str, spec: Mapping[str, Any]) -> str:
    suffix = str(spec.get("name_suffix", "")).strip()
    if not suffix:
        return source_name
    return f"{source_name} {suffix}"


def _apply_profile_variant(
    *,
    basic_profile: dict[str, Any],
    source_name: str,
    user_name: str,
    spec: Mapping[str, Any],
) -> list[str]:
    modified_fields: list[str] = []
    if _set_nested_value(
        basic_profile,
        ("DemographicData", "BasicInformation", "Name"),
        user_name,
    ):
        modified_fields.append("DemographicData.BasicInformation.Name")

    if str(spec.get("character_suffix", "")).strip():
        if _append_nested_text(
            basic_profile,
            ("DemographicData", "BasicInformation", "Character"),
            str(spec["character_suffix"]),
        ):
            modified_fields.append("DemographicData.BasicInformation.Character")

    if str(spec.get("interest_suffix", "")).strip():
        if _append_nested_text(
            basic_profile,
            ("DemographicData", "InterestActivity"),
            str(spec["interest_suffix"]),
        ):
            modified_fields.append("DemographicData.InterestActivity")

    if str(spec.get("hobby_suffix", "")).strip():
        hobby_paths = (
            ("DemographicData", "Preferences", "Lifestyle", "Hobbies"),
            ("DemographicData", "Preferences", "Lifestyle", "DietaryHabits"),
        )
        for hobby_path in hobby_paths:
            if _append_nested_text(basic_profile, hobby_path, str(spec["hobby_suffix"])):
                modified_fields.append(".".join(hobby_path))
                break

    if not modified_fields and user_name != source_name:
        modified_fields.append("user_name")
    return modified_fields


def _primary_tool_category(source_metadata: Mapping[str, Any]) -> str:
    tool_names = [str(tool).lower() for tool in source_metadata.get("available_tools", []) if str(tool).strip()]
    joined = " ".join(tool_names)
    if any(token in joined for token in ("calendar", "alarm", "event")):
        return "calendar"
    if "email" in joined or "mail" in joined:
        return "email"
    if any(token in joined for token in ("health", "workout", "mood")):
        return "health"
    if "music" in joined:
        return "music"
    if any(token in joined for token in ("shopping", "cart", "product")):
        return "shopping"
    if any(token in joined for token in ("weather",)):
        return "weather"
    if any(token in joined for token in ("home", "light", "temperature", "bathtub", "water")):
        return "smart_home"
    if any(token in joined for token in ("flight", "attraction", "accommodation", "restaurant", "navigation")):
        return "travel_and_local"
    return "general_assistant"


def _task_id_for_instruction(instruction_id: int, query: str) -> str:
    slug = _SLUG_RE.sub("_", query.lower()).strip("_")
    return f"etapp_{instruction_id:02d}_{slug[:32].rstrip('_')}"


def _user_id_for_name(user_name: str) -> str:
    slug = _SLUG_RE.sub("_", user_name.lower()).strip("_")
    return f"etapp_{slug}"


def _canonicalize_action_sequence(
    *,
    task_id: str,
    action_sequence: Iterable[Mapping[str, Any]],
    proactive: bool,
) -> str:
    components = [_sequence_component(dict(item)) for item in action_sequence if str(item.get("tool_name", "")).strip()]
    payload = {
        "intent": task_id,
        "proactive": bool(proactive or len(components) > 1),
        "tool_sequence": components,
        "key_decision_fields": components[-1]["key_decision_fields"] if components else {},
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _sequence_component(item: dict[str, Any]) -> dict[str, Any]:
    normalized_args = _normalize_value(item.get("normalized_args", {}))
    decision_keys = (
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
    return {
        "tool_name": str(item.get("tool_name", "")),
        "normalized_args": normalized_args,
        "key_decision_fields": {
            key: normalized_args[key]
            for key in decision_keys
            if isinstance(normalized_args, dict) and key in normalized_args
        },
    }


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _append_nested_text(payload: dict[str, Any], path: tuple[str, ...], suffix: str) -> bool:
    current = _get_nested_value(payload, path)
    if not isinstance(current, str) or not current.strip():
        return False
    separator = "; " if not current.rstrip().endswith(('.', '!', '?')) else " "
    return _set_nested_value(payload, path, f"{current}{separator}{suffix}")


def _get_nested_value(payload: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _set_nested_value(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> bool:
    current: Any = payload
    for key in path[:-1]:
        if not isinstance(current, dict):
            return False
        next_value = current.get(key)
        if not isinstance(next_value, dict):
            return False
        current = next_value
    if not isinstance(current, dict) or path[-1] not in current:
        return False
    current[path[-1]] = value
    return True


def _count_values(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()