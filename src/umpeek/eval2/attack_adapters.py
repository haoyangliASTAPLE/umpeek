from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Mapping, Sequence

from umpeek.attack_baselines import AttackInput, AttackPrediction, NoOpVictimClient
from umpeek.attack_baselines.adapters import build_attack_adapter
from umpeek.attack_baselines.adapters.common import (
    build_prediction_payload,
    estimate_token_count,
    extract_evidence_snippets,
)
from umpeek.attack_baselines.backend_runtime_projection import (
    BACKEND_RUNTIME_PROJECTION_VERSION,
    project_user_model_for_backend,
)
from umpeek.attack_baselines.victim import VictimObservation, VictimTurn

from .attack_registry import Exp2AttackMethodSpec, get_exp2_attack_method
from .costing import summarize_attack_cost
from .matching import atoms_from_user_model
from .schema import AttackCostRound, EXP2_EXPERIMENT_VERSION, METRIC_SCHEMA_VERSION, clone_json


METHOD_STATUS_VALUES = {"ready", "not_applicable", "blocked", "failed"}


@dataclass(frozen=True, slots=True)
class AttackTrajectoryRound:
    round_index: int
    prompt: str
    reconstruction: Any
    recovery_parse_status: str
    cost: AttackCostRound
    response_text: str = ""
    finish_reason: str = "stop"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "prompt": self.prompt,
            "response_text": self.response_text,
            "finish_reason": self.finish_reason,
            "reconstruction": clone_json(self.reconstruction),
            "recovery_parse_status": self.recovery_parse_status,
            "cost": self.cost.to_dict(),
            "metadata": clone_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class AttackTrajectory:
    method: str
    method_status: str
    sample_id: str
    backend: str
    benchmark: str
    prediction_status: str
    final_reconstruction: Any
    recovery_parse_status: str
    rounds: tuple[AttackTrajectoryRound, ...]
    source_adapter: str
    source_paths: tuple[str, ...]
    not_applicable_reason: str | None = None
    blocked_reason: str | None = None
    failed_reason: str | None = None
    curve_mode: str = "adaptive_prefix"
    actual_query_count: int = 0
    core_logic_modified: bool = False
    wrapper_modified: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    experiment_version: str = EXP2_EXPERIMENT_VERSION
    metric_schema_version: str = METRIC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.method_status not in METHOD_STATUS_VALUES:
            raise ValueError(f"Unsupported method_status: {self.method_status}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_version": self.experiment_version,
            "metric_schema_version": self.metric_schema_version,
            "method": self.method,
            "method_status": self.method_status,
            "sample_id": self.sample_id,
            "backend": self.backend,
            "benchmark": self.benchmark,
            "prediction_status": self.prediction_status,
            "not_applicable_reason": self.not_applicable_reason,
            "blocked_reason": self.blocked_reason,
            "failed_reason": self.failed_reason,
            "final_reconstruction": clone_json(self.final_reconstruction),
            "recovery_parse_status": self.recovery_parse_status,
            "rounds": [round_record.to_dict() for round_record in self.rounds],
            "round_count": len(self.rounds),
            "attack_cost": summarize_attack_cost([round_record.cost for round_record in self.rounds]),
            "curve_mode": self.curve_mode,
            "actual_query_count": int(self.actual_query_count),
            "source_adapter": self.source_adapter,
            "source_paths": list(self.source_paths),
            "core_logic_modified": bool(self.core_logic_modified),
            "wrapper_modified": bool(self.wrapper_modified),
            "metadata": clone_json(self.metadata),
        }


class TrajectoryVictimClient:
    def __init__(self, base_client: Any, sample: AttackInput, method: str) -> None:
        self._base_client = base_client
        self._sample = sample
        self._method = method
        self.rounds: list[dict[str, Any]] = []
        self._cumulative_values: list[Any] = []
        self.supports_constructed_prompts = bool(getattr(base_client, "supports_constructed_prompts", False))

    @property
    def query_count(self) -> int:
        return int(getattr(self._base_client, "query_count", len(self.rounds)) or len(self.rounds))

    @property
    def budget_exhausted(self) -> bool:
        return bool(getattr(self._base_client, "budget_exhausted", False))

    def interact(self, turns: Sequence[VictimTurn]) -> VictimObservation:
        prompt = turns[-1].prompt if turns else ""
        started = perf_counter()
        observation = self._base_client.interact(turns)
        elapsed = perf_counter() - started
        self._cumulative_values.append(observation.response_text)
        self._cumulative_values.extend(observation.visible_tool_results)
        reconstruction = build_prediction_payload(
            extract_evidence_snippets(self._cumulative_values, max_items=12),
            task_prompt=self._sample.task_prompt,
            visible_tool_results=self._sample.visible_tool_results,
            max_items_per_section=8,
        )
        parse = atoms_from_user_model(
            reconstruction,
            sample_id=self._sample.sample_id,
            source=f"{self._method}:victim_proxy_checkpoint",
        )
        round_index = len(self.rounds) + 1
        self.rounds.append(
            {
                "round_index": round_index,
                "prompt": prompt,
                "response_text": observation.response_text,
                "finish_reason": observation.finish_reason,
                "reconstruction": reconstruction,
                "recovery_parse_status": parse.parse_status,
                "cost": AttackCostRound(
                    round_index=round_index,
                    num_queries=1,
                    num_effective_queries=0 if observation.finish_reason == "budget_exhausted" else 1,
                    prompt_tokens=estimate_token_count(prompt),
                    completion_tokens=estimate_token_count(observation.response_text)
                    + sum(estimate_token_count(str(item)) for item in observation.visible_tool_results),
                    wall_time_sec=round(elapsed, 6),
                    num_timeouts=1 if observation.finish_reason == "timeout" else 0,
                    num_invalid_outputs=1 if observation.finish_reason == "invalid_output" else 0,
                    reconstruction=reconstruction,
                    metadata={
                        "reconstruction_source": "victim_proxy_checkpoint",
                        "turn_metadata": clone_json(turns[-1].metadata if turns else {}),
                    },
                ),
                "metadata": {"reconstruction_source": "victim_proxy_checkpoint"},
            }
        )
        return observation


def _coerce_attack_input(sample: Any, *, backend: str, benchmark: str) -> AttackInput:
    if isinstance(sample, AttackInput):
        return sample
    if isinstance(sample, Mapping):
        payload = dict(sample)
        payload.setdefault("backend", backend)
        payload.setdefault("benchmark", benchmark)
        return AttackInput.from_dict(payload)
    raise TypeError("sample must be an AttackInput or a mapping.")


def _build_base_victim(sample: Any, config: Mapping[str, Any] | None) -> Any:
    config = dict(config or {})
    if config.get("victim_client") is not None:
        return config["victim_client"]
    seed = config.get("seed_observation")
    if isinstance(seed, VictimObservation):
        return _StaticVictimClient(seed)
    if isinstance(seed, Mapping):
        return _StaticVictimClient(
            VictimObservation(
                response_text=str(seed.get("response_text") or ""),
                visible_tool_calls=list(seed.get("visible_tool_calls", [])),
                visible_tool_results=list(seed.get("visible_tool_results", [])),
                finish_reason=str(seed.get("finish_reason") or "stop"),
                metadata=dict(seed.get("metadata", {})) if isinstance(seed.get("metadata", {}), Mapping) else {},
            )
        )
    return NoOpVictimClient()


class _StaticVictimClient:
    supports_constructed_prompts = False

    def __init__(self, observation: VictimObservation) -> None:
        self._observation = observation
        self.query_count = 0
        self.budget_exhausted = False

    def interact(self, turns: Sequence[VictimTurn]) -> VictimObservation:
        self.query_count += 1
        return VictimObservation(
            response_text=self._observation.response_text,
            visible_tool_calls=clone_json(self._observation.visible_tool_calls),
            visible_tool_results=clone_json(self._observation.visible_tool_results),
            finish_reason=self._observation.finish_reason,
            metadata={**clone_json(self._observation.metadata), "query_count": self.query_count},
        )


def _method_status(prediction: AttackPrediction) -> tuple[str, str | None, str | None, str | None]:
    status = str(prediction.status)
    if status.startswith("not_applicable") or prediction.error_type == "not_mapped":
        return "not_applicable", prediction.error_type or status, None, None
    if status == "blocked_paid_dependency":
        return "blocked", None, prediction.error_type or status, None
    if status in {"success", "budget_exhausted", "skipped"}:
        return "ready", None, None, None
    return "failed", None, None, prediction.error_type or status


def _backend_project_final_reconstruction(
    reconstruction: Any,
    *,
    sample: AttackInput,
) -> tuple[Any, bool]:
    if not isinstance(reconstruction, Mapping):
        return reconstruction, False
    metadata = reconstruction.get("metadata")
    if isinstance(metadata, Mapping) and metadata.get("prediction_projection") == BACKEND_RUNTIME_PROJECTION_VERSION:
        return reconstruction, False
    projected = project_user_model_for_backend(
        reconstruction,
        backend=sample.backend,
        benchmark=sample.benchmark,
        task_domain=str(sample.public_context.get("task_domain") or ""),
        task_id=str(sample.task_id or ""),
        user_id=str(sample.user_id or ""),
    )
    projected["metadata"] = {
        **dict(projected.get("metadata", {})),
        "prediction_projection": BACKEND_RUNTIME_PROJECTION_VERSION,
        "projection_source": "eval2_attack_adapter_wrapper",
    }
    return projected, True


def _rounds_from_proxy(
    *,
    proxy: TrajectoryVictimClient,
    prediction: AttackPrediction,
    final_parse_status: str,
    final_reconstruction: Any,
) -> tuple[AttackTrajectoryRound, ...]:
    rows = list(proxy.rounds)
    if rows:
        previous_cost = rows[-1]["cost"]
        rows[-1] = {
            **rows[-1],
            "reconstruction": clone_json(final_reconstruction),
            "recovery_parse_status": final_parse_status,
            "cost": AttackCostRound(
                round_index=previous_cost.round_index,
                num_queries=previous_cost.num_queries,
                num_effective_queries=previous_cost.num_effective_queries,
                prompt_tokens=previous_cost.prompt_tokens,
                completion_tokens=previous_cost.completion_tokens,
                wall_time_sec=previous_cost.wall_time_sec,
                num_retries=previous_cost.num_retries,
                num_timeouts=previous_cost.num_timeouts,
                num_invalid_outputs=previous_cost.num_invalid_outputs,
                reconstruction=clone_json(final_reconstruction),
                metadata={
                    **previous_cost.metadata,
                    "reconstruction_source": "final_adapter_prediction",
                },
            ),
            "metadata": {**dict(rows[-1].get("metadata", {})), "reconstruction_source": "final_adapter_prediction"},
        }
    else:
        rows.append(
            {
                "round_index": 1,
                "prompt": "",
                "response_text": "",
                "finish_reason": "no_victim_query",
                "reconstruction": clone_json(final_reconstruction),
                "recovery_parse_status": final_parse_status,
                "cost": AttackCostRound(
                    round_index=1,
                    num_queries=0,
                    num_effective_queries=0,
                    prompt_tokens=int(prediction.metadata.get("prompt_tokens", 0) or 0),
                    completion_tokens=int(prediction.metadata.get("completion_tokens", 0) or 0),
                    wall_time_sec=0.0,
                    reconstruction=clone_json(final_reconstruction),
                    metadata={"reconstruction_source": "final_adapter_prediction", "no_victim_query": True},
                ),
                "metadata": {"reconstruction_source": "final_adapter_prediction", "no_victim_query": True},
            }
        )
    return tuple(
        AttackTrajectoryRound(
            round_index=int(row["round_index"]),
            prompt=str(row.get("prompt") or ""),
            response_text=str(row.get("response_text") or ""),
            finish_reason=str(row.get("finish_reason") or ""),
            reconstruction=clone_json(row.get("reconstruction")),
            recovery_parse_status=str(row.get("recovery_parse_status") or "unknown"),
            cost=row["cost"],
            metadata=dict(row.get("metadata", {})),
        )
        for row in rows
    )


def run_attack(
    sample: Any,
    budget: int | Mapping[str, Any],
    backend: str,
    benchmark: str,
    config: Mapping[str, Any] | None = None,
) -> AttackTrajectory:
    config = dict(config or {})
    method = str(config.get("method") or config.get("canonical_method") or "")
    if not method:
        raise ValueError("run_attack config must include method or canonical_method.")
    spec = get_exp2_attack_method(method)
    attack_input = _coerce_attack_input(sample, backend=backend, benchmark=benchmark)
    adapter = build_attack_adapter(spec.legacy_baseline)
    base_victim = _build_base_victim(sample, {**config, "max_queries": _max_queries(budget)})
    proxy = TrajectoryVictimClient(base_victim, attack_input, method)
    try:
        prediction = adapter.run(attack_input, proxy, budget)
    except Exception as exc:
        prediction = AttackPrediction(
            baseline=spec.legacy_baseline,
            sample_id=attack_input.sample_id,
            predicted_user_model={
                "facts": [],
                "preferences": [],
                "constraints": [],
                "relations": [],
                "tool_state": {},
                "raw_text": "",
                "confidence": 0.0,
            },
            status="runtime_error",
            error_type=exc.__class__.__name__,
            notes=str(exc),
            metadata={"adapter_version": adapter.__class__.__name__},
        )

    final_reconstruction, wrapper_projected_prediction = _backend_project_final_reconstruction(
        clone_json(prediction.predicted_user_model),
        sample=attack_input,
    )
    final_parse = atoms_from_user_model(
        final_reconstruction,
        sample_id=attack_input.sample_id,
        source=f"{method}:final_adapter_prediction",
    )
    rounds = _rounds_from_proxy(
        proxy=proxy,
        prediction=prediction,
        final_parse_status=final_parse.parse_status,
        final_reconstruction=final_reconstruction,
    )
    method_status, not_applicable_reason, blocked_reason, failed_reason = _method_status(prediction)
    actual_query_count = sum(round_record.cost.num_effective_queries for round_record in rounds)
    if actual_query_count == 0 and int(prediction.metadata.get("model_calls", 0) or 0) > 0:
        actual_query_count = int(prediction.metadata.get("model_calls", 0) or 0)
    adaptive_rounds = int(prediction.metadata.get("adaptive_rounds", 0) or 0)
    curve_mode = "adaptive_prefix" if adaptive_rounds > 0 and len(rounds) > 1 else "step_final_only"
    return AttackTrajectory(
        method=method,
        method_status=method_status,
        sample_id=attack_input.sample_id,
        backend=backend,
        benchmark=benchmark,
        prediction_status=str(prediction.status),
        not_applicable_reason=not_applicable_reason,
        blocked_reason=blocked_reason,
        failed_reason=failed_reason,
        final_reconstruction=final_reconstruction,
        recovery_parse_status=final_parse.parse_status,
        rounds=rounds,
        source_adapter=f"{spec.adapter_module}.{spec.adapter_class}",
        source_paths=spec.source_paths,
        curve_mode=curve_mode,
        actual_query_count=actual_query_count,
        core_logic_modified=spec.core_logic_modified,
        wrapper_modified=True,
        metadata={
            "legacy_baseline": spec.legacy_baseline,
            "adapter_prediction_status": prediction.status,
            "adapter_error_type": prediction.error_type,
            "adapter_notes": prediction.notes,
            "adapter_metadata": clone_json(prediction.metadata),
            "wrapper_projected_prediction": bool(wrapper_projected_prediction),
            "wrapper_projection_version": (
                BACKEND_RUNTIME_PROJECTION_VERSION if wrapper_projected_prediction else None
            ),
            "source_refs": list(prediction.source_refs),
            "final_parse": final_parse.to_dict(),
            "if_then_rules": [
                "IF adapter already exists THEN wrapper records cost/budget/reconstruction fields without changing core logic.",
                "IF adapter output is free-form or structured user model THEN unified recovery parser records parse status.",
                "IF victim interactions occur THEN proxy checkpoints each real round for budget curves.",
                "IF one-shot or visible-only method has no intermediate checkpoint THEN curve_mode=step_final_only with actual_query_count.",
                "IF adapter returns not_applicable THEN trajectory preserves method_status=not_applicable and reason.",
            ],
        },
    )


def _max_queries(budget: int | Mapping[str, Any]) -> int | None:
    if isinstance(budget, int):
        return budget
    value = budget.get("max_queries")
    return None if value is None else int(value)


def validate_attack_trajectory(trajectory: AttackTrajectory | Mapping[str, Any]) -> None:
    payload = trajectory.to_dict() if isinstance(trajectory, AttackTrajectory) else dict(trajectory)
    required = {
        "experiment_version",
        "metric_schema_version",
        "method",
        "method_status",
        "sample_id",
        "backend",
        "benchmark",
        "prediction_status",
        "final_reconstruction",
        "recovery_parse_status",
        "rounds",
        "attack_cost",
        "curve_mode",
        "actual_query_count",
        "source_adapter",
        "source_paths",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"AttackTrajectory missing required fields: {missing}")
    if payload["method_status"] not in METHOD_STATUS_VALUES:
        raise ValueError(f"Unsupported method_status: {payload['method_status']}")
    if payload["method_status"] == "ready" and not payload["rounds"]:
        raise ValueError("Ready attack trajectories must include at least one round record.")
    for round_record in payload["rounds"]:
        cost = dict(round_record.get("cost", {}))
        for field_name in (
            "num_queries",
            "num_effective_queries",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "wall_time_sec",
            "num_retries",
            "num_timeouts",
            "num_invalid_outputs",
        ):
            if field_name not in cost:
                raise ValueError(f"Trajectory round cost missing {field_name}")


def audit_method_inventory(project_root: Any | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in (get_exp2_attack_method(name) for name in _ordered_methods()):
        source_exists = []
        if project_root is not None:
            from pathlib import Path

            root = Path(project_root)
            source_exists = [bool((root / path).exists()) for path in spec.source_paths]
        rows.append(
            {
                **spec.to_dict(),
                "source_path_exists": source_exists,
            }
        )
    return rows


def _ordered_methods() -> tuple[str, ...]:
    from .schema import TARGET_METHODS

    return TARGET_METHODS
