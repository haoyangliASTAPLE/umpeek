from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal, Mapping, Sequence


EXP2_EXPERIMENT_VERSION = "project_a_exp2_full_comparison_metrics_v2"
METRIC_SCHEMA_VERSION = "latent_user_model_scope_v2"

TARGET_METHODS = (
    "UMPeek_final",
)
TARGET_BACKENDS = ("Mem0", "Graphiti", "LangMem+LangGraph")
TARGET_BENCHMARKS = (
    "PersonaMem-v2",
    "PersonaLens",
    "ETAPP_150x32",
    "LoCoMo_10conv_1523QA_20speakers",
)
DEFAULT_BUDGET_GRID = (1, 2, 4, 8, 16)
DEFAULT_TAU_UMR = 0.5
DEFAULT_TAU_CRS = 0.5

METRIC_NAMES = (
    "UMR-F1",
    "CRS",
    "ASR@tau",
    "Attack Cost",
    "Causal-Weighted UMR-F1",
    "HBPS",
    "DSG",
    "Budget-AUC",
)

USER_MODEL_ATOM_CATEGORIES = (
    "facts",
    "preferences",
    "constraints",
    "relations",
    "tool_state",
)

MetricStatus = Literal[
    "ok",
    "empty_gold_s",
    "missing_gold_s",
    "missing_recovered_s",
    "parse_failed",
    "replay_failed",
    "blocked_replay_unavailable",
    "insufficient_heldout",
    "split_overlap",
    "no_valid_target",
    "unsafe_external_tool_blocked",
    "missing_metric",
]

TaskType = Literal["choice", "ranking", "tool", "action", "open"]


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return value


def clone_json(value: Any) -> Any:
    return json.loads(json.dumps(to_jsonable(value), ensure_ascii=False))


def stable_atom_id(sample_id: str, category: str, text: str) -> str:
    payload = f"{sample_id}|{category}|{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class MetricAtom:
    category: str
    text: str
    sample_id: str = ""
    atom_id: str = ""
    atom_type: str = "semantic"
    semantic_group: str | None = None
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.category not in USER_MODEL_ATOM_CATEGORIES:
            raise ValueError(f"Unsupported metric atom category: {self.category!r}")
        if not str(self.text).strip():
            raise ValueError("MetricAtom.text must be non-empty.")
        if not self.atom_id:
            object.__setattr__(
                self,
                "atom_id",
                stable_atom_id(self.sample_id or "sample", self.category, self.text),
            )

    @property
    def typed_text(self) -> str:
        return f"{self.category}:{self.text}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "text": self.text,
            "typed_text": self.typed_text,
            "sample_id": self.sample_id,
            "atom_id": self.atom_id,
            "atom_type": self.atom_type,
            "semantic_group": self.semantic_group,
            "source": self.source,
            "metadata": clone_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class AtomMatch:
    gold_atom: MetricAtom
    recovered_atom: MetricAtom
    match_type: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "gold_atom": self.gold_atom.to_dict(),
            "recovered_atom": self.recovered_atom.to_dict(),
            "match_type": self.match_type,
            "score": round(float(self.score), 6),
        }


@dataclass(frozen=True, slots=True)
class AtomParseResult:
    atoms: tuple[MetricAtom, ...]
    parse_status: str
    normalization_source: str
    parse_failed: bool = False
    excluded_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "atoms": [atom.to_dict() for atom in self.atoms],
            "parse_status": self.parse_status,
            "normalization_source": self.normalization_source,
            "parse_failed": self.parse_failed,
            "excluded_count": self.excluded_count,
            "metadata": clone_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class Exp2Thresholds:
    tau_umr: float = DEFAULT_TAU_UMR
    tau_crs: float = DEFAULT_TAU_CRS
    missing_metric_policy: str = "propagate_missing"

    def __post_init__(self) -> None:
        if not 0.0 <= self.tau_umr <= 1.0:
            raise ValueError("tau_umr must be in [0, 1].")
        if not 0.0 <= self.tau_crs <= 1.0:
            raise ValueError("tau_crs must be in [0, 1].")
        if self.missing_metric_policy not in {"propagate_missing", "missing_as_failure"}:
            raise ValueError("Unsupported missing_metric_policy.")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "Exp2Thresholds":
        if payload is None:
            return cls()
        return cls(
            tau_umr=float(payload.get("tau_umr", DEFAULT_TAU_UMR)),
            tau_crs=float(payload.get("tau_crs", DEFAULT_TAU_CRS)),
            missing_metric_policy=str(payload.get("missing_metric_policy", "propagate_missing")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tau_umr": float(self.tau_umr),
            "tau_crs": float(self.tau_crs),
            "missing_metric_policy": self.missing_metric_policy,
        }


@dataclass(frozen=True, slots=True)
class AttackCostRound:
    round_index: int
    num_queries: int = 0
    num_effective_queries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_time_sec: float = 0.0
    num_retries: int = 0
    num_timeouts: int = 0
    num_invalid_outputs: int = 0
    reconstruction: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.round_index < 1:
            raise ValueError("round_index must start at 1.")
        for field_name in (
            "num_queries",
            "num_effective_queries",
            "prompt_tokens",
            "completion_tokens",
            "num_retries",
            "num_timeouts",
            "num_invalid_outputs",
        ):
            if int(getattr(self, field_name)) < 0:
                raise ValueError(f"{field_name} must be non-negative.")
        if self.wall_time_sec < 0:
            raise ValueError("wall_time_sec must be non-negative.")

    @property
    def total_tokens(self) -> int:
        return int(self.prompt_tokens + self.completion_tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": int(self.round_index),
            "num_queries": int(self.num_queries),
            "num_effective_queries": int(self.num_effective_queries),
            "prompt_tokens": int(self.prompt_tokens),
            "completion_tokens": int(self.completion_tokens),
            "total_tokens": self.total_tokens,
            "wall_time_sec": round(float(self.wall_time_sec), 6),
            "num_retries": int(self.num_retries),
            "num_timeouts": int(self.num_timeouts),
            "num_invalid_outputs": int(self.num_invalid_outputs),
            "reconstruction": clone_json(self.reconstruction),
            "metadata": clone_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ReplayEvaluationContext:
    backend: str
    benchmark: str
    sample_id: str
    task_type: TaskType
    original_behavior: Any
    no_user_behavior: Any | None = None
    target_behavior: Any | None = None
    sandbox: str = "local_benchmark_sandbox"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.backend or not self.benchmark or not self.sample_id:
            raise ValueError("ReplayEvaluationContext requires backend, benchmark, and sample_id.")
        if self.sandbox != "local_benchmark_sandbox":
            raise ValueError("CRS replay must use the local benchmark sandbox.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "benchmark": self.benchmark,
            "sample_id": self.sample_id,
            "task_type": self.task_type,
            "original_behavior": clone_json(self.original_behavior),
            "no_user_behavior": clone_json(self.no_user_behavior),
            "target_behavior": clone_json(self.target_behavior),
            "sandbox": self.sandbox,
            "metadata": clone_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class HeldoutTask:
    user_id: str
    task_id: str
    task_type: TaskType
    prompt: str = ""
    gold_behavior: Any | None = None
    split: str = "candidate"
    sort_key: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "prompt": self.prompt,
            "gold_behavior": clone_json(self.gold_behavior),
            "split": self.split,
            "sort_key": self.sort_key,
            "metadata": clone_json(self.metadata),
        }


def ensure_required_keys(payload: Mapping[str, Any], required_keys: Sequence[str], label: str) -> None:
    missing = sorted(key for key in required_keys if key not in payload)
    if missing:
        raise ValueError(f"{label} missing required keys: {missing}")
