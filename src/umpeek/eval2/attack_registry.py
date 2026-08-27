from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .schema import TARGET_METHODS, clone_json


@dataclass(frozen=True, slots=True)
class Exp2AttackMethodSpec:
    canonical_name: str
    legacy_baseline: str
    adapter_module: str
    adapter_class: str
    source_paths: tuple[str, ...]
    availability_status: str = "ready"
    threat_model: str = "black_box_visible_agent_interaction"
    wrapper_only: bool = True
    core_logic_modified: bool = False
    requires_external_content: bool = False
    adaptive: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.availability_status not in {"ready", "not_applicable", "blocked", "failed"}:
            raise ValueError(f"Unsupported availability_status: {self.availability_status}")
        if not self.canonical_name or not self.legacy_baseline:
            raise ValueError("Exp2AttackMethodSpec requires canonical_name and legacy_baseline.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_name": self.canonical_name,
            "legacy_baseline": self.legacy_baseline,
            "adapter_module": self.adapter_module,
            "adapter_class": self.adapter_class,
            "source_paths": list(self.source_paths),
            "availability_status": self.availability_status,
            "threat_model": self.threat_model,
            "wrapper_only": bool(self.wrapper_only),
            "core_logic_modified": bool(self.core_logic_modified),
            "requires_external_content": bool(self.requires_external_content),
            "adaptive": bool(self.adaptive),
            "metadata": clone_json(self.metadata),
        }


_REGISTRY: tuple[Exp2AttackMethodSpec, ...] = (
    Exp2AttackMethodSpec(
        canonical_name="UMPeek_final",
        legacy_baseline="umpeek",
        adapter_module="umpeek.attack_baselines.adapters.schema_induced_slot_probe",
        adapter_class="UMPeekAdapter",
        source_paths=("src/umpeek/attack_baselines/adapters/schema_induced_slot_probe.py",),
        adaptive=True,
        metadata={
            "final_method": True,
            "base_adapter_version": "r005_schema_induced_slot_probe_v12",
            "adapter_version": "r007_active_bayesian_profile_denoising_v004",
            "research_round": "R007",
            "metric_scope": "latent_user_model_v2",
        },
    ),
)


def list_exp2_attack_methods() -> list[Exp2AttackMethodSpec]:
    return list(_REGISTRY)


def get_exp2_attack_registry() -> dict[str, Exp2AttackMethodSpec]:
    return {spec.canonical_name: spec for spec in _REGISTRY}


def get_exp2_attack_method(method: str) -> Exp2AttackMethodSpec:
    registry = get_exp2_attack_registry()
    if method not in registry:
        raise ValueError(
            f"Method {method!r} is not included in the live artifact; expected {sorted(registry)}. "
            "Official comparison-method sources are listed in docs/BASELINES.md."
        )
    return registry[method]


def canonical_to_legacy(method: str) -> str:
    return get_exp2_attack_method(method).legacy_baseline


def validate_exp2_attack_registry(registry: Mapping[str, Exp2AttackMethodSpec] | None = None) -> None:
    resolved = dict(registry or get_exp2_attack_registry())
    expected = set(TARGET_METHODS)
    observed = set(resolved)
    if observed != expected:
        raise ValueError(f"Attack registry mismatch: expected {sorted(expected)}, observed {sorted(observed)}")
    for name, spec in resolved.items():
        if name != spec.canonical_name:
            raise ValueError(f"Registry key {name!r} does not match canonical name {spec.canonical_name!r}")


validate_exp2_attack_registry()
