"""UMPeek adapters included in the anonymous artifact."""

from .schema_induced_slot_probe import (
    EXP4_UMPEEK_ABLATION_VARIANTS,
    UMPeekAdapter,
    UMPeekAblationAdapter,
    UMPeekAblationConfig,
    build_exp4_umpeek_ablation_adapter,
    build_exp4_umpeek_ablation_config,
    build_umpeek_spec,
)


def build_attack_adapter(method: str) -> UMPeekAdapter:
    normalized = str(method).strip().lower()
    if normalized not in {"umpeek", "umpeek_final"}:
        raise ValueError(
            "Comparison-method implementations are not distributed in this artifact. "
            "Obtain them from the official sources listed in docs/BASELINES.md."
        )
    return UMPeekAdapter()


__all__ = [
    "EXP4_UMPEEK_ABLATION_VARIANTS",
    "UMPeekAdapter",
    "UMPeekAblationAdapter",
    "UMPeekAblationConfig",
    "build_attack_adapter",
    "build_exp4_umpeek_ablation_adapter",
    "build_exp4_umpeek_ablation_config",
    "build_umpeek_spec",
]
