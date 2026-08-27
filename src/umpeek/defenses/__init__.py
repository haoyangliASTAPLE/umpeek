"""Inference-time defenses connected to the real-agent victim boundary."""

from .config import configured_defense_name, load_defense_config
from .privacy_checker import PrivacyCheckerDefense
from .runtime import (
    DefendedVictimClient,
    apply_configured_defense_to_payload,
    configured_initial_query_gate,
    wrap_configured_victim_client,
)
from .schema import DEFENSE_ADAPTER_SCHEMA_VERSION, DefenseAudit, DefenseContext
from .stateful_counterfactual import StatefulCounterfactualExposureControl
from .theory_of_mind import TheoryOfMindDefense

__all__ = [
    "DEFENSE_ADAPTER_SCHEMA_VERSION",
    "DefenseAudit",
    "DefenseContext",
    "DefendedVictimClient",
    "PrivacyCheckerDefense",
    "StatefulCounterfactualExposureControl",
    "TheoryOfMindDefense",
    "apply_configured_defense_to_payload",
    "configured_defense_name",
    "configured_initial_query_gate",
    "load_defense_config",
    "wrap_configured_victim_client",
]
