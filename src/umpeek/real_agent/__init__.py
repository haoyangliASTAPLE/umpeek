"""Real LLM-backed victim-agent materialization for Exp. 2."""

from .materializer import (
    REAL_AGENT_SCHEMA_VERSION,
    RealAgentConfig,
    RealAgentVictimClient,
    RealAgentUnavailable,
    build_real_agent_sample_payload,
    real_agent_enabled,
)

__all__ = [
    "REAL_AGENT_SCHEMA_VERSION",
    "RealAgentConfig",
    "RealAgentVictimClient",
    "RealAgentUnavailable",
    "build_real_agent_sample_payload",
    "real_agent_enabled",
]
