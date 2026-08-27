from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from umpeek.attack_baselines.schema import AttackInput
from umpeek.exp1_whitebox.schema import clone_json


DEFENSE_ADAPTER_SCHEMA_VERSION = "adaptive_defense_adapter_v1"


@dataclass(frozen=True, slots=True)
class DefenseContext:
    backend: str
    benchmark: str
    task_prompt: str
    target_user_id: str = "unknown_user"
    requester_id: str = "external_requester"
    domain: str = "general"
    visible_tools: tuple[dict[str, Any], ...] = ()
    public_context: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_attack_input(cls, sample: AttackInput) -> "DefenseContext":
        requester = str(
            sample.public_context.get("requester_id")
            or sample.metadata.get("requester_id")
            or "external_requester"
        )
        domain = str(
            sample.public_context.get("domain")
            or sample.public_context.get("task_type")
            or sample.metadata.get("domain")
            or sample.benchmark
            or "general"
        )
        return cls(
            backend=str(sample.backend),
            benchmark=str(sample.benchmark),
            task_prompt=str(sample.task_prompt),
            target_user_id=str(sample.user_id or sample.sample_id or "unknown_user"),
            requester_id=requester,
            domain=domain,
            visible_tools=tuple(clone_json(sample.visible_tools)),
            public_context=clone_json(sample.public_context),
        )

    def public_description(self) -> str:
        tool_names: list[str] = []
        for tool in self.visible_tools:
            name = str(tool.get("name") or tool.get("tool_name") or "").strip()
            if name and name not in tool_names:
                tool_names.append(name)
        tools = ", ".join(tool_names[:24]) if tool_names else "no public tools"
        return (
            f"A personalized {self.benchmark} assistant backed by {self.backend}. "
            f"Its public interface exposes {tools}."
        )


@dataclass(frozen=True, slots=True)
class DefenseAudit:
    defense: str
    action: str
    model_calls: int = 0
    checks: int = 0
    revisions: int = 0
    blocked: bool = False
    parse_status: str = "ok"
    denied_flow_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_public_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": DEFENSE_ADAPTER_SCHEMA_VERSION,
            "defense": self.defense,
            "action": self.action,
            "model_calls": int(self.model_calls),
            "checks": int(self.checks),
            "revisions": int(self.revisions),
            "blocked": bool(self.blocked),
            "parse_status": self.parse_status,
            "denied_flow_count": int(self.denied_flow_count),
            **clone_json(self.metadata),
        }


def stable_public_hash(value: Any) -> str:
    payload = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def public_tool_schema(tools: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        out.append(
            {
                "name": str(tool.get("name") or tool.get("tool_name") or ""),
                "description": str(tool.get("description") or ""),
                "parameters": clone_json(tool.get("parameters") or tool.get("input_schema") or {}),
            }
        )
    return out
