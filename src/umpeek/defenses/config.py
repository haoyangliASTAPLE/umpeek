from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "adaptive_defenses.json"


def configured_defense_name() -> str:
    raw = str(os.environ.get("UMPEEK_EVAL2_DEFENSE") or "none").strip().lower().replace("-", "_")
    aliases = {
        "": "none",
        "off": "none",
        "privacychecker": "privacy_checker",
        "privacy_check": "privacy_checker",
        "tom": "theory_of_mind",
        "tom_defense": "theory_of_mind",
        "theory_of_mind_defense": "theory_of_mind",
        "stateful": "stateful_counterfactual",
        "stateful_counterfactual_exposure_control": "stateful_counterfactual",
    }
    name = aliases.get(raw, raw)
    if name not in {"none", "privacy_checker", "theory_of_mind", "stateful_counterfactual"}:
        raise ValueError(f"Unsupported adaptive defense: {raw!r}")
    return name


def load_defense_config(path: Path | str | None = None) -> dict[str, Any]:
    resolved = Path(
        path
        or os.environ.get("UMPEEK_EVAL2_DEFENSE_CONFIG")
        or DEFAULT_CONFIG_PATH
    )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "adaptive_defenses_v1":
        raise ValueError(f"Unsupported adaptive defense config schema in {resolved}")
    return dict(payload)


def benchmark_config(config: Mapping[str, Any], benchmark: str) -> dict[str, Any]:
    normalized = str(benchmark or "").strip().lower()
    if "personamem" in normalized:
        key = "PersonaMem-v2"
    elif "personalens" in normalized:
        key = "PersonaLens"
    elif "etapp" in normalized:
        key = "ETAPP"
    elif "locomo" in normalized:
        key = "LoCoMo"
    else:
        key = "default"
    section = config.get("benchmarks", {})
    return dict(section.get(key) or section.get("default") or {})
