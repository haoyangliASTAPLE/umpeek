from __future__ import annotations

import json
from pathlib import Path

import pytest

from umpeek.attack_baselines import build_attack_adapter, build_umpeek_spec, compute_umr_f1
from umpeek.defenses import load_defense_config
from umpeek.eval2.metrics import LATENT_USER_MODEL_SCOPE_VERSION


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_umpeek_interface() -> None:
    spec = build_umpeek_spec()
    assert spec.metadata["adapter_version"] == "r007_active_bayesian_profile_denoising_v004"
    assert spec.metadata["metric_scope"] == LATENT_USER_MODEL_SCOPE_VERSION


def test_only_umpeek_adapter_is_runnable() -> None:
    assert type(build_attack_adapter("umpeek")).__name__ == "UMPeekAdapter"
    with pytest.raises(ValueError, match="not distributed"):
        build_attack_adapter("adam")


def test_comparison_sources_are_external() -> None:
    path = ROOT / "configs" / "comparison_sources.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["implementations_included"] is False
    assert set(payload["methods"]) == {
        "ADAM",
        "LLM-PBE",
        "PLeak",
        "IPI",
        "Imprompter",
        "AttrInf",
        "PIE",
    }
    for source in payload["methods"].values():
        assert source.get("paper_url") or source.get("code_url")


def test_current_umr_metric_matches_identical_user_facts() -> None:
    model = {
        "facts": ["The user lives in Chicago."],
        "preferences": ["The user prefers quiet restaurants."],
        "constraints": [],
        "relations": [],
        "tool_state": [],
        "raw_text": "",
        "confidence": 1.0,
    }
    assert compute_umr_f1(model, model)["f1"] == 1.0


def test_adaptive_defense_config_loads() -> None:
    config = load_defense_config(ROOT / "configs" / "adaptive_defenses.json")
    assert config["schema_version"] == "adaptive_defenses_v1"
    assert "stateful_counterfactual" in config
