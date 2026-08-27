from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from .schema import AttackBaselineSpec, AttackInput, AttackPrediction
from .victim import VictimClient


class AttackAdapter(ABC):
    def __init__(self, spec: AttackBaselineSpec) -> None:
        self.spec = spec

    @abstractmethod
    def run(
        self,
        sample: AttackInput,
        victim_client: VictimClient,
        budget: int | Mapping[str, Any],
    ) -> AttackPrediction:
        raise NotImplementedError
