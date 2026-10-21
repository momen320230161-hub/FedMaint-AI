"""Frozen training settings and training-derived cross-client feature contracts."""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from fedmaint.data.contracts import StrictModel


class TrainingConfig(StrictModel):
    seeds: tuple[int, ...] = (42, 43, 44)
    hidden_units: int = Field(default=32, ge=1)
    max_epochs: int = Field(default=40, ge=1)
    patience: int = Field(default=6, ge=1)
    batch_size: int = Field(default=128, ge=1)
    learning_rates: tuple[float, ...] = (0.001, 0.003)
    alpha: float = Field(default=0.0001, ge=0)
    logistic_c: tuple[float, ...] = (0.1, 1.0, 10.0)
    logistic_max_iter: int = Field(default=2000, ge=1)
    rul_scale: float = Field(default=100, gt=0)

    @model_validator(mode="after")
    def validate_grids(self):
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(s < 0 or s >= 2**32 for s in self.seeds):
            raise ValueError("Seeds must be unique nonnegative 32-bit integers")
        for grid in (self.learning_rates, self.logistic_c):
            if not grid or any(v <= 0 for v in grid) or len(set(grid)) != len(grid):
                raise ValueError("Candidate grids must be nonempty, unique, and positive")
        return self

    @classmethod
    def load(cls, path: Path):
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class FeatureContract(StrictModel):
    dataset: Literal["azure", "nasa"]
    features: tuple[str, ...]
    training_factories: tuple[str, ...]
    source_factory: str
    window_length: int = Field(ge=1)
    strategy: Literal["training_union_local_zero_fill"] = "training_union_local_zero_fill"

    @model_validator(mode="after")
    def check_contract(self):
        if not self.features or len(set(self.features)) != len(self.features) or any(not f.startswith("feat_") for f in self.features):
            raise ValueError("Feature names must be unique explicit numerical features")
        if not self.training_factories or len(set(self.training_factories)) != len(self.training_factories):
            raise ValueError("Training factories must be unique and nonempty")
        if self.source_factory not in self.training_factories:
            raise ValueError("Frozen source must be a participating training factory")
        if self.dataset == "azure" and self.window_length != 1:
            raise ValueError("Azure prepared inputs are tabular window summaries")
        return self

    @property
    def input_dimension(self) -> int:
        return len(self.features) if self.dataset == "azure" else self.window_length * (len(self.features) + 1)
