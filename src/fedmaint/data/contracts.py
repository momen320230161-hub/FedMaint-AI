"""Validated, serializable experiment defaults; these are not academic mandates."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
import pandas as pd


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class AzureConfig(StrictModel):
    history_hours: int = Field(default=24, ge=1)
    horizon_hours: int = Field(default=24, ge=1)
    stride_hours: int = Field(default=24, ge=1)
    event_lag_hours: int = Field(default=1, ge=0)
    train_end: str = "2015-08-01T00:00:00"
    validation_end: str = "2015-10-01T00:00:00"
    held_out_model: Literal["model1", "model2", "model3", "model4"] = "model4"
    feature_groups: tuple[Literal["telemetry", "metadata", "errors", "maintenance"], ...] = (
        "telemetry",
    )

    @model_validator(mode="after")
    def check_protocol(self):
        start, end = pd.Timestamp(self.train_end), pd.Timestamp(self.validation_end)
        if start.tz is not None or end.tz is not None or start >= end:
            raise ValueError("Azure boundaries must be ordered, timezone-naive source timestamps")
        if not self.feature_groups or len(set(self.feature_groups)) != len(self.feature_groups):
            raise ValueError("Choose nonempty, unique feature groups")
        if self.event_lag_hours >= self.history_hours:
            raise ValueError("Event lag must be shorter than the history window")
        return self


class NasaConfig(StrictModel):
    window_cycles: int = Field(default=30, ge=1)
    stride_cycles: int = Field(default=10, ge=1)
    validation_fraction: float = Field(default=0.2, gt=0, lt=1)
    rul_cap: float | None = Field(default=None, gt=0)
    held_out_domain: Literal["FD001", "FD002", "FD003", "FD004"] = "FD004"


class PipelineConfig(StrictModel):
    seed: int = Field(default=42, ge=0)
    azure: AzureConfig = AzureConfig()
    nasa: NasaConfig = NasaConfig()

    @classmethod
    def load(cls, path: Path):
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class DataError(ValueError):
    """An invalid source or protocol, never a reason to rewrite original inputs."""
