"""Typed metadata contracts. Unknown fields fail validation instead of being discarded."""

from typing import Literal

from pydantic import Field

from fedmaint.data.contracts import PipelineConfig, StrictModel


class SourceFile(StrictModel):
    dataset: Literal["azure", "nasa"]
    name: str
    path: str
    rows: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    hash_variant: Literal["original", "repository"]
    bytes: int = Field(ge=1)
    mtime_ns: int


class Dataset(StrictModel):
    dataset_id: Literal["azure", "nasa"]
    task: Literal["binary_failure_classification", "remaining_useful_life_regression"]
    files: list[SourceFile]


class AzureSummary(StrictModel):
    machine_ids: list[int]
    age_mean: float
    sensor_means: dict[str, float]
    positive_samples: int = Field(ge=0)
    positive_fraction: float = Field(ge=0, le=1)
    time_start: str
    time_end: str


class NasaSummary(StrictModel):
    training_engine_ids: list[int]
    validation_engine_ids: list[int]
    official_test_engines: int = Field(ge=1)
    cycle_length_mean: float
    cycle_length_min: int = Field(ge=1)
    cycle_length_max: int = Field(ge=1)
    setting_means: dict[str, float]


class Factory(StrictModel):
    factory_id: str
    dataset: Literal["azure", "nasa"]
    domain: str
    role: Literal["training", "held_out"]
    units: int = Field(ge=1)
    records: int = Field(ge=1)
    samples: dict[str, int]
    summary: AzureSummary | NasaSummary


class Artifact(StrictModel):
    path: str
    factory_id: str
    source: Literal["azure", "official_train", "official_test"]
    split: str
    samples: int = Field(ge=1)
    sha256_arrays: str = Field(pattern=r"^[a-f0-9]{64}$")
    features: list[str]
    transformed: bool


class Catalog(StrictModel):
    schema_version: Literal[1]
    manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    config_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    config: PipelineConfig
    environment: dict[str, str]
    datasets: list[Dataset]
    factories: list[Factory]
    artifacts: list[Artifact]


class Health(StrictModel):
    status: Literal["ok"] = "ok"
    metadata_ready: bool


class ErrorDetail(StrictModel):
    code: str
    message: str


class ErrorResponse(StrictModel):
    error: ErrorDetail
