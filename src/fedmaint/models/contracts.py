"""Frozen V3 settings; no test-directed search."""
from typing import Literal
from pydantic import Field, model_validator
from fedmaint.data.contracts import StrictModel, PipelineConfig

class TrainingConfig(StrictModel):
    seeds: tuple[int, ...] = (42,43)
    hidden: int = Field(default=32, ge=1)
    epochs: int = Field(default=40, ge=1)
    patience: int = Field(default=6, ge=1)
    batch_size: int = Field(default=128, ge=1)
    learning_rate: float = Field(default=.001, gt=0)
    alpha: float = Field(default=.0001, ge=0)
    rul_scale: float = Field(default=100., gt=0)
    normalizer: Literal['N0','N1','N2'] = 'N2'
    nasa_architectures: tuple[Literal['gru','mlp'], ...] = ('gru','mlp')
    bin_resolution: tuple[float,float,float] = (1.,.1,1.)
    min_bin_rows: int = Field(default=128, ge=2)
    device: Literal['cpu','cuda'] = 'cpu'

    @model_validator(mode='after')
    def valid(self):
        if (not self.seeds or len(set(self.seeds)) != len(self.seeds)
            or any(s < 0 or s >= 2**32 for s in self.seeds)
            or not self.nasa_architectures or len(set(self.nasa_architectures)) != len(self.nasa_architectures)
            or any(r <= 0 for r in self.bin_resolution)):
            raise ValueError('Invalid seeds, architectures or bin resolution')
        return self

class ExperimentConfig(StrictModel):
    pipeline: PipelineConfig
    training: TrainingConfig = TrainingConfig()

    @model_validator(mode='after')
    def protocol(self):
        if (self.pipeline.seed != 42 or self.pipeline.nasa.held_out_domain != 'FD004'
            or self.pipeline.nasa.validation_fraction != .2 or self.pipeline.nasa.rul_cap is not None
            or self.pipeline.azure.held_out_model != 'model4'):
            raise ValueError('V3 retains established split/holdout/uncapped target rules')
        return self
