"""Auditable numeric transformations with explicit factory/split fit guards."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from .normalization import _statistics

from .contracts import DataError


class LocalPreprocessor:
    def fit(self, frame: pd.DataFrame, features: list[str], *, factory_id: str,
            training_factories: set[str]):
        if factory_id not in training_factories:
            raise DataError("Held-out/unknown factories may not fit preprocessing")
        if (frame.empty or not frame.factory_id.eq(factory_id).all() or not frame.split.eq("train").all()):
            raise DataError("Fit requires only one permitted factory's training records")
        if (not features or len(features) != len(set(features))
                or any(not name.startswith("feat_") for name in features)):
            raise DataError("Only explicit feat_ columns may enter preprocessing")
        values = frame[features].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise DataError("Infinite feature values")
        stats = _statistics(values, 'N0')
        self.columns = list(features)
        self.medians = np.asarray(stats['median'])
        self.mean = np.asarray(stats['center'])
        self.scale = np.asarray(stats['scale'])
        self.informative = np.asarray(stats['informative'], dtype=bool)
        self.factory_id = factory_id
        self.fit_rows = len(frame)
        self.dropped = [col for col, keep in zip(features, self.informative) if not keep]
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "columns"):
            raise DataError("Preprocessor has not been fitted")
        if not set(self.columns) <= set(frame.columns):
            raise DataError("Transform is missing fitted feature columns")
        values = frame[self.columns].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise DataError("Infinite transform features")
        result = (np.where(np.isnan(values), self.medians, values) - self.mean) / self.scale
        result[:, ~self.informative] = 0
        if not np.isfinite(result).all():
            raise DataError('Non-finite transformed input')
        return result

    def save(self, path: Path):
        state = {"schema": 3, "factory_id": self.factory_id, "fit_rows": self.fit_rows,
                 "informative": self.informative.tolist(), "columns": self.columns, "dropped": self.dropped,
                 "medians": self.medians.tolist(), "mean": self.mean.tolist(), "scale": self.scale.tolist()}
        path.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        state = json.loads(path.read_text())
        if state.pop("schema") != 3:
            raise DataError("Unknown preprocessing schema")
        result = cls()
        for name, value in state.items():
            setattr(result, name, np.asarray(value, dtype=bool if name == "informative" else float) if name in ("medians", "mean", "scale", "informative") else value)
        if (not result.columns or len(set(result.columns)) != len(result.columns)
                or any(not name.startswith("feat_") for name in result.columns)
                or any(a.shape != (len(result.columns),) for a in (result.medians, result.mean, result.scale))
                or not all(np.isfinite(a).all() for a in (result.medians, result.mean, result.scale))
                or result.informative.shape != (len(result.columns),) or (result.scale <= 0).any()):
            raise DataError("Invalid preprocessing state")
        return result
