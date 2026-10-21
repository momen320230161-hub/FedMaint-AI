"""Auditable numeric transformations with explicit factory/split fit guards."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
        # Drop all-missing columns before medians, then constants after imputation.
        available = ~np.isnan(values).all(axis=0)
        if not available.any():
            raise DataError("No observed training features")
        candidates = np.asarray(features)[available]
        values = values[:, available]
        medians = np.nanmedian(values, axis=0)
        filled = np.where(np.isnan(values), medians, values)
        informative = np.ptp(filled, axis=0) > 0
        if not informative.any():
            raise DataError("All training features are constant/uninformative")
        self.columns = candidates[informative].tolist()
        self.medians = medians[informative]
        scaler = StandardScaler().fit(filled[:, informative])
        self.mean = scaler.mean_
        self.scale = scaler.scale_
        self.factory_id = factory_id
        self.fit_rows = len(frame)
        self.dropped = [col for col in features if col not in self.columns]
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "columns"):
            raise DataError("Preprocessor has not been fitted")
        if not set(self.columns) <= set(frame.columns):
            raise DataError("Transform is missing fitted feature columns")
        values = frame[self.columns].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise DataError("Infinite transform features")
        return (np.where(np.isnan(values), self.medians, values) - self.mean) / self.scale

    def save(self, path: Path):
        state = {"schema": 1, "factory_id": self.factory_id, "fit_rows": self.fit_rows,
                 "columns": self.columns, "dropped": self.dropped,
                 "medians": self.medians.tolist(), "mean": self.mean.tolist(), "scale": self.scale.tolist()}
        path.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        state = json.loads(path.read_text())
        if state.pop("schema") != 1:
            raise DataError("Unknown preprocessing schema")
        result = cls()
        for name, value in state.items():
            setattr(result, name, np.asarray(value, dtype=float) if name in ("medians", "mean", "scale") else value)
        if (not result.columns or len(set(result.columns)) != len(result.columns)
                or any(not name.startswith("feat_") for name in result.columns)
                or any(a.shape != (len(result.columns),) for a in (result.medians, result.mean, result.scale))
                or not all(np.isfinite(a).all() for a in (result.medians, result.mean, result.scale))
                or (result.scale <= 0).any()):
            raise DataError("Invalid preprocessing state")
        return result
