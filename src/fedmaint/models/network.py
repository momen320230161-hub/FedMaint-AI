"""Small sklearn-trained networks exported as validated numeric weights, without pickle."""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from fedmaint.data.contracts import DataError


@dataclass
class FrozenNetwork:
    dataset: str
    architecture: str
    weights: list[np.ndarray]
    biases: list[np.ndarray]
    rul_scale: float = 1.0

    def validate(self, input_dimension: int):
        layers = 1 if self.architecture == "logistic" else 2
        if self.dataset not in ("azure", "nasa") or self.architecture not in ("logistic", "mlp"):
            raise DataError("Unsupported model task or architecture")
        if self.dataset == "nasa" and self.architecture != "mlp":
            raise DataError("NASA baseline must be the RUL MLP")
        if len(self.weights) != layers or len(self.biases) != layers:
            raise DataError("Invalid network layer count")
        width = input_dimension
        for w, b in zip(self.weights, self.biases, strict=True):
            if (w.ndim != 2 or w.shape[0] != width or b.shape != (w.shape[1],) or not w.shape[1]
                    or w.dtype.kind != "f" or b.dtype.kind != "f"
                    or not np.isfinite(w).all() or not np.isfinite(b).all()):
                raise DataError("Invalid network shape/weights")
            width = w.shape[1]
        if width != 1 or not np.isfinite(self.rul_scale) or self.rul_scale <= 0:
            raise DataError("Invalid scalar output or RUL scale")

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.validate(self.weights[0].shape[0])
        if x.ndim != 2 or x.shape[1] != self.weights[0].shape[0] or not np.isfinite(x).all():
            raise DataError("Model input dimension or values are invalid")
        out = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            out = out @ w + b
            if i < len(self.weights) - 1:
                out = np.maximum(out, 0)
        out = expit(out[:, 0]) if self.dataset == "azure" else np.maximum(out[:, 0] * self.rul_scale, 0)
        if not np.isfinite(out).all():
            raise DataError("Non-finite model output")
        return out

    @classmethod
    def from_estimator(cls, estimator, dataset: str, architecture: str, rul_scale: float):
        if architecture == "logistic":
            return cls(dataset, architecture, [estimator.coef_.T.copy()], [estimator.intercept_.copy()])
        return cls(dataset, architecture, [w.copy() for w in estimator.coefs_],
                   [b.copy() for b in estimator.intercepts_], rul_scale)
