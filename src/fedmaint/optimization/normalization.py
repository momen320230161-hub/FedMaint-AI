"""Private local states; public fixed dimensions; no evaluation-site fitting."""

import numpy as np

from fedmaint.data.contracts import DataError
from .common import FEATURES, SETTINGS, write_json


def _statistics(x, kind):
    available = ~np.isnan(x).all(axis=0)
    med = np.zeros(x.shape[1])
    med[available] = np.nanmedian(x[:, available], axis=0)
    filled = np.where(np.isnan(x), med, x)
    informative = available & (np.ptp(filled, axis=0) > 0)
    if kind == "N0":
        center, scale = filled.mean(0), filled.std(0)
    else:
        center = med
        q01, q25, q75, q99 = np.quantile(filled, [.01, .25, .75, .99], axis=0)
        scale = np.maximum.reduce([q75-q25, .05*(q99-q01), 1e-6*np.maximum(np.abs(med), 1)])
    scale = np.where(informative, scale, 1)
    return {"median": med.tolist(), "center": center.tolist(), "scale": scale.tolist(), "informative": informative.tolist()}


class LocalNormalizer:
    def __init__(self, kind, config):
        if kind not in ("N0", "N1", "N2"):
            raise DataError("Unknown local normalizer")
        self.kind, self.config = kind, config

    def fit(self, frame, *, domain, permitted, split="train"):
        if domain not in permitted or domain == "FD004" or split != "train" or frame.empty:
            raise DataError("Normalization fit requires permitted source-client training rows")
        if "domain" in frame and not frame.domain.eq(domain).all():
            raise DataError("Mixed-client normalization fit")
        x = frame[list(FEATURES)].to_numpy(float)
        if np.isinf(x).any():
            raise DataError("Infinite fitting values")
        self.state = {"kind": self.kind, "domain": domain, "fit_rows": len(frame),
                      "engine_ids": sorted(map(int, frame.engine_id.unique())) if "engine_id" in frame else [],
                      "features": list(FEATURES), "global": _statistics(x, self.kind), "bins": {},
                      "bin_resolution": self.config["bin_resolution"], "min_bin_rows": self.config["min_bin_rows"]}
        if self.kind == "N2":
            keys = self._keys(frame)
            for key in sorted(set(keys)):
                index = np.asarray(keys) == key
                if index.sum() >= self.config["min_bin_rows"]:
                    self.state["bins"][key] = {"rows": int(index.sum()), "stats": _statistics(x[index], "N1")}
        return self

    def _keys(self, frame):
        settings = frame[list(SETTINGS)].to_numpy(float)
        if not np.isfinite(settings).all():
            raise DataError("Operating settings must be finite")
        quantized = np.rint(settings / np.asarray(self.config["bin_resolution"])).astype(np.int64)
        return [",".join(map(str, row)) for row in quantized]

    def _apply(self, x, stats):
        x = np.where(np.isnan(x), np.asarray(stats["median"]), x)
        z = (x - np.asarray(stats["center"])) / np.asarray(stats["scale"])
        if self.kind != "N0":
            z = np.arcsinh(z)
        z[:, ~np.asarray(stats["informative"], dtype=bool)] = 0
        return z

    def transform(self, frame):
        if not hasattr(self, "state"):
            raise DataError("Unfitted normalizer")
        x = frame[list(FEATURES)].to_numpy(float)
        if np.isinf(x).any():
            raise DataError("Infinite inference values")
        out = self._apply(x, self.state["global"])
        unsupported = np.zeros(len(frame), dtype=bool)
        if self.kind == "N2":
            keys = np.asarray(self._keys(frame))
            sensor_indices = [i for i, name in enumerate(FEATURES) if name not in SETTINGS]
            for key in set(keys):
                rows = keys == key
                if key in self.state["bins"]:
                    conditional = self._apply(x[rows], self.state["bins"][key]["stats"])
                    out[np.ix_(rows, sensor_indices)] = conditional[:, sensor_indices]
                else:
                    unsupported[rows] = True
        if not np.isfinite(out).all():
            raise DataError("Non-finite transformed observations")
        return out, {"rows": len(frame), "unsupported_rows": int(unsupported.sum()),
                     "max_abs_input": float(np.max(np.abs(out))) if len(out) else 0.}

    @property
    def coverage(self):
        return sum(self.state["global"]["informative"])

    def save(self, path):
        write_json(path, self.state)

    @classmethod
    def load(cls, path):
        import json
        state = json.loads(path.read_text())
        obj = cls(state["kind"], state)
        obj.state = state
        if state["features"] != list(FEATURES):
            raise DataError("Normalizer coordinate mismatch")
        return obj
