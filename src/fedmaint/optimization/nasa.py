"""Raw source-only loading, unique-row fits, masked windows and fixed prefix endpoints."""

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd

from fedmaint.data.contracts import DataError
from fedmaint.data.nasa import COLUMNS, engine_split, validate_trajectory
from .common import DOMAINS, FEATURES, digest


@dataclass
class Windows:
    x: np.ndarray
    mask: np.ndarray
    y: np.ndarray
    keys: np.ndarray
    cycles: np.ndarray
    domain: np.ndarray
    split: str


def combine(batches):
    if len({b.split for b in batches}) != 1:
        raise DataError("Cannot combine train and evaluation windows")
    return Windows(*(np.concatenate([getattr(b, name) for b in batches]) for name in
                     ("x", "mask", "y", "keys", "cycles", "domain")), batches[0].split)


class NasaStore:
    def __init__(self, root, config, gate=None):
        self.root, self.config, self.gate = root, config, gate
        self.frames, self.access = {}, []
        self.manifest = json.loads((root / "data/manifest.json").read_text())["nasa"]["files"]

    def read(self, domain, source="train"):
        if domain not in (*DOMAINS, "FD004") or source not in ("train", "test"):
            raise DataError("Invalid NASA source")
        if domain == "FD004" or source == "test":
            if self.gate is None:
                raise DataError("Test/FD004 access is forbidden before the final freeze")
            self.gate()
        key = domain, source
        if key in self.frames:
            return self.frames[key]
        name = f"{source}_{domain}.txt"
        path = self.root / "data/CMAPSSData" / name
        if digest(path) != self.manifest[name]["sha256"]:
            raise DataError(f"NASA hash mismatch: {name}")
        frame = pd.read_csv(path, sep=r"\s+", header=None)
        if frame.shape[1] != 26 or len(frame) != self.manifest[name]["rows"]:
            raise DataError("NASA schema/row mismatch")
        frame.columns = COLUMNS
        frame = validate_trajectory(frame)
        frame["raw_rul"] = frame.groupby("engine_id").cycle.transform("max") - frame.cycle
        if source == "test":
            rul_name = f"RUL_{domain}.txt"
            rul_path = path.parent / rul_name
            if digest(rul_path) != self.manifest[rul_name]["sha256"]:
                raise DataError("Official RUL hash mismatch")
            rul = np.loadtxt(rul_path, ndmin=1)
            if (len(rul) != self.manifest[rul_name]["rows"] or not np.array_equal(frame.engine_id.unique(), np.arange(1, len(rul)+1))
                    or not np.isfinite(rul).all() or (rul < 0).any()):
                raise DataError("Official RUL ID/count mismatch")
            frame["raw_rul"] += frame.engine_id.map(dict(enumerate(rul, 1)))
        frame["domain"] = domain
        frame["source"] = source
        self.frames[key] = frame
        self.access.append({"domain": domain, "source": source, "sha256": self.manifest[name]["sha256"]})
        return frame

    def partition(self, domain):
        frame = self.read(domain)
        train_ids, val_ids = engine_split(frame.engine_id, 42, .2)
        return frame.loc[frame.engine_id.isin(train_ids)], frame.loc[frame.engine_id.isin(val_ids)]

    def windows(self, frame, normalizer, split, *, prefixes=False, terminal=False):
        normalized, diagnostic = normalizer.transform(frame)
        work = frame.reset_index(drop=True)
        xs, masks, ys, keys, cycles, domains = [], [], [], [], [], []
        length = self.config["window"]
        for engine, group in work.groupby("engine_id", sort=True):
            indices = group.index.to_numpy()
            if prefixes:
                positions = [int(c)-1 for c in self.config["prefix_cycles"] if int(c) <= len(indices)]
            elif terminal:
                positions = [len(indices)-1]
            else:
                positions = range(length-1, len(indices), self.config["stride"])
            for pos in positions:
                valid = indices[max(0, pos-length+1):pos+1]
                x = np.zeros((length, len(FEATURES)), np.float32)
                x[-len(valid):] = normalized[valid]
                mask = np.arange(length) >= length-len(valid)
                row = group.iloc[pos]
                xs.append(x); masks.append(mask); ys.append(float(row.raw_rul))
                keys.append(f"{row.domain}/{row.source}/{engine}/{int(row.cycle)}")
                cycles.append(int(row.cycle)); domains.append(row.domain)
        if not xs:
            raise DataError("No windows under the fixed protocol")
        diagnostic["engines"] = int(work.engine_id.nunique())
        diagnostic["windows"] = len(xs)
        if prefixes:
            diagnostic["endpoint_eligible_engines"] = {str(c): int(sum(v >= c for v in work.groupby("engine_id").cycle.max()))
                                                        for c in self.config["prefix_cycles"]}
        return Windows(np.stack(xs), np.stack(masks), np.array(ys), np.array(keys), np.array(cycles),
                       np.array(domains), split), diagnostic


def choose_source(states):
    return min(states, key=lambda d: (-states[d].coverage, d))
