"""C-MAPSS: domain/source-scoped engine identities, official RUL, masked windows."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import DataError, NasaConfig

DOMAINS = ("FD001", "FD002", "FD003", "FD004")
FEATURES = tuple([f"feat_setting_{i}" for i in range(1, 4)] + [f"feat_sensor_{i}" for i in range(1, 22)])
COLUMNS = ("engine_id", "cycle", *FEATURES)


@dataclass
class NasaData:
    domain: str
    train: pd.DataFrame
    test: pd.DataFrame
    official_rul: np.ndarray


def validate_trajectory(frame: pd.DataFrame) -> pd.DataFrame:
    if tuple(frame.columns) != COLUMNS or frame.empty:
        raise DataError("NASA requires 26 numeric columns and nonempty trajectories")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise DataError("NASA contains missing/non-finite values")
    for col in ("engine_id", "cycle"):
        if (frame[col] <= 0).any() or (frame[col] % 1 != 0).any():
            raise DataError("NASA engine IDs and cycles must be positive integers")
    frame = frame.astype({"engine_id": "int64", "cycle": "int64", **{col: "float64" for col in FEATURES}})
    frame = frame.sort_values(["engine_id", "cycle"], kind="stable").reset_index(drop=True)
    if frame.duplicated(["engine_id", "cycle"]).any():
        raise DataError("NASA duplicate engine/cycle")
    if not frame.groupby("engine_id").cycle.min().eq(1).all():
        raise DataError("NASA trajectories must start at cycle 1")
    if not frame.groupby("engine_id").cycle.diff().dropna().eq(1).all():
        raise DataError("NASA cycles must be contiguous")
    return frame


def load_nasa(root: Path, domain: str) -> NasaData:
    if domain not in DOMAINS:
        raise DataError("Unknown NASA domain")
    base = root / "data/CMAPSSData"
    frames = []
    for source in ("train", "test"):
        frame = pd.read_csv(base / f"{source}_{domain}.txt", sep=r"\s+", header=None)
        if frame.shape[1] != 26:
            raise DataError("NASA file must have exactly 26 numeric columns")
        frame.columns = COLUMNS
        frames.append(validate_trajectory(frame))
    rul_frame = pd.read_csv(base / f"RUL_{domain}.txt", sep=r"\s+", header=None)
    if rul_frame.shape[1] != 1:
        raise DataError("NASA official RUL requires one column")
    rul = rul_frame.iloc[:, 0].to_numpy(dtype=float)
    expected_ids = np.arange(1, len(rul) + 1)
    if not np.array_equal(frames[1].engine_id.unique(), expected_ids):
        raise DataError("NASA official RUL count/engine ID mismatch")
    if not np.isfinite(rul).all() or (rul < 0).any() or (rul % 1 != 0).any():
        raise DataError("NASA official RUL must be nonnegative integers")
    return NasaData(domain, *frames, rul)


def rul_targets(data: NasaData, source: str) -> pd.DataFrame:
    if source not in ("train", "test"):
        raise DataError("NASA source must be train or test")
    frame = getattr(data, source).copy()
    last = frame.groupby("engine_id").cycle.transform("max")
    offset = 0 if source == "train" else frame.engine_id.map(dict(enumerate(data.official_rul, 1)))
    frame["raw_rul"] = last - frame.cycle + offset
    return frame


def engine_split(ids, seed: int, validation_fraction: float) -> tuple[set[int], set[int]]:
    unique = np.sort(np.unique(ids))
    if len(unique) < 2 or not 0 < validation_fraction < 1:
        raise DataError("Engine-wise split needs at least two engines and a valid fraction")
    shuffled = np.random.default_rng(seed).permutation(unique)
    count = min(len(unique) - 1, max(1, int(np.ceil(len(unique) * validation_fraction))))
    return set(map(int, shuffled[count:])), set(map(int, shuffled[:count]))


@dataclass
class SequenceBatch:
    x: np.ndarray
    y: np.ndarray
    raw_y: np.ndarray
    mask: np.ndarray
    engine_ids: np.ndarray
    cycles: np.ndarray


def sequences(frame: pd.DataFrame, config: NasaConfig, *, terminal_only: bool = False) -> SequenceBatch:
    values, targets, raw, masks, engines, cycles = [], [], [], [], [], []
    for engine_id, group in frame.groupby("engine_id", sort=True):
        group = group.sort_values("cycle")
        positions = [len(group) - 1] if terminal_only else range(config.window_cycles - 1, len(group), config.stride_cycles)
        for position in positions:
            begin = max(0, position - config.window_cycles + 1)
            observed = group.iloc[begin:position + 1][list(FEATURES)].to_numpy(dtype=float)
            length = len(observed)
            padded = np.zeros((config.window_cycles, len(FEATURES)), dtype=float)
            padded[-length:] = observed
            mask = np.arange(config.window_cycles) >= config.window_cycles - length
            label = float(group.iloc[position].raw_rul)
            values.append(padded)
            raw.append(label)
            targets.append(label if config.rul_cap is None else min(label, config.rul_cap))
            masks.append(mask)
            engines.append(engine_id)
            cycles.append(int(group.iloc[position].cycle))
    if not values:
        raise DataError("No NASA windows; reduce window length or inspect engine split")
    return SequenceBatch(np.stack(values), np.asarray(targets), np.asarray(raw), np.stack(masks),
                         np.asarray(engines), np.asarray(cycles))
