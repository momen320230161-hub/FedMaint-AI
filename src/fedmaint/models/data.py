"""Audit prepared arrays; separate training loads from held-out transformations."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from fedmaint.api.main import read_catalog
from fedmaint.data.contracts import DataError
from fedmaint.data.preprocessing import LocalPreprocessor
from .contracts import FeatureContract


def array_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, array in sorted(arrays.items()):
        if array.dtype.hasobject:
            raise DataError("Object arrays are not permitted")
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def shared_contract(catalog: dict, dataset: str, states: dict[str, LocalPreprocessor]) -> FeatureContract:
    clients = sorted(f["factory_id"] for f in catalog["factories"] if f["dataset"] == dataset and f["role"] == "training")
    if len(clients) < 3 or any(client not in states for client in clients):
        raise DataError("Three participating training states are required")
    # Never inspect held-out columns, distributions, masks, or labels to form this union.
    union = sorted(set().union(*(states[client].columns for client in clients)))
    source = min(clients, key=lambda client: (-len(states[client].columns), client))
    window = 1 if dataset == "azure" else catalog["config"]["nasa"]["window_cycles"]
    return FeatureContract(dataset=dataset, features=tuple(union), training_factories=tuple(clients),
                           source_factory=source, window_length=window)


def align_scaled(x: np.ndarray, features: list[str], contract: FeatureContract) -> np.ndarray:
    if x.shape[-1] != len(features) or len(features) != len(set(features)) or not set(features) <= set(contract.features):
        raise DataError("Prepared feature order/dimension does not match training union")
    result = np.zeros((*x.shape[:-1], len(contract.features)), dtype=float)
    for source, name in enumerate(features):
        result[..., contract.features.index(name)] = x[..., source]
    return result


def transform_raw(x: np.ndarray, raw_features: list[str], mask: np.ndarray | None,
                  state: LocalPreprocessor, contract: FeatureContract) -> np.ndarray:
    if state.factory_id not in contract.training_factories:
        raise DataError("Only a frozen participating source transform is permitted")
    if x.shape[-1] != len(raw_features) or len(raw_features) != len(set(raw_features)):
        raise DataError("Invalid raw feature dimension")
    if contract.dataset == "azure":
        if x.ndim != 2 or mask is not None:
            raise DataError("Azure requires a tabular input without sequence masks")
        normalized = state.transform(pd.DataFrame(x, columns=raw_features))
        return align_scaled(normalized, state.columns, contract)
    validate_mask(x, mask, contract)
    normalized = np.zeros((*x.shape[:2], len(state.columns)), dtype=float)
    normalized[mask] = state.transform(pd.DataFrame(x[mask], columns=raw_features))
    return align_scaled(normalized, state.columns, contract)


def validate_mask(x, mask, contract):
    if (x.ndim != 3 or x.shape[1] != contract.window_length or mask is None
            or mask.dtype != np.bool_ or mask.shape != x.shape[:2] or not mask[:, -1].all()
            or (np.diff(mask.astype(int), axis=1) < 0).any()):
        raise DataError("NASA requires nonempty, left-padded chronological boolean masks")


def encode_input(x: np.ndarray, mask: np.ndarray | None, contract: FeatureContract) -> np.ndarray:
    if x.shape[-1] != len(contract.features):
        raise DataError("Shared feature dimension mismatch")
    if contract.dataset == "azure":
        if x.ndim != 2 or mask is not None or not np.isfinite(x).all():
            raise DataError("Azure requires finite tabular features")
        return np.asarray(x, dtype=float)
    validate_mask(x, mask, contract)
    if not np.isfinite(x[mask]).all():
        raise DataError("Non-finite observed NASA values")
    # Ignore every padded value, then append the mask so zero observations != padding.
    masked = np.where(mask[..., None], x, 0)
    return np.concatenate((masked.reshape(len(x), -1), mask.astype(float)), axis=1)


@dataclass
class Batch:
    x: np.ndarray
    y: np.ndarray
    keys: np.ndarray
    factories: np.ndarray
    split: str
    source_digest: str
    fit_y: np.ndarray | None = None

    def require(self, split: str, permitted: tuple[str, ...]):
        if (self.split != split or not len(self.y) or self.x.ndim != 2 or len(self.x) != len(self.y)
                or len(self.keys) != len(self.y) or len(self.factories) != len(self.y)
                or not set(self.factories) <= set(permitted) or len(set(self.keys)) != len(self.keys)
                or not np.isfinite(self.x).all() or not np.isfinite(self.y).all()
                or (self.fit_y is not None and (self.fit_y.shape != self.y.shape or not np.isfinite(self.fit_y).all()))):
            raise DataError(f"Invalid or contaminated {split} batch")


def pool(batches: list[Batch]) -> Batch:
    if not batches or len({b.split for b in batches}) != 1:
        raise DataError("Cannot pool different data partitions")
    return Batch(np.concatenate([b.x for b in batches]), np.concatenate([b.y for b in batches]),
                 np.concatenate([b.keys for b in batches]), np.concatenate([b.factories for b in batches]),
                 batches[0].split, hashlib.sha256("".join(b.source_digest for b in batches).encode()).hexdigest(),
                 np.concatenate([b.y if b.fit_y is None else b.fit_y for b in batches]))


class PreparedData:
    def __init__(self, root: Path, directory: Path):
        self.directory = directory.resolve()
        self.catalog = read_catalog(root, self.directory / "metadata.json").model_dump(mode="json")
        self.states = {}
        for factory in self.catalog["factories"]:
            if factory["role"] == "training":
                factory_id = factory["factory_id"]
                state = LocalPreprocessor.load(self.directory / f"{factory_id}-preprocessor.json")
                if state.factory_id != factory_id:
                    raise DataError("Preprocessor factory identity mismatch")
                self.states[factory_id] = state
        self.contracts = {dataset: shared_contract(self.catalog, dataset, self.states) for dataset in ("azure", "nasa")}

    def load(self, factory_id: str, split: str, *, source_factory: str | None = None) -> Batch:
        entries = [a for a in self.catalog["artifacts"] if a["factory_id"] == factory_id and a["split"] == split]
        if len(entries) != 1:
            raise DataError("Missing or duplicate prepared artifact")
        artifact = entries[0]
        path = (self.directory / artifact["path"]).resolve()
        if not path.is_relative_to(self.directory):
            raise DataError("Artifact path escapes preparation directory")
        with np.load(path, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        if array_digest(arrays) != artifact["sha256_arrays"]:
            raise DataError("Prepared artifact digest mismatch")
        contract = self.contracts["azure" if artifact["source"] == "azure" else "nasa"]
        mask = arrays.get("mask")
        if artifact["transformed"]:
            if factory_id not in contract.training_factories or source_factory is not None:
                raise DataError("Invalid use of a locally transformed participant artifact")
            if artifact["features"] != self.states[factory_id].columns:
                raise DataError("Prepared columns differ from fitted source state")
            x = align_scaled(arrays["x"], artifact["features"], contract)
        else:
            if factory_id in contract.training_factories or source_factory not in contract.training_factories:
                raise DataError("Held-out arrays require an explicit frozen training source")
            x = transform_raw(arrays["x"], artifact["features"], mask, self.states[source_factory], contract)
        encoded = encode_input(x, mask, contract)
        if contract.dataset == "azure":
            keys = np.asarray([f"{factory_id}/telemetry/{int(machine)}/{time}" for machine, time in
                               zip(arrays["machine_ids"], arrays["times"], strict=True)])
        else:
            keys = np.asarray([f"{factory_id}/{artifact['source']}/{int(engine)}/{int(cycle)}" for engine, cycle in
                               zip(arrays["engine_ids"], arrays["cycles"], strict=True)])
            if artifact["source"] == "official_test" and len(np.unique(arrays["engine_ids"])) != len(keys):
                raise DataError("Official test metrics require one terminal sample per engine")
        y = arrays["raw_y"] if contract.dataset == "nasa" else arrays["y"]
        if len(y) != artifact["samples"]:
            raise DataError("Prepared sample count mismatch")
        return Batch(encoded, y.astype(float), keys, np.repeat(factory_id, len(y)), split, artifact["sha256_arrays"],
                     arrays["y"].astype(float))
