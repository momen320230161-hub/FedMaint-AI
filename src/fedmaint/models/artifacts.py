"""Validated task/shape/provenance-aware model bundles for future backend use."""

from dataclasses import dataclass
import json
import zipfile
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from fedmaint.data.contracts import DataError, StrictModel
from fedmaint.data.integrity import sha256
from fedmaint.data.preprocessing import LocalPreprocessor
from .contracts import FeatureContract
from .data import encode_input, transform_raw
from .network import FrozenNetwork
from .training import TrainingResult


class BundleManifest(StrictModel):
    schema_version: Literal[1] = 1
    dataset: Literal["azure", "nasa"]
    architecture: Literal["logistic", "mlp"]
    scope: Literal["local", "centralized"]
    seed: int = Field(ge=0)
    contract: FeatureContract
    fitted_factories: tuple[str, ...]
    default_source: str
    input_dimension: int = Field(ge=1)
    rul_scale: float = Field(gt=0)
    threshold: float | None = Field(default=None, ge=0, le=1)
    weights_file: Literal["weights.npz"] = "weights.npz"
    weights_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    transforms: dict[str, str]
    transform_sha256: dict[str, str]
    selection: dict
    provenance: dict[str, str]

    @model_validator(mode="after")
    def validate_bundle_contract(self):
        factories = set(self.fitted_factories)
        if (self.contract.dataset != self.dataset or self.input_dimension != self.contract.input_dimension
                or not factories or len(factories) != len(self.fitted_factories)
                or not factories <= set(self.contract.training_factories) or self.default_source not in factories
                or set(self.transforms) != factories or set(self.transform_sha256) != factories):
            raise ValueError("Inconsistent model/preprocessing contract")
        if (self.scope == "local" and len(factories) != 1) or (self.scope == "centralized" and factories != set(self.contract.training_factories)):
            raise ValueError("Invalid local/centralized training membership")
        if (self.dataset == "azure") != (self.threshold is not None):
            raise ValueError("Only classification bundles require a threshold")
        if self.dataset == "nasa" and self.architecture != "mlp":
            raise ValueError("Unsupported NASA architecture")
        return self


def _child(directory: Path, relative: str) -> Path:
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()) or path == directory.resolve():
        raise DataError("Bundle path escapes artifact directory")
    return path


def save_bundle(directory: Path, result: TrainingResult, contract: FeatureContract, scope: str, seed: int,
                fitted_factories: tuple[str, ...], states: dict[str, LocalPreprocessor], provenance: dict[str, str]):
    directory.mkdir(parents=True, exist_ok=False)
    network = result.network
    network.validate(contract.input_dimension)
    arrays = {f"weight_{i}": w for i, w in enumerate(network.weights)}
    arrays.update({f"bias_{i}": b for i, b in enumerate(network.biases)})
    np.savez_compressed(directory / "weights.npz", **arrays)
    transforms, digests = {}, {}
    for client in fitted_factories:
        path = directory / f"transform-{client}.json"
        states[client].save(path)
        transforms[client], digests[client] = path.name, sha256(path)
    source = fitted_factories[0] if scope == "local" else contract.source_factory
    manifest = BundleManifest(dataset=contract.dataset, architecture=network.architecture, scope=scope, seed=seed,
                              contract=contract, fitted_factories=fitted_factories, default_source=source,
                              input_dimension=contract.input_dimension, rul_scale=network.rul_scale,
                              threshold=result.threshold, weights_sha256=sha256(directory / "weights.npz"),
                              transforms=transforms, transform_sha256=digests, selection=result.selection,
                              provenance=provenance)
    (directory / "model.json").write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest


@dataclass
class LoadedModel:
    manifest: BundleManifest
    network: FrozenNetwork
    transforms: dict[str, LocalPreprocessor]

    def predict_encoded(self, x):
        if x.ndim != 2 or x.shape[1] != self.manifest.input_dimension:
            raise DataError("Encoded input does not match saved model")
        return self.network.predict(x)

    def predict_raw(self, x, features: list[str], mask=None, *, source_factory: str | None = None):
        source = source_factory or self.manifest.default_source
        if source not in self.transforms:
            raise DataError("Requested source was not frozen into this model")
        normalized = transform_raw(x, features, mask, self.transforms[source], self.manifest.contract)
        return self.predict_encoded(encode_input(normalized, mask, self.manifest.contract))


def load_bundle(directory: Path, *, expected_dataset: str | None = None) -> LoadedModel:
    directory = directory.resolve()
    try:
        manifest = BundleManifest.model_validate_json((directory / "model.json").read_text(encoding="utf-8"))
        if expected_dataset is not None and manifest.dataset != expected_dataset:
            raise DataError("Model task differs from requested dataset")
        weights_path = _child(directory, manifest.weights_file)
        if sha256(weights_path) != manifest.weights_sha256:
            raise DataError("Checkpoint hash mismatch")
        layers = 1 if manifest.architecture == "logistic" else 2
        with np.load(weights_path, allow_pickle=False) as archive:
            if set(archive.files) != {f"{kind}_{i}" for kind in ("weight", "bias") for i in range(layers)}:
                raise DataError("Invalid checkpoint keys")
            network = FrozenNetwork(manifest.dataset, manifest.architecture,
                                    [archive[f"weight_{i}"] for i in range(layers)],
                                    [archive[f"bias_{i}"] for i in range(layers)], manifest.rul_scale)
        network.validate(manifest.input_dimension)
        states = {}
        for client, relative in manifest.transforms.items():
            path = _child(directory, relative)
            if sha256(path) != manifest.transform_sha256[client]:
                raise DataError("Frozen transform hash mismatch")
            state = LocalPreprocessor.load(path)
            if state.factory_id != client or not set(state.columns) <= set(manifest.contract.features):
                raise DataError("Frozen transform identity/feature mismatch")
            states[client] = state
        return LoadedModel(manifest, network, states)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, EOFError) as exc:
        raise DataError(f"Invalid model artifact: {exc}") from exc
