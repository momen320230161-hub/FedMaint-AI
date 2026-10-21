"""Reproducible preparation orchestration. No models, training, or held-out fitting."""

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .azure import MODELS, SENSORS, load_azure, make_azure_samples
from .contracts import DataError, PipelineConfig
from .integrity import sha256, verify_manifest
from .nasa import DOMAINS, FEATURES, engine_split, load_nasa, rul_targets, sequences
from .preprocessing import LocalPreprocessor


def safe_output(root: Path, output: Path) -> Path:
    output = output.resolve()
    for protected in (root / "data", root / "docs", root / "src", root / ".git"):
        protected = protected.resolve()
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise DataError("Generated outputs must be outside source/input directories")
    return output


def _save_batch(path: Path, **arrays) -> str:
    digest = hashlib.sha256()
    for name, array in sorted(arrays.items()):
        array = np.asarray(array)
        if array.dtype.hasobject:
            array = array.astype(str)
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
        arrays[name] = array
    np.savez_compressed(path, **arrays)
    return digest.hexdigest()


def prepare(root: Path, config: PipelineConfig, output: Path) -> dict:
    root = root.resolve()
    output = safe_output(root, output)
    records = verify_manifest(root)
    output.mkdir(parents=True, exist_ok=True)
    config_json = config.model_dump_json()
    config_digest = hashlib.sha256(config_json.encode()).hexdigest()
    # Revoke an old catalog before replacing samples: failed runs must not appear ready.
    catalog_path = output / "metadata.json"
    if catalog_path.exists():
        previous = json.loads(catalog_path.read_text(encoding="utf-8"))
        if previous["config_sha256"] != config_digest:
            raise DataError("Use a new output directory for a changed configuration")
    catalog_path.unlink(missing_ok=True)
    factories, artifacts, datasets = [], [], []
    azure = load_azure(root)
    samples = make_azure_samples(azure, config.azure)
    features = [name for name in samples if name.startswith("feat_")]
    training_factories = {f"azure-{model}" for model in MODELS if model != config.azure.held_out_model}
    for model in MODELS:
        factory_id = f"azure-{model}"
        held_out = factory_id not in training_factories
        machine_ids = azure.machines.loc[azure.machines.model == model, "machineID"].tolist()
        frame = samples.loc[samples.factory_id == factory_id]
        if frame.empty:
            raise DataError(f"No samples in {factory_id}")
        scaler = None
        if not held_out:
            if set(frame.split) != {"train", "validation", "test"}:
                raise DataError(f"Incomplete chronological splits: {factory_id}")
            scaler = LocalPreprocessor().fit(frame.loc[frame.split == "train"], features,
                                            factory_id=factory_id, training_factories=training_factories)
            scaler.save(output / f"{factory_id}-preprocessor.json")
        counts = {}
        for split, group in frame.groupby("split", sort=True):
            counts[split] = len(group)
            arrays = {"x": group[features].to_numpy(dtype=float) if held_out else scaler.transform(group),
                      "y": group.target.to_numpy(), "machine_ids": group.machineID.to_numpy(),
                      "times": group.datetime.to_numpy(dtype="datetime64[ns]"),
                      "history_start": group.history_start.to_numpy(dtype="datetime64[ns]"),
                      "target_end": group.target_end.to_numpy(dtype="datetime64[ns]")}
            name = f"{factory_id}-{split}.npz"
            digest = _save_batch(output / name, **arrays)
            artifacts.append({"path": name, "factory_id": factory_id, "source": "azure",
                              "split": split, "samples": len(group), "sha256_arrays": digest,
                              "features": features if held_out else scaler.columns,
                              "transformed": not held_out})
        telemetry = azure.telemetry.loc[azure.telemetry.machineID.isin(machine_ids)]
        factories.append({"factory_id": factory_id, "dataset": "azure", "domain": model,
                          "role": "held_out" if held_out else "training", "units": len(machine_ids),
                          "records": len(telemetry), "samples": counts,
                          "summary": {"machine_ids": machine_ids,
                                      "age_mean": float(azure.machines.loc[azure.machines.model == model, "age"].mean()),
                                      "sensor_means": {s: float(telemetry[s].mean()) for s in SENSORS},
                                      "positive_samples": int(frame.target.sum()),
                                      "positive_fraction": float(frame.target.mean()),
                                      "time_start": str(telemetry.datetime.min()), "time_end": str(telemetry.datetime.max())}})
    datasets.append({"dataset_id": "azure", "task": "binary_failure_classification",
                     "files": [r for r in records if r["dataset"] == "azure"]})
    del azure, samples
    nasa_training = {f"nasa-{d}" for d in DOMAINS if d != config.nasa.held_out_domain}
    for domain in DOMAINS:
        data = load_nasa(root, domain)
        factory_id = f"nasa-{domain}"
        held_out = factory_id not in nasa_training
        train = rul_targets(data, "train")
        official_test = rul_targets(data, "test")
        scaler = None
        if held_out:
            groups = [("held_out_run_to_failure", train, False), ("held_out_official_test", official_test, True)]
            train_ids, val_ids = set(), set()
        else:
            train_ids, val_ids = engine_split(train.engine_id, config.seed, config.nasa.validation_fraction)
            fit_frame = train.loc[train.engine_id.isin(train_ids)].assign(factory_id=factory_id, split="train")
            scaler = LocalPreprocessor().fit(fit_frame, list(FEATURES), factory_id=factory_id,
                                            training_factories=nasa_training)
            scaler.save(output / f"{factory_id}-preprocessor.json")
            groups = [("train", train.loc[train.engine_id.isin(train_ids)], False),
                      ("validation", train.loc[train.engine_id.isin(val_ids)], False),
                      ("official_test", official_test, True)]
        counts = {}
        for split, group, terminal in groups:
            # Normalize observed rows before padding. Padding remains zero in normalized space.
            transformed = group.copy()
            if scaler is not None:
                normalized = scaler.transform(group)
                transformed.loc[:, list(FEATURES)] = 0.0
                transformed.loc[:, scaler.columns] = normalized
            batch = sequences(transformed, config.nasa, terminal_only=terminal)
            if scaler is not None:
                kept_indices = [FEATURES.index(col) for col in scaler.columns]
                batch.x = batch.x[:, :, kept_indices]
            counts[split] = len(batch.y)
            name = f"{factory_id}-{split}.npz"
            digest = _save_batch(output / name, x=batch.x, y=batch.y, raw_y=batch.raw_y, mask=batch.mask,
                                 engine_ids=batch.engine_ids, cycles=batch.cycles)
            artifacts.append({"path": name, "factory_id": factory_id,
                              "source": "official_test" if terminal else "official_train",
                              "split": split, "samples": len(batch.y), "sha256_arrays": digest,
                              "features": list(FEATURES) if held_out else scaler.columns,
                              "transformed": not held_out})
        factories.append({"factory_id": factory_id, "dataset": "nasa", "domain": domain,
                          "role": "held_out" if held_out else "training",
                          "units": int(data.train.engine_id.nunique()), "records": len(train), "samples": counts,
                          "summary": {"training_engine_ids": sorted(train_ids), "validation_engine_ids": sorted(val_ids),
                                      "official_test_engines": int(data.test.engine_id.nunique()),
                                      "cycle_length_mean": float(train.groupby("engine_id").cycle.max().mean()),
                                      "cycle_length_min": int(train.groupby("engine_id").cycle.max().min()),
                                      "cycle_length_max": int(train.groupby("engine_id").cycle.max().max()),
                                      "setting_means": {col: float(train[col].mean()) for col in FEATURES[:3]}}})
    datasets.append({"dataset_id": "nasa", "task": "remaining_useful_life_regression",
                     "files": [r for r in records if r["dataset"] == "nasa"]})
    # Check source identity again before publishing a catalog of the completed run.
    for record in records:
        stat = (root / record["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (record["bytes"], record["mtime_ns"]):
            raise DataError("Raw inputs changed during preparation")
    catalog = {"schema_version": 1, "manifest_sha256": sha256(root / "data/manifest.json"),
               "config_sha256": config_digest, "config": config.model_dump(mode="json"),
               "environment": {"python": platform.python_version(), **{p: version(p) for p in
                              ("numpy", "pandas", "scikit-learn", "fastapi", "pydantic")}},
               "datasets": datasets, "factories": factories, "artifacts": artifacts}
    encoded = json.dumps(catalog, indent=2, allow_nan=False) + "\n"
    temporary = output / "metadata.json.tmp"
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(catalog_path)
    return catalog
