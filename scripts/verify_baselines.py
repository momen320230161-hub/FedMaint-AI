"""Reload every completed model; reproduce saved predictions, metrics, and thresholds."""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedmaint.data.contracts import DataError
from fedmaint.data.integrity import sha256, verify_manifest
from fedmaint.models.artifacts import load_bundle
from fedmaint.models.contracts import TrainingConfig
from fedmaint.models.data import PreparedData, pool
from fedmaint.models.metrics import classification_metrics, regression_metrics, validation_threshold


def verify(experiment: Path, prepared: Path) -> dict:
    experiment = experiment.resolve()
    plan = json.loads((experiment / "plan.json").read_text())
    config = TrainingConfig.model_validate(plan["config"])
    if plan["status"] != "completed" or len(plan["runs"]) != 12 * len(config.seeds):
        raise DataError("Experiment plan is not fully completed")
    records = verify_manifest(ROOT)
    if (sha256(prepared / "metadata.json") != plan["provenance"]["phase01_catalog_sha256"]
            or sha256(experiment / "phase01-catalog.json") != plan["provenance"]["phase01_catalog_sha256"]):
        raise DataError("Preparation catalog provenance changed")
    for relative, digest in plan["source_files"].items():
        if sha256(ROOT / relative) != digest:
            raise DataError(f"Model source changed since training: {relative}")
    data = PreparedData(ROOT, prepared)
    if {d: c.model_dump(mode="json") for d, c in data.contracts.items()} != plan["contracts"]:
        raise DataError("Shared feature contract changed")
    cache = {}
    def batch(factory, split, source=None):
        key = factory, split, source
        if key not in cache:
            cache[key] = data.load(factory, split, source_factory=source)
        return cache[key]
    checked = 0
    with threadpool_limits(limits=1):
        for run in plan["runs"]:
            directory = (experiment / run["model_directory"]).resolve()
            if not directory.is_relative_to(experiment) or run["status"] != "completed":
                raise DataError("Invalid run directory or completion state")
            bundle = load_bundle(directory, expected_dataset=run["dataset"])
            for client, frozen in bundle.transforms.items():
                original = data.states[client]
                if frozen.columns != original.columns or frozen.fit_rows != original.fit_rows:
                    raise DataError("Frozen source schema differs from Phase 01")
                for field in ("medians", "mean", "scale"):
                    np.testing.assert_array_equal(getattr(frozen, field), getattr(original, field))
            val = pool([batch(client, "validation") for client in bundle.manifest.fitted_factories])
            if run["dataset"] == "azure":
                threshold = validation_threshold(val, bundle.predict_encoded(val.x), bundle.manifest.fitted_factories)
                if threshold != bundle.manifest.threshold:
                    raise DataError("Saved threshold differs from validation-only rule")
            for evaluation in run["evaluations"]:
                path = (directory / evaluation["predictions"]).resolve()
                if not path.is_relative_to(directory) or sha256(path) != evaluation["predictions_sha256"]:
                    raise DataError("Prediction file path/hash mismatch")
                saved = pd.read_csv(path, float_precision="round_trip")
                current = val if evaluation["split"] == "validation" else batch(
                    evaluation["factory_id"], evaluation["split"], evaluation["frozen_source"])
                prediction = bundle.predict_encoded(current.x)
                np.testing.assert_array_equal(saved.sample_key.to_numpy(), current.keys)
                np.testing.assert_array_equal(saved.factory_id.to_numpy(), current.factories)
                np.testing.assert_array_equal(saved.truth.to_numpy(), current.y)
                np.testing.assert_array_equal(saved.prediction.to_numpy(), prediction)
                if run["dataset"] == "azure":
                    np.testing.assert_array_equal(saved.decision.to_numpy(), prediction >= bundle.manifest.threshold)
                    metrics = classification_metrics(current.y, prediction, bundle.manifest.threshold)
                else:
                    metrics = regression_metrics(current.y, prediction)
                if current.source_digest != evaluation["prepared_arrays_sha256"]:
                    raise DataError("Evaluation input provenance changed")
                for name, value in metrics.items():
                    recorded = evaluation["metrics"][name]
                    if isinstance(value, float):
                        if not math.isclose(value, recorded, rel_tol=1e-12, abs_tol=1e-12):
                            raise DataError(f"Metric changed: {name}")
                    elif value != recorded:
                        raise DataError(f"Metric changed: {name}")
                checked += 1
    return {"verified_models": len(plan["runs"]), "verified_prediction_files": checked,
            "raw_files_verified": len(records), "status": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / "outputs/phase02")
    parser.add_argument("--prepared", type=Path, default=ROOT / "outputs/phase01")
    args = parser.parse_args()
    result = verify(args.experiment, args.prepared)
    path = args.experiment / "verification.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
