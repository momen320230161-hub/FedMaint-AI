"""Train all frozen plans before opening any test/held-out arrays for evaluation."""

import hashlib
import csv
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import pandas as pd
from threadpoolctl import threadpool_limits

from fedmaint.data.contracts import DataError
from fedmaint.data.integrity import sha256, verify_manifest
from fedmaint.data.prepare import safe_output
from .artifacts import load_bundle, save_bundle
from .contracts import TrainingConfig
from .data import PreparedData, pool
from .metrics import classification_metrics, regression_metrics
from .training import train_model


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_experiments(root: Path, prepared: Path, output: Path, config: TrainingConfig) -> dict:
    root = root.resolve()
    output = safe_output(root, output)
    if (output / "plan.json").exists():
        raise DataError("Experiment directory already has a plan; use a new output directory")
    verify_manifest(root)
    data = PreparedData(root, prepared)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(prepared / "metadata.json", output / "phase01-catalog.json")
    code_files = sorted([*root.glob("src/**/*.py"), root / "scripts/run_baselines.py", root / "pyproject.toml"])
    source_hashes = {str(p.relative_to(root)): sha256(p) for p in code_files}
    code_digest = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    provenance = {"source_revision": revision, "source_sha256": code_digest,
                  "phase01_catalog_sha256": sha256(prepared / "metadata.json"),
                  "manifest_sha256": data.catalog["manifest_sha256"],
                  "training_config_sha256": hashlib.sha256(config.model_dump_json().encode()).hexdigest()}
    plan = {"schema_version": 1, "status": "training", "config": config.model_dump(mode="json"),
            "provenance": provenance, "source_files": source_hashes,
            "environment": {"python": platform.python_version(), **{p: version(p) for p in
                            ("numpy", "scipy", "scikit-learn", "pandas", "threadpoolctl")}},
            "contracts": {d: c.model_dump(mode="json") for d, c in data.contracts.items()}, "runs": []}
    write_json(output / "plan.json", plan)
    started = time.perf_counter()
    cache = {}

    def load(factory, split, source=None):
        key = factory, split, source
        if key not in cache:
            cache[key] = data.load(factory, split, source_factory=source)
        return cache[key]

    try:
        for dataset, contract in data.contracts.items():
            for architecture in (("logistic", "mlp") if dataset == "azure" else ("mlp",)):
                for seed in config.seeds:
                    for members in [*( (client,) for client in contract.training_factories), contract.training_factories]:
                        scope = "local" if len(members) == 1 else "centralized"
                        name = members[0] if scope == "local" else "pooled"
                        run_id = f"{dataset}-{architecture}-{name}-seed{seed}"
                        # Only these two split names are loaded anywhere in the training stage.
                        train = pool([load(client, "train") for client in members])
                        validation = pool([load(client, "validation") for client in members])
                        result = train_model(train, validation, contract, architecture, config, seed, fitted_factories=members)
                        model_dir = output / run_id
                        manifest = save_bundle(model_dir, result, contract, scope, seed, members, data.states, provenance)
                        plan["runs"].append({"run_id": run_id, "dataset": dataset, "architecture": architecture,
                                             "scope": scope, "seed": seed, "status": "trained",
                                             "model_directory": run_id, "training_seconds": result.selection["training_seconds"],
                                             "default_source": manifest.default_source, "evaluations": []})
                        write_json(output / "plan.json", plan)
                        print(f"TRAINED {run_id}: {result.selection['selected']}", flush=True)
        # No model/threshold/schema choice is permitted below this boundary.
        plan["status"] = "evaluating"
        write_json(output / "plan.json", plan)
        with threadpool_limits(limits=1):
            for run in plan["runs"]:
                bundle = load_bundle(output / run["model_directory"])
                members = bundle.manifest.fitted_factories
                groups = [("pooled" if run["scope"] == "centralized" else members[0], "validation",
                           pool([load(client, "validation") for client in members]), None)]
                local_test = "test" if run["dataset"] == "azure" else "official_test"
                groups.extend((client, local_test, load(client, local_test), None) for client in members)
                for factory in data.catalog["factories"]:
                    if factory["dataset"] == run["dataset"] and factory["role"] == "held_out":
                        for split in factory["samples"]:
                            groups.append((factory["factory_id"], split,
                                           load(factory["factory_id"], split, bundle.manifest.default_source),
                                           bundle.manifest.default_source))
                for factory, split, batch, source in groups:
                    prediction = bundle.predict_encoded(batch.x)
                    metrics = (classification_metrics(batch.y, prediction, bundle.manifest.threshold) if run["dataset"] == "azure"
                               else regression_metrics(batch.y, prediction))
                    name = f"predictions-{factory}-{split}.csv"
                    rows = {"sample_key": batch.keys, "factory_id": batch.factories,
                            "truth": batch.y, "prediction": prediction}
                    if bundle.manifest.threshold is not None:
                        rows["decision"] = prediction >= bundle.manifest.threshold
                    path = output / run["run_id"] / name
                    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.17g")
                    run["evaluations"].append({"factory_id": factory, "split": split, "frozen_source": source,
                                               "metrics": metrics, "predictions": name, "predictions_sha256": sha256(path),
                                               "prepared_arrays_sha256": batch.source_digest})
                run["status"] = "completed"
                write_json(output / "plan.json", plan)
                print(f"EVALUATED {run['run_id']}", flush=True)
        expected = 12 * len(config.seeds)
        if len(plan["runs"]) != expected or any(r["status"] != "completed" for r in plan["runs"]):
            raise DataError("Mandatory local/centralized runs are missing")
        plan["status"] = "completed"
        plan["runtime_seconds"] = time.perf_counter() - started
        write_json(output / "plan.json", plan)
        _write_metrics_table(plan, output / "metrics.csv")
    except Exception as exc:
        plan["status"] = "failed"
        plan["error"] = f"{type(exc).__name__}: {exc}"
        write_json(output / "plan.json", plan)
        raise
    return plan


def _write_metrics_table(plan, path):
    rows = []
    for run in plan["runs"]:
        for evaluation in run["evaluations"]:
            rows.append({**{k: run[k] for k in ("run_id", "dataset", "architecture", "scope", "seed")},
                         "factory_id": evaluation["factory_id"], "split": evaluation["split"],
                         "frozen_source": evaluation["frozen_source"], **evaluation["metrics"]})
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})
