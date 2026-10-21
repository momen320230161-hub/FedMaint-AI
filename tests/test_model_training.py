import json
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from fedmaint.api.artifacts import load_model_artifact
from fedmaint.data.contracts import DataError
from fedmaint.data.integrity import sha256
from fedmaint.models.artifacts import load_bundle, save_bundle
from fedmaint.models.contracts import FeatureContract, TrainingConfig
from fedmaint.models.data import Batch, PreparedData
from fedmaint.models.experiments import run_experiments
from fedmaint.models.training import train_model
from fedmaint.models.network import FrozenNetwork
from test_model_contracts import state

ROOT = Path(__file__).resolve().parents[1]


def small_config():
    return TrainingConfig(seeds=(42,), hidden_units=4, max_epochs=2, patience=2,
                          learning_rates=(.001,), logistic_c=(1.,), batch_size=16)


def synthetic(dataset="azure"):
    contract = FeatureContract(dataset=dataset, features=("feat_a", "feat_b"), training_factories=("client",),
                               source_factory="client", window_length=1 if dataset == "azure" else 3)
    rng = np.random.default_rng(10)
    batches = []
    for split, size in (("train", 40), ("validation", 20)):
        x = rng.normal(size=(size, contract.input_dimension))
        y = (x[:, 0] > 0).astype(float) if dataset == "azure" else 100 + 20 * x[:, 0]
        batches.append(Batch(x, y, np.array([f"{split}-{i}" for i in range(size)]), np.repeat("client", size), split, split))
    return contract, *batches


@pytest.mark.parametrize("dataset,architecture", [("azure", "logistic"), ("azure", "mlp"), ("nasa", "mlp")])
def test_seeded_weights_outputs_and_roundtrip(dataset, architecture, tmp_path):
    contract, train, val = synthetic(dataset)
    config = small_config()
    a = train_model(train, val, contract, architecture, config, 42, fitted_factories=("client",))
    b = train_model(train, val, contract, architecture, config, 42, fitted_factories=("client",))
    for wa, wb in zip(a.network.weights, b.network.weights, strict=True):
        np.testing.assert_array_equal(wa, wb)
    save_bundle(tmp_path / "model", a, contract, "local", 42, ("client",), {"client": state("client", list(contract.features))}, {})
    loaded = load_model_artifact(tmp_path / "model", expected_dataset=dataset)
    with threadpool_limits(limits=1):
        np.testing.assert_array_equal(loaded.predict_encoded(val.x), a.network.predict(val.x))
    assert loaded.predict_encoded(val.x).shape == (20,)
    with pytest.raises(DataError):
        loaded.predict_encoded(val.x[:, :-1])
    with pytest.raises(DataError):
        load_model_artifact(tmp_path / "model", expected_dataset="nasa" if dataset == "azure" else "azure")


@pytest.mark.parametrize("defect", ["test_split", "held_out_client", "duplicate_keys"])
def test_contamination_rejected_before_training(defect):
    contract, train, val = synthetic()
    if defect == "test_split":
        train.split = "test"
    elif defect == "held_out_client":
        train.factories[0] = "held"
    else:
        val.keys[0] = train.keys[0]
    with pytest.raises(DataError):
        train_model(train, val, contract, "logistic", small_config(), 42, fitted_factories=("client",))


def test_capped_training_target_does_not_replace_raw_validation_truth(monkeypatch):
    from sklearn.neural_network import MLPRegressor
    contract, train, val = synthetic("nasa")
    train.fit_y = np.minimum(train.y, 50.)
    original_y = val.y.copy()
    observed = []
    real = MLPRegressor.partial_fit
    def capture(self, x, y):
        observed.append(y.copy())
        return real(self, x, y)
    monkeypatch.setattr(MLPRegressor, "partial_fit", capture)
    result = train_model(train, val, contract, "mlp", small_config(), 42, fitted_factories=("client",))
    assert observed and all((y <= .5).all() for y in observed)
    np.testing.assert_array_equal(val.y, original_y)
    assert result.selection["validation_target_source"] == "raw_y"


def test_tampered_or_malformed_checkpoint_is_rejected(tmp_path):
    contract, train, val = synthetic()
    result = train_model(train, val, contract, "logistic", small_config(), 42, fitted_factories=("client",))
    directory = tmp_path / "model"
    save_bundle(directory, result, contract, "local", 42, ("client",), {"client": state("client", list(contract.features))}, {})
    manifest = json.loads((directory / "model.json").read_text())
    weights = directory / "weights.npz"
    np.savez(weights, weight_0=np.zeros((999, 1)), bias_0=np.zeros(1))
    with pytest.raises(DataError, match="hash mismatch"):
        load_bundle(directory)
    manifest["weights_sha256"] = sha256(weights)
    (directory / "model.json").write_text(json.dumps(manifest))
    with pytest.raises(DataError, match="shape"):
        load_bundle(directory)


@pytest.mark.parametrize("dataset,architecture", [("azure", "logistic"), ("azure", "mlp"), ("nasa", "mlp")])
def test_exported_numeric_inference_matches_sklearn(dataset, architecture):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    contract, train, val = synthetic(dataset)
    with threadpool_limits(limits=1):
        if architecture == "logistic":
            estimator = LogisticRegression(solver="newton-cholesky").fit(train.x, train.y)
        elif dataset == "azure":
            estimator = MLPClassifier(hidden_layer_sizes=(4,), batch_size=16, random_state=42)
            estimator.partial_fit(train.x, train.y, classes=np.array([0., 1.]))
        else:
            estimator = MLPRegressor(hidden_layer_sizes=(4,), batch_size=16, random_state=42)
            estimator.partial_fit(train.x, train.y / 100)
        scale = 100 if dataset == "nasa" else 1
        frozen = FrozenNetwork.from_estimator(estimator, dataset, architecture, scale)
        expected = estimator.predict_proba(val.x)[:, 1] if dataset == "azure" else np.maximum(estimator.predict(val.x) * 100, 0)
        np.testing.assert_allclose(frozen.predict(val.x), expected, rtol=1e-13, atol=1e-13)


def test_real_end_to_end_plan_trains_before_evaluating_and_loads_frozen_raw_interface(prepared, tmp_path, monkeypatch):
    output, _ = prepared
    import fedmaint.models.experiments as experiments
    saved_count = 0
    real_save, real_load = experiments.save_bundle, PreparedData.load
    def audited_save(*args, **kwargs):
        nonlocal saved_count
        result = real_save(*args, **kwargs)
        saved_count += 1
        return result
    def audited_load(self, factory, split, **kwargs):
        if split not in ("train", "validation"):
            assert saved_count == 12, "All selections must finish before reading test/held-out arrays"
        return real_load(self, factory, split, **kwargs)
    monkeypatch.setattr(experiments, "save_bundle", audited_save)
    monkeypatch.setattr(PreparedData, "load", audited_load)
    run_dir = tmp_path / "smoke"
    plan = run_experiments(ROOT, output, run_dir, small_config())
    assert plan["status"] == "completed"
    assert len(plan["runs"]) == 12
    assert all(run["status"] == "completed" for run in plan["runs"])
    for run in plan["runs"]:
        assert any(e["split"].startswith("held_out") for e in run["evaluations"])
        loaded = load_bundle(run_dir / run["run_id"])
        if run["scope"] == "centralized":
            assert len(loaded.manifest.fitted_factories) == 3
    data = PreparedData(ROOT, output)
    nasa_run = next(r for r in plan["runs"] if r["dataset"] == "nasa" and r["scope"] == "centralized")
    loaded = load_bundle(run_dir / nasa_run["run_id"])
    artifact = next(a for a in data.catalog["artifacts"] if a["factory_id"] == "nasa-FD004" and a["split"] == "held_out_official_test")
    with np.load(output / artifact["path"]) as arrays:
        observed = loaded.predict_raw(arrays["x"], artifact["features"], arrays["mask"])
    expected = loaded.predict_encoded(data.load("nasa-FD004", "held_out_official_test", source_factory=loaded.manifest.default_source).x)
    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)
    with pytest.raises(DataError):
        run_experiments(ROOT, output, run_dir, small_config())
