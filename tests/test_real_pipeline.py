from pathlib import Path

import numpy as np

from fedmaint.data.contracts import PipelineConfig
from fedmaint.data.nasa import DOMAINS, load_nasa, rul_targets
from fedmaint.data.prepare import prepare
from fedmaint.data.preprocessing import LocalPreprocessor

ROOT = Path(__file__).resolve().parents[1]


def test_real_schema_and_official_targets_for_all_domains():
    counts = {"FD001": (100, 100), "FD002": (260, 259), "FD003": (100, 100), "FD004": (249, 248)}
    for domain in DOMAINS:
        data = load_nasa(ROOT, domain)
        assert (data.train.engine_id.nunique(), data.test.engine_id.nunique()) == counts[domain]
        frame = rul_targets(data, "test")
        np.testing.assert_array_equal(frame.groupby("engine_id").raw_rul.last(), data.official_rul)


def test_real_client_isolation_local_fit_and_finite_samples(prepared):
    output, catalog = prepared
    assert len(catalog["factories"]) == 8
    for dataset in ("azure", "nasa"):
        clients = [f for f in catalog["factories"] if f["dataset"] == dataset]
        assert sum(f["role"] == "training" for f in clients) == 3
        assert sum(f["role"] == "held_out" for f in clients) == 1
    for factory in catalog["factories"]:
        factory_id = factory["factory_id"]
        state = output / f"{factory_id}-preprocessor.json"
        if factory["role"] == "held_out":
            assert not state.exists()
            assert not {"train", "validation", "test"} & set(factory["samples"])
        else:
            pre = LocalPreprocessor.load(state)
            assert pre.factory_id == factory_id
            assert pre.fit_rows > 0
        if factory["dataset"] == "nasa":
            summary = factory["summary"]
            assert not set(summary["training_engine_ids"]) & set(summary["validation_engine_ids"])
            if factory["role"] == "training":
                assert len(summary["training_engine_ids"]) + len(summary["validation_engine_ids"]) == factory["units"]
                train = np.load(output / f"{factory_id}-train.npz")
                val = np.load(output / f"{factory_id}-validation.npz")
                assert not set(train["engine_ids"]) & set(val["engine_ids"])
    for artifact in catalog["artifacts"]:
        with np.load(output / artifact["path"], allow_pickle=False) as arrays:
            assert len(arrays["x"]) == artifact["samples"]
            assert arrays["x"].shape[-1] == len(artifact["features"])
            assert np.isfinite(arrays["y"]).all()
            if artifact["transformed"]:
                assert np.isfinite(arrays["x"]).all()
            if "mask" in arrays:
                assert (arrays["x"][~arrays["mask"]] == 0).all()


def test_two_full_runs_are_reproducible(prepared, tmp_path):
    first_output, first = prepared
    second = prepare(ROOT, PipelineConfig.load(ROOT / "configs/v3_data.json"), tmp_path / "repeat")
    assert first == second  # Metadata includes split membership and numeric-array digests.
    for path in first_output.glob("*-preprocessor.json"):
        assert path.read_bytes() == (tmp_path / "repeat" / path.name).read_bytes()


def test_real_heterogeneity_is_reported(prepared):
    _, catalog = prepared
    azure = [f for f in catalog["factories"] if f["dataset"] == "azure"]
    assert [f["units"] for f in azure] == [16, 17, 35, 32]
    assert len({f["summary"]["age_mean"] for f in azure}) > 1
    nasa = [f for f in catalog["factories"] if f["dataset"] == "nasa"]
    assert len({f["summary"]["cycle_length_mean"] for f in nasa}) > 1
