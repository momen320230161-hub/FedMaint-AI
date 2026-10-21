from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from fedmaint.data.contracts import DataError
from fedmaint.data.preprocessing import LocalPreprocessor
from fedmaint.models.contracts import FeatureContract
from fedmaint.models.data import (PreparedData, align_scaled, encode_input, shared_contract, transform_raw)


def state(client, columns):
    pre = LocalPreprocessor()
    pre.columns = columns
    pre.factory_id = client
    pre.fit_rows = 4
    pre.dropped = []
    pre.medians = np.ones(len(columns))
    pre.mean = np.ones(len(columns))
    pre.scale = np.full(len(columns), 2.)
    return pre


def test_training_only_union_has_stable_order_and_source_selection():
    catalog = {"factories": [{"factory_id": c, "dataset": "azure", "role": "training"} for c in ("a", "b", "c")]
               + [{"factory_id": "held", "dataset": "azure", "role": "held_out"}], "config": {}}
    states = {"a": state("a", ["feat_a"]), "b": state("b", ["feat_b", "feat_a"]), "c": state("c", ["feat_b"])}
    contract = shared_contract(catalog, "azure", states)
    assert contract.features == ("feat_a", "feat_b")
    assert contract.source_factory == "b"
    for client in states:
        aligned = align_scaled(np.ones((5, len(states[client].columns))), states[client].columns, contract)
        assert aligned.shape == (5, 2)
    changed = deepcopy(states)
    changed["held"] = state("held", ["feat_future_or_held_out_secret"])
    assert shared_contract(catalog, "azure", changed) == contract


def test_frozen_transform_never_fits_or_adapts_to_held_out(monkeypatch):
    contract = FeatureContract(dataset="azure", features=("feat_a", "feat_b"), training_factories=("a",),
                               source_factory="a", window_length=1)
    pre = state("a", ["feat_a"])
    before = pre.mean.copy(), pre.scale.copy(), pre.medians.copy()
    def forbidden(*args, **kwargs):
        raise AssertionError("No fitting during held-out inference")
    monkeypatch.setattr(LocalPreprocessor, "fit", forbidden)
    result = transform_raw(np.array([[1e8, 7.], [np.nan, 99.]]), ["feat_a", "feat_b"], None, pre, contract)
    assert result[0, 0] == (1e8 - 1) / 2
    assert result[1, 0] == 0
    assert (result[:, 1] == 0).all()  # Locally dropped features stay neutral, even if held-out variation exists.
    for now, old in zip((pre.mean, pre.scale, pre.medians), before, strict=True):
        np.testing.assert_array_equal(now, old)
    with pytest.raises(DataError):
        transform_raw(np.ones((2, 1)), ["feat_a"], None, state("held", ["feat_a"]), contract)


def test_nasa_encoding_obeys_padding_and_keeps_mask_information():
    contract = FeatureContract(dataset="nasa", features=("feat_a", "feat_b"), training_factories=("a",),
                               source_factory="a", window_length=3)
    x = np.ones((2, 3, 2))
    mask = np.array([[False, True, True], [True, True, True]])
    encoded = encode_input(x, mask, contract)
    changed = x.copy()
    changed[~mask] = np.nan
    np.testing.assert_array_equal(encode_input(changed, mask, contract), encoded)
    assert encoded.shape == (2, 9)
    assert encoded[0, -3:].tolist() == [0, 1, 1]
    with pytest.raises(DataError):
        encode_input(x, np.array([[True, False, True], [True, True, True]]), contract)


def test_real_dimensions_identity_and_engine_source_isolation(prepared):
    output, _ = prepared
    from pathlib import Path
    data = PreparedData(Path(__file__).resolve().parents[1], output)
    assert {d: c.input_dimension for d, c in data.contracts.items()} == {"azure": 21, "nasa": 750}
    assert data.contracts["nasa"].source_factory == "nasa-FD002"
    for dataset, contract in data.contracts.items():
        for client in contract.training_factories:
            train, val = data.load(client, "train"), data.load(client, "validation")
            test = data.load(client, "test" if dataset == "azure" else "official_test")
            assert train.x.shape[1] == val.x.shape[1] == test.x.shape[1] == contract.input_dimension
            assert not set(train.keys) & set(val.keys)
            assert not set(train.keys) & set(test.keys)
            assert not set(val.keys) & set(test.keys)
