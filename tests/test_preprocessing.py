import numpy as np
import pandas as pd
import pytest

from fedmaint.data.contracts import DataError
from fedmaint.data.preprocessing import LocalPreprocessor


def training_frame():
    return pd.DataFrame({"factory_id": ["client"] * 3, "split": ["train"] * 3,
                         "feat_signal": [1., np.nan, 3.], "feat_constant": [5.] * 3,
                         "feat_missing": [np.nan] * 3, "target": [0, 1, 0]})


def fit(frame=None):
    return LocalPreprocessor().fit(training_frame() if frame is None else frame,
                                  ["feat_signal", "feat_constant", "feat_missing"],
                                  factory_id="client", training_factories={"client"})


def test_train_only_medians_constants_and_normalization():
    pre = fit()
    assert pre.columns == ["feat_signal", "feat_constant", "feat_missing"]
    assert pre.dropped == ["feat_constant", "feat_missing"]
    assert pre.medians.tolist() == [2, 5, 0]
    assert pre.mean.tolist() == [2, 5, 0]
    np.testing.assert_allclose(pre.transform(training_frame()).mean(axis=0), 0, atol=1e-15)
    np.testing.assert_allclose(pre.transform(training_frame()).std(axis=0), [1, 0, 0])


def test_adversarial_validation_and_held_out_values_do_not_affect_fit():
    pre = fit()
    mean, scale = pre.mean.copy(), pre.scale.copy()
    unseen = pd.DataFrame({"feat_signal": [1e15, np.nan], "feat_constant": [1e20] * 2, "feat_missing": [np.nan] * 2})
    transformed = pre.transform(unseen)
    assert transformed[0, 0] > 1e14
    assert transformed[1, 0] == 0
    np.testing.assert_array_equal(pre.mean, mean)
    np.testing.assert_array_equal(pre.scale, scale)


@pytest.mark.parametrize("defect", ["held_out_factory", "validation", "test", "held_out", "mixed_factory", "target_column"])
def test_fit_rejects_forbidden_records_or_columns(defect):
    frame = training_frame()
    features = ["feat_signal"]
    allowed = {"client"}
    if defect == "held_out_factory":
        allowed = set()
    elif defect == "mixed_factory":
        frame.loc[1, "factory_id"] = "other"
    elif defect == "target_column":
        features.append("target")
    else:
        frame.loc[1, "split"] = defect
    with pytest.raises(DataError):
        LocalPreprocessor().fit(frame, features, factory_id="client", training_factories=allowed)


def test_serialization_roundtrip(tmp_path):
    pre = fit()
    path = tmp_path / "preprocessor.json"
    pre.save(path)
    restored = LocalPreprocessor.load(path)
    np.testing.assert_array_equal(pre.transform(training_frame()), restored.transform(training_frame()))
    assert restored.factory_id == "client"


def test_missing_transform_column_and_unfitted_guard():
    with pytest.raises(DataError):
        LocalPreprocessor().transform(training_frame())
    with pytest.raises(DataError):
        fit().transform(pd.DataFrame({"target": [1]}))
