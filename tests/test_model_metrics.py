import math

import numpy as np
import pytest

from fedmaint.data.contracts import DataError
from fedmaint.data.batches import Batch
from fedmaint.models.metrics import classification_metrics, regression_metrics, validation_threshold


def test_classification_definitions_and_confusion_order():
    result = classification_metrics([0, 0, 1, 1], [.1, .6, .7, .8], .5)
    assert result["pr_auc"] == result["average_precision"] == result["roc_auc"] == 1
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["precision"] == 2 / 3
    assert result["recall"] == 1
    assert result["f1"] == .8
    assert result["positive_prevalence"] == .5


def test_undefined_classification_metrics_are_explicit():
    result = classification_metrics([0, 0], [.1, .2], .5)
    assert result["roc_auc"] is result["pr_auc"] is result["average_precision"] is None
    assert result["f1"] == result["precision"] == result["recall"] == 0


def test_raw_rul_errors_and_official_asymmetric_nasa_score():
    result = regression_metrics([100, 100], [87, 110])
    assert result["rmse"] == math.sqrt((13**2 + 10**2) / 2)
    assert result["mae"] == 11.5
    assert result["nasa_score"] == pytest.approx(2 * math.expm1(1))
    assert result["target_convention"] == "raw_uncapped"
    assert regression_metrics([0, 1000], [0, 1000])["nasa_score"] == 0


def test_nasa_overflow_retains_log_and_scientific_score_instead_of_infinity():
    result = regression_metrics([0], [20000])
    assert result["nasa_score"] is None
    assert result["nasa_score_overflow"] is True
    assert result["nasa_score_log"] == pytest.approx(2000)
    assert "e+868" in result["nasa_score_scientific"]


def test_threshold_is_selected_only_from_permitted_validation():
    batch = Batch(np.zeros((4, 1)), np.array([0, 1, 0, 1]), np.array(["a", "b", "c", "d"]),
                  np.repeat("client", 4), "validation")
    assert validation_threshold(batch, np.array([.1, .4, .35, .8]), ("client",)) == .4
    for split in ("train", "test", "held_out"):
        batch.split = split
        with pytest.raises(DataError):
            validation_threshold(batch, np.array([.1, .4, .35, .8]), ("client",))
    batch.split = "validation"
    with pytest.raises(DataError):
        validation_threshold(batch, np.array([.1, .4, .35, .8]), ("other",))
