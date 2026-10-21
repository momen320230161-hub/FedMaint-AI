"""Explicit PR area/AP and raw-RUL metrics, including stable NASA score sums."""

import math

import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import (auc, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)

from fedmaint.data.contracts import DataError
from .data import Batch


def _vectors(y, prediction):
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.ndim != 1 or y.shape != prediction.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise DataError("Metrics require nonempty, aligned finite vectors")
    return y, prediction


def classification_metrics(y, probability, threshold: float) -> dict:
    y, probability = _vectors(y, probability)
    if not np.isin(y, [0, 1]).all() or not 0 <= threshold <= 1 or ((probability < 0) | (probability > 1)).any():
        raise DataError("Invalid classification labels/probabilities/threshold")
    predicted = probability >= threshold
    positive = int(y.sum())
    if positive:
        precision, recall, _ = precision_recall_curve(y, probability)
        pr_auc = float(auc(recall, precision))
        ap = float(average_precision_score(y, probability))
    else:
        pr_auc, ap = None, None  # Undefined, not a fabricated successful score.
    return {"samples": len(y), "positive_prevalence": float(y.mean()), "pr_auc": pr_auc,
            "average_precision": ap, "f1": float(f1_score(y, predicted, zero_division=0)),
            "precision": float(precision_score(y, predicted, zero_division=0)),
            "recall": float(recall_score(y, predicted, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, probability)) if 0 < positive < len(y) else None,
            "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(), "threshold": threshold}


def validation_threshold(validation: Batch, probability: np.ndarray, permitted: tuple[str, ...]) -> float:
    validation.require("validation", permitted)
    y, probability = _vectors(validation.y, probability)
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2 or ((probability < 0) | (probability > 1)).any():
        raise DataError("Threshold selection needs both validation classes and valid probabilities")
    precision, recall, thresholds = precision_recall_curve(y, probability)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denominator, out=np.zeros_like(denominator), where=denominator > 0)
    # Exact best F1, deterministic conservative tie-break, never held-out/test tuning.
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def regression_metrics(y, prediction) -> dict:
    y, prediction = _vectors(y, prediction)
    if (y < 0).any():
        raise DataError("RUL truth must be nonnegative")
    error = prediction - y
    rmse = float(np.sqrt(np.mean(np.square(error))))
    if not np.isfinite(rmse):
        raise DataError("Regression errors exceed supported numeric range")
    # Official asymmetric score: exp(-d/13)-1 for early d<0; exp(d/10)-1 for late d>=0.
    exponent = np.where(error < 0, -error / 13, error / 10)
    positive = exponent > 0
    if positive.any():
        a = exponent[positive]
        log_terms = a + np.log(-np.expm1(-a))
        log_score = float(logsumexp(log_terms))
        score = math.exp(log_score) if log_score < math.log(np.finfo(float).max) else None
        power = math.floor(log_score / math.log(10))
        mantissa = math.exp(log_score - power * math.log(10))
        scientific = f"{mantissa:.10g}e{power:+d}"
    else:
        log_score, score, scientific = None, 0.0, "0"
    return {"samples": len(y), "rmse": rmse, "mae": float(np.mean(np.abs(error))),
            "nasa_score": score, "nasa_score_log": log_score, "nasa_score_scientific": scientific,
            "nasa_score_overflow": score is None, "target_convention": "raw_uncapped"}
