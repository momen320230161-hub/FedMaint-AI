"""Bounded, comparable budgets and explicit validation-only model/checkpoint selection."""

from dataclasses import dataclass
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from threadpoolctl import threadpool_limits

from fedmaint.data.contracts import DataError
from .contracts import FeatureContract, TrainingConfig
from .data import Batch
from .metrics import validation_threshold
from .network import FrozenNetwork


@dataclass
class TrainingResult:
    network: FrozenNetwork
    threshold: float | None
    selection: dict


def train_model(train: Batch, validation: Batch, contract: FeatureContract, architecture: str,
                config: TrainingConfig, seed: int, *, fitted_factories: tuple[str, ...]) -> TrainingResult:
    if not set(fitted_factories) <= set(contract.training_factories) or not fitted_factories:
        raise DataError("Training factory selection is not permitted")
    train.require("train", fitted_factories)
    validation.require("validation", fitted_factories)
    if set(train.keys) & set(validation.keys) or set(train.factories) != set(validation.factories):
        raise DataError("Training/validation overlap or client mismatch")
    if train.x.shape[1] != contract.input_dimension or validation.x.shape[1] != contract.input_dimension:
        raise DataError("Model input differs from shared feature contract")
    if seed not in config.seeds or architecture not in (("logistic", "mlp") if contract.dataset == "azure" else ("mlp",)):
        raise DataError("Unsupported seed/architecture for this task")
    classification = contract.dataset == "azure"
    fit_target = train.y if train.fit_y is None else train.fit_y
    if classification and not np.array_equal(fit_target, train.y):
        raise DataError("Classification fit labels differ from binary truth")
    if not classification and ((train.y < 0).any() or (validation.y < 0).any() or (fit_target < 0).any()):
        raise DataError("Negative RUL labels")
    if classification and (not np.isin(train.y, [0, 1]).all() or len(np.unique(train.y)) != 2
                           or not np.isin(validation.y, [0, 1]).all() or len(np.unique(validation.y)) != 2):
        raise DataError("Both training and validation classes are required")
    started = time.perf_counter()
    histories, best_network, best_score, selected = [], None, -np.inf, None
    candidates = config.logistic_c if architecture == "logistic" else config.learning_rates
    with threadpool_limits(limits=1):
        for candidate in candidates:
            if architecture == "logistic":
                estimator = LogisticRegression(C=candidate, max_iter=config.logistic_max_iter, tol=1e-6,
                                               random_state=seed, solver="newton-cholesky")
                estimator.fit(train.x, train.y)
                network = FrozenNetwork.from_estimator(estimator, contract.dataset, architecture, 1)
                score = float(average_precision_score(validation.y, network.predict(validation.x)))
                history = {"C": candidate, "epochs": [{"epoch": 1, "validation_score": score}],
                           "optimizer_iterations": int(estimator.n_iter_[0])}
                if score > best_score:
                    best_score, best_network = score, network
                    selected = {"C": candidate, "epoch": 1}
            else:
                estimator_type = MLPClassifier if classification else MLPRegressor
                estimator = estimator_type(hidden_layer_sizes=(config.hidden_units,), activation="relu", solver="adam",
                                           alpha=config.alpha, batch_size=min(config.batch_size, len(train.y)),
                                           learning_rate_init=candidate, random_state=seed, shuffle=True, early_stopping=False)
                history = {"learning_rate": candidate, "epochs": []}
                candidate_best, stale = -np.inf, 0
                for epoch in range(1, config.max_epochs + 1):
                    if classification:
                        estimator.partial_fit(train.x, train.y, classes=np.array([0., 1.]))
                    else:
                        estimator.partial_fit(train.x, fit_target / config.rul_scale)
                    network = FrozenNetwork.from_estimator(estimator, contract.dataset, architecture, 1.0 if classification else config.rul_scale)
                    prediction = network.predict(validation.x)
                    score = (float(average_precision_score(validation.y, prediction)) if classification else
                             -float(np.sqrt(np.mean(np.square(validation.y - prediction)))))
                    history["epochs"].append({"epoch": epoch, "validation_score": score, "training_loss": float(estimator.loss_)})
                    if score > best_score:
                        best_score, best_network = score, network
                        selected = {"learning_rate": candidate, "epoch": epoch}
                    if score > candidate_best + 1e-10:
                        candidate_best, stale = score, 0
                    else:
                        stale += 1
                    if stale >= config.patience:
                        break
            histories.append(history)
        threshold = validation_threshold(validation, best_network.predict(validation.x), fitted_factories) if classification else None
    return TrainingResult(best_network, threshold,
                          {"selected": selected, "validation_score": best_score,
                           "criterion": "average_precision" if classification else "negative_raw_window_rmse",
                           "threshold_rule": "validation_max_f1_highest_threshold_tie" if classification else None,
                           "train_samples": len(train.y), "validation_samples": len(validation.y),
                           "train_keys_sha256": _keys_digest(train.keys), "validation_keys_sha256": _keys_digest(validation.keys),
                           "train_arrays_sha256": train.source_digest, "validation_arrays_sha256": validation.source_digest,
                           "fitted_factories": list(fitted_factories), "candidate_history": histories,
                           "fit_target_source": "prepared_y", "validation_target_source": "raw_y" if not classification else "target",
                           "training_seconds": time.perf_counter() - started})


def _keys_digest(keys):
    import hashlib
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()
