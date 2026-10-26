"""Exploratory LODO screening/confirmation only; no FD004 data dependency."""

import json

import numpy as np
import pandas as pd

from fedmaint.models.metrics import regression_metrics
from .common import Budget, DOMAINS, digest, write_json
from .nasa import NasaStore, choose_source, combine
from .normalization import LocalNormalizer
from .torch_models import predict
from .training import fit


def score_predictions(directory, name, batch, prediction, *, prefixes=False):
    path = directory / f"{name}-predictions.csv"
    pd.DataFrame({"key": batch.keys, "domain": batch.domain, "cycle": batch.cycles, "raw_truth": batch.y,
                  "raw_prediction": prediction, "zero_floor_prediction": np.maximum(prediction, 0)}).to_csv(
                      path, index=False, float_format="%.17g")
    groups = [(str(c), batch.cycles == c) for c in (50, 100)] if prefixes else [("all", np.ones(len(batch.y), bool))]
    metrics = {}
    for group, index in groups:
        if not index.any():
            raise ValueError("Missing required evaluation prefix")
        metrics[group] = {"raw": regression_metrics(batch.y[index], prediction[index]),
                          "zero_floor_diagnostic": regression_metrics(batch.y[index], np.maximum(prediction[index], 0)),
                          "negative_rate": float((prediction[index] < 0).mean()),
                          "minimum_prediction": float(prediction[index].min()), "maximum_prediction": float(prediction[index].max()),
                          "max_absolute_error": float(np.abs(prediction[index]-batch.y[index]).max()),
                          "absolute_error_p95": float(np.quantile(np.abs(prediction[index]-batch.y[index]), .95))}
    return {"predictions": path.name, "sha256": digest(path), "metrics": metrics}


def candidate_summary(runs, kind, architecture, seed):
    selected = [r for r in runs if r["normalizer"] == kind and r["architecture"] == architecture and r["seed"] == seed]
    if len(selected) != 3 or any(r["status"] != "completed" for r in selected):
        return {"status": "incomplete"}
    folds = {r["held_out"]: float(np.mean([r["outer"]["metrics"][str(c)]["raw"]["rmse"] for c in (50, 100)])) for r in selected}
    return {"status": "completed", "macro_rmse": float(np.mean(list(folds.values()))),
            "worst_domain_rmse": max(folds.values()), "fold_rmse": folds}


def supports(candidate, baseline):
    return (candidate["status"] == baseline["status"] == "completed"
            and candidate["macro_rmse"] < baseline["macro_rmse"]
            and candidate["worst_domain_rmse"] <= baseline["worst_domain_rmse"]
            and sum(candidate["fold_rmse"][d] < baseline["fold_rmse"][d] for d in DOMAINS) >= 2)


def run_source(root, output, config):
    target = output / "selection.json"
    if target.exists():
        raise ValueError("Source selection has already been recorded; no silent repeat")
    budget, store = Budget(output, config), NasaStore(root, config)
    runs, cache = [], {}
    def data(kind, held):
        key = kind, held
        if key not in cache:
            members = tuple(d for d in DOMAINS if d != held)
            states, train, validation = {}, [], []
            for domain in members:
                fit_rows, val_rows = store.partition(domain)
                states[domain] = LocalNormalizer(kind, config).fit(fit_rows, domain=domain, permitted=members)
                train.append(store.windows(fit_rows, states[domain], "train")[0])
                validation.append(store.windows(val_rows, states[domain], "validation", prefixes=True)[0])
            cache[key] = states, combine(train), combine(validation)
        return cache[key]

    def one(kind, architecture, held, seed, stage):
        run = {"normalizer": kind, "architecture": architecture, "held_out": held, "seed": seed, "stage": stage}
        directory = output / stage / f"{kind}-{architecture}-{held}-seed{seed}"
        try:
            states, train, val = data(kind, held)
            model, meta = fit(train, val, architecture, seed, config, directory, budget)
            source = choose_source(states)
            for domain, state in states.items():
                state.save(directory / f"private-{domain}.json")
            outer, diagnostics = store.windows(store.read(held), states[source], "outer", prefixes=True)
            run.update(status="completed", source=source, training=meta, diagnostic=diagnostics,
                       outer=score_predictions(directory, "outer", outer, predict(model, outer), prefixes=True),
                       directory=str(directory.relative_to(output)))
        except Exception as exc:
            run.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        runs.append(run)
        write_json(output / "source-progress.json", {"runs": runs, "access": store.access})
        print(f"{stage} {kind}/{architecture}/{held}/{seed}: {run['status']}", flush=True)

    for kind in config["normalizers"]:
        for architecture in config["architectures"]:
            for held in DOMAINS:
                one(kind, architecture, held, 42, "screen")
    summaries = {f"{k}-{a}": candidate_summary(runs, k, a, 42) for k in config["normalizers"] for a in config["architectures"]}
    eligible = [(name, score) for name,score in summaries.items() if score["status"] == "completed"]
    if not eligible or summaries["N0-mlp"]["status"] != "completed":
        write_json(target, {"status": "incomplete", "runs": runs, "screen": summaries})
        raise ValueError("Required source screening baseline is incomplete")
    # Rank by source macro error; ties use worst-domain error, then fixed lexical ordering.
    best = min(eligible, key=lambda item: (item[1]["macro_rmse"], item[1]["worst_domain_rmse"], item[0]))[0]
    normalization = best.split("-")[0]
    panel = sorted(set([("N0", "mlp"), (normalization, "mlp"), (normalization, "gru")]))
    for kind, architecture in panel:
        for held in DOMAINS:
            one(kind, architecture, held, 43, "confirm")
    confirmation = {f"{k}-{a}": candidate_summary(runs, k, a, 43) for k,a in panel}
    promoted = [name for name in confirmation if name != "N0-mlp" and
                supports(summaries[name], summaries["N0-mlp"]) and supports(confirmation[name], confirmation["N0-mlp"])]
    winner = min(promoted, key=lambda name: np.mean([summaries[name]["macro_rmse"], confirmation[name]["macro_rmse"]])) if promoted else "N0-mlp"
    report = {"status": "completed" if all(r["status"] == "completed" for r in runs) else "partial",
              "screen": summaries, "confirmation": confirmation, "winner": winner, "promoted": promoted,
              "panel": [f"{k}-{a}" for k,a in panel], "shortlisted_normalizer": normalization,
              "selection_data": list(DOMAINS), "exploratory_not_unbiased": True, "runs": runs, "access": store.access}
    write_json(target, report)
    return report
