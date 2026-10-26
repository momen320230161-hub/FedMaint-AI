import copy
import hashlib
import time

import numpy as np
import torch

from fedmaint.data.contracts import DataError
from .common import write_json
from .torch_models import RULModel, parameter_count, predict, save_weights


def validation_score(batch, prediction):
    values = []
    for domain in sorted(set(batch.domain)):
        for cycle in (50, 100):
            index = (batch.domain == domain) & (batch.cycles == cycle)
            if not index.any():
                raise DataError("A required fixed validation prefix is missing")
            values.append(float(np.sqrt(np.mean((prediction[index]-batch.y[index])**2))))
    return float(np.mean(values))


def fit(train, validation, architecture, seed, config, directory, budget):
    if train.split != "train" or validation.split != "validation" or set(train.keys) & set(validation.keys):
        raise DataError("Fit/selection contamination")
    if set(train.domain) != set(validation.domain) or "FD004" in set(train.domain):
        raise DataError("Invalid fit/validation membership")
    run_id = str(directory.relative_to(budget.path.parent))
    started = budget.start(run_id)
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = RULModel(architecture, hidden=config["hidden"], window=config["window"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    x, mask = torch.from_numpy(train.x), torch.from_numpy(train.mask)
    truth = torch.tensor(train.y/config["rul_scale"], dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    best_score, best, stale, history, selected = float("inf"), None, 0, [], None
    try:
        for epoch in range(1, config["epochs"]+1):
            model.train()
            order = torch.randperm(len(truth), generator=generator)
            for indices in order.split(config["batch_size"]):
                if time.perf_counter()-started >= budget.remaining:
                    raise DataError("Global training-time cap reached")
                optimizer.zero_grad(set_to_none=True)
                result = model(x[indices], mask[indices])
                penalty = sum((parameter.square().sum() for name, parameter in model.named_parameters() if "weight" in name))
                loss = .5*(result-truth[indices]).square().mean() + config["alpha"]/(2*len(truth))*penalty
                if not bool(torch.isfinite(loss)):
                    raise DataError("Non-finite training loss")
                loss.backward()
                optimizer.step()
            prediction = predict(model, validation, config["rul_scale"])
            score = validation_score(validation, prediction)
            history.append({"epoch": epoch, "inner_macro_prefix_rmse": score, "last_batch_loss": float(loss.detach())})
            if score < best_score:
                best_score, best, stale, selected = score, copy.deepcopy(model.state_dict()), 0, epoch
            else:
                stale += 1
            if stale >= config["patience"]:
                break
        model.load_state_dict(best)
        save_weights(directory / "weights.npz", model)
        meta = {"architecture": architecture, "parameters": parameter_count(model), "seed": seed,
                "selected_epoch": selected, "inner_macro_prefix_rmse": best_score, "history": history,
                "train_domains": sorted(set(train.domain)), "train_samples": len(train.y),
                "validation_samples": len(validation.y), "postprocessing": config["postprocessing"]}
        for name, batch in (("fit", train), ("inner_validation", validation)):
            meta[name + "_keys_sha256"] = hashlib.sha256("\n".join(batch.keys).encode()).hexdigest()
            meta[name + "_engine_membership"] = sorted(set("/".join(key.split("/")[:3]) for key in batch.keys))
        write_json(directory / "training.json", meta)
        budget.finish(started, "completed", selected_epoch=selected)
        return model, meta
    except Exception as exc:
        budget.finish(started, "failed", error=f"{type(exc).__name__}: {exc}")
        write_json(directory / "failure.json", {"error": str(exc), "history": history})
        raise
