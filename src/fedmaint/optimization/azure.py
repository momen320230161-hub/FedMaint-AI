"""Controlled source-model ablations; model4 is removed before sample generation."""

import hashlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from fedmaint.data.azure import SCHEMAS, validate_azure, make_azure_samples
from fedmaint.data.contracts import PipelineConfig
from fedmaint.data.preprocessing import LocalPreprocessor
from fedmaint.models.data import Batch
from fedmaint.models.metrics import classification_metrics, validation_threshold
from .common import Budget, digest, write_json


def source_azure(root):
    manifest = json.loads((root / "data/manifest.json").read_text())["azure"]["files"]
    for name, record in manifest.items():
        if digest(root / "data/Azure Data" / name) not in {record["sha256"], record.get("repository_sha256")}:
            raise ValueError(f"Azure provenance mismatch: {name}")
    machines = pd.read_csv(root / "data/Azure Data/PdM_machines.csv")
    machines = machines.loc[machines.model != "model4"]
    allowed = set(machines.machineID)
    tables = {"machines": machines}
    for name in SCHEMAS:
        if name == "machines":
            continue
        chunks = [c.loc[c.machineID.isin(allowed)] for c in pd.read_csv(root / f"data/Azure Data/PdM_{name}.csv", chunksize=50000)]
        tables[name] = pd.concat(chunks, ignore_index=True)
    # Reading CSV chunks is an ingestion filter, never computation of model4 statistics.
    return validate_azure(tables)


def run_azure(root, output, config):
    path = output / "azure.json"
    if path.exists():
        raise ValueError("Azure audit already recorded")
    raw = source_azure(root)
    timing = {}
    failure_keys = set(zip(raw.failures.machineID, raw.failures.datetime))
    for name in ("errors", "maint"):
        events = getattr(raw, name)
        timing[name] = {"events": len(events), "exact_failure_time_events": sum((m,t) in failure_keys for m,t in zip(events.machineID,events.datetime)),
                        "posting_timestamps_available": False, "duplicate_events": int(events.duplicated().sum())}
    base = PipelineConfig.load(root / "configs/phase01.json").azure
    arms = [("telemetry", ("telemetry",), 1), ("errors", ("telemetry", "errors"), 1),
            ("maintenance", ("telemetry", "maintenance"), 1), ("full", ("telemetry", "errors", "maintenance"), 1),
            ("full_lag6", ("telemetry", "errors", "maintenance"), 6),
            ("full_lag12", ("telemetry", "errors", "maintenance"), 12)]
    budget, runs, population = Budget(output, config), [], None
    for arm, groups, lag in arms:
        cfg = base.model_copy(update={"feature_groups": groups, "event_lag_hours": lag})
        frame = make_azure_samples(raw, cfg)
        keys = frame.machineID.astype(str) + "/" + frame.datetime.astype(str)
        if population is None:
            population = keys.tolist()
        if population != keys.tolist():
            raise ValueError("Ablations changed sample populations")
        features = [c for c in frame if c.startswith("feat_")]
        for held in ("azure-model1", "azure-model2", "azure-model3"):
            members = tuple(f"azure-model{i}" for i in (1, 2, 3) if f"azure-model{i}" != held)
            states = {}
            for client in members:
                rows = frame.loc[(frame.factory_id == client) & (frame.split == "train")]
                states[client] = LocalPreprocessor().fit(rows, features, factory_id=client, training_factories=set(members))
            union = sorted(set().union(*(s.columns for s in states.values())))
            source = min(members, key=lambda client: (-len(states[client].columns), client))
            def transformed(rows, state):
                local = state.transform(rows)
                aligned = np.zeros((len(rows), len(union)))
                for i,c in enumerate(state.columns):
                    aligned[:, union.index(c)] = local[:, i]
                return aligned
            def pool(split):
                groups = [frame.loc[(frame.factory_id == client) & (frame.split == split)] for client in members]
                x = np.concatenate([transformed(g, states[c]) for g,c in zip(groups,members)])
                rows = pd.concat(groups)
                return x, rows
            x, train = pool("train")
            vx, val = pool("validation")
            test = frame.loc[(frame.factory_id == held) & (frame.split == "test")]
            tx = transformed(test, states[source])
            directory = output / "azure" / f"{arm}-{held}"
            directory.mkdir(parents=True, exist_ok=False)
            started = budget.start(str(directory.relative_to(output)))
            try:
                model = MLPClassifier(hidden_layer_sizes=(32,), activation="relu", solver="adam", alpha=config["alpha"],
                                      batch_size=128, learning_rate_init=.001, random_state=42, early_stopping=False)
                best, best_score, stale, selected, history = None, -np.inf, 0, None, []
                with threadpool_limits(limits=1):
                    for epoch in range(1, config["epochs"]+1):
                        if time.perf_counter()-started >= budget.remaining:
                            raise ValueError("Training-time budget exhausted")
                        model.partial_fit(x, train.target.to_numpy(), classes=np.array([0,1]))
                        probability = model.predict_proba(vx)[:,1]
                        score = float(average_precision_score(val.target, probability))
                        history.append({"epoch":epoch,"inner_validation_ap":score})
                        if score > best_score:
                            import copy
                            best, best_score, stale, selected = copy.deepcopy(model), score, 0, epoch
                        else:
                            stale += 1
                        if stale >= config["patience"]:
                            break
                    vbatch = Batch(vx, val.target.to_numpy(),
                                   (val.machineID.astype(str)+"/"+val.datetime.astype(str)).to_numpy(),
                                   val.factory_id.to_numpy(), "validation", "source-only")
                    threshold = validation_threshold(vbatch, best.predict_proba(vx)[:,1], members)
                    probability = best.predict_proba(tx)[:,1]
                metrics = classification_metrics(test.target.to_numpy(), probability, threshold)
                pd.DataFrame({"machine_id":test.machineID,"time":test.datetime,"truth":test.target,
                              "probability":probability,"decision":probability>=threshold}).to_csv(directory/"predictions.csv",index=False,float_format="%.17g")
                np.savez_compressed(directory/"weights.npz", **{f"weight_{i}":w for i,w in enumerate(best.coefs_)},
                                    **{f"bias_{i}":b for i,b in enumerate(best.intercepts_)})
                for client,state in states.items():
                    state.save(directory/f"private-{client}.json")
                runs.append({"arm":arm,"groups":list(groups),"lag_hours":lag,"held_model":held,"fit_models":list(members),
                             "source":source,"seed":42,"status":"completed","selected_epoch":selected,
                             "history":history,"metrics":metrics,"no_skill_ap":float(test.target.mean()),
                             "directory":str(directory.relative_to(output)), "features":union,
                             "samples":{s:len(r) for s,r in (("train",train),("validation",val),("outer_test",test))},
                             "population_sha256":hashlib.sha256("\n".join(population).encode()).hexdigest(),
                             "fit_keys_sha256":hashlib.sha256("\n".join(train.machineID.astype(str)+"/"+train.datetime.astype(str)).encode()).hexdigest(),
                             "validation_keys_sha256":hashlib.sha256("\n".join(val.machineID.astype(str)+"/"+val.datetime.astype(str)).encode()).hexdigest()})
                budget.finish(started,"completed")
            except Exception as exc:
                budget.finish(started,"failed",error=str(exc))
                runs.append({"arm":arm,"held_model":held,"status":"failed","error":str(exc)})
            write_json(output/"azure-progress.json",{"runs":runs})
            print(f"AZURE {arm}/{held}: {runs[-1]['status']}",flush=True)
    result = {"status":"completed" if all(r["status"]=="completed" for r in runs) else "partial", "runs":runs,
              "identical_sample_population":True,"model4_used_for_selection":False,"availability_measured":False,
              "reference_recommendation":"telemetry_only_conservative_until_event_posting_is_known"}
    result["source_event_timing"] = timing
    result["raw_sha256"] = {p.name:digest(p) for p in (root / "data/Azure Data").glob("*.csv")}
    write_json(path,result)
    return result
