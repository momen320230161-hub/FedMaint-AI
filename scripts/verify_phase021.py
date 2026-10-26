"""Audit frozen artifacts and saved predictions without another model evaluation."""

import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from fedmaint.models.metrics import regression_metrics,classification_metrics
from fedmaint.optimization.audit import verify_preservation
from fedmaint.optimization.common import write_json,load_config
from fedmaint.optimization.torch_models import load_weights,parameter_count
from fedmaint.optimization.workflow import verify_freeze


def equal(actual,expected):
    assert actual.keys()==expected.keys()
    for key,value in expected.items():
        result=actual[key]
        if isinstance(value,float):
            assert math.isclose(result,value,rel_tol=1e-12,abs_tol=1e-12),(key,result,value)
        else:
            assert result==value,(key,result,value)


def verify():
    out=ROOT/"outputs/phase021"
    frozen=verify_freeze(ROOT,out)
    cfg=load_config(ROOT)
    selection=json.loads((out/"selection.json").read_text())
    evaluation=json.loads((out/"evaluation.json").read_text())
    azure=json.loads((out/"azure.json").read_text())
    budget=json.loads((out/"budget.json").read_text())
    assert evaluation["status"]==azure["status"]==selection["status"]=="completed"
    assert len(budget["fits"])<=69 and budget["training_seconds"]<=2700
    assert all(r["status"]=="completed" for r in budget["fits"])
    assert all(r["domain"]!="FD004" and r["source"]=="train" for r in selection["access"])
    count=0
    for run in selection["runs"]:
        assert run["held_out"] not in run["training"]["train_domains"]
        assert len(run["training"]["train_domains"])==2
        fit=set(run["training"]["fit_engine_membership"])
        val=set(run["training"]["inner_validation_engine_membership"])
        assert not fit&val
        assert all(not k.startswith(run["held_out"]+"/") for k in fit|val)
        path=out/run["directory"]
        for state in path.glob("private-*.json"):
            normal=json.loads(state.read_text())
            assert normal["domain"] in run["training"]["train_domains"]
            assert {f"{normal['domain']}/train/{i}" for i in normal["engine_ids"]} <= fit
        frame=pd.read_csv(path/run["outer"]["predictions"],float_precision="round_trip")
        assert not frame.key.duplicated().any()
        for cycle in (50,100):
            group=frame.loc[frame.cycle==cycle]
            for policy,column in (("raw","raw_prediction"),("zero_floor_diagnostic","zero_floor_prediction")):
                equal(regression_metrics(group.raw_truth,group[column]),run["outer"]["metrics"][str(cycle)][policy])
        count+=1
    for run in evaluation["runs"]:
        frame=pd.read_csv(out/run["directory"]/run["predictions"],float_precision="round_trip")
        assert not frame.key.duplicated().any()
        assert np.isfinite(frame.raw_prediction).all()
        np.testing.assert_array_equal(frame.zero_floor_prediction,np.maximum(frame.raw_prediction,0))
        for policy,column in (("raw","raw_prediction"),("zero_floor_diagnostic","zero_floor_prediction")):
            equal(regression_metrics(frame.raw_truth,frame[column]),run["metrics"]["all"][policy])
        if run["pool"]=="held_out_official_test":
            assert len(frame)==248
        count+=1
    for run in azure["runs"]:
        assert run["held_model"] not in run["fit_models"] and len(run["fit_models"])==2
        assert "azure-model4" not in run["fit_models"]
        frame=pd.read_csv(out/run["directory"]/"predictions.csv",float_precision="round_trip")
        assert not frame.duplicated(["machine_id","time"]).any()
        equal(classification_metrics(frame.truth,frame.probability,run["metrics"]["threshold"]),run["metrics"])
        np.testing.assert_array_equal(frame.decision,frame.probability>=run["metrics"]["threshold"])
        count+=1
    nasa_models=0
    for path in (out/"final-models").glob("*/model.json"):
        record=json.loads(path.read_text())
        model=load_weights(path.parent/"weights.npz",record["architecture"],cfg)
        assert parameter_count(model)==record["training"]["parameters"]
        assert all("FD004" not in str(k) for k in record["members"])
        nasa_models+=1
    preservation=verify_preservation(ROOT,out)
    result={"status":"verified","frozen_files":len(frozen["files"]),"saved_prediction_files_recomputed":count,
            "final_nasa_models_reloaded":nasa_models,"fits":len(budget["fits"]),
            "failed_fits":sum(r["status"]!="completed" for r in budget["fits"]),
            "training_seconds":budget["training_seconds"],"preservation":preservation,
            "audit_only_no_raw_heldout_read_or_model_forward":True}
    write_json(out/"verification.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    verify()
