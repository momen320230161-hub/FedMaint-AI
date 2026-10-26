"""Predeclared descriptive source-only timing audit; no fit or model evaluation."""

import json
from pathlib import Path
import sys

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from fedmaint.data.azure import make_azure_samples
from fedmaint.data.contracts import PipelineConfig
from fedmaint.optimization.azure import source_azure
from fedmaint.optimization.common import digest,write_json
from fedmaint.optimization.workflow import verify_freeze


def main():
    output=ROOT/"outputs/phase021"
    verify_freeze(ROOT,output)
    raw=source_azure(ROOT)
    frame=make_azure_samples(raw,PipelineConfig.load(ROOT/"configs/phase01.json").azure)
    error_columns=[c for c in frame if c.startswith("feat_errors_")]
    error_present=frame[error_columns].sum(axis=1)>0
    conditions={}
    for label,index in (("error_present",error_present),("no_error",~error_present)):
        conditions[label]={"samples":int(index.sum()),"next_24h_failures":int(frame.loc[index,"target"].sum()),
                           "failure_prevalence":float(frame.loc[index,"target"].mean())}
    histograms={}
    for name,series in (("prediction",frame.datetime),("error_event",raw.errors.datetime),
                        ("maintenance_event",raw.maint.datetime),("failure_event",raw.failures.datetime)):
        histograms[name]={str(k):int(v) for k,v in series.dt.hour.value_counts().sort_index().items()}
    result={"status":"descriptive_audit_only","source_models":sorted(raw.machines.model.unique()),
            "model4_used":False,"samples":len(frame),"splits":frame.split.value_counts().to_dict(),
            "hour_of_day_histograms":histograms,"error_conditioned_target_prevalence":conditions,
            "posting_timestamps_available":False,"selection_or_configuration_changes":False,
            "interpretation":"Descriptive source association on a synthetic daily grid, not proof of event availability or causal leakage.",
            "code_sha256":digest(Path(__file__)),"freeze_sha256":digest(output/"freeze.json")}
    write_json(output/"availability-audit.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
