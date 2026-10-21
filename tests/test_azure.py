from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from fedmaint.data.azure import make_azure_samples, validate_azure
from fedmaint.data.contracts import DataError


def test_targets_open_left_closed_right_and_censoring(azure_data, azure_config):
    times = pd.to_datetime(["2015-01-01 10:00", "2015-01-01 14:00"])
    failures = pd.DataFrame({"machineID": [1, 1], "datetime": times, "failure": ["comp1", "comp2"]})
    samples = make_azure_samples(replace(azure_data, failures=failures), azure_config)
    machine = samples.loc[samples.machineID == 1].set_index("datetime")
    assert machine.loc[times[0], "target"] == 1  # Event at t+H included.
    assert machine.loc[times[1], "target"] == 0  # Failure at t excluded.
    assert machine.loc[times[0] - pd.Timedelta(hours=1), "target"] == 1
    assert (machine.target_end <= azure_data.telemetry.datetime.max()).all()


def test_exact_history_and_delayed_event_availability(azure_data, azure_config):
    frame = make_azure_samples(azure_data, azure_config)
    row = frame.loc[(frame.machineID == 1) & (frame.datetime == "2015-01-01 10:00")].iloc[0]
    assert row.feat_volt_mean == 10  # values at hours 8,9,10 are 9,10,11
    assert row.feat_errors_error1_count == 0
    assert np.isnan(row.feat_maintenance_comp1_hours_since)
    next_row = frame.loc[(frame.machineID == 1) & (frame.datetime == "2015-01-01 11:00")].iloc[0]
    assert next_row.feat_errors_error1_count == 1
    assert next_row.feat_maintenance_comp1_hours_since == 1


def test_adversarial_future_changes_cannot_change_past_features(azure_data, azure_config):
    original = make_azure_samples(azure_data, azure_config)
    cutoff = pd.Timestamp("2015-01-01 20:00")
    altered = replace(azure_data, telemetry=azure_data.telemetry.copy(), errors=azure_data.errors.copy(),
                      maint=azure_data.maint.copy(), failures=azure_data.failures.copy())
    altered.telemetry.loc[altered.telemetry.datetime > cutoff, "volt"] = 1e12
    for name, column, value in (("errors", "errorID", "error5"), ("maint", "comp", "comp4"),
                                 ("failures", "failure", "comp3")):
        table = getattr(altered, name)
        table.loc[len(table)] = {"datetime": cutoff + pd.Timedelta(hours=1), "machineID": 1, column: value}
    changed = make_azure_samples(altered, azure_config)
    cols = [c for c in original if c.startswith("feat_")]
    pd.testing.assert_frame_equal(original.loc[original.datetime <= cutoff, cols],
                                  changed.loc[changed.datetime <= cutoff, cols])


def test_temporal_embargo_and_held_out_isolation(azure_data, azure_config):
    frame = make_azure_samples(azure_data, azure_config)
    assert set(frame.loc[frame.factory_id == "azure-model4", "split"]) == {"held_out"}
    assert not frame.loc[frame.split == "train"].factory_id.eq("azure-model4").any()
    train_end, val_end = pd.Timestamp(azure_config.train_end), pd.Timestamp(azure_config.validation_end)
    assert (frame.loc[frame.split == "train", "target_end"] < train_end).all()
    validation = frame.loc[frame.split == "validation"]
    assert (validation.history_start >= train_end).all()
    assert (validation.target_end < val_end).all()
    assert (frame.loc[frame.split == "test", "history_start"] >= val_end).all()


def test_feature_group_ablation(azure_data, azure_config):
    config = azure_config.model_copy(update={"feature_groups": ("telemetry",)})
    sample = make_azure_samples(azure_data, config)
    assert not any("maintenance" in c or "errors" in c or c == "feat_age" for c in sample)
    assert all("failure" not in c and "target" not in c for c in sample if c.startswith("feat_"))


@pytest.mark.parametrize("defect", ["duplicate", "gap", "unknown_machine", "missing_timestamp", "invalid_event", "infinite", "invalid_model"])
def test_schema_rejects_invalid_inputs(azure_data, defect):
    tables = {name: getattr(azure_data, name).copy() for name in ("telemetry", "machines", "errors", "maint", "failures")}
    if defect == "duplicate":
        tables["telemetry"] = pd.concat([tables["telemetry"], tables["telemetry"].iloc[:1]])
    elif defect == "gap":
        tables["telemetry"] = tables["telemetry"].drop(2)
    elif defect == "unknown_machine":
        tables["errors"].loc[0, "machineID"] = 1000
    elif defect == "missing_timestamp":
        tables["telemetry"].loc[0, "datetime"] = pd.NaT
    elif defect == "invalid_event":
        tables["maint"].loc[0, "comp"] = "unknown"
    elif defect == "infinite":
        tables["telemetry"].loc[0, "volt"] = np.inf
    else:
        tables["machines"].loc[0, "model"] = "unknown"
    with pytest.raises((DataError, ValueError)):
        validate_azure(tables)


def test_missing_numeric_values_remain_missing_for_train_only_imputation(azure_data):
    tables = {name: getattr(azure_data, name).copy() for name in ("telemetry", "machines", "errors", "maint", "failures")}
    tables["telemetry"].loc[0, "volt"] = np.nan
    clean = validate_azure(tables)
    assert np.isnan(clean.telemetry.iloc[0].volt)
