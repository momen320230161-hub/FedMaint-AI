"""Hourly, machine-local Azure features and right-censored binary targets."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import AzureConfig, DataError

SENSORS = ("volt", "rotate", "pressure", "vibration")
MODELS = ("model1", "model2", "model3", "model4")
COMPONENTS = ("comp1", "comp2", "comp3", "comp4")
ERRORS = ("error1", "error2", "error3", "error4", "error5")
SCHEMAS = {
    "telemetry": ("datetime", "machineID", *SENSORS),
    "machines": ("machineID", "model", "age"),
    "errors": ("datetime", "machineID", "errorID"),
    "maint": ("datetime", "machineID", "comp"),
    "failures": ("datetime", "machineID", "failure"),
}


@dataclass
class AzureData:
    telemetry: pd.DataFrame
    machines: pd.DataFrame
    errors: pd.DataFrame
    maint: pd.DataFrame
    failures: pd.DataFrame


def validate_azure(tables: dict[str, pd.DataFrame]) -> AzureData:
    cleaned = {}
    for name, columns in SCHEMAS.items():
        frame = tables[name].copy()
        if set(frame.columns) != set(columns) or (frame.empty and name in ("telemetry", "machines")):
            raise DataError(f"Azure {name}: invalid columns or empty table")
        ids = pd.to_numeric(frame.machineID, errors="raise")
        if ids.isna().any() or (ids <= 0).any() or (ids % 1 != 0).any():
            raise DataError(f"Azure {name}: invalid machine IDs")
        frame["machineID"] = ids.astype("int64")
        if "datetime" in frame:
            frame["datetime"] = pd.to_datetime(frame.datetime, errors="raise")
            if frame.datetime.isna().any() or frame.datetime.dt.tz is not None:
                raise DataError(f"Azure {name}: invalid timestamps")
            if (frame.datetime != frame.datetime.dt.floor("h")).any():
                raise DataError(f"Azure {name}: timestamps must be hourly")
            frame = frame.sort_values(["machineID", "datetime"], kind="stable").reset_index(drop=True)
        if name == "telemetry":
            if frame.duplicated(["machineID", "datetime"]).any():
                raise DataError("Duplicate machine/timestamp telemetry")
            for col in SENSORS:
                frame[col] = pd.to_numeric(frame[col], errors="raise")
                if np.isinf(frame[col].to_numpy(dtype=float)).any():
                    raise DataError(f"Non-finite telemetry: {col}")
            deltas = frame.groupby("machineID").datetime.diff().dropna()
            if not deltas.eq(pd.Timedelta(hours=1)).all():
                raise DataError("Telemetry has missing hourly rows; do not silently bridge gaps")
        elif name == "machines":
            frame["age"] = pd.to_numeric(frame.age, errors="raise")
            if (frame.machineID.duplicated().any() or not frame.model.isin(MODELS).all()
                    or frame.age.isna().any() or not np.isfinite(frame.age).all() or (frame.age < 0).any()):
                raise DataError("Invalid machine metadata")
        else:
            column, values = {"errors": ("errorID", ERRORS), "maint": ("comp", COMPONENTS),
                              "failures": ("failure", COMPONENTS)}[name]
            if not frame[column].isin(values).all() or frame.duplicated().any():
                raise DataError(f"Invalid or duplicate Azure {name} events")
        cleaned[name] = frame
    known = set(cleaned["machines"].machineID)
    if set(cleaned["telemetry"].machineID) != known:
        raise DataError("Telemetry and metadata machine sets differ")
    for name in ("errors", "maint", "failures"):
        if not set(cleaned[name].machineID) <= known:
            raise DataError(f"Unknown machines in {name}")
    return AzureData(**cleaned)


def load_azure(root: Path) -> AzureData:
    return validate_azure({name: pd.read_csv(root / "data/Azure Data" / f"PdM_{name}.csv")
                           for name in SCHEMAS})


def _ns(values) -> np.ndarray:
    return np.asarray(values, dtype="datetime64[ns]").astype("int64")


def make_azure_samples(data: AzureData, config: AzureConfig) -> pd.DataFrame:
    """Feature interval (t-W, t]; target is any component failure in (t, t+H].

    No failure columns enter features. Event records have a configurable availability lag.
    Full history AND target must fit strictly inside each chronological partition.
    """
    outputs = []
    metadata = data.machines.set_index("machineID")
    for machine_id, frame in data.telemetry.groupby("machineID", sort=True):
        frame = frame.reset_index(drop=True)
        times = pd.DatetimeIndex(frame.datetime)
        if len(times) < config.history_hours:
            continue
        start, end = times.min(), times.max()
        train_end, val_end = pd.Timestamp(config.train_end), pd.Timestamp(config.validation_end)
        if not start < train_end < val_end < end:
            raise DataError("Azure split boundaries fall outside a machine's observation range")
        indices = np.arange(config.history_hours - 1, len(frame), config.stride_hours)
        t = times[indices]
        history_start = t - pd.Timedelta(hours=config.history_hours - 1)
        target_end = t + pd.Timedelta(hours=config.horizon_hours)
        model = metadata.loc[machine_id, "model"]
        held_out = model == config.held_out_model
        splits = np.full(len(t), "purged", dtype=object)
        if held_out:
            splits[target_end <= end] = "held_out"
        else:
            splits[(history_start >= start) & (target_end < train_end)] = "train"
            splits[(history_start >= train_end) & (target_end < val_end)] = "validation"
            splits[(history_start >= val_end) & (target_end <= end)] = "test"
        sample = pd.DataFrame({"factory_id": f"azure-{model}", "machineID": machine_id,
                               "datetime": t, "history_start": history_start,
                               "target_end": target_end, "split": splits})
        failure_times = _ns(data.failures.loc[data.failures.machineID == machine_id, "datetime"])
        failure_times.sort()
        right = np.searchsorted(failure_times, _ns(target_end), side="right")
        left = np.searchsorted(failure_times, _ns(t), side="right")
        sample["target"] = (right > left).astype("int8")
        if "telemetry" in config.feature_groups:
            for sensor in SENSORS:
                series = frame[sensor]
                window = series.rolling(config.history_hours, min_periods=1)
                for stat in ("mean", "min", "max"):
                    sample[f"feat_{sensor}_{stat}"] = getattr(window, stat)().iloc[indices].to_numpy()
                sample[f"feat_{sensor}_count"] = window.count().iloc[indices].to_numpy()
        if "metadata" in config.feature_groups:
            sample["feat_age"] = float(metadata.loc[machine_id, "age"])
        for group, table, column, codes in (
            ("errors", data.errors, "errorID", ERRORS),
            ("maintenance", data.maint, "comp", COMPONENTS),
        ):
            if group not in config.feature_groups:
                continue
            events = table.loc[table.machineID == machine_id]
            lower = _ns(t - pd.Timedelta(hours=config.history_hours))
            available = _ns(t - pd.Timedelta(hours=config.event_lag_hours))
            for code in codes:
                event_times = np.sort(_ns(events.loc[events[column] == code, "datetime"]))
                past = np.searchsorted(event_times, available, side="right")
                count = past - np.searchsorted(event_times, lower, side="right")
                sample[f"feat_{group}_{code}_count"] = count
                if group == "maintenance":
                    age = np.full(len(t), np.nan)
                    valid = count > 0
                    age[valid] = (_ns(t)[valid] - event_times[past[valid] - 1]) / 3.6e12
                    sample[f"feat_{group}_{code}_hours_since"] = age
        outputs.append(sample.loc[sample.split != "purged"])
    if not outputs:
        raise DataError("No complete Azure windows")
    result = pd.concat(outputs, ignore_index=True)
    if result.empty:
        raise DataError("No Azure samples survive temporal purging")
    return result
