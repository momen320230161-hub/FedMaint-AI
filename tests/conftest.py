from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fedmaint.data.azure import AzureData, MODELS, SCHEMAS, SENSORS
from fedmaint.data.contracts import AzureConfig, PipelineConfig
from fedmaint.data.nasa import COLUMNS, FEATURES, NasaData
from fedmaint.data.prepare import prepare

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def azure_data():
    frames = []
    times = pd.date_range("2015-01-01", periods=120, freq="h")
    for machine in range(1, 5):
        frame = pd.DataFrame({"machineID": machine, "datetime": times})
        for i, sensor in enumerate(SENSORS):
            frame[sensor] = np.arange(120, dtype=float) + machine * (i + 1)
        frames.append(frame)
    machines = pd.DataFrame({"machineID": range(1, 5), "model": MODELS, "age": range(1, 5)})
    def events(column, value):
        return pd.DataFrame({"datetime": [times[10]] * 4, "machineID": range(1, 5), column: value})
    return AzureData(pd.concat(frames, ignore_index=True), machines, events("errorID", "error1"),
                     events("comp", "comp1"), events("failure", "comp1"))


@pytest.fixture
def azure_config():
    return AzureConfig(feature_groups=("telemetry","errors","maintenance"), history_hours=3, horizon_hours=4, stride_hours=1,
                       train_end="2015-01-02T16:00:00", validation_end="2015-01-04T08:00:00")


@pytest.fixture
def nasa_data():
    def trajectory(lengths):
        rows = []
        for engine, length in enumerate(lengths, 1):
            for cycle in range(1, length + 1):
                rows.append([engine, cycle, *[float(cycle * (i + 1) + engine) for i in range(24)]])
        return pd.DataFrame(rows, columns=COLUMNS)
    return NasaData("FD001", trajectory([8] * 6), trajectory([2, 5]), np.array([7, 11]))


@pytest.fixture(scope="session")
def prepared(tmp_path_factory):
    output = tmp_path_factory.mktemp("real-preparation")
    config = PipelineConfig.load(ROOT / "configs/v3_data.json")
    return output, prepare(ROOT, config, output)
