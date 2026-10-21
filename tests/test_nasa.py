import numpy as np
import pytest

from fedmaint.data.contracts import DataError, NasaConfig
from fedmaint.data.nasa import engine_split, rul_targets, sequences, validate_trajectory


def test_training_rul_is_observed_final_cycle_minus_cycle(nasa_data):
    frame = rul_targets(nasa_data, "train")
    assert frame.groupby("engine_id").raw_rul.last().eq(0).all()
    assert frame.groupby("engine_id").raw_rul.first().eq(7).all()


def test_official_test_rul_offsets_every_cycle(nasa_data):
    frame = rul_targets(nasa_data, "test")
    assert frame.groupby("engine_id").raw_rul.last().tolist() == [7, 11]
    assert frame.groupby("engine_id").raw_rul.first().tolist() == [8, 15]


def test_short_test_trajectories_are_left_padded_and_masked(nasa_data):
    frame = rul_targets(nasa_data, "test")
    batch = sequences(frame, NasaConfig(window_cycles=4), terminal_only=True)
    assert batch.x.shape == (2, 4, 24)
    assert batch.mask.tolist() == [[False, False, True, True], [True] * 4]
    assert (batch.x[0, :2] == 0).all()
    assert batch.y.tolist() == [7, 11]
    assert batch.cycles.tolist() == [2, 5]


def test_capping_retains_uncapped_evaluation_targets(nasa_data):
    batch = sequences(rul_targets(nasa_data, "test"), NasaConfig(window_cycles=4, rul_cap=9), terminal_only=True)
    assert batch.y.tolist() == [7, 9]
    assert batch.raw_y.tolist() == [7, 11]


def test_engine_split_is_disjoint_complete_and_seeded(nasa_data):
    a, b = engine_split(nasa_data.train.engine_id, 42, 0.2)
    assert not a & b
    assert a | b == set(range(1, 7))
    assert (a, b) == engine_split(nasa_data.train.engine_id[::-1], 42, 0.2)
    assert (a, b) != engine_split(nasa_data.train.engine_id, 43, 0.2)


@pytest.mark.parametrize("defect", ["duplicate", "gap", "nan", "fractional", "start"])
def test_trajectory_validation(nasa_data, defect):
    frame = nasa_data.train.copy()
    if defect == "duplicate":
        frame.loc[len(frame)] = frame.iloc[0]
    elif defect == "gap":
        frame = frame.drop(2)
    elif defect == "nan":
        frame.loc[0, "feat_sensor_1"] = np.nan
    elif defect == "fractional":
        frame["cycle"] = frame.cycle.astype(float)
        frame.loc[0, "cycle"] = 0.5
    else:
        frame = frame.drop(0)
    with pytest.raises(DataError):
        validate_trajectory(frame)
