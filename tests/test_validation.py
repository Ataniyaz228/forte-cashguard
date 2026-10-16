import numpy as np
import pandas as pd
import pytest

from datathon.data import prepare_features
from datathon.splits import make_split
from datathon.train import choose_threshold


def config(strategy):
    return {
        "task": "classification",
        "target": "target",
        "seed": 42,
        "split": {"strategy": strategy, "time_column": "time", "group_column": "group"},
    }


def test_no_customer_crosses_group_partitions():
    frame = pd.DataFrame({"group": np.repeat(np.arange(100), 3), "target": [0, 1, 0] * 100})
    partitions = make_split(frame, config("group"))
    groups = [set(frame.iloc[idx].group) for idx in partitions]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
    assert sum(map(len, partitions)) == len(frame)


def test_identical_timestamps_never_cross_time_boundaries():
    frame = pd.DataFrame({"time": np.repeat(pd.date_range("2025-01-01", periods=40), 4)})
    cfg = config("time")
    cfg["split"]["gap_periods"] = 2
    train, validation, holdout = make_split(frame, cfg)
    assert frame.iloc[train].time.max() < frame.iloc[validation].time.min()
    assert frame.iloc[validation].time.max() < frame.iloc[holdout].time.min()
    assert len(set(train) | set(validation) | set(holdout)) < len(frame)


def test_missing_group_fails_instead_of_silent_leakage():
    frame = pd.DataFrame({"group": [1, None, 3, 4, 5]})
    with pytest.raises(ValueError, match="Missing group"):
        make_split(frame, config("group"))


def test_categorical_nulls_and_numeric_infinity_are_supported():
    frame = pd.DataFrame({"segment": ["A", None], "amount": [np.inf, 2]})
    result = prepare_features(frame, ["segment", "amount"], ["segment"])
    assert result.segment.isna().sum() == 0
    assert pd.isna(result.amount.iloc[0])


def test_numeric_strings_fail_if_they_hide_categories():
    with pytest.raises(ValueError):
        prepare_features(pd.DataFrame({"x": ["retail", "business"]}), ["x"], [])


def test_threshold_can_choose_reject_all_even_for_probability_one():
    threshold, table = choose_threshold(
        [0, 0], np.array([0.6, 1.0]), 1, {"false_positive": 10, "false_negative": 1}
    )
    assert threshold > 1
    assert table.cost.min() == 0


def test_bad_time_or_invalid_fraction_is_rejected():
    cfg = config("time")
    cfg["split"]["validation_size"] = 0.9
    with pytest.raises(ValueError):
        make_split(pd.DataFrame({"time": ["bad"]}), cfg)
