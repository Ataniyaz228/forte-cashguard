import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def make_split(frame, config):
    """Three disjoint partitions. Early stopping never sees the final holdout."""
    cfg = config["split"]
    strategy = cfg["strategy"]
    valid = float(cfg.get("validation_size", 0.2))
    test = float(cfg.get("holdout_size", 0.2))
    if valid <= 0 or test <= 0 or valid + test >= 0.8:
        raise ValueError("Require positive validation/holdout sizes with sum < 0.8.")
    idx = np.arange(len(frame))
    seed = config.get("seed", 42)
    if strategy == "time":
        times = pd.to_datetime(frame[cfg["time_column"]], utc=True, errors="raise")
        if times.isna().any():
            raise ValueError("Missing timestamps cannot be split safely.")
        unique = np.sort(times.unique())
        n = len(unique)
        a, b = int(n * (1 - valid - test)), int(n * (1 - test))
        gap = int(cfg.get("gap_periods", 0))
        if gap < 0 or a - gap < 1 or b - gap <= a or b >= n:
            raise ValueError("Too few unique timestamps for these fractions/gap.")
        train = idx[times.isin(unique[: a - gap])]
        validation = idx[times.isin(unique[a : b - gap])]
        holdout = idx[times.isin(unique[b:])]
    elif strategy == "group":
        groups = frame[cfg["group_column"]]
        if groups.isna().any():
            raise ValueError("Missing group IDs cannot be split safely.")
        remain, holdout = next(
            GroupShuffleSplit(n_splits=1, test_size=test, random_state=seed).split(
                idx, groups=groups
            )
        )
        tr, va = next(
            GroupShuffleSplit(n_splits=1, test_size=valid / (1 - test), random_state=seed).split(
                remain, groups=groups.iloc[remain]
            )
        )
        train, validation = remain[tr], remain[va]
    elif strategy == "random":
        stratify = frame[config["target"]] if config["task"] == "classification" else None
        remain, holdout = train_test_split(
            idx, test_size=test, stratify=stratify, random_state=seed
        )
        train, validation = train_test_split(
            remain,
            test_size=valid / (1 - test),
            stratify=None if stratify is None else stratify.iloc[remain],
            random_state=seed,
        )
    else:
        raise ValueError("split.strategy must be random, group or time.")
    if min(map(len, [train, validation, holdout])) == 0:
        raise ValueError("A partition is empty.")
    return np.asarray(train), np.asarray(validation), np.asarray(holdout)
