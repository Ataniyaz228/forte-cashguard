import json

import numpy as np
import pandas as pd
import pytest
import yaml

from datathon.predict import predict
from datathon.train import train


@pytest.mark.parametrize(
    "task,strategy",
    [
        ("binary", "group"),
        ("regression", "time"),
        ("multiclass", "random"),
    ],
)
def test_full_training_and_submission_contract(tmp_path, task, strategy):
    rng = np.random.default_rng(123)
    n = 600
    signal = rng.normal(size=n)
    target = signal + rng.normal(0, 0.2, n)
    if task == "binary":
        target = (target > 0.4).astype(int)
    elif task == "multiclass":
        target = np.where(signal < -0.5, "low", np.where(signal > 0.5, "high", "mid"))
    frame = pd.DataFrame(
        {
            "id": np.arange(n),
            "x": signal,
            "cat": rng.choice(["retail", "business", None], n),
            "group": np.repeat(np.arange(n // 3), 3),
            "time": pd.date_range("2025-01-01", periods=n, freq="h"),
            "target": target,
        }
    )
    source = tmp_path / "data.csv"
    frame.to_csv(source, index=False)
    cfg = {
        "data": str(source),
        "target": "target",
        "task": "regression" if task == "regression" else "classification",
        "positive_label": 1,
        "id_column": "id",
        "seed": 42,
        "features": ["x", "cat"],
        "categorical": ["cat"],
        "split": {"strategy": strategy, "group_column": "group", "time_column": "time"},
        "model": {"iterations": 15, "depth": 3, "thread_count": 2, "early_stopping_rounds": 5},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    run = train(config_path, tmp_path / "runs")
    metrics = json.loads((run / "metrics.json").read_text())
    assert sum(metrics["partition_sizes"].values()) == n
    assert all(np.isfinite(value) for value in metrics["holdout"].values())
    test = frame.sample(30, random_state=99).drop(columns="target")
    test_path, sample_path, output = [
        tmp_path / name for name in ["test.csv", "sample.csv", "submission.csv"]
    ]
    test.to_csv(test_path, index=False)
    sample = pd.DataFrame({"id": test.id.iloc[::-1], "target": 0})
    sample.to_csv(sample_path, index=False)
    kind = "probability" if task == "binary" else "label"
    predict(run, test_path, sample_path, output, "target", kind=kind)
    result = pd.read_csv(output)
    assert result.id.tolist() == sample.id.tolist()
    assert result.target.notna().all()
    if kind == "probability":
        assert result.target.between(0, 1).all()
    # Change sample ordering again: predictions must remain attached to the same IDs.
    sample.iloc[::-1].to_csv(sample_path, index=False)
    second = tmp_path / "second.csv"
    predict(run, test_path, sample_path, second, "target", kind=kind)
    pd.testing.assert_series_equal(
        result.set_index("id").target.sort_index(),
        pd.read_csv(second).set_index("id").target.sort_index(),
    )
    # Duplicate IDs must fail, never silently shift predictions.
    sample.iloc[0, 0] = sample.iloc[1, 0]
    sample.to_csv(sample_path, index=False)
    with pytest.raises(ValueError, match="unique"):
        predict(run, test_path, sample_path, output, "target", kind=kind)
