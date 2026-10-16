"""Practice data. No connection to the real datathon case or ForteBank customers."""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1800
signal_a = rng.normal(size=n)
signal_b = rng.normal(size=n)
segment = rng.choice(["alpha", "beta", "gamma"], n)
logit = 1.5 * signal_a - signal_b + (segment == "beta") * 0.8 - 1.7
frame = pd.DataFrame(
    {
        "row_id": np.arange(n),
        "customer_id": np.repeat(np.arange(n // 3), 3),
        "event_time": pd.date_range("2025-01-01", periods=n, freq="h"),
        "signal_a": signal_a,
        "signal_b": signal_b,
        "segment": segment,
        "amount": rng.lognormal(6, 1, n),
        "target": rng.binomial(1, 1 / (1 + np.exp(-logit))),
        "continuous_target": 2 * signal_a - signal_b + rng.normal(scale=0.5, size=n),
        "multi_target": np.where(signal_a < -0.4, "low", np.where(signal_a > 0.6, "high", "mid")),
    }
)
frame.loc[rng.choice(n, 60, replace=False), "segment"] = None
frame.loc[rng.choice(n, 50, replace=False), "amount"] = np.nan
out = Path("data/synthetic")
out.mkdir(parents=True, exist_ok=True)
frame.iloc[:1500].to_csv(out / "train.csv", index=False)
frame.iloc[1500:].drop(columns=["target", "continuous_target", "multi_target"]).to_csv(
    out / "test.csv", index=False
)
pd.DataFrame(
    {"row_id": frame.iloc[1500:].row_id.sample(frac=1, random_state=42), "target": 0}
).to_csv(out / "sample_submission.csv", index=False)
print("Created synthetic practice files in data/synthetic; NOT competition data.")
