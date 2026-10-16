import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import load_table


def audit(path, target=None, output="reports/eda", **load_options):
    frame = load_table(path, **load_options)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    profile = pd.DataFrame(
        {
            "dtype": frame.dtypes.astype(str),
            "missing": frame.isna().sum(),
            "missing_share": frame.isna().mean(),
            "unique": frame.nunique(dropna=False),
        }
    )
    profile.to_csv(out / "columns.csv", index_label="column")
    numeric = frame.select_dtypes(include="number")
    if not numeric.empty:
        numeric.describe(percentiles=[0.01, 0.5, 0.99]).to_csv(out / "numeric_summary.csv")
    warnings = []
    for col in frame.columns:
        if profile.loc[col, "unique"] <= 1:
            warnings.append(f"Constant/all-missing column: {col}")
        if frame[col].nunique() == len(frame):
            warnings.append(f"Unique values in every row: {col}; verify ID / timing / leakage.")
    duplicates = int(frame.duplicated().sum())
    if duplicates:
        warnings.append(f"Exact duplicate rows: {duplicates}; check split contamination.")
    result = {
        "rows": len(frame),
        "columns": len(frame.columns),
        "memory_mib": float(frame.memory_usage(deep=True).sum() / 1024**2),
        "duplicate_rows": duplicates,
        "warnings": warnings,
        "scope": "Full-table descriptive audit. Restrict hypothesis testing to train after splitting.",
    }
    if target:
        if target not in frame:
            raise ValueError(f"Target column does not exist: {target}")
        result["target_missing"] = int(frame[target].isna().sum())
        frame[target].value_counts(dropna=False).head(100).to_csv(out / "target_counts.csv")
    (out / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    missing = profile.missing_share.nlargest(20).sort_values()
    fig, ax = plt.subplots(figsize=(10, max(4, len(missing) * 0.3)))
    ax.barh(missing.index, missing.values, color="#9D194B")
    ax.set_xlabel("Missing fraction")
    ax.set_xlim(0, max(0.05, missing.max()))
    fig.tight_layout()
    fig.savefig(out / "missing.png", dpi=160)
    plt.close(fig)
    if not numeric.empty:
        corr = numeric.iloc[:, :30].replace([np.inf, -np.inf], np.nan).corr()
        corr.to_csv(out / "numeric_correlations.csv")
    print(f"Audit saved to {out}; {len(frame):,} rows x {len(frame.columns)} columns")
    return result
