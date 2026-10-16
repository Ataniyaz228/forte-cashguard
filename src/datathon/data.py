from pathlib import Path

import numpy as np
import pandas as pd


def load_table(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path, **kwargs)
    elif suffix in {".xlsx", ".xlsm"}:
        frame = pd.read_excel(path, **kwargs)
    elif suffix in {".csv", ".tsv"}:
        frame = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", **kwargs)
    else:
        raise ValueError(f"Unsupported file: {suffix}. Use CSV, TSV, Parquet or XLSX.")
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names; correct the schema before analysis.")
    if frame.empty:
        raise ValueError("The data table is empty.")
    return frame


def prepare_features(frame, features, categorical):
    """No fitted statistics, no target encoding, no preprocessing across partitions."""
    missing = set(features) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")
    if set(categorical) - set(features):
        raise ValueError("Every categorical column must also be in features.")
    out = frame[features].copy()
    for col in features:
        if col in categorical:
            out[col] = out[col].astype("string").fillna("__MISSING__").astype(str)
        else:
            out[col] = pd.to_numeric(out[col], errors="raise").astype(float)
            out[col] = out[col].replace([np.inf, -np.inf], np.nan)
    return out
