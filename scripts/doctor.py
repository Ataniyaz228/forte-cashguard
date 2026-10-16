"""Offline readiness check: imports, file IO, real CPU training, optional GPU training."""

import argparse
import importlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true")
args = parser.parse_args()
modules = [
    "numpy",
    "pandas",
    "polars",
    "pyarrow",
    "duckdb",
    "sklearn",
    "catboost",
    "matplotlib",
    "seaborn",
    "plotly",
    "optuna",
    "jupyterlab",
    "ipykernel",
    "streamlit",
    "openpyxl",
    "httpx",
    "yaml",
    "pytest",
]
results = {"python": platform.python_version(), "imports": {}, "checks": {}}
for module in modules:
    importlib.import_module(module)
    results["imports"][module] = "ok"

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostError

with tempfile.TemporaryDirectory() as temp:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "s": ["a", "b", "c"]})
    for suffix, writer, reader in [
        ("parquet", lambda p: frame.to_parquet(p, index=False), pd.read_parquet),
        ("xlsx", lambda p: frame.to_excel(p, index=False), pd.read_excel),
    ]:
        path = Path(temp) / f"check.{suffix}"
        writer(path)
        pd.testing.assert_frame_equal(frame, reader(path), check_dtype=False)
        results["checks"][suffix] = "ok"
    x = np.random.default_rng(42).normal(size=(200, 3))
    y = (x[:, 0] > 0).astype(int)
    model = CatBoostClassifier(
        iterations=8, thread_count=2, verbose=False, allow_writing_files=False
    ).fit(x, y)
    model.save_model(str(Path(temp) / "test.cbm"))
    results["checks"]["cpu_catboost_fit_and_save"] = "ok"
    if args.gpu:
        try:
            CatBoostClassifier(
                iterations=5,
                task_type="GPU",
                devices="0",
                verbose=False,
                allow_writing_files=False,
                gpu_ram_part=0.3,
            ).fit(x, y)
            results["checks"]["gpu_catboost"] = "ok"
        except CatBoostError as error:
            results["checks"]["gpu_catboost"] = str(error)
if shutil.which("nvidia-smi"):
    results["gpu"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
results["versions"] = {
    p: importlib.metadata.version(p)
    for p in ["catboost", "pandas", "scikit-learn", "jupyterlab", "duckdb", "streamlit"]
}
Path("reports").mkdir(exist_ok=True)
Path("reports/readiness.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
