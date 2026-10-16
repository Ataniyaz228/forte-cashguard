from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier, CatBoostRegressor

from .data import load_table, prepare_features


def predict(run, data, sample, output, prediction_column, kind="label", positive_label=None):
    """Align on unique IDs, never silently assume competition sample order."""
    import json

    run = Path(run)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    metrics = json.loads((run / "metrics.json").read_text())
    frame, template = load_table(data), load_table(sample)
    id_col = cfg.get("id_column")
    if not id_col:
        raise ValueError("Set a unique id_column in training config for safe submission alignment.")
    for name, table in [("data", frame), ("sample", template)]:
        if id_col not in table or table[id_col].isna().any() or not table[id_col].is_unique:
            raise ValueError(f"{name} needs unique, non-null IDs in {id_col}.")
    if set(frame[id_col]) != set(template[id_col]):
        raise ValueError("Input IDs and sample IDs differ.")
    if prediction_column not in template or prediction_column == id_col:
        raise ValueError("prediction_column must be a non-ID column in the sample.")
    x = prepare_features(frame, cfg["features"], cfg.get("categorical", []))
    model = CatBoostClassifier() if cfg["task"] == "classification" else CatBoostRegressor()
    model.load_model(str(run / "model.cbm"))
    if kind == "probability":
        if cfg["task"] != "classification":
            raise ValueError("Probabilities require classification.")
        label = positive_label if positive_label is not None else metrics["positive_label"]
        labels = list(model.classes_)
        matches = [i for i, item in enumerate(labels) if str(item) == str(label)]
        if len(matches) != 1:
            raise ValueError(f"Choose a positive label from {labels}.")
        values = model.predict_proba(x)[:, matches[0]]
    elif kind == "label":
        if cfg["task"] == "classification" and len(model.classes_) == 2:
            labels = list(model.classes_)
            pos = labels.index(metrics["positive_label"])
            values = np.where(
                model.predict_proba(x)[:, pos] >= metrics["decision_threshold"],
                labels[pos],
                labels[1 - pos],
            )
        else:
            values = np.asarray(model.predict(x)).ravel()
    else:
        raise ValueError("kind must be label or probability.")
    mapping = pd.Series(values, index=frame[id_col])
    template[prediction_column] = template[id_col].map(mapping)
    if template[prediction_column].isna().any():
        raise ValueError("Submission contains missing predictions.")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output, index=False)
    print(f"Validated {len(template)} predictions; written to {output}")
