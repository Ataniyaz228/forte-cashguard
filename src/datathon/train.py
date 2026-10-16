import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from .data import load_table, prepare_features
from .splits import make_split


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=json_value, allow_nan=False),
        encoding="utf-8",
    )


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classification_metrics(y, probabilities, labels, positive_index, threshold):
    pred = labels[probabilities.argmax(axis=1)]
    binary = len(labels) == 2
    if binary:
        pred = np.where(
            probabilities[:, positive_index] >= threshold,
            labels[positive_index],
            labels[1 - positive_index],
        )
    scores = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "log_loss": log_loss(y, probabilities, labels=labels),
    }
    if binary:
        truth = np.asarray(y) == labels[positive_index]
        decision = pred == labels[positive_index]
        scores.update(
            {
                "roc_auc": roc_auc_score(truth, probabilities[:, positive_index]),
                "average_precision": average_precision_score(
                    truth, probabilities[:, positive_index]
                ),
                "precision": precision_score(truth, decision, zero_division=0),
                "recall": recall_score(truth, decision, zero_division=0),
                "f1": f1_score(truth, decision, zero_division=0),
            }
        )
    return scores, pred


def regression_metrics(y, pred):
    return {
        "mae": mean_absolute_error(y, pred),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": r2_score(y, pred),
    }


def choose_threshold(y_validation, probability, positive_label, costs):
    if not costs:
        return 0.5, None
    fp_cost = float(costs["false_positive"])
    fn_cost = float(costs["false_negative"])
    if min(fp_cost, fn_cost) < 0 or fp_cost + fn_cost == 0:
        raise ValueError("Costs must be nonnegative and at least one must be positive.")
    truth = np.asarray(y_validation) == positive_label
    rows = []
    for threshold in np.r_[np.linspace(0, 1, 201), np.nextafter(1.0, 2.0)]:
        decision = probability >= threshold
        fp = int((decision & ~truth).sum())
        fn = int((~decision & truth).sum())
        rows.append(
            {"threshold": threshold, "fp": fp, "fn": fn, "cost": fp * fp_cost + fn * fn_cost}
        )
    table = pd.DataFrame(rows)
    return float(table.loc[table.cost.idxmin(), "threshold"]), table


def train(config_path, output_root="reports/runs"):
    cfg = yaml.safe_load(Path(config_path).read_text())
    if cfg["task"] not in {"classification", "regression"}:
        raise ValueError("task must be classification or regression.")
    features = cfg.get("features", [])
    if not features or len(features) != len(set(features)):
        raise ValueError("List unique, decision-time features explicitly in config.features.")
    forbidden = {cfg["target"], cfg.get("id_column")}
    if cfg["split"]["strategy"] == "group":
        forbidden.add(cfg["split"]["group_column"])
    if set(features) & forbidden:
        raise ValueError("Target / row ID / split group ID must not be model features.")
    frame = load_table(cfg["data"])
    if frame[cfg["target"]].isna().any():
        raise ValueError("Missing targets: separate unlabeled rows before training.")
    x = prepare_features(frame, features, cfg.get("categorical", []))
    y = frame[cfg["target"]]
    tr, va, ho = make_split(frame, cfg)
    task = cfg["task"]
    positive_index = 1
    if task == "classification":
        classes = np.unique(y.iloc[tr])
        if len(classes) < 2:
            raise ValueError("Training needs at least two classes.")
        for name, rows in [("validation", va), ("holdout", ho)]:
            if set(y.iloc[rows]) != set(classes):
                raise ValueError(f"{name} class coverage differs from train; revise split.")
        if len(classes) == 2:
            positive = cfg.get("positive_label", classes[1])
            if positive not in classes:
                raise ValueError("positive_label must match an observed class, including its type.")
            positive_index = list(classes).index(positive)
    else:
        y = pd.to_numeric(y, errors="raise")
        if not np.isfinite(y).all():
            raise ValueError("Regression targets must be finite.")

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out = Path(output_root) / run_id
    out.mkdir(parents=True)
    yaml.safe_dump(cfg, (out / "config.yaml").open("w"), allow_unicode=True, sort_keys=False)
    membership = np.full(len(frame), "gap", dtype=object)
    for name, indices in [("train", tr), ("validation", va), ("holdout", ho)]:
        membership[indices] = name
    pd.DataFrame({"row_position": np.arange(len(frame)), "partition": membership}).to_csv(
        out / "split.csv", index=False
    )
    model_cfg = dict(cfg.get("model", {}))
    patience = model_cfg.pop("early_stopping_rounds", 50)
    params = {"iterations": 500, "depth": 6, "learning_rate": 0.05, "thread_count": 4}
    params.update(model_cfg)
    params.update(random_seed=cfg.get("seed", 42), allow_writing_files=False, verbose=False)
    if task == "classification":
        model = CatBoostClassifier(**params)
        dummy = DummyClassifier(strategy="prior")
    else:
        model = CatBoostRegressor(**params, loss_function="RMSE")
        dummy = DummyRegressor(strategy="mean")
    started = time.perf_counter()
    model.fit(
        x.iloc[tr],
        y.iloc[tr],
        cat_features=cfg.get("categorical", []),
        eval_set=(x.iloc[va], y.iloc[va]),
        use_best_model=True,
        early_stopping_rounds=patience,
    )
    training_seconds = time.perf_counter() - started
    dummy.fit(np.zeros((len(tr), 1)), y.iloc[tr])
    threshold = 0.5
    validation_metrics = None
    if task == "classification":
        labels = np.asarray(model.classes_)
        validation_proba = model.predict_proba(x.iloc[va])
        if len(labels) == 2:
            threshold, curve = choose_threshold(
                y.iloc[va],
                validation_proba[:, positive_index],
                labels[positive_index],
                cfg.get("costs"),
            )
            if curve is not None:
                curve.to_csv(out / "validation_thresholds.csv", index=False)
        validation_metrics, _ = classification_metrics(
            y.iloc[va], validation_proba, labels, positive_index, threshold
        )
        start_prediction = time.perf_counter()
        proba = model.predict_proba(x.iloc[ho])
        inference_seconds = time.perf_counter() - start_prediction
        metrics, pred = classification_metrics(y.iloc[ho], proba, labels, positive_index, threshold)
        dummy_proba = dummy.predict_proba(np.zeros((len(ho), 1)))
        # Dummy uses its own conventional decision threshold, not the tuned model threshold.
        baseline, _ = classification_metrics(y.iloc[ho], dummy_proba, labels, positive_index, 0.5)
        cm = confusion_matrix(y.iloc[ho], pred, labels=labels)
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(out / "confusion_matrix.csv")
        probabilities = pd.DataFrame(proba, columns=[f"probability_{v}" for v in labels])
        metadata_labels = labels.tolist()
    else:
        validation_metrics = regression_metrics(y.iloc[va], model.predict(x.iloc[va]))
        start_prediction = time.perf_counter()
        pred = model.predict(x.iloc[ho])
        inference_seconds = time.perf_counter() - start_prediction
        metrics = regression_metrics(y.iloc[ho], pred)
        baseline = regression_metrics(y.iloc[ho], dummy.predict(np.zeros((len(ho), 1))))
        probabilities = pd.DataFrame(index=range(len(ho)))
        metadata_labels = None
    predictions = pd.DataFrame(
        {
            "row_position": ho,
            "actual": y.iloc[ho].to_numpy(),
            "prediction": np.asarray(pred).ravel(),
        }
    )
    predictions = pd.concat([predictions, probabilities], axis=1)
    predictions.to_csv(out / "holdout_predictions.csv", index=False)
    importance = pd.DataFrame({"feature": features, "importance": model.feature_importances_})
    importance.sort_values("importance", ascending=False).to_csv(
        out / "feature_importance.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(9, max(3, min(10, len(features) * 0.45))))
    top = importance.nlargest(15, "importance").sort_values("importance")
    ax.barh(top.feature, top.importance, color="#9D194B")
    ax.set_xlabel("CatBoost feature importance (not a causal effect)")
    fig.tight_layout()
    fig.savefig(out / "feature_importance.png", dpi=160)
    plt.close(fig)
    model.save_model(str(out / "model.cbm"))
    result = {
        "run_id": run_id,
        "task": task,
        "dataset_sha256": file_hash(cfg["data"]),
        "partition_sizes": {"train": len(tr), "validation": len(va), "holdout": len(ho)},
        "split_strategy": cfg["split"]["strategy"],
        "seed": cfg.get("seed", 42),
        "holdout": metrics,
        "baseline_holdout": baseline,
        "validation": validation_metrics,
        "decision_threshold": threshold if task == "classification" else None,
        "positive_label": metadata_labels[positive_index]
        if task == "classification" and len(metadata_labels) == 2
        else None,
        "class_labels": metadata_labels,
        "trees": model.tree_count_,
        "training_seconds": training_seconds,
        "holdout_inference_seconds": inference_seconds,
        "python": platform.python_version(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["catboost", "pandas", "scikit-learn", "numpy"]
        },
        "caveat": "Single holdout estimate. Repeated optimization against it invalidates independence.",
    }
    if cfg.get("costs") and task == "classification" and len(metadata_labels) == 2:
        actual_pos = y.iloc[ho].to_numpy() == result["positive_label"]
        predicted_pos = np.asarray(pred) == result["positive_label"]
        result["holdout_cost"] = float(
            (predicted_pos & ~actual_pos).sum() * cfg["costs"]["false_positive"]
            + (~predicted_pos & actual_pos).sum() * cfg["costs"]["false_negative"]
        )
        result["cost_units"] = "As configured; scenario costs, not observed business impact."
    write_json(out / "metrics.json", result)
    rows = "\n".join(f"| {k} | {baseline[k]:.5f} | {v:.5f} |" for k, v in metrics.items())
    (out / "summary.md").write_text(
        f"# Run {run_id}\n\nSplit: {result['partition_sizes']}; strategy: {result['split_strategy']}.\n\n"
        "Early stopping and threshold use validation only. Metrics below use a separate holdout.\n\n"
        f"| Metric | Dummy | CatBoost |\n|---|---:|---:|\n{rows}\n\n"
        "Feature importance is association, not causation. A single split is not a confidence interval.\n",
        encoding="utf-8",
    )
    print(f"Run saved to {out}")
    return out
