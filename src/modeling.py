"""Reproducible model comparison, selection, evaluation and inference."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
import joblib
import sklearn
import sys
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, brier_score_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from .constants import FEATURES, MODEL_PATH, METRICS_PATH, TARGET
from .data import load_training_data, file_sha256, dataset_path
from .preprocessing import MachiningFeatures, make_preprocessor

SEED = 42


def candidates():
    return {
        "Logistic Regression": LogisticRegression(max_iter=800, class_weight="balanced", random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=9, min_samples_leaf=8, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
    }


def pipeline_for(model):
    return Pipeline([("engineering", MachiningFeatures()), ("preprocess", make_preprocessor()), ("classifier", model)])


def scored_metrics(y_true, probability, threshold):
    classification = (np.asarray(probability) >= float(threshold)).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, classification)), 5),
        "precision": round(float(precision_score(y_true, classification, zero_division=0)), 5),
        "recall": round(float(recall_score(y_true, classification, zero_division=0)), 5),
        "f1": round(float(f1_score(y_true, classification, zero_division=0)), 5),
        "roc_auc": round(float(roc_auc_score(y_true, probability)), 5),
        "average_precision": round(float(average_precision_score(y_true, probability)), 5),
        "brier_score": round(float(brier_score_loss(y_true, probability)), 5),
        "confusion_matrix": confusion_matrix(y_true, classification).astype(int).tolist(),
    }


def choose_threshold(y_true, probability):
    # Chosen exclusively using validation data; held-out test untouched.
    choices = np.linspace(.08, .85, 78)
    return float(max(choices, key=lambda th: (f1_score(y_true, probability >= th, zero_division=0), -abs(th - .5))))


def train_and_save(data_path=None, verbose=True):
    frame, dataset_summary = load_training_data(data_path)
    X = frame[FEATURES]
    y = frame[TARGET]
    X_fit, X_test, y_fit, y_test = train_test_split(X, y, test_size=.20, stratify=y, random_state=SEED)
    X_train, X_valid, y_train, y_valid = train_test_split(X_fit, y_fit, test_size=.25, stratify=y_fit, random_state=SEED)
    comparison = []
    selection = []
    for label, algorithm in candidates().items():
        calibrated = CalibratedClassifierCV(estimator=pipeline_for(algorithm), cv=3, method="sigmoid")
        calibrated.fit(X_train, y_train)
        pred = calibrated.predict_proba(X_valid)[:, 1]
        scores = scored_metrics(y_valid, pred, .5)
        comparison.append({"model": label, **{k: v for k, v in scores.items() if k != "confusion_matrix"}})
        selection.append((scores["average_precision"], scores["roc_auc"], label, calibrated, pred))
        if verbose:
            print(f"{label:22s} validation AP={scores['average_precision']:.4f}, ROC-AUC={scores['roc_auc']:.4f}")
    _, _, name, best, valid_probability = max(selection, key=lambda item: (item[0], item[1]))
    threshold = choose_threshold(y_valid, valid_probability)
    test_probability = best.predict_proba(X_test)[:, 1]
    metrics = scored_metrics(y_test, test_probability, threshold)
    importance = permutation_importance(
        best, X_valid, y_valid, n_repeats=3, random_state=SEED,
        scoring="average_precision", n_jobs=1,
    )
    influences = sorted([
        {"feature": col, "mean_drop_in_average_precision": round(float(value), 5)}
        for col, value in zip(FEATURES, importance.importances_mean)
    ], key=lambda item: item["mean_drop_in_average_precision"], reverse=True)
    typical = {col: (X_train[col].mode().iloc[0] if col == "Type" else float(X_train[col].median())) for col in FEATURES}
    package = {
        "model": best,
        "threshold": threshold,
        "model_name": name,
        "feature_columns": FEATURES,
        "typical_values": typical,
        "dataset_kind": dataset_summary["dataset_kind"],
        "dataset_hash": file_sha256(data_path or dataset_path()),
        "sklearn_version": sklearn.__version__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": 1,
    }
    report = {
        "title": "MachineGuard AI: Intelligent CNC Machine Failure Prediction and Maintenance",
        "data": dataset_summary,
        "environment": {"python": sys.version.split()[0], "scikit_learn": sklearn.__version__},
        "split": {"train": len(X_train), "validation": len(X_valid), "test": len(X_test)},
        "validation_comparison": comparison,
        "selected_model": name,
        "selection_basis": "highest validation average precision; ROC-AUC tie breaker",
        "decision_threshold": round(threshold, 5),
        "test_metrics": metrics,
        "test_positive_count": int(y_test.sum()),
        "validation_permutation_importance": influences,
        "validation_threshold_is_test_independent": True,
        "calibration": "3-fold sigmoid calibration fitted on training partition; not proof of deployment reliability",
        "provenance_note": "Generated simulator data are NOT from UCI unless dataset_kind is official_uci_synthetic",
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, MODEL_PATH, compress=3)
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if verbose:
        print("Selected:", name, "threshold:", round(threshold, 4))
        print("Held-out test:", json.dumps(metrics, indent=2))
    return report


def load_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("The trained model is missing. Run bootstrap.py.")
    artifact = joblib.load(MODEL_PATH)
    if artifact.get("feature_columns") != FEATURES:
        raise ValueError("Saved model uses a different feature schema. Retrain using bootstrap.py.")
    return artifact


def load_metrics():
    if not METRICS_PATH.exists():
        raise FileNotFoundError("Training metrics missing. Run bootstrap.py.")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
