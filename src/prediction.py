from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"


def load_model_artifacts():
    """Load the trained scaler, model, and feature metadata from disk."""
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    feature_columns = joblib.load(MODEL_DIR / "feature_columns.joblib")
    xgb_model = joblib.load(MODEL_DIR / "xgboost_model.joblib")
    isolation_model = joblib.load(MODEL_DIR / "isolation_forest.joblib")
    with (MODEL_DIR / "model_metrics.json").open("r", encoding="utf-8") as file:
        metrics = json.load(file)
    return {
        "scaler": scaler,
        "feature_columns": feature_columns,
        "xgb_model": xgb_model,
        "isolation_model": isolation_model,
        "metrics": metrics,
    }


def prepare_feature_row(feature_values: dict, feature_columns: list[str]) -> pd.DataFrame:
    """Create a row-shaped DataFrame from a dictionary of raw features."""
    record = {}
    for column in feature_columns:
        if column not in feature_values:
            raise ValueError(f"Missing required field: {column}")
        record[column] = float(feature_values[column])
    return pd.DataFrame([record], columns=feature_columns)


def predict_transaction(feature_values: dict):
    """Predict whether a transaction is legitimate or fraudulent using both models."""
    artifacts = load_model_artifacts()
    feature_frame = prepare_feature_row(feature_values, artifacts["feature_columns"])
    scaled_features = artifacts["scaler"].transform(feature_frame)

    xgb_prob = float(artifacts["xgb_model"].predict_proba(scaled_features)[0, 1])
    xgb_prediction = int(xgb_prob >= 0.5)

    isolation_score = float(artifacts["isolation_model"].score_samples(scaled_features)[0])
    isolation_prediction = int(artifacts["isolation_model"].predict(scaled_features)[0] == -1)

    return {
        "xgboost_probability": xgb_prob,
        "xgboost_prediction": xgb_prediction,
        "isolation_prediction": isolation_prediction,
        "isolation_score": isolation_score,
    }
