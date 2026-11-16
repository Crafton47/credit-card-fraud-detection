from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from src.evaluation import compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_xgboost_model(X_train, X_test, y_train, y_test, random_state: int = 42):
    """Train a supervised XGBoost classifier using SMOTE on the training split only."""
    n_positive = int(y_train.sum())
    n_negative = len(y_train) - n_positive
    scale_pos_weight = n_negative / n_positive if n_positive else 1.0

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        learning_rate=0.05,
        max_depth=6,
        n_estimators=500,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_resampled, y_resampled)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = compute_metrics(y_test, predictions, probabilities)
    metrics["smote_applied"] = True

    joblib.dump(model, MODEL_DIR / "xgboost_model.joblib")
    joblib.dump(smote, MODEL_DIR / "smote.joblib")
    return model, metrics


def load_xgboost_model(path: str | Path | None = None):
    if path is None:
        path = MODEL_DIR / "xgboost_model.joblib"
    return joblib.load(path)
