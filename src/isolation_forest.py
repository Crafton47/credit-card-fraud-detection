from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import IsolationForest

from src.evaluation import compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_isolation_forest(X_train, X_test, y_train, y_test, contamination: float = 0.01, random_state: int = 42):
    """Train an unsupervised Isolation Forest and convert its anomaly scores to fraud labels."""
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X_train)

    anomaly_scores = -model.score_samples(X_test)
    predictions = np.where(model.predict(X_test) == -1, 1, 0)
    metrics = compute_metrics(y_test, predictions, anomaly_scores)
    metrics["contamination"] = contamination

    joblib.dump(model, MODEL_DIR / "isolation_forest.joblib")
    return model, metrics


def load_isolation_forest(path: str | Path | None = None):
    if path is None:
        path = MODEL_DIR / "isolation_forest.joblib"
    return joblib.load(path)
