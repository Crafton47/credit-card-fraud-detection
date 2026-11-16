from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Calculate the core metrics used in fraud detection evaluation."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    return metrics


def get_confusion_matrix(y_true, y_pred, labels=(0, 1)):
    """Return a confusion matrix for the model output."""
    return confusion_matrix(y_true, y_pred, labels=list(labels))


def plot_confusion_matrix(y_true, y_pred, labels=(0, 1), title="Confusion Matrix"):
    """Create a confusion matrix heatmap."""
    cm = get_confusion_matrix(y_true, y_pred, labels)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def create_classification_report(y_true, y_pred):
    """Create a detailed text classification report."""
    return classification_report(
        y_true,
        y_pred,
        target_names=["Legitimate", "Fraud"],
        zero_division=0,
    )


def save_metrics_report(metrics: dict, path: str | Path) -> None:
    """Persist a metrics dictionary to JSON so the dashboard can load it."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def build_model_comparison_report(isolation_metrics: dict, xgboost_metrics: dict) -> dict:
    """Create a side-by-side comparison summary for both models."""
    return {
        "Isolation Forest": isolation_metrics,
        "XGBoost": xgboost_metrics,
    }
