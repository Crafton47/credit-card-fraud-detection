from __future__ import annotations

import json
from pathlib import Path

import joblib

from src.evaluation import build_model_comparison_report, compute_metrics, save_metrics_report
from src.isolation_forest import train_isolation_forest
from src.preprocessing import (
    apply_smote,
    ensure_dataset,
    get_class_distribution,
    get_features_and_target,
    load_dataset,
    save_training_summary,
    split_data,
    standardize_features,
    summarize_dataset,
)
from src.xgboost_model import train_xgboost_model

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Train and evaluate the Isolation Forest and XGBoost fraud detection models."""
    dataset_path = ensure_dataset()
    df = load_dataset(dataset_path)
    summary = summarize_dataset(df)
    print("Dataset Summary:")
    print(json.dumps(summary, indent=2))

    X, y, feature_columns = get_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    test_split_path = PROJECT_ROOT / "dataset" / "test_transactions.csv"
    test_split = X_test.copy()
    test_split["Class"] = y_test.values
    test_split_path.parent.mkdir(parents=True, exist_ok=True)
    test_split.to_csv(test_split_path, index=False)

    scaler, X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(feature_columns, MODEL_DIR / "feature_columns.joblib")

    before_smote = get_class_distribution(y_train)
    _, X_smote_train, y_smote_train = apply_smote(X_train_scaled, y_train, random_state=42)
    after_smote = get_class_distribution(y_smote_train)

    print("\nClass Distribution Before SMOTE:")
    print(json.dumps(before_smote, indent=2))
    print("\nClass Distribution After SMOTE:")
    print(json.dumps(after_smote, indent=2))

    isolation_model, isolation_metrics = train_isolation_forest(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        contamination=0.01,
        random_state=42,
    )

    xgboost_model, xgboost_metrics = train_xgboost_model(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        random_state=42,
    )

    comparison_report = build_model_comparison_report(isolation_metrics, xgboost_metrics)
    save_metrics_report(comparison_report, MODEL_DIR / "model_metrics.json")

    training_summary = {
        "dataset_path": str(dataset_path),
        "data_summary": summary,
        "class_distribution_before_smote": before_smote,
        "class_distribution_after_smote": after_smote,
        "comparison_report": comparison_report,
    }
    save_training_summary(MODEL_DIR / "training_summary.json", training_summary)

    print("\nIsolation Forest Metrics:")
    print(json.dumps(isolation_metrics, indent=2))
    print("\nXGBoost Metrics:")
    print(json.dumps(xgboost_metrics, indent=2))

    print("\nTraining complete. Models saved in the 'models/' directory.")


if __name__ == "__main__":
    main()
