from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
DATASET_PATH = DATASET_DIR / "creditcard.csv"
MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_DATASET_URL = (
    "https://raw.githubusercontent.com/PacktPublishing/"
    "Machine-Learning-for-Credit-Card-Fraud-Detection/master/creditcard.csv"
)


def generate_synthetic_dataset(path: Path) -> Path:
    """Create a realistic synthetic fraud dataset with the standard credit-card schema."""
    rng = np.random.default_rng(42)
    fraud_ratio = 0.002
    n_samples = 284_807
    n_fraud = max(1, int(n_samples * fraud_ratio))
    n_legit = n_samples - n_fraud

    legit = rng.normal(0, 1, size=(n_legit, 28))
    fraud = rng.normal(0.8, 1.2, size=(n_fraud, 28))
    features = np.vstack([legit, fraud])

    time_values = np.concatenate([
        rng.uniform(0, 172_800, size=n_legit),
        rng.uniform(0, 172_800, size=n_fraud),
    ])
    amount_values = np.concatenate([
        rng.lognormal(mean=0.0, sigma=1.0, size=n_legit),
        rng.lognormal(mean=1.5, sigma=1.2, size=n_fraud),
    ])
    amount_values[n_legit:] *= 4.5

    df = pd.DataFrame(features, columns=[f"V{i}" for i in range(1, 29)])
    df["Time"] = time_values
    df["Amount"] = amount_values
    df["Class"] = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])

    df = df[[* [f"V{i}" for i in range(1, 29)], "Time", "Amount", "Class"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def ensure_dataset() -> Path:
    """Download the public dataset when available, or generate a realistic offline fallback."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    if not DATASET_PATH.exists():
        try:
            urlretrieve(DEFAULT_DATASET_URL, DATASET_PATH)
        except Exception:
            return generate_synthetic_dataset(DATASET_PATH)
    return DATASET_PATH


def load_dataset(dataset_path: str | Path | None = None) -> pd.DataFrame:
    """Load and validate the dataset."""
    actual_path = Path(dataset_path) if dataset_path else ensure_dataset()
    df = pd.read_csv(actual_path)
    if "Class" not in df.columns:
        raise ValueError("The dataset must include a 'Class' column with labels 0/1.")
    return df


def summarize_dataset(df: pd.DataFrame) -> dict:
    """Return a compact dataset overview for reporting and dashboards."""
    fraud_count = int(df["Class"].sum())
    legit_count = int((1 - df["Class"]).sum())
    return {
        "total_transactions": int(len(df)),
        "legitimate_transactions": legit_count,
        "fraudulent_transactions": fraud_count,
        "fraud_percentage": round((fraud_count / len(df)) * 100, 4),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "feature_columns": [col for col in df.columns if col != "Class"],
    }


def get_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Separate the feature matrix from the target vector."""
    feature_columns = [column for column in df.columns if column != "Class"]
    X = df[feature_columns].copy()
    y = df["Class"].astype(int).copy()
    return X, y, feature_columns


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Create stratified train/test splits without any leakage."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def standardize_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fit a scaler on the training data and transform both splits."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
    """Apply SMOTE only to the training split."""
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return smote, X_resampled, y_resampled


def save_training_summary(path: str | Path, payload: dict) -> None:
    """Persist a JSON summary of the training run."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def get_class_distribution(y_series: pd.Series) -> dict:
    """Return class counts for model analysis and reports."""
    counts = y_series.value_counts(normalize=False).to_dict()
    return {str(key): int(value) for key, value in sorted(counts.items())}
