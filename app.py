from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DATASET_DIR = ROOT / "dataset"

st.set_page_config(
    page_title="Credit Card Fraud Detection (FinTech)",
    page_icon="💳",
    layout="wide",
)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    dataset_path = DATASET_DIR / "creditcard.csv"
    if not dataset_path.exists():
        st.error("The dataset is missing. Please run `python train.py` first to load and train the models.")
        return pd.DataFrame()
    return pd.read_csv(dataset_path)


@st.cache_data
def load_test_data() -> pd.DataFrame:
    dataset_path = DATASET_DIR / "test_transactions.csv"
    if not dataset_path.exists():
        if DATASET_DIR.exists() and "Class" in pd.read_csv(DATASET_DIR / "creditcard.csv").columns:
            df = pd.read_csv(DATASET_DIR / "creditcard.csv")
            from sklearn.model_selection import train_test_split
            X = df.drop(columns=["Class"])
            y = df["Class"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            test_df = X_test.copy()
            test_df["Class"] = y_test.values
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            test_df.to_csv(dataset_path, index=False)
            return test_df
        st.error("The test split is missing. Please run `python train.py` first.")
        return pd.DataFrame()
    df = pd.read_csv(dataset_path)
    if "Class" not in df.columns:
        st.error("The saved test split does not contain the target column `Class`.")
        return pd.DataFrame()
    return df


@st.cache_resource
def load_saved_artifacts():
    if not MODEL_DIR.exists():
        st.error("The trained models are missing. Please run `python train.py` first.")
        return None
    try:
        scaler = joblib.load(MODEL_DIR / "scaler.joblib")
        feature_columns = joblib.load(MODEL_DIR / "feature_columns.joblib")
        xgboost_model = joblib.load(MODEL_DIR / "xgboost_model.joblib")
        isolation_model = joblib.load(MODEL_DIR / "isolation_forest.joblib")
        with (MODEL_DIR / "model_metrics.json").open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        return {
            "scaler": scaler,
            "feature_columns": feature_columns,
            "xgboost_model": xgboost_model,
            "isolation_model": isolation_model,
            "metrics": metrics,
        }
    except FileNotFoundError:
        st.error("One or more model files are missing. Please run `python train.py` first.")
        return None


st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            color: #e5e7eb;
        }
        div[data-testid="stMetricLabel"] {
            color: #cbd5e1;
        }
        div[data-testid="stMetricValue"] {
            color: #f8fafc;
            font-size: 2rem;
        }
        .block-container {
            padding-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_confusion_matrix(y_true, y_pred, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 4))
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


# Load the dataset and saved models
raw_df = load_dataset()
artifacts = load_saved_artifacts()
if raw_df.empty or artifacts is None:
    st.stop()

base_df = raw_df.copy()
feature_columns = [col for col in base_df.columns if col != "Class"]

test_df = load_test_data()
if test_df.empty or "Class" not in test_df.columns:
    st.error("No valid test split is available. Please run `python train.py` first.")
    st.stop()

X_test = test_df.drop(columns=["Class"], errors="ignore")
y_test = test_df["Class"].astype(int)

scaler = artifacts["scaler"]
feature_columns = artifacts["feature_columns"]
xgb_model = artifacts["xgboost_model"]
isolation_model = artifacts["isolation_model"]

X_test_scaled = scaler.transform(X_test[feature_columns])

xgb_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]
xgb_pred = (xgb_prob >= 0.5).astype(int)
iso_anomaly_scores = -isolation_model.score_samples(X_test_scaled)
iso_pred = (isolation_model.predict(X_test_scaled) == -1).astype(int)

fraud_count = int(base_df["Class"].sum())
legit_count = int((1 - base_df["Class"]).sum())
fraud_ratio = (fraud_count / len(base_df)) * 100

st.title("Credit Card Fraud Detection System (FinTech)")
st.caption("Advanced anomaly detection and supervised fraud classification for finance risk monitoring.")
st.divider()

home, overview, visualization, fraud_detection, performance, comparison, predict = st.tabs(
    [
        "Home Dashboard",
        "Dataset Overview",
        "Data Visualization",
        "Fraud Detection",
        "Model Performance",
        "Model Comparison",
        "Transaction Prediction",
    ]
)

with home:
    st.header("Executive Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{len(base_df):,}")
    col2.metric("Legitimate Transactions", f"{legit_count:,}")
    col3.metric("Fraudulent Transactions", f"{fraud_count:,}")
    col4.metric("Fraud Percentage", f"{fraud_ratio:.2f}%")

    st.subheader("Fraud Distribution")
    chart_df = pd.DataFrame({
        "Transaction Type": ["Legitimate", "Fraudulent"],
        "Count": [legit_count, fraud_count],
    }).set_index("Transaction Type")
    st.bar_chart(chart_df)

    st.info("The dataset is highly imbalanced, so Recall and F1-score are prioritized over raw accuracy.")

with overview:
    st.header("Dataset Overview")
    st.dataframe(base_df.head(10), use_container_width=True)

    missing_values = int(base_df.isnull().sum().sum())
    duplicate_rows = int(base_df.duplicated().sum())

    col1, col2 = st.columns(2)
    col1.metric("Missing Values", missing_values)
    col2.metric("Duplicate Rows", duplicate_rows)

    st.subheader("Class Summary")
    summary_df = pd.DataFrame(
        {
            "Label": ["Legitimate", "Fraudulent"],
            "Count": [legit_count, fraud_count],
            "Percentage": [100 - fraud_ratio, fraud_ratio],
        }
    )
    st.dataframe(summary_df, use_container_width=True)

with visualization:
    st.header("Exploratory Data Analysis")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(base_df["Amount"], bins=40, kde=True, ax=axes[0], color="#60a5fa")
    axes[0].set_title("Transaction Amount Distribution")
    sns.histplot(base_df["Time"], bins=40, kde=True, ax=axes[1], color="#34d399")
    axes[1].set_title("Transaction Time Distribution")
    plt.tight_layout()
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.countplot(x="Class", data=base_df, palette=["#10b981", "#ef4444"], ax=ax2)
    ax2.set_title("Fraud vs Legitimate Distribution")
    ax2.set_xticklabels(["Legitimate", "Fraud"])
    st.pyplot(fig2)

with fraud_detection:
    st.header("Fraud Detection Review")
    st.subheader("XGBoost Fraud Probability")
    risk_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": xgb_pred,
        "Probability": xgb_prob,
    })
    st.dataframe(risk_df.head(10), use_container_width=True)

    st.subheader("Isolation Forest Anomaly Detection")
    anomaly_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted_Anomaly": iso_pred,
        "Anomaly_Score": iso_anomaly_scores,
    })
    st.dataframe(anomaly_df.head(10), use_container_width=True)

with performance:
    st.header("Model Performance")
    if "XGBoost" in artifacts["metrics"]:
        xgb_metrics = artifacts["metrics"]["XGBoost"]
        iso_metrics = artifacts["metrics"]["Isolation Forest"]
    else:
        xgb_metrics = {}
        iso_metrics = {}

    metric_cols = st.columns(4)
    metric_cols[0].metric("XGBoost Precision", f"{xgb_metrics.get('precision', 0.0):.2%}")
    metric_cols[1].metric("XGBoost Recall", f"{xgb_metrics.get('recall', 0.0):.2%}")
    metric_cols[2].metric("XGBoost F1-Score", f"{xgb_metrics.get('f1_score', 0.0):.2%}")
    metric_cols[3].metric("XGBoost ROC-AUC", f"{xgb_metrics.get('roc_auc', 0.0):.2%}")

    st.subheader("XGBoost Confusion Matrix")
    st.pyplot(render_confusion_matrix(y_test, xgb_pred, "XGBoost Confusion Matrix"))

    st.subheader("Isolation Forest Confusion Matrix")
    st.pyplot(render_confusion_matrix(y_test, iso_pred, "Isolation Forest Confusion Matrix"))

    st.subheader("Key Definitions")
    st.markdown(
        """
        - True Positive (TP): actual fraud and predicted fraud
        - True Negative (TN): actual legitimate and predicted legitimate
        - False Positive (FP): actual legitimate but predicted fraud
        - False Negative (FN): actual fraud but predicted legitimate
        """
    )

with comparison:
    st.header("Model Comparison")
    comparison_rows = [
        {
            "Model": "Isolation Forest",
            "Accuracy": iso_metrics.get("accuracy", 0.0),
            "Precision": iso_metrics.get("precision", 0.0),
            "Recall": iso_metrics.get("recall", 0.0),
            "F1-Score": iso_metrics.get("f1_score", 0.0),
            "ROC-AUC": iso_metrics.get("roc_auc", 0.0),
            "PR-AUC": iso_metrics.get("pr_auc", 0.0),
        },
        {
            "Model": "XGBoost",
            "Accuracy": xgb_metrics.get("accuracy", 0.0),
            "Precision": xgb_metrics.get("precision", 0.0),
            "Recall": xgb_metrics.get("recall", 0.0),
            "F1-Score": xgb_metrics.get("f1_score", 0.0),
            "ROC-AUC": xgb_metrics.get("roc_auc", 0.0),
            "PR-AUC": xgb_metrics.get("pr_auc", 0.0),
        },
    ]
    comparison_df = pd.DataFrame(comparison_rows)
    st.dataframe(comparison_df, use_container_width=True)

    chart_df = comparison_df.set_index("Model")
    fig, ax = plt.subplots(figsize=(10, 5))
    chart_df[["Recall", "F1-Score", "Precision"]].plot(kind="bar", ax=ax, colormap="Set2")
    ax.set_title("Recall and F1-Score: Model Comparison")
    ax.set_ylabel("Score")
    plt.tight_layout()
    st.pyplot(fig)

with predict:
    st.header("Transaction Prediction")
    st.write("Enter the transaction feature values below to predict whether a payment is legitimate or fraudulent.")

    default_values = {}
    for column in feature_columns:
        if column == "Time":
            default_values[column] = float(base_df[column].median())
        elif column == "Amount":
            default_values[column] = float(base_df[column].median())
        else:
            default_values[column] = 0.0

    user_input = {}
    for column in feature_columns:
        user_input[column] = st.number_input(
            column,
            value=float(default_values[column]),
            format="%.6f",
            step=0.01,
        )

    if st.button("Predict Transaction"):
        from src.prediction import predict_transaction

        result = predict_transaction(user_input)
        xgb_probability = result["xgboost_probability"]
        xgb_label = "Fraudulent" if result["xgboost_prediction"] == 1 else "Legitimate"
        iso_label = "Fraudulent" if result["isolation_prediction"] == 1 else "Legitimate"

        st.metric("XGBoost Fraud Probability", f"{xgb_probability * 100:.2f}%")
        st.write("XGBoost Prediction:", xgb_label)
        st.write("Isolation Forest Prediction:", iso_label)

        if xgb_probability >= 0.5 or result["isolation_prediction"] == 1:
            st.error("High-risk transaction: suspicious activity detected.")
        else:
            st.success("Low-risk transaction: transaction appears legitimate.")

        st.caption("This is a model-driven estimate and should be paired with a broader fraud risk workflow.")

