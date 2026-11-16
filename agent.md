# Banking Fraud Detection System — Agent Guide

**What this project does:** Classifies credit card transactions as legitimate or fraudulent using two ML models (Isolation Forest + XGBoost) and serves an interactive Streamlit dashboard and standalone web UI.

## Quick Start

```powershell
cd "fraud_detect\Banking-Fraud-Detection-and-Risk-Analysis-System-main"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py           # trains both models, saves artifacts to models/
streamlit run app.py      # launches dashboard
```

## Architecture

| File | Purpose |
|---|---|
| `train.py` | Main training pipeline — orchestrates preprocessing, SMOTE, model training, evaluation, artifact saving |
| `app.py` | Streamlit dashboard — loads saved models, shows metrics, enables live prediction |
| `frontend.html` | High-performance responsive web dashboard with live simulator and risk inspector |
| `main.py` | Quick EDA script (not part of pipeline) |
| `train_model.py` | Legacy RandomForest-based training script |
| `debug_fetch_dataset.py` | One-off dataset fetch diagnostic script |

### Source Modules (`src/`)

| Module | Role |
|---|---|
| `src/preprocessing.py` | Dataset loading (with fallback synthetic generation), splitting, scaling, SMOTE application |
| `src/xgboost_model.py` | Trains XGBoost classifier with balanced class weights and SMOTE on training data only |
| `src/isolation_forest.py` | Trains unsupervised Isolation Forest anomaly detector |
| `src/evaluation.py` | Metric computation (precision, recall, F1, ROC-AUC, PR-AUC), confusion matrix plotting |
| `src/prediction.py` | Loads artifacts and runs inference for a single transaction row |

## Data Flow

```
creditcard.csv (ULB dataset)
    → stratified train/test split (80/20)
    → StandardScaler fit on train only
    → SMOTE applied ONLY to X_train_scaled
    → XGBoost trained on resampled data
    → Isolation Forest trained on full X_train_scaled
    → All artifacts saved to models/
```

Key rule: **no data leakage** — SMOTE never touches test data; scaler never fitted on test.

## Saved Artifacts (`models/`)

| File | Content |
|---|---|
| `scaler.joblib` | Fitted StandardScaler |
| `feature_columns.joblib` | List of 30 feature column names |
| `xgboost_model.joblib` | Trained XGBClassifier |
| `isolation_forest.joblib` | Trained IsolationForest |
| `smote.joblib` | Trained SMOTE object |
| `model_metrics.json` | Evaluation metrics for both models |
| `training_summary.json` | Full training run summary |
| `fraud_model.pkl` | Legacy RandomForest model |

## Model Performance (current best)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| XGBoost | 99.86% | 61.0% | **85.1%** | 71.1% | 99.76% | 85.79% |
| Isolation Forest | 99.21% | 18.7% | 88.6% | 30.9% | 99.42% | 76.87% |

**XGBoost is the primary model.** Isolation Forest provides anomaly detection complement.

## Dataset Schema

31 columns: `V1–V28` (PCA components), `Time`, `Amount`, `Class` (0=legitimate, 1=fraud). ~284K rows, 0.2% fraud rate.

## Dashboard Tabs (`app.py`)

1. **Home Dashboard** — executive summary with counts and fraud %
2. **Dataset Overview** — raw data preview, missing values, duplicates
3. **Data Visualization** — amount/time distributions, class count plot
4. **Fraud Detection** — per-transaction predicted probabilities and anomaly scores
5. **Model Performance** — confusion matrices, metric cards
6. **Model Comparison** — side-by-side table and bar chart
7. **Transaction Prediction** — manual input form → dual-model prediction

## Key Dependencies

- `xgboost==2.1.1`, `imbalanced-learn==0.12.4`, `scikit-learn==1.5.2`
- `streamlit==1.40.1`, `pandas==2.2.3`, `joblib==1.4.2`
- Python 3.10+ (uses type hints with `__future__.annotations`)

## Files NOT to modify (artifacts)

`models/*.joblib`, `models/*.json`, `models/*.pkl` — these are outputs of `train.py`. Regenerate by re-running training.

## Frontend Web UI

A standalone web UI is available at:
- `frontend.html`

### Features:
- **Executive Overview:** Key metrics, SMOTE rebalancing chart, and ensemble architecture.
- **Model Comparison:** Performance tables and confusion matrix cards.
- **Risk Queue Inspector:** Real-time transaction review with search and risk filters.
- **Interactive Risk Simulator:** Transaction scoring tester with presets for legitimate and high-risk fraud cases.
