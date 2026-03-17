# -*- coding: utf-8 -*-
"""
config.py — Cau hinh tap trung cho toan bo MLOps project.

Tat ca constants, paths, va settings duoc khai bao tai day
thay vi hardcode rai rac trong tung file.
"""

import os

# ==============================================================
# BASE DIRECTORIES
# ==============================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ==============================================================
# DATA FILES
# ==============================================================
RAW_TRANSACTIONS_FILE = os.path.join(RAW_DATA_DIR, "transactions.csv")

# Cot mac dinh khi CSV khong co header
CSV_DEFAULT_COLUMNS = [
    "transaction_id", "user_id", "category_id", "amount", "date", "note"
]

# ==============================================================
# LSTM TRAINING
# ==============================================================
LSTM_WINDOW_SIZE = 4        # 4 tuan input
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 8
LSTM_UNITS = 32
LSTM_DENSE_UNITS = 16

# ==============================================================
# LIGHTGBM TRAINING
# ==============================================================
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_estimators": 200,
}

LGBM_FEATURE_COLS = [
    "category_id", "month", "year",
    "monthly_spending", "transaction_count", "avg_transaction",
    "max_transaction", "avg_day_of_week", "avg_day_of_month",
    "total_all_categories", "category_ratio",
    "prev_month_spending", "prev_month_count", "prev_month_ratio",
    "avg_monthly_spending_3m", "spending_trend",
]

# ==============================================================
# ISOLATION FOREST
# ==============================================================
IFOREST_CONTAMINATION = 0.05
IFOREST_N_ESTIMATORS = 100

IFOREST_FEATURE_COLS = [
    "amount", "category_id", "day_of_week", "day_of_month", "amount_vs_category_avg"
]

# ==============================================================
# MODEL FILENAMES
# ==============================================================
LSTM_TFLITE_FILE = "expense_model.tflite"
LSTM_META_FILE = "meta_lstm.joblib"
LGBM_MODEL_FILE = "category_forecast_lgbm.joblib"
IFOREST_MODEL_FILE = "anomaly_iforest.joblib"
BEST_METRICS_FILE = "best_metrics.json"

# Full paths
LSTM_TFLITE_PATH = os.path.join(MODELS_DIR, LSTM_TFLITE_FILE)
LSTM_META_PATH = os.path.join(MODELS_DIR, LSTM_META_FILE)
LGBM_MODEL_PATH = os.path.join(MODELS_DIR, LGBM_MODEL_FILE)
IFOREST_MODEL_PATH = os.path.join(MODELS_DIR, IFOREST_MODEL_FILE)
BEST_METRICS_PATH = os.path.join(MODELS_DIR, BEST_METRICS_FILE)

# ==============================================================
# DEMO DATA
# ==============================================================
DEMO_DATA_THRESHOLD = 2000   # Mix demo data neu real data < threshold
DEMO_NUM_USERS = 20
DEMO_MONTHS = 12

# ==============================================================
# API / SERVER
# ==============================================================
PREDICT_SERVER_HOST = "0.0.0.0"
PREDICT_SERVER_PORT = 8001
PREDICT_SERVER_RELOAD_URL = f"http://127.0.0.1:{PREDICT_SERVER_PORT}/reload"

BACKEND_INGEST_URL = "https://localhost:8080/api/internal/transactions"
BACKEND_DEPLOY_URL = "https://localhost:8080/api/internal/model/update"
API_KEY_HEADER = "X-API-KEY"
DEFAULT_API_KEY = "secret_mlops_key"

# ==============================================================
# PREFECT
# ==============================================================
PREFECT_PORT = 4200
PIPELINE_CRON = "0 2 * * 0"   # 2:00 sang Chu Nhat hang tuan

# ==============================================================
# MLFLOW EXPERIMENT NAMES
# ==============================================================
MLFLOW_LSTM_EXPERIMENT = "Expense Forecasting - LSTM"
MLFLOW_LGBM_EXPERIMENT = "Expense Forecasting - LightGBM"
MLFLOW_ANOMALY_EXPERIMENT = "Expense Forecasting - Anomaly Detection"
