# -*- coding: utf-8 -*-
"""
train_iforest.py — Huấn luyện Isolation Forest cho phát hiện giao dịch bất thường.

Mô hình này chạy SERVER-SIDE để:
  - Phát hiện giao dịch bất thường (anomaly detection)
  - Cảnh báo chi tiêu lạ dựa trên pattern lịch sử

Features cho mỗi giao dịch:
  - amount: Số tiền giao dịch
  - category_id: Mã danh mục
  - day_of_week: Thứ trong tuần (0-6)
  - day_of_month: Ngày trong tháng (1-31)
  - amount_vs_category_avg: Tỉ số amount / TB danh mục (phát hiện chi tiêu cao bất thường)
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import mlflow
from prefect import task

INPUT_FILE = "data/raw/transactions.csv"
OUTPUT_DIR = "models"
MODEL_FILENAME = "anomaly_iforest.joblib"


def _load_transactions():
    """Đọc file transactions.csv và chuẩn hóa."""
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Không tìm thấy {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    if "date" not in df.columns:
        df = pd.read_csv(INPUT_FILE, names=["transaction_id", "user_id", "category_id", "amount", "date", "note"])

    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    return df


def _engineer_anomaly_features(df):
    """
    Feature engineering cho Isolation Forest.
    Tính ratio amount / TB category để phát hiện chi tiêu cao bất thường.
    """
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day

    # Tính trung bình chi tiêu theo category (toàn bộ data)
    category_avg = df.groupby("category_id")["amount"].mean().to_dict()
    df["category_avg"] = df["category_id"].map(category_avg)
    df["amount_vs_category_avg"] = df["amount"] / df["category_avg"].replace(0, 1)

    feature_cols = ["amount", "category_id", "day_of_week", "day_of_month", "amount_vs_category_avg"]
    return df, feature_cols


@task(name="Train Isolation Forest Anomaly Detection Model")
def train_iforest():
    """Huấn luyện Isolation Forest cho phát hiện bất thường."""
    print("=== ISOLATION FOREST TRAINING ===")
    print(f"Loading data from {INPUT_FILE}...")

    df = _load_transactions()

    if len(df) < 50:
        print("[WARNING] Dữ liệu quá ít (<50). Tạo demo data...")
        df = _generate_demo_data()

    print(f"Loaded {len(df)} transactions. Engineering features...")
    df, feature_cols = _engineer_anomaly_features(df)

    X = df[feature_cols].values

    print(f"Training Isolation Forest on {len(X)} samples...")

    mlflow.set_experiment("Expense Forecasting - Anomaly Detection")

    with mlflow.start_run():
        # contamination=0.05 nghĩa là kỳ vọng 5% giao dịch là bất thường
        model = IsolationForest(
            contamination=0.05,
            n_estimators=100,
            max_samples='auto',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X)

        # Đánh giá: tính số lượng anomaly phát hiện được
        predictions = model.predict(X)
        n_anomalies = np.sum(predictions == -1)
        n_normal = np.sum(predictions == 1)
        anomaly_ratio = n_anomalies / len(X) * 100

        print(f"[RESULT] Anomalies: {n_anomalies} ({anomaly_ratio:.1f}%), Normal: {n_normal}")

        # Log MLflow
        mlflow.log_param("contamination", 0.05)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("n_samples", len(X))
        mlflow.log_metric("n_anomalies", n_anomalies)
        mlflow.log_metric("n_normal", n_normal)
        mlflow.log_metric("anomaly_ratio_percent", anomaly_ratio)

        # Lưu model
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model_path = os.path.join(OUTPUT_DIR, MODEL_FILENAME)

        model_bundle = {
            "model": model,
            "feature_cols": feature_cols,
        }
        joblib.dump(model_bundle, model_path)
        print(f"[OK] Anomaly model saved at: {model_path}")

        mlflow.log_artifact(model_path)

    print("=== ISOLATION FOREST TRAINING COMPLETE ===")


def _generate_demo_data():
    """Tạo dữ liệu demo khi chưa có đủ data."""
    import random
    from datetime import datetime, timedelta

    random.seed(42)
    np.random.seed(42)

    records = []
    users = [1, 2, 3]
    categories = [1, 2, 3, 4, 5]
    start_date = datetime(2024, 1, 1)

    for user_id in users:
        base = random.randint(50000, 200000)
        for day_offset in range(180):
            date = start_date + timedelta(days=day_offset)
            n_tx = random.randint(1, 3)
            for _ in range(n_tx):
                cat = random.choice(categories)
                amount = int(base * random.uniform(0.3, 2.0))
                # 5% chance tạo giao dịch bất thường (x3-x5 lần bình thường)
                if random.random() < 0.05:
                    amount *= random.randint(3, 5)
                records.append({
                    "transaction_id": len(records) + 1,
                    "user_id": user_id,
                    "category_id": cat,
                    "amount": amount,
                    "date": date.strftime("%Y-%m-%d"),
                    "note": "Demo"
                })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Đã tạo {len(df)} giao dịch demo.")
    return df


if __name__ == "__main__":
    from prefect import flow

    @flow(name="Isolation Forest Training Flow")
    def run_iforest_training():
        train_iforest()

    run_iforest_training()
