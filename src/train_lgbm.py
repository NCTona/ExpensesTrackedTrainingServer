# -*- coding: utf-8 -*-
"""
train_lightgbm.py — Huấn luyện LightGBM cho bài toán dự báo chi tiêu theo danh mục.

Mô hình này chạy SERVER-SIDE (trên Spring Boot) để:
  - Dự báo chi tiêu tháng tiếp theo theo từng danh mục
  - Sử dụng behavioral features thay vì user_id để generalize cho mọi user

Input features (dữ liệu dạng bảng - tabular):
  - category_id, month, year
  - monthly_spending, transaction_count, avg_transaction, max_transaction
  - avg_day_of_week, avg_day_of_month
  - total_all_categories, category_ratio
  - prev_month_spending, prev_month_count, prev_month_ratio
  - avg_monthly_spending_3m (behavioral feature mới)
  - spending_trend (behavioral feature mới)

Target:
  - next_month_spending: Tổng chi tiêu tháng tiếp theo của category đó
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import mlflow
import mlflow.lightgbm
from prefect import task

INPUT_FILE = "data/raw/transactions.csv"
OUTPUT_DIR = "models"
MODEL_FILENAME = "category_forecast_lgbm.joblib"


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


def _engineer_features(df):
    """
    Feature Engineering: Tạo bảng đặc trưng từ dữ liệu giao dịch thô.
    Mỗi hàng đại diện cho (user, category, month) với các features tổng hợp.
    
    Không dùng user_id làm feature — thay bằng behavioral features
    để model generalize cho mọi user (bao gồm user mới).
    """
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day

    # === Tổng hợp theo (user, category, year-month) ===
    df["year_month"] = df["date"].dt.to_period("M")

    monthly_cat = df.groupby(["user_id", "category_id", "year_month"]).agg(
        monthly_spending=("amount", "sum"),
        transaction_count=("amount", "count"),
        avg_transaction=("amount", "mean"),
        max_transaction=("amount", "max"),
        avg_day_of_week=("day_of_week", "mean"),
        avg_day_of_month=("day_of_month", "mean"),
    ).reset_index()

    # Tổng chi tiêu toàn bộ categories trong tháng đó (để tính tỉ lệ)
    monthly_total = df.groupby(["user_id", "year_month"]).agg(
        total_all_categories=("amount", "sum")
    ).reset_index()

    monthly_cat = monthly_cat.merge(monthly_total, on=["user_id", "year_month"], how="left")

    # Tỉ lệ chi tiêu của category so với tổng
    monthly_cat["category_ratio"] = (
        monthly_cat["monthly_spending"] / monthly_cat["total_all_categories"].replace(0, 1)
    )

    # Sắp xếp theo thời gian
    monthly_cat = monthly_cat.sort_values(["user_id", "category_id", "year_month"])

    # === Tạo lag features (tháng trước) ===
    monthly_cat["prev_month_spending"] = monthly_cat.groupby(
        ["user_id", "category_id"]
    )["monthly_spending"].shift(1)

    monthly_cat["prev_month_count"] = monthly_cat.groupby(
        ["user_id", "category_id"]
    )["transaction_count"].shift(1)

    monthly_cat["prev_month_ratio"] = monthly_cat.groupby(
        ["user_id", "category_id"]
    )["category_ratio"].shift(1)

    # === BEHAVIORAL FEATURES MỚI ===
    
    # Trung bình chi tiêu 3 tháng gần nhất cho category
    monthly_cat["avg_monthly_spending_3m"] = monthly_cat.groupby(
        ["user_id", "category_id"]
    )["monthly_spending"].transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))
    
    # Xu hướng chi tiêu: last_month / avg_3m (> 1 = tăng, < 1 = giảm)
    monthly_cat["spending_trend"] = (
        monthly_cat["prev_month_spending"] / monthly_cat["avg_monthly_spending_3m"].replace(0, 1)
    )
    # Clip để tránh giá trị cực đoan
    monthly_cat["spending_trend"] = monthly_cat["spending_trend"].clip(0, 5)

    # === Tạo target: chi tiêu tháng SAU ===
    monthly_cat["next_month_spending"] = monthly_cat.groupby(
        ["user_id", "category_id"]
    )["monthly_spending"].shift(-1)

    # Trích xuất month number từ year_month period
    monthly_cat["month"] = monthly_cat["year_month"].dt.month
    monthly_cat["year"] = monthly_cat["year_month"].dt.year

    # Loại bỏ rows thiếu (đầu & cuối chuỗi không có lag/target)
    monthly_cat = monthly_cat.dropna(subset=["prev_month_spending", "next_month_spending", "avg_monthly_spending_3m"])

    return monthly_cat


@task(name="Train LightGBM Category Forecast Model")
def train_lightgbm():
    """Huấn luyện model LightGBM dự báo chi tiêu theo danh mục."""
    print("=== LIGHTGBM TRAINING ===")
    print(f"Loading data from {INPUT_FILE}...")

    df = _load_transactions()

    if len(df) < 100:
        print("[WARNING] Du lieu qua it (<100). Tao du lieu mo phong de demo...")
        df = _generate_demo_data()

    print(f"Loaded {len(df)} transactions. Engineering features...")
    features_df = _engineer_features(df)

    if len(features_df) < 5:
        print("[WARNING] Du lieu sau feature engineering qua it de train. Bo qua LightGBM.")
        return

    # Columns dùng để train — KHÔNG có user_id
    feature_cols = [
        "category_id", "month", "year",
        "monthly_spending", "transaction_count", "avg_transaction",
        "max_transaction", "avg_day_of_week", "avg_day_of_month",
        "total_all_categories", "category_ratio",
        "prev_month_spending", "prev_month_count", "prev_month_ratio",
        "avg_monthly_spending_3m", "spending_trend"
    ]
    target_col = "next_month_spending"

    X = features_df[feature_cols].values
    y = features_df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"Features: {feature_cols}")

    # === Huấn luyện LightGBM ===
    mlflow.set_experiment("Expense Forecasting - LightGBM")

    with mlflow.start_run():
        params = {
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

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.log_evaluation(period=50)]
        )

        # Đánh giá
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        print(f"[RESULT] LightGBM - RMSE: {rmse:.2f}, MAE: {mae:.2f}")

        # Log MLflow
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

        # Feature importance
        importance = model.feature_importances_
        for col, imp in zip(feature_cols, importance):
            mlflow.log_metric(f"fi_{col}", imp)

        # Lưu model
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model_path = os.path.join(OUTPUT_DIR, MODEL_FILENAME)

        # Lưu model + metadata (feature columns & tên)
        model_bundle = {
            "model": model,
            "feature_cols": feature_cols,
            "target_col": target_col,
        }
        joblib.dump(model_bundle, model_path)
        print(f"[OK] Model saved at: {model_path}")

        mlflow.log_artifact(model_path)
        mlflow.lightgbm.log_model(model, artifact_path="lgbm_model",
                                   registered_model_name="CategoryForecastLGBM")

    print("=== LIGHTGBM TRAINING COMPLETE ===")


def _generate_demo_data():
    """Tạo dữ liệu mô phỏng khi chưa có đủ data thực."""
    import random
    from datetime import datetime, timedelta

    random.seed(42)
    np.random.seed(42)

    records = []
    users = [1, 2, 3, 4, 5]
    categories = [1, 2, 3, 4, 5]  # Ăn uống, Đi lại, Giải trí, Mua sắm, Khác
    start_date = datetime(2024, 1, 1)

    for user_id in users:
        # Mỗi user có pattern riêng
        base_spending = random.randint(50000, 300000)
        for day_offset in range(365):
            date = start_date + timedelta(days=day_offset)
            # Mỗi ngày có 1-3 giao dịch
            n_transactions = random.randint(1, 3)
            for _ in range(n_transactions):
                cat = random.choice(categories)
                # Thêm seasonality: cuối tháng chi nhiều hơn
                day_factor = 1.0 + (date.day / 30.0) * 0.3
                amount = int(base_spending * day_factor * random.uniform(0.3, 2.0))
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
    print(f"Đã tạo {len(df)} giao dịch demo cho {len(users)} users.")
    return df


if __name__ == "__main__":
    from prefect import flow

    @flow(name="LightGBM Training Flow")
    def run_lgbm_training():
        train_lightgbm()

    run_lgbm_training()
