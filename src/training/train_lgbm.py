# -*- coding: utf-8 -*-
"""
train_lgbm.py — Train LightGBM cho bai toan du bao chi tieu theo danh muc.

Model nay chay SERVER-SIDE (tren Spring Boot) de:
  - Du bao chi tieu thang tiep theo theo tung danh muc
  - Su dung behavioral features thay vi user_id de generalize cho moi user

Input features (tabular data):
  - category_id, month, year
  - monthly_spending, transaction_count, avg_transaction, max_transaction
  - avg_day_of_week, avg_day_of_month
  - total_all_categories, category_ratio
  - prev_month_spending, prev_month_count, prev_month_ratio
  - avg_monthly_spending_3m, spending_trend

Target:
  - next_month_spending: Tong chi tieu cho category do trong thang tiep theo
"""

import logging
import os

import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from prefect import task
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from src.config import (
    MODELS_DIR,
    LGBM_MODEL_PATH,
    LGBM_PARAMS,
    LGBM_FEATURE_COLS,
    DEMO_DATA_THRESHOLD,
    DEMO_NUM_USERS,
    DEMO_MONTHS,
    MLFLOW_LGBM_EXPERIMENT,
)
from src.data.loader import load_transactions
from src.data.generate_demo import generate_realistic_expenses

logger = logging.getLogger(__name__)


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Engineering: Tao bang dac trung tu du lieu giao dich tho.

    Moi hang dai dien cho (user, category, month) voi cac features tong hop.
    Khong dung user_id lam feature — thay bang behavioral features
    de model generalize cho moi user (bao gom user moi).

    Args:
        df: DataFrame giao dich voi cot date, amount, user_id, category_id.

    Returns:
        DataFrame voi cac feature columns va target next_month_spending.
    """
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["year_month"] = df["date"].dt.to_period("M")

    # Tong hop theo (user, category, year-month)
    monthly_cat = df.groupby(["user_id", "category_id", "year_month"]).agg(
        monthly_spending=("amount", "sum"),
        transaction_count=("amount", "count"),
        avg_transaction=("amount", "mean"),
        max_transaction=("amount", "max"),
        avg_day_of_week=("day_of_week", "mean"),
        avg_day_of_month=("day_of_month", "mean"),
    ).reset_index()

    # Tong chi tieu toan bo categories trong thang
    monthly_total = df.groupby(["user_id", "year_month"]).agg(
        total_all_categories=("amount", "sum")
    ).reset_index()

    monthly_cat = monthly_cat.merge(
        monthly_total, on=["user_id", "year_month"], how="left"
    )

    # Ti le chi tieu cua category so voi tong
    monthly_cat["category_ratio"] = (
        monthly_cat["monthly_spending"]
        / monthly_cat["total_all_categories"].replace(0, 1)
    )

    monthly_cat = monthly_cat.sort_values(["user_id", "category_id", "year_month"])

    # Lag features (thang truoc)
    group_cols = ["user_id", "category_id"]
    monthly_cat["prev_month_spending"] = monthly_cat.groupby(group_cols)[
        "monthly_spending"
    ].shift(1)
    monthly_cat["prev_month_count"] = monthly_cat.groupby(group_cols)[
        "transaction_count"
    ].shift(1)
    monthly_cat["prev_month_ratio"] = monthly_cat.groupby(group_cols)[
        "category_ratio"
    ].shift(1)

    # Behavioral features
    monthly_cat["avg_monthly_spending_3m"] = monthly_cat.groupby(group_cols)[
        "monthly_spending"
    ].transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))

    monthly_cat["spending_trend"] = (
        monthly_cat["prev_month_spending"]
        / monthly_cat["avg_monthly_spending_3m"].replace(0, 1)
    ).clip(0, 5)

    # Target: chi tieu thang SAU
    monthly_cat["next_month_spending"] = monthly_cat.groupby(group_cols)[
        "monthly_spending"
    ].shift(-1)

    # Extract month/year tu period
    monthly_cat["month"] = monthly_cat["year_month"].dt.month
    monthly_cat["year"] = monthly_cat["year_month"].dt.year

    # Loai bo rows thieu
    monthly_cat = monthly_cat.dropna(
        subset=["prev_month_spending", "next_month_spending", "avg_monthly_spending_3m"]
    )

    return monthly_cat


@task(name="Train LightGBM Category Forecast Model")
def train_lightgbm() -> None:
    """Train model LightGBM du bao chi tieu theo danh muc."""
    logger.info("=== LIGHTGBM TRAINING ===")

    df = load_transactions()

    # Mix demo data if real data is scarce
    if len(df) < DEMO_DATA_THRESHOLD:
        logger.warning("Real data small. Mixing in demo data for training...")
        df_demo = generate_realistic_expenses(
            num_users=DEMO_NUM_USERS, months=DEMO_MONTHS
        )
        df_demo["date"] = pd.to_datetime(df_demo["date"])
        df = pd.concat([df, df_demo], ignore_index=True)

    logger.info(f"Loaded {len(df)} transactions. Engineering features...")
    features_df = _engineer_features(df)

    if len(features_df) < 2:
        logger.error("Insufficient data for training. Saving placeholder model...")
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(
            {"model": None, "is_dummy": True}, LGBM_MODEL_PATH
        )
        return

    target_col = "next_month_spending"
    X = features_df[LGBM_FEATURE_COLS].values
    y_vals = features_df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_vals, test_size=0.2, random_state=42
    )

    logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    logger.info(f"Features: {LGBM_FEATURE_COLS}")

    # Train LightGBM
    mlflow.set_experiment(MLFLOW_LGBM_EXPERIMENT)

    with mlflow.start_run():
        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.log_evaluation(period=50)],
        )

        # Evaluate
        y_pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))

        logger.info(f"[RESULT] LightGBM - RMSE: {rmse:.2f}, MAE: {mae:.2f}")

        # Log MLflow
        mlflow.log_params(LGBM_PARAMS)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

        # Feature importance
        for col, imp in zip(LGBM_FEATURE_COLS, model.feature_importances_):
            mlflow.log_metric(f"fi_{col}", float(imp))

        # Save model
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_bundle = {
            "model": model,
            "feature_cols": LGBM_FEATURE_COLS,
            "target_col": target_col,
        }
        joblib.dump(model_bundle, LGBM_MODEL_PATH)
        logger.info(f"[OK] Model saved at: {LGBM_MODEL_PATH}")

        mlflow.log_artifact(LGBM_MODEL_PATH)
        mlflow.lightgbm.log_model(
            model,
            artifact_path="lgbm_model",
            registered_model_name="CategoryForecastLGBM",
        )

    logger.info("=== LIGHTGBM TRAINING COMPLETE ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from prefect import flow

    @flow(name="LightGBM Training Flow")
    def run_lgbm_training():
        train_lightgbm()

    run_lgbm_training()
