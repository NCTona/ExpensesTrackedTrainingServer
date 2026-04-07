# -*- coding: utf-8 -*-
"""
train_lgbm.py — Train LightGBM cho bai toan du bao chi tieu theo danh muc.

Model nay chay SERVER-SIDE (tren FastAPI) de:
  - Du bao tong chi tieu CUOI THANG HIEN TAI dua tren chi tieu dang dien ra
  - Giup canh bao user khi sap vuot ngan sach

Input features:
  - category_id: danh muc chi tieu
  - days_passed: so ngay da qua trong thang
  - days_remaining: so ngay con lai
  - current_spent: da chi bao nhieu cho category nay
  - current_tx_count: so giao dich da thuc hien
  - daily_rate: toc do chi binh quan/ngay = current_spent / days_passed
  - category_ratio: ti le category / tong chi tat ca categories

Target:
  - month_end_spending: Tong chi tieu THUC TE cuoi thang cho category do
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
    Feature Engineering: Tao training data tu giao dich.

    Voi moi thang da hoan tat, cat tai nhieu thoi diem (moi 3 ngay)
    de tao training samples. Moi sample mo phong:
      "Tai ngay X, user da chi Y cho category Z → cuoi thang thuc te = W"

    Args:
        df: DataFrame giao dich voi cot date, amount, user_id, category_id.

    Returns:
        DataFrame voi feature columns va target month_end_spending.
    """
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["year_month"] = df["date"].dt.to_period("M")

    # Tinh tong chi tieu THUC TE cuoi thang cho moi (user, category, month)
    month_end_totals = df.groupby(["user_id", "category_id", "year_month"]).agg(
        month_end_spending=("amount", "sum"),
        total_tx_count=("amount", "count"),
    ).reset_index()

    # Tong chi tieu tat ca categories trong thang (cho category_ratio)
    month_totals_all = df.groupby(["user_id", "year_month"]).agg(
        month_total_all=("amount", "sum"),
    ).reset_index()

    month_end_totals = month_end_totals.merge(
        month_totals_all, on=["user_id", "year_month"], how="left"
    )

    samples = []
    cut_days = list(range(3, 29, 3))  # [3, 6, 9, ..., 27]

    for _, row in month_end_totals.iterrows():
        user_id = row["user_id"]
        cat_id = row["category_id"]
        ym = row["year_month"]
        month_end = row["month_end_spending"]
        month_total_all = row["month_total_all"]

        # So ngay trong thang
        total_days = ym.to_timestamp().days_in_month

        # Lay giao dich cua (user, category, month) nay
        mask = (
            (df["user_id"] == user_id)
            & (df["category_id"] == cat_id)
            & (df["year_month"] == ym)
        )
        user_cat_tx = df[mask]

        for cut_day in cut_days:
            if cut_day > total_days:
                continue

            # Chi tinh giao dich TU NGAY 1 DEN cut_day
            tx_until_cut = user_cat_tx[user_cat_tx["day"] <= cut_day]

            current_spent = float(tx_until_cut["amount"].sum())
            current_tx_count = len(tx_until_cut)
            days_passed = cut_day
            days_remaining = total_days - cut_day
            daily_rate = current_spent / days_passed if days_passed > 0 else 0

            # Tong tat ca categories den ngay cut_day
            mask_all = (
                (df["user_id"] == user_id)
                & (df["year_month"] == ym)
                & (df["day"] <= cut_day)
            )
            total_all_until_cut = float(df[mask_all]["amount"].sum())
            category_ratio = (
                current_spent / total_all_until_cut
                if total_all_until_cut > 0
                else 0
            )

            samples.append({
                "category_id": cat_id,
                "days_passed": days_passed,
                "days_remaining": days_remaining,
                "current_spent": current_spent,
                "current_tx_count": current_tx_count,
                "daily_rate": daily_rate,
                "category_ratio": category_ratio,
                "month_end_spending": month_end,  # TARGET
            })

    return pd.DataFrame(samples)


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

    target_col = "month_end_spending"
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
