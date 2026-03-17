# -*- coding: utf-8 -*-
"""
train_iforest.py — Train Isolation Forest cho phat hien giao dich bat thuong.

Model nay chay SERVER-SIDE de:
  - Phat hien cac giao dich bat thuong
  - Canh bao ve chi tieu bat thuong dua tren lich su

Features:
  - amount: Gia tri giao dich
  - category_id: ID danh muc
  - day_of_week: Ngay trong tuan (0-6)
  - day_of_month: Ngay trong thang (1-31)
  - amount_vs_category_avg: Ti le amount / TB category
"""

import logging
import os
from typing import List, Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd
from prefect import task
from sklearn.ensemble import IsolationForest

from src.config import (
    MODELS_DIR,
    IFOREST_MODEL_PATH,
    IFOREST_CONTAMINATION,
    IFOREST_N_ESTIMATORS,
    IFOREST_FEATURE_COLS,
    DEMO_DATA_THRESHOLD,
    DEMO_NUM_USERS,
    DEMO_MONTHS,
    MLFLOW_ANOMALY_EXPERIMENT,
)
from src.data.loader import load_transactions
from src.data.generate_demo import generate_realistic_expenses

logger = logging.getLogger(__name__)


def _engineer_anomaly_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Feature engineering cho Isolation Forest.

    Tinh ti le amount / TB category de phat hien chi tieu cao bat thuong.

    Args:
        df: DataFrame giao dich.

    Returns:
        Tuple (df voi features moi, danh sach feature columns).
    """
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day

    category_avg = df.groupby("category_id")["amount"].mean().to_dict()
    df["category_avg"] = df["category_id"].map(category_avg)
    df["amount_vs_category_avg"] = df["amount"] / df["category_avg"].replace(0, 1)

    return df, IFOREST_FEATURE_COLS


@task(name="Train Isolation Forest Anomaly Detection Model")
def train_iforest() -> None:
    """Train Isolation Forest cho phat hien bat thuong."""
    logger.info("=== ISOLATION FOREST TRAINING ===")

    df = load_transactions()

    # Merge demo data if real data is small
    if len(df) < DEMO_DATA_THRESHOLD:
        logger.warning("Real data small. Mixing in realistic demo data...")
        df_demo = generate_realistic_expenses(
            num_users=DEMO_NUM_USERS, months=DEMO_MONTHS
        )
        df_demo["date"] = pd.to_datetime(df_demo["date"])
        df = pd.concat([df, df_demo], ignore_index=True)

    logger.info(f"Loaded {len(df)} transactions. Engineering features...")
    df, feature_cols = _engineer_anomaly_features(df)

    X = df[feature_cols].values
    logger.info(f"Training Isolation Forest on {len(X)} samples...")

    mlflow.set_experiment(MLFLOW_ANOMALY_EXPERIMENT)

    with mlflow.start_run():
        model = IsolationForest(
            contamination=IFOREST_CONTAMINATION,
            n_estimators=IFOREST_N_ESTIMATORS,
            max_samples="auto",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X)

        # Evaluation
        predictions = model.predict(X)
        n_anomalies = int(np.sum(predictions == -1))
        n_normal = int(np.sum(predictions == 1))
        anomaly_ratio = n_anomalies / len(X) * 100

        logger.info(
            f"[RESULT] Anomalies: {n_anomalies} ({anomaly_ratio:.1f}%), "
            f"Normal: {n_normal}"
        )

        # Log MLflow
        mlflow.log_param("contamination", IFOREST_CONTAMINATION)
        mlflow.log_param("n_estimators", IFOREST_N_ESTIMATORS)
        mlflow.log_param("n_samples", len(X))
        mlflow.log_metric("n_anomalies", n_anomalies)
        mlflow.log_metric("n_normal", n_normal)
        mlflow.log_metric("anomaly_ratio_percent", anomaly_ratio)

        # Save model
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_bundle = {
            "model": model,
            "feature_cols": feature_cols,
        }
        joblib.dump(model_bundle, IFOREST_MODEL_PATH)
        logger.info(f"[OK] Anomaly model saved at: {IFOREST_MODEL_PATH}")

        mlflow.log_artifact(IFOREST_MODEL_PATH)

    logger.info("=== ISOLATION FOREST TRAINING COMPLETE ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from prefect import flow

    @flow(name="Isolation Forest Training Flow")
    def run_iforest_training():
        train_iforest()

    run_iforest_training()
