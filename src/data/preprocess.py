# -*- coding: utf-8 -*-
"""
preprocess.py — Tien xu ly du lieu giao dich thanh time series theo tuan.

Pipeline:
  1. Doc transactions.csv
  2. Filter CHI category "An uong" (category_id=2) cho LSTM
  3. Aggregate theo tuan (weekly spending)
  4. Tao sequences (window 4 tuan) voi per-window max normalization
  5. Luu X_ts.npy, y.npy, meta.joblib vao data/processed/

Ly do chi dung category An uong:
  - Cac category khac co chi tieu dot ngot, bat thuong -> LSTM khong hoc duoc trend
  - An uong co tan suat deu, amount on dinh -> phu hop voi LSTM forecasting
"""

import logging
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from prefect import task

from src.config import (
    RAW_TRANSACTIONS_FILE,
    PROCESSED_DATA_DIR,
    LSTM_WINDOW_SIZE,
    CSV_DEFAULT_COLUMNS,
    DEMO_NUM_USERS,
    DEMO_MONTHS,
    LSTM_FOOD_CATEGORY_ID,
)
from src.data.generate_demo import generate_realistic_expenses

logger = logging.getLogger(__name__)

WINDOW_SIZE = LSTM_WINDOW_SIZE


@task(name="Create Weekly Time Series Sequences (Per-Window Max Normalization)")
def create_sequences_normalized(
    raw_values: np.ndarray,
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tao sequences voi per-window max normalization.

    Moi window duoc normalize boi max cua 4 tuan input,
    dong bo voi cach Android TFLite normalize khi predict.

    Args:
        raw_values: Mang gia tri chi tieu theo tuan, shape (n, 1).
        window: So tuan input (mac dinh 4).

    Returns:
        Tuple (X, y) — X shape (samples, window, 1), y shape (samples,).
    """
    X, y = [], []
    for i in range(len(raw_values) - window):
        window_vals = raw_values[i : i + window].flatten()
        target_val = raw_values[i + window].flatten()[0]

        max_val = window_vals.max()
        if max_val <= 0:
            continue

        scaled_window = np.clip(window_vals / max_val, 0, 3.0)
        scaled_target = np.clip(target_val / max_val, 0, 3.0)

        X.append(scaled_window.reshape(-1, 1))
        y.append(scaled_target)

    if len(X) == 0:
        return np.array([]), np.array([])
    return np.array(X), np.array(y)


def _process_users_weekly(
    df: pd.DataFrame,
    window_size: int,
) -> Tuple[list, list]:
    """Xu ly tung user: aggregate weekly va tao sequences."""
    X_all, y_all = [], []

    for user_id in df["user_id"].unique():
        df_user = df[df["user_id"] == user_id]

        if "type" in df.columns:
            df_user = df_user[df_user["type"].str.lower() == "expense"]

        if len(df_user) == 0:
            continue

        weekly = df_user.set_index("date").resample("W")["amount"].sum().fillna(0)
        raw_values = weekly.values.reshape(-1, 1)

        if len(raw_values) <= window_size:
            continue

        X_ts, y = create_sequences_normalized(raw_values, window_size)
        if len(X_ts) > 0:
            X_all.append(X_ts)
            y_all.append(y)

    return X_all, y_all


@task(name="Preprocess Transactions Data (Weekly - Food Category Only)")
def preprocess() -> None:
    """
    Tien xu ly du lieu transactions thanh sequences cho LSTM training.

    - Doc data/raw/transactions.csv
    - Filter CHI category An uong (LSTM_FOOD_CATEGORY_ID)
    - Aggregate theo tuan
    - Per-window max normalization (dong bo voi TFLite)
    - Luu vao data/processed/
    """
    logger.info(f"Preprocessing {RAW_TRANSACTIONS_FILE} (weekly aggregation)...")

    if not os.path.exists(RAW_TRANSACTIONS_FILE):
        raise FileNotFoundError(f"Missing {RAW_TRANSACTIONS_FILE}")

    # Read CSV
    df = pd.read_csv(RAW_TRANSACTIONS_FILE)
    if "date" not in df.columns:
        df = pd.read_csv(RAW_TRANSACTIONS_FILE, names=CSV_DEFAULT_COLUMNS)

    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)

    # Filter CHI category An uong cho LSTM
    total_before = len(df)
    df_food = df[df["category_id"] == LSTM_FOOD_CATEGORY_ID].copy()
    logger.info(
        f"Filtered category_id={LSTM_FOOD_CATEGORY_ID} (An uong): "
        f"{len(df_food)}/{total_before} transactions"
    )

    # Process real data (chi An uong)
    X_ts_all, y_all = _process_users_weekly(df_food, WINDOW_SIZE)

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Fallback: generate demo data if real data insufficient
    if len(X_ts_all) == 0:
        logger.warning("Real food data insufficient. Generating demo data for stability...")
        df_demo = generate_realistic_expenses(
            num_users=DEMO_NUM_USERS, months=DEMO_MONTHS
        )
        df_demo["date"] = pd.to_datetime(df_demo["date"])
        # Demo data cung chi lay An uong
        if "category_id" in df_demo.columns:
            df_demo = df_demo[df_demo["category_id"] == LSTM_FOOD_CATEGORY_ID]
        demo_X, demo_y = _process_users_weekly(df_demo, WINDOW_SIZE)
        X_ts_all.extend(demo_X)
        y_all.extend(demo_y)

    if len(X_ts_all) == 0:
        logger.warning("Still insufficient food data after demo generation.")
        np.save(os.path.join(PROCESSED_DATA_DIR, "X_ts.npy"), np.array([]))
        np.save(os.path.join(PROCESSED_DATA_DIR, "y.npy"), np.array([]))
    else:
        all_X = np.concatenate(X_ts_all)
        all_y = np.concatenate(y_all)
        np.save(os.path.join(PROCESSED_DATA_DIR, "X_ts.npy"), all_X)
        np.save(os.path.join(PROCESSED_DATA_DIR, "y.npy"), all_y)
        logger.info(f"Total food samples for training: {len(all_X)}")

    # Save meta
    joblib.dump(
        {
            "window_size": WINDOW_SIZE,
            "normalization": "per_window_max",
            "category_id": LSTM_FOOD_CATEGORY_ID,
            "category_name": "An uong",
        },
        os.path.join(PROCESSED_DATA_DIR, "meta.joblib"),
    )
    logger.info(
        f"Preprocess complete! Window size: {WINDOW_SIZE} weeks, "
        f"normalization: per_window_max, category: An uong (id={LSTM_FOOD_CATEGORY_ID})"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from prefect import flow

    @flow(name="Data Preprocessing Flow")
    def run_preprocessing():
        preprocess()

    run_preprocessing()
