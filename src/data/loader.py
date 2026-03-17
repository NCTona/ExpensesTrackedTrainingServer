# -*- coding: utf-8 -*-
"""
data_loader.py — Module doc va chuan hoa du lieu transactions.

Gom chung logic _load_transactions() truoc day trung lap
giua train_lgbm.py va train_iforest.py.
"""

import logging
import os
from typing import List, Optional

import pandas as pd

from src.config import RAW_TRANSACTIONS_FILE, CSV_DEFAULT_COLUMNS

logger = logging.getLogger(__name__)


def load_transactions(
    filepath: Optional[str] = None,
    expense_only: bool = True,
) -> pd.DataFrame:
    """
    Doc file transactions.csv va chuan hoa.

    Args:
        filepath: Duong dan toi file CSV. Mac dinh dung RAW_TRANSACTIONS_FILE.
        expense_only: Neu True, chi giu lai cac giao dich type='expense'.

    Returns:
        DataFrame da chuan hoa voi cac cot: date (datetime), amount (numeric).

    Raises:
        FileNotFoundError: Neu file khong ton tai.
    """
    filepath = filepath or RAW_TRANSACTIONS_FILE

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing {filepath}")

    df = pd.read_csv(filepath)

    # Auto-detect columns format
    if "date" not in df.columns:
        df = pd.read_csv(filepath, names=CSV_DEFAULT_COLUMNS)

    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)

    # Filter expense only
    if expense_only and "type" in df.columns:
        df = df[df["type"].str.lower() == "expense"]

    logger.info(f"Loaded {len(df)} transactions from {filepath}")
    return df
