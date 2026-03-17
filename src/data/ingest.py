# -*- coding: utf-8 -*-
"""
ingest.py — Lay du lieu giao dich tu Spring Boot Backend.

Goi API de tai du lieu moi nhat va luu vao data/raw/transactions.csv.
"""

import logging
import os

import pandas as pd
import requests
from prefect import task

from src.config import (
    BACKEND_INGEST_URL,
    RAW_TRANSACTIONS_FILE,
    API_KEY_HEADER,
    DEFAULT_API_KEY,
)

logger = logging.getLogger(__name__)


@task(name="Fetch Data from Backend", retries=2, retry_delay_seconds=10)
def fetch_data() -> None:
    """
    Goi HTTP GET toi Spring Boot API de lay du lieu giao dich moi.

    Du lieu duoc luu thanh CSV tai data/raw/transactions.csv.

    Raises:
        Exception: Neu khong ket noi duoc hoac API tra loi loi.
    """
    logger.info(f"Fetching data from {BACKEND_INGEST_URL}...")
    try:
        headers = {API_KEY_HEADER: DEFAULT_API_KEY}
        response = requests.get(
            BACKEND_INGEST_URL,
            headers=headers,
            timeout=10,
            verify=False,
        )
        response.raise_for_status()
        data = response.json()

        if not data:
            logger.info("No new data found from Backend.")
        else:
            df = pd.DataFrame(data)
            os.makedirs(os.path.dirname(RAW_TRANSACTIONS_FILE), exist_ok=True)
            df.to_csv(RAW_TRANSACTIONS_FILE, index=False)
            logger.info(
                f"Successfully saved {len(df)} rows into {RAW_TRANSACTIONS_FILE}"
            )
    except Exception as e:
        logger.error(f"Error fetching data from Backend: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from prefect import flow

    @flow(name="Data Ingestion Flow")
    def run_ingestion():
        fetch_data()

    run_ingestion()
