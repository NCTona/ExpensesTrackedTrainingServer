# -*- coding: utf-8 -*-
"""
serve_predict.py — FastAPI Prediction Service cho LightGBM + Isolation Forest.

Chay nhu mot microservice ben canh Spring Boot.
Spring Boot goi API nay de lay ket qua du doan xu huong chi tieu.

Endpoints:
  POST /predict/category   -> Du doan chi tieu thang toi theo tung danh muc
  POST /predict/bulk       -> Du doan nhieu categories cung luc
  POST /predict/trend      -> Phan tich xu huong chi tieu so dong
  POST /predict/anomaly    -> Phat hien giao dich bat thuong (Isolation Forest)
  GET  /health             -> Kiem tra trang thai server
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import (
    LGBM_MODEL_PATH,
    IFOREST_MODEL_PATH,
    PREDICT_SERVER_HOST,
    PREDICT_SERVER_PORT,
)

logger = logging.getLogger(__name__)

# ================= Global State =================
model_bundle: Optional[dict] = None
anomaly_bundle: Optional[dict] = None


def load_model() -> None:
    """Load model LightGBM tu file."""
    global model_bundle
    if os.path.exists(LGBM_MODEL_PATH):
        model_bundle = joblib.load(LGBM_MODEL_PATH)
        logger.info(f"Loaded LightGBM model from {LGBM_MODEL_PATH}")
    else:
        logger.warning(f"Model not found at {LGBM_MODEL_PATH}")


def load_anomaly_model() -> None:
    """Load model Isolation Forest tu file."""
    global anomaly_bundle
    if os.path.exists(IFOREST_MODEL_PATH):
        anomaly_bundle = joblib.load(IFOREST_MODEL_PATH)
        logger.info(f"Loaded Isolation Forest model from {IFOREST_MODEL_PATH}")
    else:
        logger.warning(f"Anomaly model not found at {IFOREST_MODEL_PATH}")


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup/shutdown lifecycle."""
    load_model()
    load_anomaly_model()
    yield


app = FastAPI(
    title="Expense Forecasting API",
    description="LightGBM + Isolation Forest prediction service cho du bao chi tieu",
    version="3.0.0",
    lifespan=lifespan,
)


@app.post("/reload")
async def reload_models():
    """Reload toan bo model tu file. Goi sau khi MLOps cap nhat model."""
    load_model()
    load_anomaly_model()
    return {"status": "reloaded"}


# ================= Request/Response Models =================


class CategoryPredictRequest(BaseModel):
    """Request body cho du doan chi tieu 1 category."""
    category_id: int
    month: int
    year: int
    monthly_spending: float
    transaction_count: int
    avg_transaction: float
    max_transaction: float
    avg_day_of_week: float
    avg_day_of_month: float
    total_all_categories: float
    category_ratio: float
    prev_month_spending: float
    prev_month_count: int
    prev_month_ratio: float
    avg_monthly_spending_3m: float
    spending_trend: float


class CategoryPredictResponse(BaseModel):
    """Response body cho du doan chi tieu."""
    category_id: int
    predicted_spending: float
    current_spending: float
    trend: str
    change_percent: float


class BulkPredictRequest(BaseModel):
    """Request body cho du doan nhieu categories cung luc."""
    predictions: List[CategoryPredictRequest]


class TrendAnalysisRequest(BaseModel):
    """Request body cho phan tich xu huong so dong."""
    category_id: int
    monthly_averages: List[float]
    user_current_spending: float


class TrendAnalysisResponse(BaseModel):
    """Response body cho phan tich xu huong."""
    category_id: int
    population_average: float
    user_spending: float
    deviation_percent: float
    status: str
    message: str


class AnomalyTransactionRequest(BaseModel):
    """Request body cho 1 giao dich can check bat thuong."""
    transaction_id: int
    amount: float
    category_id: int
    day_of_week: int
    day_of_month: int
    amount_vs_category_avg: float


class AnomalyCheckRequest(BaseModel):
    """Request body cho nhieu giao dich can check."""
    transactions: List[AnomalyTransactionRequest]


class AnomalyResult(BaseModel):
    """Ket qua cho 1 giao dich."""
    transaction_id: int
    is_anomaly: bool
    anomaly_score: float
    message: str


# ================= API Endpoints =================


@app.get("/health")
async def health_check():
    """Kiem tra trang thai server va cac model."""
    return {
        "status": "ok",
        "lgbm_model_loaded": model_bundle is not None,
        "anomaly_model_loaded": anomaly_bundle is not None,
    }


@app.post("/predict/category", response_model=CategoryPredictResponse)
async def predict_category(request: CategoryPredictRequest):
    """Du doan chi tieu thang tiep theo cho 1 category."""
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model chua duoc load")

    model = model_bundle["model"]

    features = np.array([[
        request.category_id,
        request.month,
        request.year,
        request.monthly_spending,
        request.transaction_count,
        request.avg_transaction,
        request.max_transaction,
        request.avg_day_of_week,
        request.avg_day_of_month,
        request.total_all_categories,
        request.category_ratio,
        request.prev_month_spending,
        request.prev_month_count,
        request.prev_month_ratio,
        request.avg_monthly_spending_3m,
        request.spending_trend,
    ]])

    predicted = max(0.0, float(model.predict(features)[0]))

    # Tinh xu huong so voi thang truoc da ket thuc
    reference = request.monthly_spending
    if reference > 0:
        change_pct = ((predicted - reference) / reference) * 100
    elif request.avg_monthly_spending_3m > 0:
        change_pct = (
            (predicted - request.avg_monthly_spending_3m)
            / request.avg_monthly_spending_3m
            * 100
        )
    else:
        change_pct = 0.0

    if change_pct > 10:
        trend = "increasing"
    elif change_pct < -10:
        trend = "decreasing"
    else:
        trend = "stable"

    return CategoryPredictResponse(
        category_id=request.category_id,
        predicted_spending=round(predicted, 0),
        current_spending=reference,
        trend=trend,
        change_percent=round(change_pct, 1),
    )


@app.post("/predict/bulk", response_model=List[CategoryPredictResponse])
async def predict_bulk(request: BulkPredictRequest):
    """Du doan chi tieu cho nhieu categories cung luc."""
    results = []
    for pred_req in request.predictions:
        result = await predict_category(pred_req)
        results.append(result)
    return results


@app.post("/predict/trend", response_model=TrendAnalysisResponse)
async def analyze_trend(request: TrendAnalysisRequest):
    """Phan tich xu huong chi tieu cua user so voi so dong."""
    if not request.monthly_averages:
        raise HTTPException(
            status_code=400, detail="Can it nhat 1 gia tri trung binh"
        )

    pop_avg = float(np.mean(request.monthly_averages))
    user_spending = request.user_current_spending

    if pop_avg > 0:
        deviation = ((user_spending - pop_avg) / pop_avg) * 100
    else:
        deviation = 0.0

    if deviation < -20:
        status = "below_average"
        message = (
            f"Ban dang chi tieu it hon {abs(deviation):.0f}% so voi muc trung binh. "
            f"Rat tiet kiem!"
        )
    elif deviation <= 20:
        status = "average"
        message = f"Chi tieu cua ban nam trong muc trung binh ({deviation:+.0f}%)."
    elif deviation <= 50:
        status = "above_average"
        message = (
            f"Ban dang chi tieu nhieu hon {deviation:.0f}% so voi muc trung binh. "
            f"Can nhac dieu chinh."
        )
    else:
        status = "warning"
        message = f"Canh bao: Chi tieu vuot {deviation:.0f}% so voi xu huong so dong!"

    return TrendAnalysisResponse(
        category_id=request.category_id,
        population_average=round(pop_avg, 0),
        user_spending=user_spending,
        deviation_percent=round(deviation, 1),
        status=status,
        message=message,
    )


@app.post("/predict/anomaly", response_model=List[AnomalyResult])
async def check_anomalies(request: AnomalyCheckRequest):
    """Phat hien giao dich bat thuong bang Isolation Forest."""
    if anomaly_bundle is None:
        raise HTTPException(
            status_code=503, detail="Anomaly model chua duoc load"
        )

    model = anomaly_bundle["model"]
    results = []

    for tx in request.transactions:
        features = np.array([[
            tx.amount,
            tx.category_id,
            tx.day_of_week,
            tx.day_of_month,
            tx.amount_vs_category_avg,
        ]])

        prediction = model.predict(features)[0]
        score = float(model.decision_function(features)[0])
        is_anomaly = prediction == -1

        if is_anomaly:
            if tx.amount_vs_category_avg > 3:
                message = (
                    f"Giao dich {tx.amount:,.0f}d cao gap "
                    f"{tx.amount_vs_category_avg:.1f}x so voi muc binh thuong "
                    f"cua danh muc nay!"
                )
            elif tx.amount_vs_category_avg > 2:
                message = (
                    f"Giao dich {tx.amount:,.0f}d cao hon dang ke "
                    f"so voi thoi quen chi tieu cua ban."
                )
            else:
                message = (
                    f"Giao dich {tx.amount:,.0f}d co dau hieu bat thuong "
                    f"ve thoi diem hoac muc chi."
                )
        else:
            message = "Giao dich binh thuong."

        results.append(AnomalyResult(
            transaction_id=tx.transaction_id,
            is_anomaly=is_anomaly,
            anomaly_score=round(score, 4),
            message=message,
        ))

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=PREDICT_SERVER_HOST, port=PREDICT_SERVER_PORT)
