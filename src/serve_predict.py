"""
predict_server.py — FastAPI Prediction Service cho LightGBM + LSTM + Isolation Forest.

Chạy như một microservice bên cạnh Spring Boot.
Spring Boot sẽ gọi API này để lấy kết quả dự đoán xu hướng chi tiêu.

Endpoints:
  POST /predict/category   → Dự đoán chi tiêu tháng tới theo từng danh mục
  POST /predict/bulk        → Dự đoán nhiều categories cùng lúc
  POST /predict/trend       → Phân tích xu hướng chi tiêu số đông
  POST /predict/anomaly     → Phát hiện giao dịch bất thường (Isolation Forest)
  POST /predict/weekly      → Dự đoán chi tiêu tuần tới (LSTM)
  GET  /health              → Kiểm tra trạng thái server
"""

import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(
    title="Expense Forecasting API",
    description="LightGBM + LSTM + Isolation Forest prediction service cho dự báo chi tiêu",
    version="3.0.0"
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "category_forecast_lgbm.joblib")
ANOMALY_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "anomaly_iforest.joblib")
LSTM_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "expense_model.h5")
LSTM_META_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "meta_lstm.joblib")
model_bundle = None
anomaly_bundle = None
lstm_model = None
lstm_meta = None


def load_model():
    """Load model LightGBM từ file."""
    global model_bundle
    if os.path.exists(MODEL_PATH):
        model_bundle = joblib.load(MODEL_PATH)
        print(f"✅ Loaded LightGBM model from {MODEL_PATH}")
    else:
        print(f"⚠️ Model not found at {MODEL_PATH}")


def load_anomaly_model():
    """Load model Isolation Forest từ file."""
    global anomaly_bundle
    if os.path.exists(ANOMALY_MODEL_PATH):
        anomaly_bundle = joblib.load(ANOMALY_MODEL_PATH)
        print(f"✅ Loaded Isolation Forest model from {ANOMALY_MODEL_PATH}")
    else:
        print(f"⚠️ Anomaly model not found at {ANOMALY_MODEL_PATH}")


def load_lstm_model():
    """Load model LSTM (Keras .h5) từ file."""
    global lstm_model, lstm_meta
    if os.path.exists(LSTM_MODEL_PATH):
        import tensorflow as tf
        # compile=False vì server chỉ cần predict, không cần optimizer/loss
        lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH, compile=False)
        print(f"✅ Loaded LSTM model from {LSTM_MODEL_PATH}")
    else:
        print(f"⚠️ LSTM model not found at {LSTM_MODEL_PATH}")

    if os.path.exists(LSTM_META_PATH):
        lstm_meta = joblib.load(LSTM_META_PATH)
        print(f"✅ Loaded LSTM meta: {lstm_meta}")
    else:
        print(f"⚠️ LSTM meta not found at {LSTM_META_PATH}")


# ================= Request/Response Models =================

class CategoryPredictRequest(BaseModel):
    """Request body cho dự đoán chi tiêu 1 category (không có user_id)."""
    category_id: int
    month: int
    year: int
    monthly_spending: float       # Tổng chi tiêu tháng trước cho category này
    transaction_count: int        # Số giao dịch trong tháng trước
    avg_transaction: float        # Trung bình mỗi giao dịch
    max_transaction: float        # Giao dịch lớn nhất
    avg_day_of_week: float        # Trung bình ngày trong tuần (0-6)
    avg_day_of_month: float       # Trung bình ngày trong tháng
    total_all_categories: float   # Tổng chi tiêu tất cả categories
    category_ratio: float         # Tỉ lệ category/tổng
    prev_month_spending: float    # Chi tiêu tháng trước nữa cho category này
    prev_month_count: int         # Số giao dịch tháng trước nữa
    prev_month_ratio: float       # Tỉ lệ tháng trước nữa
    avg_monthly_spending_3m: float  # TB chi tiêu 3 tháng gần nhất
    spending_trend: float          # Xu hướng: last_month / avg_3m


class CategoryPredictResponse(BaseModel):
    """Response body cho dự đoán chi tiêu."""
    category_id: int
    predicted_spending: float
    current_spending: float
    trend: str  # "increasing", "decreasing", "stable"
    change_percent: float


class BulkPredictRequest(BaseModel):
    """Request body cho dự đoán nhiều categories cùng lúc."""
    predictions: List[CategoryPredictRequest]


class TrendAnalysisRequest(BaseModel):
    """Request body cho phân tích xu hướng số đông."""
    category_id: int
    monthly_averages: List[float]
    user_current_spending: float


class TrendAnalysisResponse(BaseModel):
    """Response body cho phân tích xu hướng."""
    category_id: int
    population_average: float
    user_spending: float
    deviation_percent: float
    status: str
    message: str


class AnomalyTransactionRequest(BaseModel):
    """Request body cho 1 giao dịch cần check bất thường."""
    transaction_id: int
    amount: float
    category_id: int
    day_of_week: int
    day_of_month: int
    amount_vs_category_avg: float  # amount / TB category


class AnomalyCheckRequest(BaseModel):
    """Request body cho nhiều giao dịch cần check."""
    transactions: List[AnomalyTransactionRequest]


class AnomalyResult(BaseModel):
    """Kết quả cho 1 giao dịch."""
    transaction_id: int
    is_anomaly: bool
    anomaly_score: float
    message: str


class WeeklyPredictRequest(BaseModel):
    """Request body cho dự đoán chi tiêu tuần tới (LSTM)."""
    weekly_spending: List[float]  # 4 tuần chi tiêu gần nhất (VNĐ)


class WeeklyPredictResponse(BaseModel):
    """Response body cho dự đoán tuần."""
    predicted_spending: float
    input_weeks: List[float]
    trend: str  # "increasing", "decreasing", "stable"
    change_percent: float


# ================= API Endpoints =================

@app.on_event("startup")
async def startup():
    load_model()
    load_anomaly_model()
    load_lstm_model()


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "lgbm_model_loaded": model_bundle is not None,
        "anomaly_model_loaded": anomaly_bundle is not None,
        "lstm_model_loaded": lstm_model is not None,
    }


@app.post("/predict/category", response_model=CategoryPredictResponse)
async def predict_category(request: CategoryPredictRequest):
    """Dự đoán chi tiêu tháng tiếp theo cho 1 category."""
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model chưa được load")

    model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]

    # Tạo feature vector theo đúng thứ tự — KHÔNG có user_id
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

    predicted = float(model.predict(features)[0])
    predicted = max(0, predicted)  # Không cho số âm

    # FIX: Tính xu hướng so với THÁNG TRƯỚC ĐÃ KẾT THÚC (monthly_spending)
    # thay vì so sánh với tháng đang diễn ra
    reference = request.monthly_spending  # Đây giờ là chi tiêu tháng trước đã hoàn tất
    if reference > 0:
        change_pct = ((predicted - reference) / reference) * 100
    else:
        # Nếu tháng trước không chi, so với avg_3m
        if request.avg_monthly_spending_3m > 0:
            change_pct = ((predicted - request.avg_monthly_spending_3m) / request.avg_monthly_spending_3m) * 100
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
        change_percent=round(change_pct, 1)
    )


@app.post("/predict/bulk", response_model=List[CategoryPredictResponse])
async def predict_bulk(request: BulkPredictRequest):
    """Dự đoán chi tiêu cho nhiều categories cùng lúc."""
    results = []
    for pred_req in request.predictions:
        result = await predict_category(pred_req)
        results.append(result)
    return results


@app.post("/predict/trend", response_model=TrendAnalysisResponse)
async def analyze_trend(request: TrendAnalysisRequest):
    """Phân tích xu hướng chi tiêu của user so với số đông."""
    if not request.monthly_averages:
        raise HTTPException(status_code=400, detail="Cần ít nhất 1 giá trị trung bình")

    pop_avg = np.mean(request.monthly_averages)
    user_spending = request.user_current_spending

    if pop_avg > 0:
        deviation = ((user_spending - pop_avg) / pop_avg) * 100
    else:
        deviation = 0.0

    # Phân loại
    if deviation < -20:
        status = "below_average"
        message = f"Bạn đang chi tiêu ít hơn {abs(deviation):.0f}% so với mức trung bình. Rất tiết kiệm!"
    elif deviation <= 20:
        status = "average"
        message = f"Chi tiêu của bạn nằm trong mức trung bình ({deviation:+.0f}%)."
    elif deviation <= 50:
        status = "above_average"
        message = f"Bạn đang chi tiêu nhiều hơn {deviation:.0f}% so với mức trung bình. Cân nhắc điều chỉnh."
    else:
        status = "warning"
        message = f"⚠️ Cảnh báo: Chi tiêu vượt {deviation:.0f}% so với xu hướng số đông!"

    return TrendAnalysisResponse(
        category_id=request.category_id,
        population_average=round(pop_avg, 0),
        user_spending=user_spending,
        deviation_percent=round(deviation, 1),
        status=status,
        message=message
    )


@app.post("/predict/anomaly", response_model=List[AnomalyResult])
async def check_anomalies(request: AnomalyCheckRequest):
    """Phát hiện giao dịch bất thường bằng Isolation Forest."""
    if anomaly_bundle is None:
        raise HTTPException(status_code=503, detail="Anomaly model chưa được load")

    model = anomaly_bundle["model"]
    feature_cols = anomaly_bundle["feature_cols"]

    results = []
    for tx in request.transactions:
        features = np.array([[
            tx.amount,
            tx.category_id,
            tx.day_of_week,
            tx.day_of_month,
            tx.amount_vs_category_avg,
        ]])

        # Isolation Forest: -1 = anomaly, 1 = normal
        prediction = model.predict(features)[0]
        score = model.decision_function(features)[0]

        is_anomaly = prediction == -1

        if is_anomaly:
            if tx.amount_vs_category_avg > 3:
                message = f"Giao dịch {tx.amount:,.0f}₫ cao gấp {tx.amount_vs_category_avg:.1f}x so với mức bình thường của danh mục này!"
            elif tx.amount_vs_category_avg > 2:
                message = f"Giao dịch {tx.amount:,.0f}₫ cao hơn đáng kể so với thói quen chi tiêu của bạn."
            else:
                message = f"Giao dịch {tx.amount:,.0f}₫ có dấu hiệu bất thường về thời điểm hoặc mức chi."
        else:
            message = "Giao dịch bình thường."

        results.append(AnomalyResult(
            transaction_id=tx.transaction_id,
            is_anomaly=is_anomaly,
            anomaly_score=round(float(score), 4),
            message=message
        ))

    return results


@app.post("/predict/weekly", response_model=WeeklyPredictResponse)
async def predict_weekly(request: WeeklyPredictRequest):
    """Dự đoán chi tiêu tuần tới dựa trên LSTM model."""
    if lstm_model is None:
        raise HTTPException(status_code=503, detail="LSTM model chưa được load")

    weeks = request.weekly_spending
    if len(weeks) != 4:
        raise HTTPException(status_code=400, detail="Cần chính xác 4 tuần dữ liệu")

    # Normalize: dùng max(weeks) làm scale (giống logic Android)
    max_val = max(weeks) if max(weeks) > 0 else 1.0
    scaled = [min(w / max_val, 3.0) for w in weeks]  # clip ở 3 như preprocess.py

    # Reshape cho LSTM: (1, 4, 1)
    input_data = np.array(scaled).reshape(1, 4, 1).astype(np.float32)
    prediction_scaled = float(lstm_model.predict(input_data, verbose=0)[0][0])

    # Denormalize
    predicted = max(prediction_scaled * max_val, 0)

    # Trend so với tuần gần nhất
    last_week = weeks[-1]
    if last_week > 0:
        change_pct = ((predicted - last_week) / last_week) * 100
    else:
        avg = sum(weeks) / len(weeks) if sum(weeks) > 0 else 0
        change_pct = ((predicted - avg) / avg * 100) if avg > 0 else 0.0

    if change_pct > 10:
        trend = "increasing"
    elif change_pct < -10:
        trend = "decreasing"
    else:
        trend = "stable"

    return WeeklyPredictResponse(
        predicted_spending=round(predicted, 0),
        input_weeks=weeks,
        trend=trend,
        change_percent=round(change_pct, 1)
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
