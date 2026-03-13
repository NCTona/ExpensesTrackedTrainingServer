# 🤖 Expenses Tracked Training Server (MLOps)

MLOps pipeline phục vụ hệ thống quản lý chi tiêu — train và serve 3 model AI:

| Model | Mục đích | Output |
|---|---|---|
| **LSTM** | Dự đoán chi tiêu tuần tới | `.tflite` (on-device) + `.h5` (server) |
| **LightGBM** | Dự đoán chi tiêu theo danh mục (tháng) | `.joblib` |
| **Isolation Forest** | Phát hiện giao dịch bất thường | `.joblib` |

## 📋 Yêu cầu

- **Python** 3.10+
- **Git** (đã cài)
- **DVC** (quản lý pipeline ML)
- **Prefect** (orchestrate flow — tùy chọn)

## 🚀 Cài đặt

### 1. Clone repo

```bash
git clone https://github.com/NCTona/ExpensesTrackedTrainingServer.git
cd ExpensesTrackedTrainingServer
```

### 2. Tạo virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

### 3. Cài dependencies

```bash
pip install -r requirement.txt
```

### 4. Tạo file `.env`

```bash
echo PYTHONPATH=. > .env
```

### 5. Chuẩn bị dữ liệu

Tạo thư mục data nếu chưa có:

```bash
mkdir -p data/raw
```

Đặt file `transactions.csv` vào `data/raw/`. File phải chứa các cột:
- `user_id`, `category_id`, `amount`, `transaction_date`, `type`

> 💡 Nếu chưa có dữ liệu, script train sẽ tự tạo demo data.

## 🏋️ Train models

### Chạy qua DVC (khuyến nghị)

DVC tự động phát hiện thay đổi và chỉ train lại model cần thiết:

```bash
dvc repro
```

Pipeline sẽ chạy theo thứ tự:
1. `preprocess` → Tiền xử lý dữ liệu
2. `train_lstm` → Train LSTM model
3. `train_lgbm` → Train LightGBM model
4. `train_iforest` → Train Isolation Forest model

### Chạy từng script riêng lẻ

```bash
python src/preprocess.py
python src/train_lstm.py
python src/train_lgbm.py
python src/train_iforest.py
```

### Output sau khi train

```
models/
├── expense_model.tflite      # LSTM cho Android (on-device)
├── expense_model.h5          # LSTM cho FastAPI (server-side)
├── meta_lstm.joblib           # Metadata LSTM
├── category_forecast_lgbm.joblib  # LightGBM
└── anomaly_iforest.joblib     # Isolation Forest
```

## 🌐 Chạy Prediction Server

```bash
python src/serve_predict.py
```

Server sẽ chạy tại `http://localhost:8001` với các endpoint:

| Method | Endpoint | Mô tả |
|---|---|---|
| `GET` | `/health` | Kiểm tra trạng thái server |
| `POST` | `/predict/category` | Dự đoán 1 category |
| `POST` | `/predict/bulk` | Dự đoán nhiều categories |
| `POST` | `/predict/trend` | Phân tích xu hướng số đông |
| `POST` | `/predict/anomaly` | Phát hiện bất thường |
| `POST` | `/predict/weekly` | Dự đoán chi tiêu tuần tới (LSTM) |

> 📖 API docs tự động tại `http://localhost:8001/docs`

## 🔄 Chạy MLOps Flow (Prefect)

Orchestrate toàn bộ pipeline (ingest data → train → deploy model):

```bash
python flows/main_flow.py
```

Flow bao gồm:
1. **Ingest** — Lấy dữ liệu giao dịch từ Spring Boot API
2. **DVC Repro** — Train lại model nếu data thay đổi
3. **Deploy** — Upload model TFLite lên Spring Boot server

## 📊 MLflow Tracking

Xem kết quả thí nghiệm:

```bash
mlflow ui
```

Truy cập `http://localhost:5000` để xem metrics, parameters, và artifacts.

## 🏗️ Cấu trúc thư mục

```
.
├── src/
│   ├── preprocess.py        # Tiền xử lý dữ liệu
│   ├── train_lstm.py        # Train LSTM
│   ├── train_lgbm.py        # Train LightGBM
│   ├── train_iforest.py     # Train Isolation Forest
│   ├── ingest.py            # Lấy data từ API
│   └── serve_predict.py     # FastAPI server
├── flows/
│   └── main_flow.py         # Prefect orchestration
├── models/                  # Model outputs (gitignored, DVC tracked)
├── data/
│   ├── raw/                 # Dữ liệu gốc
│   └── processed/           # Dữ liệu đã xử lý
├── dvc.yaml                 # Pipeline definition
├── requirement.txt          # Python dependencies
└── .env                     # Environment variables
```

## ⚙️ Kết nối với hệ thống

Server này hoạt động cùng:
- **Spring Boot Backend** — gọi `/predict/*` endpoints để phục vụ app
- **Android App** — sử dụng model TFLite trực tiếp trên device

Cấu hình URL trong Spring Boot `application.properties`:

```properties
mlops.predict-server-url=http://localhost:8001
```
