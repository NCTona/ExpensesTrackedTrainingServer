from prefect import task, flow
import subprocess
import os
import sys

# Ensure src path is available
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.ingest import fetch_data

@task(name="Ingest Data from Spring Boot")
def task_ingest_data():
    """Gọi HTTP request về Spring Boot để cập nhật data/raw/transactions.csv"""
    fetch_data()
    return True

@task(name="Run DVC Pipeline (Preprocess & Train Models)")
def task_run_dvc():
    """
    Chạy DVC repro để xem data có thay đổi không.
    Nếu data mới, DVC sẽ tự động train lại model LSTM và LightGBM (theo file dvc.yaml).
    """
    print("Triggering DVC reproduction...")
    try:
        # Cần chỉ định working directory là gốc repo MLOps
        cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        result = subprocess.run(
            ["dvc", "repro"], cwd=cwd, check=True, 
            capture_output=True, text=True, encoding="utf-8"
        )
        print("DVC Repro Output:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print("DVC Repro Failed:\n", e.stderr)
        raise e

@task(name="Deploy Latest Model to Spring Boot")
def task_deploy_model():
    """
    Kéo model tốt nhất từ MLflow hoặc file dvc output và push ngược lại cho Backend
    """
    import requests
    model_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), "models", "expense_model.tflite")
    
    # Endpoint trên Spring Boot (dùng HTTPS vì server bật SSL/TLS)
    SPRING_BOOT_API_URL = "https://localhost:8080/api/internal/model/update"
    
    if os.path.exists(model_path):
        print(f"Bắt đầu upload model từ {model_path} lên {SPRING_BOOT_API_URL}...")
        try:
            with open(model_path, "rb") as f:
                # Gửi model dưới dạng file đính kèm multipart
                files = {"model_file": ("expense_model.tflite", f, "application/octet-stream")}
                headers = {"X-API-KEY": "secret_mlops_key"}
                
                response = requests.post(SPRING_BOOT_API_URL, files=files, headers=headers, timeout=30, verify=False)
                
            response.raise_for_status()
            print("🚀 Upload model thành công! Backend đã nhận model mới.")
            # Nhanh chóng in log phản hồi từ Java Backend
            print("Response:", response.json() if response.text else "OK")
        except requests.exceptions.RequestException as e:
            print(f"❌ Upload model thất bại! Lỗi kết nối hoặc API phản hồi lỗi: {e}")
            if e.response is not None:
                print(f"Chi tiết response lỗi: {e.response.text}")
    else:
        print("❌ Không tìm thấy model mới được tạo ra ở thư mục models/")

@flow(name="MLOps Expense Forecasting Pipeline")
def mlops_pipeline():
    # 1. Tải dữ liệu từ backend
    task_ingest_data()
    
    # 2. Xử lý & Train (dvc repro quản lý state cả LSTM + LightGBM)
    task_run_dvc()
    
    # 3. Đẩy model LSTM (.tflite) lên backend cho Android app
    task_deploy_model()

if __name__ == "__main__":
    import sys
    # Nếu truyền cờ --run-once thì chỉ chạy 1 lần
    if len(sys.argv) > 1 and sys.argv[1] == "--run-once":
        print("Chạy pipeline thủ công 1 lần...")
        mlops_pipeline()
    else:
        # Nếu không truyền gì, lên lịch chạy theo dạng Server định kỳ
        print("Đang khởi động MLOps Server tuần hoàn...")
        mlops_pipeline.serve(
            name="expense-forecasting-weekly",
            cron="0 2 * * 0", # Mặc định chạy vào lúc 2:00 sáng Chủ Nhật hàng tuần
            tags=["mlops", "training"],
            description="Pipeline định kỳ kéo dữ liệu mới và huấn luyện mô hình dự báo chi tiêu."
        )
