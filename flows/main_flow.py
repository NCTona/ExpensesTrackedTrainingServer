"""
main_flow.py — Prefect MLOps Pipeline cho Expense Forecasting.

Flow chính:
  1. Ingest data từ Spring Boot
  2. DVC repro (preprocess + train models)
  3. Evaluate metrics — so sánh với best model
  4. Deploy nếu model mới tốt hơn

Chạy:
  python flows/main_flow.py              → Chạy scheduled (CN 2:00 sáng)
  python flows/main_flow.py --run-once   → Chạy 1 lần thủ công
"""

from prefect import task, flow, get_run_logger
import subprocess
import os
import sys
import json
import socket
import time

# Ensure src path is available
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.ingest import fetch_data

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BEST_METRICS_PATH = os.path.join(BASE_DIR, "models", "best_metrics.json")


def load_best_metrics():
    """Đọc metrics tốt nhất hiện tại từ file."""
    if os.path.exists(BEST_METRICS_PATH):
        with open(BEST_METRICS_PATH, "r") as f:
            return json.load(f)
    return {}


def save_best_metrics(metrics):
    """Lưu metrics tốt nhất."""
    os.makedirs(os.path.dirname(BEST_METRICS_PATH), exist_ok=True)
    with open(BEST_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)


@task(name="Ingest Data from Spring Boot", retries=3, retry_delay_seconds=30)
def task_ingest_data():
    """Gọi HTTP request về Spring Boot để cập nhật data/raw/transactions.csv"""
    logger = get_run_logger()
    logger.info("Bắt đầu kéo dữ liệu từ Spring Boot...")
    fetch_data()
    logger.info("Ingest data hoan tat.")
    return True


@task(name="Run DVC Pipeline (Preprocess & Train Models)", retries=2, retry_delay_seconds=60)
def task_run_dvc():
    """
    Chạy DVC repro để xem data có thay đổi không.
    Nếu data mới, DVC sẽ tự động train lại model (theo dvc.yaml).
    """
    logger = get_run_logger()
    logger.info("Triggering DVC reproduction...")
    try:
        result = subprocess.run(
            ["dvc", "repro"], cwd=BASE_DIR, check=True,
            capture_output=True, text=True, encoding="utf-8"
        )
        logger.info(f"DVC Repro Output:\n{result.stdout}")

        # Kiểm tra xem có stage nào thực sự chạy không
        if "didn't change" in result.stdout.lower() or "stage" not in result.stdout.lower():
            logger.info("Khong co thay doi du lieu, model giu nguyen.")
            return False  # Không có training mới
        return True  # Có training mới
    except subprocess.CalledProcessError as e:
        logger.error(f"DVC Repro Failed:\n{e.stderr}")
        raise e


@task(name="Evaluate Model Metrics")
def task_evaluate_metrics():
    """
    Đọc metrics từ MLflow run mới nhất và so sánh với best_metrics.json.
    Trả về dict chứa kết quả đánh giá.
    """
    logger = get_run_logger()

    try:
        import mlflow

        best = load_best_metrics()
        evaluation = {"should_deploy": False, "details": {}}

        # === Đánh giá LSTM ===
        mlflow.set_experiment("Expense Forecasting - LSTM")
        runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)
        if not runs.empty:
            new_rmse = runs.iloc[0].get("metrics.rmse", None)
            new_mae = runs.iloc[0].get("metrics.mae", None)
            run_id = runs.iloc[0]["run_id"]

            old_rmse = best.get("lstm", {}).get("rmse", float("inf"))

            if new_rmse is not None:
                improvement = ((old_rmse - new_rmse) / old_rmse * 100) if old_rmse != float("inf") else 100
                evaluation["details"]["lstm"] = {
                    "new_rmse": round(new_rmse, 4),
                    "old_rmse": round(old_rmse, 4) if old_rmse != float("inf") else "N/A",
                    "improvement_pct": round(improvement, 1),
                    "run_id": run_id
                }

                # Deploy nếu model mới tốt hơn hoặc chưa có best
                if new_rmse < old_rmse or old_rmse == float("inf"):
                    evaluation["should_deploy"] = True
                    best["lstm"] = {
                        "rmse": round(new_rmse, 4),
                        "mae": round(new_mae, 4) if new_mae else 0,
                        "run_id": run_id
                    }
                    logger.info(f"LSTM cai thien: RMSE {old_rmse} -> {new_rmse} ({improvement:+.1f}%)")
                else:
                    logger.info(f"LSTM khong cai thien: RMSE {new_rmse} >= {old_rmse}")

        # === Đánh giá LightGBM ===
        mlflow.set_experiment("Expense Forecasting - LightGBM")
        runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)
        if not runs.empty:
            new_rmse = runs.iloc[0].get("metrics.rmse", None)
            run_id = runs.iloc[0]["run_id"]
            old_rmse = best.get("lgbm", {}).get("rmse", float("inf"))

            if new_rmse is not None and (new_rmse < old_rmse or old_rmse == float("inf")):
                best["lgbm"] = {
                    "rmse": round(new_rmse, 4),
                    "run_id": run_id
                }
                logger.info(f"LightGBM cai thien: RMSE {old_rmse} -> {new_rmse}")

        # Lưu metrics mới
        if evaluation["should_deploy"]:
            save_best_metrics(best)
            logger.info(f"Da cap nhat best_metrics.json")

        return evaluation

    except Exception as e:
        logger.warning(f"Khong the doc MLflow metrics: {e}. Deploy mac dinh.")
        return {"should_deploy": True, "details": {"error": str(e)}}


@task(name="Deploy Model to Spring Boot", retries=2, retry_delay_seconds=30)
def task_deploy_model():
    """Upload model .tflite mới nhất lên Spring Boot Backend."""
    import requests
    logger = get_run_logger()

    model_path = os.path.join(BASE_DIR, "models", "expense_model.tflite")
    SPRING_BOOT_API_URL = "https://localhost:8080/api/internal/model/update"

    if not os.path.exists(model_path):
        logger.error("Khong tim thay models/expense_model.tflite")
        return False

    logger.info(f"Uploading model tu {model_path}...")
    try:
        with open(model_path, "rb") as f:
            files = {"model_file": ("expense_model.tflite", f, "application/octet-stream")}
            headers = {"X-API-KEY": os.getenv("MLOPS_API_KEY", "secret_mlops_key")}

            response = requests.post(
                SPRING_BOOT_API_URL, files=files, headers=headers,
                timeout=30, verify=False
            )
        response.raise_for_status()
        logger.info("Upload model thanh cong!")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Upload model that bai: {e}")
        raise e


@flow(name="MLOps Expense Forecasting Pipeline")
def mlops_pipeline():
    logger = get_run_logger()
    logger.info("=" * 50)
    logger.info("Bat dau MLOps Pipeline")
    logger.info("=" * 50)

    # 1. Tải dữ liệu từ backend
    task_ingest_data()

    # 2. Xử lý & Train (DVC quản lý LSTM + LightGBM + IForest)
    has_new_training = task_run_dvc()

    if not has_new_training:
        logger.info("Khong co training moi, pipeline ket thuc.")
        return

    # 3. Đánh giá metrics — so sánh model mới vs cũ
    evaluation = task_evaluate_metrics()

    # 4. Deploy CHỈ KHI model mới tốt hơn
    if evaluation.get("should_deploy", False):
        logger.info("Model moi tot hon -> Deploying...")
        task_deploy_model()
    else:
        logger.info("Model moi khong tot hon -> Bo qua deploy.")

    logger.info("Pipeline hoan tat!")


if __name__ == "__main__":
    import sys
    
    # ---------------------------------------------------------
    # TỰ ĐỘNG BẬT PREFECT SERVER NẾU CHƯA CHẠY
    # ---------------------------------------------------------
    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('127.0.0.1', port)) == 0

    PREFECT_PORT = 4200
    if not is_port_in_use(PREFECT_PORT):
        print(f"[*] Prefect Server chua chay tren cong {PREFECT_PORT}. Dang khoi dong tu dong...")
        # Mở một cửa sổ cmd mới chạy ngầm server (Windows specific)
        subprocess.Popen(
            'start "Prefect Server" cmd /c "prefect server start"',
            shell=True
        )
        
        # Chờ server bật lên (timeout 15 giây)
        print("[*] Vui long doi Server khoi dong...")
        for _ in range(15):
            if is_port_in_use(PREFECT_PORT):
                print(f"[+] Prefect Server da san sang tren cong {PREFECT_PORT}!")
                break
            time.sleep(1)
            
    # Đảm bảo Prefect cấu hình vào Server (Bằng lệnh CLI ưu tiên hơn là os.environ ở Prefect v3)
    os.environ["PREFECT_API_URL"] = f"http://127.0.0.1:{PREFECT_PORT}/api"
    subprocess.run(["prefect", "config", "set", f"PREFECT_API_URL=http://127.0.0.1:{PREFECT_PORT}/api"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # ---------------------------------------------------------

    if len(sys.argv) > 1 and sys.argv[1] == "--run-once":
        print("Chạy pipeline thủ công 1 lần...")
        mlops_pipeline()
    else:
        print("Đang khởi động MLOps Server tuần hoàn...")
        mlops_pipeline.serve(
            name="expense-forecasting-weekly",
            cron="0 2 * * 0",  # 2:00 sáng Chủ Nhật hàng tuần
            tags=["mlops", "training"],
            description="Pipeline định kỳ kéo dữ liệu mới và huấn luyện mô hình dự báo chi tiêu."
        )
