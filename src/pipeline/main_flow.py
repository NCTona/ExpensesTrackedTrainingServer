# -*- coding: utf-8 -*-
"""
main_flow.py — Prefect MLOps Pipeline cho Expense Forecasting.

Flow chinh:
  1. Ingest data tu Spring Boot
  2. DVC repro (preprocess + train models)
  3. Evaluate metrics — so sanh voi best model, rollback neu te hon
  4. Deploy .tflite neu LSTM model moi tot hon
  5. Reload FastAPI predict server

Chay:
  python -m src.pipeline.main_flow --run-once   -> Chay 1 lan thu cong
  python -m src.pipeline.main_flow              -> Chay scheduled (CN 2:00 sang)
"""

import json
import logging
import os
import socket
import subprocess
import sys
import time
from typing import Any, Dict

from prefect import flow, get_run_logger, task

# Ensure src path is available
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import shutil

from src.config import (
    BASE_DIR,
    BEST_METRICS_PATH,
    LSTM_TFLITE_PATH,
    LSTM_TFLITE_FILE,
    LGBM_MODEL_PATH,
    LGBM_MODEL_FILE,
    BACKEND_DEPLOY_URL,
    DEFAULT_API_KEY,
    API_KEY_HEADER,
    PREFECT_PORT,
    PIPELINE_CRON,
    MLFLOW_LSTM_EXPERIMENT,
    MLFLOW_LGBM_EXPERIMENT,
    MLFLOW_ANOMALY_EXPERIMENT,
    PREDICT_SERVER_RELOAD_URL,
)
from src.data.ingest import fetch_data

logger = logging.getLogger(__name__)


def load_best_metrics() -> Dict[str, Any]:
    """Doc best metrics tu file JSON."""
    if os.path.exists(BEST_METRICS_PATH):
        with open(BEST_METRICS_PATH, "r") as f:
            return json.load(f)
    return {}


def save_best_metrics(metrics: Dict[str, Any]) -> None:
    """Luu best metrics."""
    os.makedirs(os.path.dirname(BEST_METRICS_PATH), exist_ok=True)
    with open(BEST_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)


def rollback_model_from_mlflow(
    experiment_name: str,
    run_id: str,
    artifact_filename: str,
    target_path: str,
    task_logger,
) -> bool:
    """
    Rollback model: lay model cu tu mlruns/ (theo run_id) ghi de lai models/.
    Tra ve True neu rollback thanh cong.
    """
    import mlflow

    try:
        mlflow.set_experiment(experiment_name)
        artifact_uri = mlflow.get_run(run_id).info.artifact_uri
        # artifact_uri co dang: file:///path/to/mlruns/<exp>/<run>/artifacts
        # Can URL-decode vi co the chua %20 thay vi dau cach
        from urllib.parse import unquote, urlparse
        parsed = urlparse(artifact_uri)
        artifact_dir = unquote(parsed.path).lstrip("/")
        source_path = os.path.join(artifact_dir, artifact_filename)

        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
            task_logger.info(
                f"Rollback: Da khoi phuc {artifact_filename} "
                f"tu run {run_id[:8]}... -> {target_path}"
            )
            return True
        else:
            task_logger.warning(
                f"Rollback: Khong tim thay {source_path}"
            )
            return False
    except Exception as e:
        task_logger.warning(f"Rollback that bai: {e}")
        return False


@task(name="Ingest Data from Spring Boot", retries=3, retry_delay_seconds=30)
def task_ingest_data() -> bool:
    """Goi HTTP request ve Spring Boot de cap nhat data/raw/transactions.csv."""
    task_logger = get_run_logger()
    task_logger.info("Starting data ingestion from Spring Boot...")
    fetch_data()
    task_logger.info("Ingest data hoan tat.")
    return True


@task(name="Run DVC Pipeline (Preprocess & Train Models)", retries=2, retry_delay_seconds=60)
def task_run_dvc() -> bool:
    """
    Chay DVC repro de xem data co thay doi khong.
    Neu data moi, DVC se tu dong train lai model (theo dvc.yaml).
    """
    task_logger = get_run_logger()
    task_logger.info("Triggering DVC reproduction...")
    try:
        result = subprocess.run(
            ["dvc", "repro"],
            cwd=BASE_DIR,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        task_logger.info(f"DVC Repro Output:\n{result.stdout}")

        # Kiem tra xem co stage nao thuc su CHAY khong
        # "Running stage" xuat hien khi DVC thuc su execute 1 stage
        if "running stage" in result.stdout.lower():
            task_logger.info("DVC da chay lai it nhat 1 stage.")
            return True
        else:
            task_logger.info("No data changes, model unchanged.")
            return False
    except subprocess.CalledProcessError as e:
        task_logger.error(f"DVC Repro Failed:\n{e.stderr}")
        raise


@task(name="Evaluate Model Metrics")
def task_evaluate_metrics() -> Dict[str, Any]:
    """
    Doc metrics tu MLflow run moi nhat va so sanh voi best_metrics.json.
    Tra ve dict chua ket qua danh gia.
    """
    task_logger = get_run_logger()

    try:
        import mlflow

        best = load_best_metrics()
        evaluation: Dict[str, Any] = {"should_deploy": False, "details": {}}
        metrics_changed = False

        # === Danh gia LSTM ===
        task_logger.info("--- Danh gia LSTM ---")
        mlflow.set_experiment(MLFLOW_LSTM_EXPERIMENT)
        runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)

        if not runs.empty:
            new_rmse = runs.iloc[0].get("metrics.rmse", None)
            new_mae = runs.iloc[0].get("metrics.mae", None)
            run_id = runs.iloc[0]["run_id"]
            old_rmse = best.get("lstm", {}).get("rmse", float("inf"))

            if new_rmse is not None:
                improvement = (
                    ((old_rmse - new_rmse) / old_rmse * 100)
                    if old_rmse != float("inf")
                    else 100
                )
                evaluation["details"]["lstm"] = {
                    "new_rmse": round(new_rmse, 4),
                    "old_rmse": round(old_rmse, 4) if old_rmse != float("inf") else "N/A",
                    "improvement_pct": round(improvement, 1),
                    "run_id": run_id,
                }

                if new_rmse < old_rmse or old_rmse == float("inf"):
                    evaluation["should_deploy"] = True
                    metrics_changed = True
                    best["lstm"] = {
                        "rmse": round(new_rmse, 4),
                        "mae": round(new_mae, 4) if new_mae else 0,
                        "run_id": run_id,
                    }
                    task_logger.info(
                        f"LSTM cai thien: RMSE {old_rmse} -> {new_rmse} "
                        f"({improvement:+.1f}%)"
                    )
                else:
                    task_logger.info(
                        f"LSTM khong cai thien: RMSE {new_rmse} >= {old_rmse}"
                    )
                    # Rollback: khoi phuc model cu tu MLflow
                    old_run_id = best.get("lstm", {}).get("run_id")
                    if old_run_id:
                        rollback_model_from_mlflow(
                            MLFLOW_LSTM_EXPERIMENT, old_run_id,
                            LSTM_TFLITE_FILE, LSTM_TFLITE_PATH, task_logger,
                        )
        else:
            task_logger.warning("Khong tim thay LSTM run nao trong MLflow.")

        # === Danh gia LightGBM ===
        task_logger.info("--- Danh gia LightGBM ---")
        mlflow.set_experiment(MLFLOW_LGBM_EXPERIMENT)
        runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)

        if not runs.empty:
            new_rmse = runs.iloc[0].get("metrics.rmse", None)
            new_mae = runs.iloc[0].get("metrics.mae", None)
            run_id = runs.iloc[0]["run_id"]
            old_rmse = best.get("lgbm", {}).get("rmse", float("inf"))

            if new_rmse is not None:
                if new_rmse < old_rmse or old_rmse == float("inf"):
                    improvement = (
                        ((old_rmse - new_rmse) / old_rmse * 100)
                        if old_rmse != float("inf")
                        else 100
                    )
                    metrics_changed = True
                    best["lgbm"] = {
                        "rmse": round(new_rmse, 4),
                        "mae": round(new_mae, 4) if new_mae else 0,
                        "run_id": run_id,
                    }
                    task_logger.info(
                        f"LightGBM cai thien: RMSE {old_rmse} -> {new_rmse} "
                        f"({improvement:+.1f}%)"
                    )
                else:
                    task_logger.info(
                        f"LightGBM khong cai thien: RMSE {new_rmse} >= {old_rmse}"
                    )
                    # Rollback: khoi phuc model cu tu MLflow
                    old_run_id = best.get("lgbm", {}).get("run_id")
                    if old_run_id:
                        rollback_model_from_mlflow(
                            MLFLOW_LGBM_EXPERIMENT, old_run_id,
                            LGBM_MODEL_FILE, LGBM_MODEL_PATH, task_logger,
                        )
        else:
            task_logger.warning("Khong tim thay LightGBM run nao trong MLflow.")

        # === Danh gia Isolation Forest ===
        task_logger.info("--- Danh gia Isolation Forest ---")
        mlflow.set_experiment(MLFLOW_ANOMALY_EXPERIMENT)
        runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)

        if not runs.empty:
            n_anomalies = runs.iloc[0].get("metrics.n_anomalies", None)
            n_normal = runs.iloc[0].get("metrics.n_normal", None)
            anomaly_ratio = runs.iloc[0].get("metrics.anomaly_ratio_percent", None)
            n_samples = runs.iloc[0].get("params.n_samples", "N/A")
            run_id = runs.iloc[0]["run_id"]

            old_ratio = best.get("iforest", {}).get("anomaly_ratio", None)

            evaluation["details"]["iforest"] = {
                "n_anomalies": int(n_anomalies) if n_anomalies else 0,
                "n_normal": int(n_normal) if n_normal else 0,
                "anomaly_ratio": round(anomaly_ratio, 2) if anomaly_ratio else 0,
                "n_samples": n_samples,
                "run_id": run_id,
            }

            if anomaly_ratio is not None:
                metrics_changed = True
                best["iforest"] = {
                    "anomaly_ratio": round(anomaly_ratio, 2),
                    "n_anomalies": int(n_anomalies) if n_anomalies else 0,
                    "n_samples": int(n_samples) if n_samples != "N/A" else 0,
                    "run_id": run_id,
                }
                task_logger.info(
                    f"IForest: {int(n_anomalies)} bat thuong / {n_samples} mau "
                    f"({anomaly_ratio:.1f}%)"
                )

                if old_ratio is not None:
                    ratio_change = anomaly_ratio - old_ratio
                    if abs(ratio_change) > 2.0:
                        task_logger.warning(
                            f"IForest: Ti le bat thuong thay doi dang ke: "
                            f"{old_ratio:.1f}% -> {anomaly_ratio:.1f}% "
                            f"({ratio_change:+.1f}%)"
                        )
                    else:
                        task_logger.info(
                            f"IForest: Ti le bat thuong on dinh "
                            f"({old_ratio:.1f}% -> {anomaly_ratio:.1f}%)"
                        )
        else:
            task_logger.warning("Khong tim thay IForest run nao trong MLflow.")

        # === Luu best metrics ===
        if metrics_changed:
            save_best_metrics(best)
            task_logger.info("Updated best_metrics.json")

        # === Tong ket ===
        task_logger.info("=" * 40)
        task_logger.info("TONG KET DANH GIA MODEL:")
        lstm_status = "Cai thien" if evaluation.get("should_deploy") else "Giu nguyen"
        lgbm_rmse = best.get("lgbm", {}).get("rmse", "N/A")
        iforest_ratio = best.get("iforest", {}).get("anomaly_ratio", "N/A")
        task_logger.info(f"  LSTM:    {lstm_status} (best RMSE: {best.get('lstm', {}).get('rmse', 'N/A')}) ")
        task_logger.info(f"  LightGBM: best RMSE = {lgbm_rmse}")
        task_logger.info(f"  IForest:  anomaly ratio = {iforest_ratio}%")
        task_logger.info("=" * 40)

        return evaluation

    except Exception as e:
        task_logger.warning(
            f"Could not read MLflow metrics: {e}. Defaulting to deploy."
        )
        return {"should_deploy": True, "details": {"error": str(e)}}


@task(name="Deploy Model to Spring Boot", retries=2, retry_delay_seconds=30)
def task_deploy_model() -> bool:
    """Upload model .tflite moi nhat len Spring Boot Backend."""
    import requests

    task_logger = get_run_logger()

    if not os.path.exists(LSTM_TFLITE_PATH):
        task_logger.error(f"Khong tim thay {LSTM_TFLITE_PATH}")
        return False

    task_logger.info(f"Uploading model tu {LSTM_TFLITE_PATH}...")
    try:
        with open(LSTM_TFLITE_PATH, "rb") as f:
            files = {
                "model_file": ("expense_model.tflite", f, "application/octet-stream")
            }
            headers = {
                API_KEY_HEADER: os.getenv("MLOPS_API_KEY", DEFAULT_API_KEY)
            }
            response = requests.post(
                BACKEND_DEPLOY_URL,
                files=files,
                headers=headers,
                timeout=30,
                verify=False,
            )
        response.raise_for_status()
        task_logger.info("Upload model thanh cong!")
        return True
    except requests.exceptions.RequestException as e:
        task_logger.error(f"Upload model that bai: {e}")
        raise


@flow(name="MLOps Expense Forecasting Pipeline")
def mlops_pipeline():
    """Flow chinh: Ingest -> DVC Repro -> Evaluate -> Deploy."""
    flow_logger = get_run_logger()
    flow_logger.info("=" * 50)
    flow_logger.info("Starting MLOps Pipeline")
    flow_logger.info("=" * 50)

    # 1. Tai du lieu tu backend
    task_ingest_data()

    # 2. Xu ly & Train (DVC quan ly LSTM + LightGBM + IForest)
    has_new_training = task_run_dvc()

    if not has_new_training:
        flow_logger.info("Khong co training moi, pipeline ket thuc.")
        return

    # 3. Danh gia metrics
    evaluation = task_evaluate_metrics()

    # 4. Deploy LSTM len Spring Boot CHI KHI model moi tot hon
    if evaluation.get("should_deploy", False):
        flow_logger.info("Model moi tot hon -> Deploying...")
        task_deploy_model()
    else:
        flow_logger.info("Model moi khong tot hon -> Bo qua deploy.")

    # 5. Reload FastAPI predict server (neu dang chay)
    try:
        import requests
        resp = requests.post(PREDICT_SERVER_RELOAD_URL, timeout=5)
        if resp.status_code == 200:
            flow_logger.info("FastAPI predict server: Reload model thanh cong!")
        else:
            flow_logger.warning(f"FastAPI reload tra ve status {resp.status_code}")
    except Exception:
        flow_logger.info("FastAPI predict server khong chay -> bo qua reload.")

    flow_logger.info("Pipeline hoan tat!")


def _is_port_in_use(port: int) -> bool:
    """Kiem tra port co dang duoc su dung khong."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _ensure_prefect_server() -> None:
    """Tu dong bat Prefect Server neu chua chay."""
    if not _is_port_in_use(PREFECT_PORT):
        print(
            f"[*] Prefect Server chua chay tren cong {PREFECT_PORT}. "
            f"Dang khoi dong tu dong..."
        )
        subprocess.Popen(
            'start "Prefect Server" cmd /c "prefect server start"',
            shell=True,
        )

        print("[*] Vui long doi Server khoi dong...")
        for _ in range(15):
            if _is_port_in_use(PREFECT_PORT):
                print(f"[+] Prefect Server da san sang tren cong {PREFECT_PORT}!")
                break
            time.sleep(1)

    # Config Prefect API URL
    api_url = f"http://127.0.0.1:{PREFECT_PORT}/api"
    os.environ["PREFECT_API_URL"] = api_url
    subprocess.run(
        ["prefect", "config", "set", f"PREFECT_API_URL={api_url}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


if __name__ == "__main__":
    _ensure_prefect_server()

    if len(sys.argv) > 1 and sys.argv[1] == "--run-once":
        print("Running pipeline manually once...")
        mlops_pipeline()
    else:
        print("Starting continuous MLOps Server...")
        mlops_pipeline.serve(
            name="expense-forecasting-weekly",
            cron=PIPELINE_CRON,
            tags=["mlops", "training"],
            description="Pipeline dinh ky keo du lieu moi va huan luyen mo hinh du bao chi tieu.",
        )
