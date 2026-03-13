@echo off
echo ==============================================
echo KHOI DONG HEO THONG MLOps (PREFECT + WORKER)
echo ==============================================

:: 1. Set môi trường mặc định từ xa cho Prefect UI
set PREFECT_API_URL=http://127.0.0.1:4200/api
echo [OK] Da set bien moi truong PREFECT_API_URL

:: 2. Mở một cửa sổ mới để chạy Prefect Server ngầm
echo [OK] Dang mo Server Prefect o cua so moi...
start "Prefect Server (DO NOT CLOSE)" cmd /k "E:\AntigravityWorkspace\MLOPs\.venv\Scripts\activate.bat && prefect server start"

:: 3. Đợi vài giây cho Server khởi động xong
timeout /t 5

:: 4. Kích hoạt môi trường ảo và chạy DVC (nếu có dvc) sau đó gọi main_flow
echo [OK] Kich hoat DVC va chay worker MLOps chinh...
call E:\AntigravityWorkspace\MLOPs\.venv\Scripts\activate.bat

:: Nếu bạn muốn DVC cập nhật/pull dữ liệu tự động, bỏ comment dòng dưới đây:
:: dvc pull

:: Chạy flow chính!
python flows\main_flow.py

pause
