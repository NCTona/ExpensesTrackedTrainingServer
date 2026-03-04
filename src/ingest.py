import requests
import pandas as pd
import os
from prefect import task

# Đường dẫn tĩnh tới Backend (dùng HTTPS vì server bật SSL/TLS)
BACKEND_URL = "https://localhost:8080/api/internal/transactions"
OUTPUT_FILE = "data/raw/transactions.csv"

@task(name="Fetch Data from Backend", retries=2, retry_delay_seconds=10)
def fetch_data():
    print(f"Fetching data from {BACKEND_URL}...")
    try:
        # Gọi thử với secret_key cơ bản (nếu Backend cấu hình app-key)
        headers = { "X-API-KEY": "secret_mlops_key" }
        response = requests.get(BACKEND_URL, headers=headers, timeout=10, verify=False)
        response.raise_for_status()
        data = response.json()
        
        # Giả định dữ liệu lấy về là danh sách dictionary
        if not data:
            print("No new data found from Backend.")
        else:
            df = pd.DataFrame(data)
            os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
            df.to_csv(OUTPUT_FILE, index=False)
            print(f"Khoan thành công {len(df)} dòng dữ liệu vào {OUTPUT_FILE}")
    except Exception as e:
        print(f"Lỗi khi kéo data từ Backend: {e}")
        # Tạm thời phục hồi bằng cách tạo dữ liệu dummy nếu gọi không thành công (cho mục đích demo/dev)
        if not os.path.exists(OUTPUT_FILE):
             print("Tạo dữ liệu demo (fallback) để pipeline vẫn chạy đủ cho WINDOW_SIZE.")
             os.makedirs("data/raw", exist_ok=True)
             
             # Tạo data ngẫu nhiên
             import random
             from datetime import datetime, timedelta
             
             dates = [(datetime(2023, 10, 1) + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(20)]
             demo_data = {
                 "transaction_id": list(range(1, 21)),
                 "user_id": [101] * 10 + [102] * 10, # Mỗi user 10 ngày
                 "category_id": [random.randint(1,5) for _ in range(20)],
                 "amount": [random.randint(20000, 200000) for _ in range(20)],
                 "date": dates,
                 "note": ["Demo data"] * 20
             }
             pd.DataFrame(demo_data).to_csv(OUTPUT_FILE, index=False)
             print(f"Đã tạo demo transactions.csv với {len(demo_data['transaction_id'])} dòng.")

if __name__ == "__main__":
    from prefect import flow
    @flow(name="Data Ingestion Flow")
    def run_ingestion():
        fetch_data()
    run_ingestion()
