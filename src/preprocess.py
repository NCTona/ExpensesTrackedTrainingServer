import pandas as pd
import numpy as np
import os
import joblib
from prefect import task

INPUT_FILE = "data/raw/transactions.csv"
OUTPUT_DIR = "data/processed"

# Dùng 4 tuần thay vì 7 ngày — dữ liệu weekly mượt hơn, ít noise hơn
WINDOW_SIZE = 4

@task(name="Create Weekly Time Series Sequences")
def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    if len(X) == 0:
        return np.array([]), np.array([])
    return np.array(X), np.array(y)

@task(name="Preprocess Transactions Data (Weekly)")
def preprocess():
    print(f"Preprocessing {INPUT_FILE} (weekly aggregation)...")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Missing {INPUT_FILE}")

    # Cần kiểm tra columns format vì fallback vs old format
    df = pd.read_csv(INPUT_FILE)
    if "date" not in df.columns:
        df = pd.read_csv(INPUT_FILE, names=["transaction_id", "user_id", "category_id", "amount", "date", "note"])
    else:
        df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    
    user_ids = df["user_id"].unique()
    
    X_ts_all, y_all = []  , []

    for user_id in user_ids:
        df_user = df[df["user_id"] == user_id]
        
        # Aggregate theo TUẦN thay vì ngày — giảm noise đáng kể
        weekly = df_user.set_index("date").resample("W")["amount"].sum()
        weekly = weekly.fillna(0)
        
        values = weekly.values.reshape(-1, 1)
        if len(values) <= WINDOW_SIZE:
            continue
        
        # Normalize bằng percentile 95 thay vì MAX cứng 2M
        # Mỗi user có scale riêng, linh hoạt hơn
        p95 = np.percentile(values[values > 0], 95) if np.any(values > 0) else 1.0
        p95 = max(p95, 1.0)  # Tránh chia cho 0
        scaled = np.clip(values / p95, 0, 3)  # Cho phép vượt 1 nhưng cap ở 3
        
        X_ts, y = create_sequences(scaled, WINDOW_SIZE)
        if len(X_ts) == 0:
            continue
            
        # LỌC BỎ CHUỖI TĨNH (INACTIVE): Loại bỏ sequences mà cả window + target đều = 0
        active_indices = [i for i in range(len(X_ts)) if np.sum(X_ts[i]) > 0.001 or y[i] > 0.001]
        if len(active_indices) == 0:
            continue
            
        X_ts = X_ts[active_indices]
        y = y[active_indices]
        
        # KHÔNG dùng data augmentation giả nữa
        # Weekly aggregation đã cho đủ samples tự nhiên hơn
        
        X_ts_all.append(X_ts)
        y_all.append(y)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if len(X_ts_all) == 0:
        print("Dữ liệu quá ít để tạo sequence. Sẽ lưu dạng rỗng.")
        np.save(os.path.join(OUTPUT_DIR, "X_ts.npy"), np.array([]))
        np.save(os.path.join(OUTPUT_DIR, "y.npy"), np.array([]))
    else:
        np.save(os.path.join(OUTPUT_DIR, "X_ts.npy"), np.concatenate(X_ts_all))
        np.save(os.path.join(OUTPUT_DIR, "y.npy"), np.concatenate(y_all))

    # Lưu meta — KHÔNG CÒN num_users vì bỏ user embedding
    joblib.dump({"window_size": WINDOW_SIZE}, os.path.join(OUTPUT_DIR, "meta.joblib"))
    print(f"Preprocess hoàn tất! Window size: {WINDOW_SIZE} weeks")

if __name__ == "__main__":
    from prefect import flow
    @flow(name="Data Preprocessing Flow")
    def run_preprocessing():
        preprocess()
    run_preprocessing()
