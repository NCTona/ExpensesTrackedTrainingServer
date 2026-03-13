import pandas as pd
import numpy as np
import mlflow.tensorflow
from sklearn.preprocessing import MinMaxScaler

# =========================
# CONFIG
# =========================
CSV_PATH = "data_predict.csv"
WINDOW_SIZE = 7

# RUN_ID
RUN_ID = "a727a0846c0f4678bf51b0d98b1f2f89"

# =========================
# LOAD MODEL FROM MLFLOW
# =========================
print("Loading model from MLflow...")
model = mlflow.tensorflow.load_model(
    f"runs:/{RUN_ID}/model"
)
print("Model loaded successfully!")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(
    CSV_PATH,
    names=["transaction_id", "user_id", "category_id", "amount", "date", "note"]
)

df["date"] = pd.to_datetime(df["date"])

# Map user_id → index (PHẢI GIỐNG LÚC TRAIN)
user_ids = df["user_id"].unique()
user2idx = {u: i for i, u in enumerate(user_ids)}

# =========================
# TEST PREDICTION
# =========================
def predict_next_day(user_id):
    if user_id not in user2idx:
        print("User không tồn tại!")
        return

    user_idx = user2idx[user_id]

    df_user = df[df["user_id"] == user_id]
    daily = df_user.groupby("date")["amount"].sum().sort_index()
    daily = daily.asfreq("D", fill_value=0)

    if len(daily) < WINDOW_SIZE:
        print("Không đủ dữ liệu để dự đoán")
        return

    # ngày cuối cùng & ngày dự đoán
    last_date = daily.index[-1]
    predict_date = last_date + pd.Timedelta(days=1)

    scaler = MinMaxScaler()
    values = scaler.fit_transform(daily.values.reshape(-1, 1))

    x_ts = values[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)
    x_user = np.array([[user_idx]])

    y_pred = model.predict([x_ts, x_user])
    y_real = scaler.inverse_transform(y_pred)

    print(f"User {user_id}")
    print(f"Ngày dự đoán: {predict_date.date()}")
    print(f"Dự đoán chi tiêu: {y_real[0][0]:.2f}")

# =========================
# RUN TEST
# =========================
if __name__ == "__main__":
    for user_id in user_ids:
        predict_next_day(user_id=user_id)
        print("-" * 40)
