import pandas as pd
import numpy as np
import tensorflow as tf

from keras.layers import (
    Input, GRU, Dense, Embedding, Concatenate, Flatten
)
from keras.models import Model

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import mlflow
import mlflow.tensorflow


# =========================
# CONFIG
# =========================
CSV_PATH = "transactions.csv"
WINDOW_SIZE = 7
EPOCHS = 30
BATCH_SIZE = 16

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(
    CSV_PATH,
    names=["transaction_id", "user_id", "category_id", "amount", "date", "note"]
)

df["date"] = pd.to_datetime(df["date"])

user_ids = df["user_id"].unique()
user2idx = {u: i for i, u in enumerate(user_ids)}
num_users = len(user2idx)

# =========================
# HELPER FUNCTIONS
# =========================
def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)

# =========================
# BUILD DATASET
# =========================
X_ts_all = []
X_user_all = []
y_all = []

for user_id in user_ids:
    df_user = df[df["user_id"] == user_id]

    daily = (
        df_user.groupby("date")["amount"]
        .sum()
        .sort_index()
    )

    # Ensure continuous dates
    daily = daily.asfreq("D", fill_value=0)

    # Normalize per user
    scaler = MinMaxScaler()
    values = scaler.fit_transform(daily.values.reshape(-1, 1))

    X_ts, y = create_sequences(values, WINDOW_SIZE)

    if len(X_ts) == 0:
        continue

    user_idx = user2idx[user_id]
    user_arr = np.full((len(X_ts), 1), user_idx)

    X_ts_all.append(X_ts)
    X_user_all.append(user_arr)
    y_all.append(y)

X_ts_all = np.concatenate(X_ts_all)
X_user_all = np.concatenate(X_user_all)
y_all = np.concatenate(y_all)

# Train / test split
split = int(0.8 * len(X_ts_all))
X_ts_train, X_ts_test = X_ts_all[:split], X_ts_all[split:]
X_user_train, X_user_test = X_user_all[:split], X_user_all[split:]
y_train, y_test = y_all[:split], y_all[split:]

# =========================
# BUILD MODEL (GRU + USER EMBEDDING)
# =========================
ts_input = Input(shape=(WINDOW_SIZE, 1), name="time_series")
user_input = Input(shape=(1,), name="user_id")

user_embedding = Embedding(
    input_dim=num_users,
    output_dim=8,
    name="user_embedding"
)(user_input)

user_embedding = Flatten(name="user_embedding_flat")(user_embedding)

# GRU thay cho LSTM
ts_features = GRU(64, activation="tanh", name="gru_layer")(ts_input)

x = Concatenate()([ts_features, user_embedding])
output = Dense(1, name="prediction")(x)

model = Model(
    inputs=[ts_input, user_input],
    outputs=output
)

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

# =========================
# TRAIN + MLFLOW
# =========================
mlflow.set_experiment("Expense Forecasting - GRU User Embedding")

with mlflow.start_run():
    model.fit(
        [X_ts_train, X_user_train],
        y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    y_pred = model.predict([X_ts_test, X_user_test])

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # Log params
    mlflow.log_param("window_size", WINDOW_SIZE)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("embedding_dim", 8)
    mlflow.log_param("num_users", num_users)
    mlflow.log_param("model_type", "GRU")

    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)

    # Log model
    mlflow.tensorflow.log_model(
        model,
        artifact_path="model",
        registered_model_name="ExpenseForecastingGRU"
    )

    print("RMSE:", rmse)
    print("MAE:", mae)
