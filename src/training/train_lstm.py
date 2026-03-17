# -*- coding: utf-8 -*-
"""
train_lstm.py — Train LSTM model du bao chi tieu theo tuan.

Model nay chay on-device (TFLite) va server-side (.h5):
  - Input: 4 tuan chi tieu gan nhat (per-window max normalized)
  - Output: du doan chi tieu tuan tiep theo

Khong dung User Embedding vi model chay tren device,
moi device phuc vu 1 user duy nhat.
"""

import logging
import os

import joblib
import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input, LSTM
from keras.models import Model
from prefect import task
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    LSTM_EPOCHS,
    LSTM_BATCH_SIZE,
    LSTM_UNITS,
    LSTM_DENSE_UNITS,
    LSTM_TFLITE_PATH,
    LSTM_META_PATH,
    MLFLOW_LSTM_EXPERIMENT,
)

logger = logging.getLogger(__name__)


@task(name="Build LSTM Model (No User Embedding)")
def build_model(window_size: int) -> Model:
    """
    Xay dung LSTM model don gian cho du bao chi tieu.

    Args:
        window_size: So tuan input (mac dinh 4).

    Returns:
        Keras Model da compile.
    """
    ts_input = Input(shape=(window_size, 1), name="time_series")
    ts_features = LSTM(LSTM_UNITS, activation="tanh")(ts_input)
    x = Dense(LSTM_DENSE_UNITS, activation="relu", name="hidden_layer")(ts_features)
    output = Dense(1, name="prediction")(x)

    model = Model(inputs=ts_input, outputs=output)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


@task(name="Train TFLite Weekly Expense Model")
def train() -> None:
    """
    Train LSTM, luu .h5 (server) va .tflite (on-device).

    Pipeline:
      1. Load preprocessed data (X_ts.npy, y.npy)
      2. Train voi EarlyStopping
      3. Log metrics len MLflow
      4. Export .tflite va .h5
    """
    logger.info("Loading preprocessed data (weekly)...")
    X_ts = np.load(os.path.join(PROCESSED_DATA_DIR, "X_ts.npy"))
    y = np.load(os.path.join(PROCESSED_DATA_DIR, "y.npy"))
    meta = joblib.load(os.path.join(PROCESSED_DATA_DIR, "meta.joblib"))

    if len(X_ts) == 0:
        logger.warning("Dataset is empty, skipping training.")
        return

    window_size = meta["window_size"]
    split = int(0.8 * len(X_ts))
    X_ts_train, X_ts_test = X_ts[:split], X_ts[split:]
    y_train, y_test = y[:split], y[split:]

    logger.info(f"Data ready. Window: {window_size} weeks. Samples: {len(X_ts)}")

    mlflow.set_experiment(MLFLOW_LSTM_EXPERIMENT)

    with mlflow.start_run():
        model = build_model(window_size)
        model.summary()

        # Log params
        mlflow.log_param("window_size", window_size)
        mlflow.log_param("epochs", LSTM_EPOCHS)
        mlflow.log_param("batch_size", LSTM_BATCH_SIZE)
        mlflow.log_param("user_embedding", False)
        mlflow.log_param("aggregation", "weekly")

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        model.fit(
            X_ts_train,
            y_train,
            validation_split=0.1,
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH_SIZE,
            callbacks=[early_stop],
            verbose=1,
        )

        # Evaluate
        y_pred = model.predict(X_ts_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        logger.info(f"Model Evaluate: RMSE={rmse:.4f}, MAE={mae:.4f}")

        mlflow.tensorflow.log_model(
            model,
            artifact_path="model",
            registered_model_name="ExpenseForecastingLSTM",
        )

        # Convert TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter._experimental_lower_tensor_list_ops = False

        tflite_model = converter.convert()
        os.makedirs(MODELS_DIR, exist_ok=True)

        with open(LSTM_TFLITE_PATH, "wb") as f:
            f.write(tflite_model)

        logger.info(
            f"TFLite model saved: {LSTM_TFLITE_PATH} ({len(tflite_model)} bytes)"
        )
        mlflow.log_artifact(LSTM_TFLITE_PATH)

        # Save meta
        lstm_meta = {
            "window_size": window_size,
            "model_type": "lstm_weekly",
        }
        joblib.dump(lstm_meta, LSTM_META_PATH)
        logger.info(f"LSTM meta saved: {LSTM_META_PATH}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from prefect import flow

    @flow(name="Model Training Flow")
    def run_training():
        train()

    run_training()
