import os
import numpy as np
import tensorflow as tf
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import mlflow
import mlflow.tensorflow
from prefect import task

INPUT_DIR = "data/processed"
EPOCHS = 50
BATCH_SIZE = 8

@task(name="Build LSTM Model (No User Embedding)")
def build_model(window_size):
    """
    Model LSTM đơn giản — BỎ User Embedding.
    
    Lý do: Model chạy on-device (TFLite), mỗi thiết bị chỉ phục vụ 1 user.
    Cá nhân hóa thông qua dữ liệu input (mỗi user có chuỗi chi tiêu riêng)
    thay vì thông qua embedding vector.
    """
    # Input: chuỗi chi tiêu 4 tuần gần nhất
    ts_input = Input(shape=(window_size, 1), name="time_series")
    
    # LSTM layer
    ts_features = LSTM(32, activation="tanh")(ts_input)
    
    # Hidden layer
    x = Dense(16, activation="relu", name="hidden_layer")(ts_features)
    
    # Linear Output — dự đoán chi tiêu tuần tới
    output = Dense(1, name="prediction")(x)

    model = Model(inputs=ts_input, outputs=output)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

@task(name="Train TFLite Weekly Expense Model")
def train():
    print("Loading preprocessed data (weekly)...")
    X_ts = np.load(os.path.join(INPUT_DIR, "X_ts.npy"))
    y = np.load(os.path.join(INPUT_DIR, "y.npy"))
    meta = joblib.load(os.path.join(INPUT_DIR, "meta.joblib"))
    
    if len(X_ts) == 0:
        print("Tập dữ liệu đang trống, bỏ qua thư mục huấn luyện.")
        return

    window_size = meta["window_size"]
    
    split = int(0.8 * len(X_ts))
    X_ts_train, X_ts_test = X_ts[:split], X_ts[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Data ready. Window size: {window_size} weeks. Samples: {len(X_ts)}")

    mlflow.set_experiment("Expense Forecasting - LSTM")
    
    with mlflow.start_run():
        model = build_model(window_size)
        
        model.summary()
        
        # log params
        mlflow.log_param("window_size", window_size)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("user_embedding", False)
        mlflow.log_param("aggregation", "weekly")
        
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        
        # Single input — không cần multi-input nữa
        model.fit(
            X_ts_train,
            y_train,
            validation_split=0.1,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=1
        )
        
        y_pred = model.predict(X_ts_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        print(f"Model Evaluate: RMSE={rmse:.4f}, MAE={mae:.4f}")
        
        mlflow.tensorflow.log_model(
            model,
            artifact_path="model",
            registered_model_name="ExpenseForecastingLSTM"
        )
        
        # Convert TFLite — thêm các ops cần thiết cho LSTM
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
        ]
        converter._experimental_lower_tensor_list_ops = False
        
        tflite_model = converter.convert()
        os.makedirs("models", exist_ok=True)
        tflite_path = "models/expense_model.tflite"
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        
        print(f"TFLite model saved: {tflite_path} ({len(tflite_model)} bytes)")
        mlflow.log_artifact(tflite_path)
        
        # Lưu thêm Keras model (.h5) để serve_predict.py load server-side
        keras_path = "models/expense_model.h5"
        model.save(keras_path)
        print(f"Keras model saved: {keras_path}")
        mlflow.log_artifact(keras_path)
        
        # Lưu meta cho LSTM server-side (window_size để serve_predict biết input shape)
        lstm_meta = {
            "window_size": window_size,
            "model_type": "lstm_weekly",
        }
        lstm_meta_path = "models/meta_lstm.joblib"
        joblib.dump(lstm_meta, lstm_meta_path)
        print(f"LSTM meta saved: {lstm_meta_path}")

if __name__ == "__main__":
    from prefect import flow
    @flow(name="Model Training Flow")
    def run_training():
        train()
    run_training()
