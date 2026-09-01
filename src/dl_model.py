from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from src.evaluate import regression_metrics
from src.utils import load_config, ensure_directories

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def make_sequences(values, targets, sequence_length):
    X, y = [], []
    for i in range(sequence_length, len(values)):
        X.append(values[i-sequence_length:i])
        y.append(targets[i])
    return np.asarray(X), np.asarray(y)


def train_lstm(config=None):
    config = config or load_config()
    ensure_directories(config)

    path = Path(config["data"]["processed_dir"]) / "flowcast_features.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values(["road_id", "timestamp"]).copy()

    # This LSTM forecasts next-window traffic volume per road. It is trained
    # from scratch (new weights), with no pretrained model.
    numeric = [
        "traffic_volume", "avg_speed", "occupancy", "travel_time",
        "rainfall", "visibility", "temperature", "road_capacity",
        "signal_timing", "hour", "day_of_week", "is_peak_hour",
        "is_weekend", "public_holiday", "event_flag", "roadwork_flag"
    ]
    features = [c for c in numeric if c in df.columns]

    clean = df.dropna(subset=["traffic_volume"] + features).copy()

    # Use each road independently so sequences do not cross road boundaries.
    all_predictions = []
    histories = []

    out = Path(config["models"]["deep_learning_dir"])

    for road_id, road_df in clean.groupby("road_id"):
        road_df = road_df.sort_values("timestamp")
        if len(road_df) < 80:
            continue

        vals = road_df[features].astype(float).values
        target = road_df["traffic_volume"].astype(float).values.reshape(-1, 1)

        split = int(len(vals) * 0.8)
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        X_train_raw = scaler_x.fit_transform(vals[:split])
        X_test_raw = scaler_x.transform(vals[split:])
        y_train_raw = scaler_y.fit_transform(target[:split]).ravel()
        y_test_raw = scaler_y.transform(target[split:]).ravel()

        seq = config["forecast"]["sequence_length"]

        X_train, y_train = make_sequences(
            X_train_raw, y_train_raw, seq
        )
        X_test, y_test = make_sequences(
            X_test_raw, y_test_raw, seq
        )

        if len(X_train) < 20 or len(X_test) < 5:
            continue

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(seq, len(features))),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1)
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"]
        )

        early = EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=60,
            batch_size=32,
            callbacks=[early],
            verbose=0
        )

        pred_scaled = model.predict(X_test, verbose=0)
        pred = scaler_y.inverse_transform(pred_scaled).ravel()
        actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

        metrics = regression_metrics(actual, pred)
        metrics["road_id"] = road_id

        all_predictions.append(
            pd.DataFrame({
                "road_id": road_id,
                "actual": actual,
                "predicted": pred,
                "residual": actual - pred
            })
        )

        histories.append(pd.DataFrame(history.history).assign(road_id=road_id))

        # Save the model and scalers for each road.
        safe = str(road_id).replace("/", "_")
        model.save(out / f"lstm_{safe}.keras")
        joblib.dump(scaler_x, out / f"scaler_x_{safe}.joblib")
        joblib.dump(scaler_y, out / f"scaler_y_{safe}.joblib")

    if not all_predictions:
        raise RuntimeError("No road had enough valid observations for LSTM training.")

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(out / "lstm_predictions.csv", index=False)

    history_df = pd.concat(histories, ignore_index=True)
    history_df.to_csv(out / "lstm_training_history.csv", index=False)

    summary = (
        predictions.groupby("road_id")
        .apply(
            lambda g: pd.Series(
                regression_metrics(g["actual"], g["predicted"])
            ),
            include_groups=False
        )
        .reset_index()
    )

    summary.to_csv(out / "lstm_scoreboard.csv", index=False)

    plt.figure(figsize=(10, 5))
    for road_id, g in history_df.groupby("road_id"):
        plt.plot(g["loss"], alpha=0.25)
    plt.title("LSTM Training Loss by Road")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.tight_layout()
    plt.savefig(
        Path(config["reports"]["figures_dir"]) / "lstm_training_loss.png",
        dpi=150
    )
    plt.close()

    return summary


if __name__ == "__main__":
    print(train_lstm())
