import json
import pandas as pd
import numpy as np
from src.utils import load_config


def parse_vehicle_distribution(df):
    df = df.copy()

    def get_value(text, key):
        try:
            obj = json.loads(text) if isinstance(text, str) else {}
            return float(obj.get(key, 0))
        except Exception:
            return 0.0

    if "vehicle_type_dist" in df:
        for key in ["2W", "Car", "LCV", "HCV"]:
            df[f"vehicle_{key.lower()}"] = df["vehicle_type_dist"].apply(
                lambda x: get_value(x, key)
            )

    return df


def add_time_features(df):
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])

    df["hour"] = ts.dt.hour
    df["minute"] = ts.dt.minute
    df["day_of_week"] = ts.dt.dayofweek
    df["day_of_month"] = ts.dt.day
    df["month"] = ts.dt.month
    df["week_of_year"] = ts.dt.isocalendar().week.astype(int)

    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_morning_peak"] = df["hour"].isin([7, 8, 9]).astype(int)
    df["is_evening_peak"] = df["hour"].isin([16, 17, 18, 19]).astype(int)
    df["is_peak_hour"] = (
        (df["is_morning_peak"] == 1) |
        (df["is_evening_peak"] == 1)
    ).astype(int)

    # Cyclic encodings avoid an artificial jump between 23:00 and 00:00.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df


def add_lag_and_rolling_features(df, config):
    df = df.copy()
    df = df.sort_values(["road_id", "timestamp"])

    lags = config["features"]["lags"]
    windows = config["features"]["rolling_windows"]

    for target in ["traffic_volume", "avg_speed", "occupancy", "travel_time"]:
        if target not in df:
            continue

        for lag in lags:
            df[f"{target}_lag_{lag}"] = (
                df.groupby("road_id")[target].shift(lag)
            )

        for window in windows:
            shifted = df.groupby("road_id")[target].shift(1)
            df[f"{target}_rolling_mean_{window}"] = (
                shifted.groupby(df["road_id"]).rolling(window).mean()
                .reset_index(level=0, drop=True)
            )
            df[f"{target}_rolling_std_{window}"] = (
                shifted.groupby(df["road_id"]).rolling(window).std()
                .reset_index(level=0, drop=True)
            )

    return df


def add_weather_flags(df):
    df = df.copy()
    df["is_raining"] = (df["rainfall"] > 0).astype(int)
    df["heavy_rain"] = (df["rainfall"] >= 2.5).astype(int)
    df["low_visibility"] = (df["visibility"] < 5000).astype(int)
    df["bad_weather"] = (
        (df["is_raining"] == 1) |
        (df["low_visibility"] == 1)
    ).astype(int)

    return df


def add_interactions(df):
    df = df.copy()
    df["volume_capacity_ratio"] = (
        df["traffic_volume"] / df["road_capacity"].replace(0, np.nan)
    )
    df["speed_occupancy_interaction"] = (
        df["avg_speed"] * df["occupancy"]
    )
    df["rain_peak_interaction"] = (
        df["rainfall"] * df["is_peak_hour"]
    )
    return df


def build_features(df, config=None):
    config = config or load_config()

    df = df.copy()
    df = parse_vehicle_distribution(df)
    df = add_time_features(df)
    df = add_lag_and_rolling_features(df, config)
    df = add_weather_flags(df)
    df = add_interactions(df)

    # Accident risk is a binary supervised-learning target derived from the
    # actual accident_count supplied by the traffic sensor data.
    df["accident_risk"] = (df["accident_count"] > 0).astype(int)

    df = df.replace([np.inf, -np.inf], np.nan)

    # Only lag/rolling-derived rows are removed. Raw target rows can still be
    # missing and are handled separately by each model.
    feature_columns = [
        c for c in df.columns
        if "_lag_" in c or "_rolling_" in c
    ]
    if feature_columns:
        df = df.dropna(subset=feature_columns)

    return df


if __name__ == "__main__":
    config = load_config()
    path = config["data"]["processed_dir"] + "/flowcast_processed.csv"
    out = config["data"]["processed_dir"] + "/flowcast_features.csv"

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = build_features(df, config)
    df.to_csv(out, index=False)
    print(df.shape)
