from pathlib import Path
import pandas as pd
import numpy as np
from src.utils import load_config, ensure_directories


def clean_traffic(df):
    df = df.copy()
    df = df.dropna(subset=["timestamp", "road_id", "weather_station_id"])
    df = df.sort_values(["road_id", "timestamp"])

    # Remove duplicate road/time observations, retaining the first record.
    df = df.drop_duplicates(
        subset=["road_id", "timestamp"],
        keep="first"
    )

    # Domain-safe bounds; extreme values are treated as missing, not clipped.
    for col, low, high in [
        ("avg_speed", 0, 130),
        ("occupancy", 0, 100),
        ("traffic_volume", 0, None),
        ("travel_time", 0, None),
    ]:
        if col in df:
            if low is not None:
                df.loc[df[col] < low, col] = np.nan
            if high is not None:
                df.loc[df[col] > high, col] = np.nan

    # Interpolate numeric sensor readings within each road.
    numeric = ["traffic_volume", "avg_speed", "occupancy"]
    for col in numeric:
        if col in df:
            df[col] = (
                df.groupby("road_id")[col]
                .transform(lambda s: s.interpolate(limit_direction="both"))
            )

    # Categorical congestion is kept missing if it was not supplied;
    # the modelling stage handles missing labels explicitly.
    return df


def clean_weather(df):
    df = df.copy()
    df = df.dropna(subset=["timestamp", "station_id"])
    df = df.sort_values(["station_id", "timestamp"])

    df["weather_condition"] = (
        df["weather_condition"].astype(str).str.strip().str.lower()
        .replace({
            "clear sky": "clear",
            "sunny": "clear",
            "light rain": "rain",
            "heavy rain": "heavy_rain",
            "stormy": "storm",
        })
    )

    for col in ["temperature", "rainfall", "visibility"]:
        if col in df:
            df[col] = (
                df.groupby("station_id")[col]
                .transform(lambda s: s.interpolate(limit_direction="both"))
            )

    return df


def clean_calendar(df):
    df = df.copy()
    for col in ["public_holiday", "event_flag", "roadwork_flag"]:
        df[col] = df[col].fillna(0).astype(int)
    return df


def merge_sources(traffic, weather, calendar):

    # Convert timestamp columns
    traffic["timestamp"] = pd.to_datetime(traffic["timestamp"])
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])

    # Sort timestamps for merge_asof
    traffic = traffic.sort_values(
        by="timestamp"
    ).reset_index(drop=True)

    weather = weather.sort_values(
        by="timestamp"
    ).reset_index(drop=True)

    # Merge traffic and weather
    merged = pd.merge_asof(
        traffic,
        weather,
        on="timestamp",
        direction="nearest"
    )

    # Calendar merge
    if "date" not in merged.columns:
        merged["date"] = merged["timestamp"].dt.date

    calendar["date"] = pd.to_datetime(calendar["date"]).dt.date

    merged = merged.merge(
        calendar,
        on="date",
        how="left"
    )

    return merged

def run_cleaning(config=None):
    config = config or load_config()
    ensure_directories(config)
    interim = Path(config["data"]["interim_dir"])
    processed = Path(config["data"]["processed_dir"])

    traffic = pd.read_csv(interim / "traffic_valid.csv", parse_dates=["timestamp"])
    weather = pd.read_csv(interim / "weather_valid.csv", parse_dates=["timestamp"])
    calendar = pd.read_csv(interim / "calendar_valid.csv", parse_dates=["date"])

    traffic = clean_traffic(traffic)
    weather = clean_weather(weather)
    calendar = clean_calendar(calendar)

    merged = merge_sources(traffic, weather, calendar)

    # Final missing-value treatment for modelling inputs.
    for col in ["temperature", "rainfall", "visibility"]:
        if col in merged:
            merged[col] = merged[col].fillna(merged[col].median())

    merged.to_csv(processed / "flowcast_processed.csv", index=False)
    return merged


if __name__ == "__main__":
    df = run_cleaning()
    print(df.shape)
