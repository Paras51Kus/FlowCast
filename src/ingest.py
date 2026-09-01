from pathlib import Path
import pandas as pd
import numpy as np
from src.utils import load_config, ensure_directories


def parse_traffic(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df["timestamp"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        errors="coerce"
    )
    return df


def parse_weather(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    # Weather uses DD/MM/YYYY.
    df["timestamp"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        dayfirst=True,
        errors="coerce"
    )
    return df


def parse_calendar(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    return df


def range_invalid_mask(df):
    bad = pd.Series(False, index=df.index)
    rules = {
        "traffic_volume": df.get("traffic_volume", pd.Series(index=df.index, dtype=float)) < 0,
        "avg_speed": df.get("avg_speed", pd.Series(index=df.index, dtype=float)) <= 0,
        "occupancy": (df.get("occupancy", pd.Series(index=df.index, dtype=float)) < 0) |
                     (df.get("occupancy", pd.Series(index=df.index, dtype=float)) > 100),
        "travel_time": df.get("travel_time", pd.Series(index=df.index, dtype=float)) <= 0,
        "road_capacity": df.get("road_capacity", pd.Series(index=df.index, dtype=float)) <= 0,
        "accident_count": df.get("accident_count", pd.Series(index=df.index, dtype=float)) < 0,
    }
    for mask in rules.values():
        bad |= mask.fillna(False)
    return bad


def validate_and_quarantine(df, name, config):
    invalid = range_invalid_mask(df) if name == "traffic" else pd.Series(False, index=df.index)
    valid = df.loc[~invalid].copy()
    invalid_rows = df.loc[invalid].copy()

    out = Path(config["data"]["interim_dir"])
    valid.to_csv(out / f"{name}_valid.csv", index=False)
    invalid_rows.to_csv(out / f"{name}_quarantine.csv", index=False)

    return {
        "dataset": name,
        "raw_rows": len(df),
        "valid_rows": len(valid),
        "quarantined_rows": len(invalid_rows),
        "duplicates": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "invalid_timestamps": int(df["timestamp"].isna().sum()) if "timestamp" in df else 0,
    }


def run_ingestion(config=None):
    config = config or load_config()
    ensure_directories(config)
    raw = Path(config["data"]["raw_dir"])

    traffic = parse_traffic(raw / config["data"]["traffic_file"])
    weather = parse_weather(raw / config["data"]["weather_file"])
    calendar = parse_calendar(raw / config["data"]["calendar_file"])

    reports = [
        validate_and_quarantine(traffic, "traffic", config),
        validate_and_quarantine(weather, "weather", config),
        validate_and_quarantine(calendar, "calendar", config),
    ]

    return reports


if __name__ == "__main__":
    for report in run_ingestion():
        print(report)
