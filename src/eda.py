from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import load_config, ensure_directories


def run_eda(config=None):
    config = config or load_config()
    ensure_directories(config)

    path = Path(config["data"]["processed_dir"]) / "flowcast_features.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    figdir = Path(config["reports"]["figures_dir"])

    # Volume distribution.
    plt.figure(figsize=(10, 5))
    sns.histplot(df["traffic_volume"].dropna(), bins=50)
    plt.title("Traffic Volume Distribution")
    plt.tight_layout()
    plt.savefig(figdir / "traffic_volume_distribution.png", dpi=150)
    plt.close()

    # Correlation heatmap for important numerical fields.
    cols = [
        "traffic_volume", "avg_speed", "occupancy", "travel_time",
        "rainfall", "temperature", "visibility", "road_capacity",
        "signal_timing", "accident_count"
    ]
    cols = [c for c in cols if c in df.columns]

    plt.figure(figsize=(12, 8))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Traffic and Environmental Correlations")
    plt.tight_layout()
    plt.savefig(figdir / "correlation_heatmap.png", dpi=150)
    plt.close()

    # Hourly traffic profile.
    hourly = df.groupby("hour")["traffic_volume"].mean()
    plt.figure(figsize=(10, 5))
    hourly.plot(marker="o")
    plt.title("Average Traffic Volume by Hour")
    plt.xlabel("Hour")
    plt.ylabel("Average Traffic Volume")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(figdir / "hourly_traffic_profile.png", dpi=150)
    plt.close()

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
    }


if __name__ == "__main__":
    print(run_eda())
