from pathlib import Path
import pandas as pd

from src.utils import load_config, ensure_directories
from src.ingest import run_ingestion
from src.clean import run_cleaning
from src.features import build_features
from src.eda import run_eda
from src.ml_models import run_all_classical
from src.dl_model import train_lstm


def main():
    config = load_config()
    ensure_directories(config)

    print("=== M1: INGESTION & VALIDATION ===")
    for report in run_ingestion(config):
        print(report)

    print("\n=== M2: CLEANING & MERGING ===")
    run_cleaning(config)

    print("\n=== M3: FEATURE ENGINEERING ===")
    processed = Path(config["data"]["processed_dir"])
    df = pd.read_csv(
        processed / "flowcast_processed.csv",
        parse_dates=["timestamp"]
    )
    df = build_features(df, config)
    df.to_csv(processed / "flowcast_features.csv", index=False)
    print("Feature dataset:", df.shape)

    print("\n=== M4: EDA & REPORTING ===")
    print(run_eda(config))

    print("\n=== M5: CLASSICAL ML ===")
    classical = run_all_classical(config)
    for result in classical:
        print(result)

    print("\n=== M6: DEEP LEARNING ===")
    lstm = train_lstm(config)
    print(lstm)

    print("\n=== FLOWCAST PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
