from pathlib import Path
import json
import yaml
import joblib


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_directories(config):
    dirs = [
        config["data"]["interim_dir"],
        config["data"]["processed_dir"],
        config["models"]["classical_dir"],
        config["models"]["deep_learning_dir"],
        config["models"]["model_cards_dir"],
        config["reports"]["figures_dir"],
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def save_joblib(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
