from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor, XGBClassifier

from src.evaluate import regression_metrics, classification_metrics
from src.utils import load_config, ensure_directories


# ============================================================
# COLUMNS TO EXCLUDE FROM MODEL FEATURES
# ============================================================

DROP_FOR_MODEL = {
    "timestamp",
    "date",
    "time",
    "road_id",
    "road_name",
    "weather_station_id",
    "station_id",
    "vehicle_type_dist",

    # Target / leakage columns
    "congestion_level",
    "accident_count",
    "accident_risk",
    "traffic_volume",
    "avg_speed",
    "occupancy",
    "travel_time",
}


# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================

def prepare_xy(df, target):
    """
    Prepare feature matrix X and target y.

    Removes target leakage columns and converts categorical
    weather/calendar columns into numeric dummy variables.
    """

    data = df.copy()

    # Remove rows where target is missing
    data = data.dropna(subset=[target])

    # Start with columns that should always be excluded
    drop = set(DROP_FOR_MODEL)

    # Keep the requested target
    drop.discard(target)

    # Remove unwanted columns
    X = data.drop(
        columns=[c for c in drop if c in data.columns]
    )

    # Target
    y = data[target].copy()

    # Convert useful categorical columns to dummy variables
    categorical_cols = [
        c for c in [
            "weather_condition",
            "holiday_name",
            "event_name"
        ]
        if c in X.columns
    ]

    if categorical_cols:
        X = pd.get_dummies(
            X,
            columns=categorical_cols,
            dummy_na=True
        )

    # Keep numeric columns only
    X = X.select_dtypes(include=[np.number]).copy()

    # Replace infinity values
    X = X.replace([np.inf, -np.inf], np.nan)

    return X, y, data.index


# ============================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

def chronological_split(X, y, test_size=0.2):
    """
    Split time-series data chronologically.

    Earlier data -> training
    Later data -> testing
    """

    n = len(X)

    cut = int(n * (1 - test_size))

    return (
        X.iloc[:cut],
        X.iloc[cut:],
        y.iloc[:cut],
        y.iloc[cut:]
    )


# ============================================================
# REGRESSION MODELS
# ============================================================

def train_regression_suite(
    df,
    target="traffic_volume",
    config=None
):

    config = config or load_config()

    ensure_directories(config)

    # Sort chronologically
    data = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # Prepare features
    X, y, _ = prepare_xy(data, target)

    # Align indices
    X, y = X.align(
        y,
        join="inner",
        axis=0
    )

    # Train-test split
    X_train, X_test, y_train, y_test = chronological_split(
        X,
        y,
        config["validation"]["test_size"]
    )

    # ========================================================
    # REGRESSION MODELS
    # ========================================================

    models = {

        "Ridge": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                Ridge(alpha=10.0)
            )
        ]),

        "RandomForest": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=250,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]),

        "XGBoost": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "model",
                XGBRegressor(
                    n_estimators=400,
                    learning_rate=0.04,
                    max_depth=6,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ])
    }

    results = []

    out = Path(
        config["models"]["classical_dir"]
    )

    # ========================================================
    # TRAIN REGRESSION MODELS
    # ========================================================

    for name, model in models.items():

        print(f"Training regression model: {name}")

        # Train
        model.fit(
            X_train,
            y_train
        )

        # Predict
        pred = model.predict(
            X_test
        )

        # Metrics
        metrics = regression_metrics(
            y_test,
            pred
        )

        metrics["Model"] = name

        results.append(metrics)

        # Save model
        joblib.dump(
            {
                "model": model,
                "features": X.columns.tolist(),
                "target": target
            },
            out / f"{name.lower()}_volume.joblib"
        )

        # Save predictions and residuals
        predictions_df = pd.DataFrame({
            "timestamp": data.loc[
                X_test.index,
                "timestamp"
            ].values,

            "actual": y_test.values,

            "predicted": pred,

            "residual": y_test.values - pred
        })

        predictions_df.to_csv(
            out / f"{name.lower()}_volume_predictions.csv",
            index=False
        )

    # ========================================================
    # SCOREBOARD
    # ========================================================

    scoreboard = pd.DataFrame(results)[
        [
            "Model",
            "MAE",
            "RMSE",
            "MAPE",
            "R2"
        ]
    ]

    scoreboard.to_csv(
        out / "regression_scoreboard.csv",
        index=False
    )

    return scoreboard


# ============================================================
# CLASSIFICATION MODELS
# ============================================================

def train_classification_suite(
    df,
    target,
    model_suffix,
    config=None
):

    config = config or load_config()
    ensure_directories(config)

    # Sort chronologically
    data = df.sort_values("timestamp").reset_index(drop=True)

    # Prepare features and target
    X, y, _ = prepare_xy(data, target)

    # Align indices
    X, y = X.align(y, join="inner", axis=0)

    print(f"\n=== CLASSIFICATION: {target} ===")
    print("Original dtype:", y.dtype)
    print("Original samples:", len(y))
    print("\nOriginal class distribution:")
    print(y.value_counts(dropna=False))

    # Remove missing target values
    valid_mask = y.notna()
    X = X.loc[valid_mask]
    y = y.loc[valid_mask].copy()

    # --------------------------------------------------------
    # ENCODE CATEGORICAL TARGET
    # --------------------------------------------------------

    label_encoder = None

    if not pd.api.types.is_numeric_dtype(y):

        print("\nEncoding categorical labels...")

        label_encoder = LabelEncoder()

        y = pd.Series(
            label_encoder.fit_transform(y.astype(str)),
            index=y.index,
            name=target
        )

        print("\nLabel mapping:")

        for i, label in enumerate(label_encoder.classes_):
            print(f"{i} -> {label}")

    else:

        y = pd.to_numeric(y, errors="coerce")

        valid_mask = y.notna()

        X = X.loc[valid_mask]
        y = y.loc[valid_mask].astype(int)

    # --------------------------------------------------------
    # VALIDATE DATA
    # --------------------------------------------------------

    print("\nAfter processing:")

    print("Samples:", len(X))
    print("Features:", X.shape[1])
    print("Number of classes:", y.nunique())
    print("Classes:", sorted(y.unique()))

    if len(X) == 0:
        raise ValueError(
            f"No samples available for classification target: {target}"
        )

    if y.nunique() < 2:
        raise ValueError(
            f"Classification requires at least 2 classes. "
            f"Found {y.nunique()}."
        )

    # --------------------------------------------------------
    # CHRONOLOGICAL SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = chronological_split(
        X,
        y,
        config["validation"]["test_size"]
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTesting class distribution:")
    print(y_test.value_counts().sort_index())

    num_classes = y.nunique()

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    models = {

        "RandomForest": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]),

        "XGBoost": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.85,
                    colsample_bytree=0.85,

                    objective=(
                        "multi:softprob"
                        if num_classes > 2
                        else "binary:logistic"
                    ),

                    eval_metric=(
                        "mlogloss"
                        if num_classes > 2
                        else "logloss"
                    ),

                    random_state=42,
                    n_jobs=-1
                )
            )
        ])
    }

    results = []

    out = Path(config["models"]["classical_dir"])

    # --------------------------------------------------------
    # TRAIN MODELS
    # --------------------------------------------------------

    for name, model in models.items():

        print(f"\nTraining classification model: {name}")

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        pred = np.asarray(pred).astype(int)

        metrics = classification_metrics(y_test, pred)

        metrics["Model"] = name

        results.append(metrics)

        # Save model
        model_data = {
            "model": model,
            "features": X.columns.tolist(),
            "target": target,
            "num_classes": int(num_classes)
        }

        # Save label mapping
        if label_encoder is not None:
            model_data["label_classes"] = label_encoder.classes_.tolist()

        joblib.dump(
            model_data,
            out / f"{name.lower()}_{model_suffix}.joblib"
        )

        # Classification report
        if label_encoder is not None:

            report = classification_report(
                y_test,
                pred,
                labels=list(range(num_classes)),
                target_names=label_encoder.classes_,
                zero_division=0
            )

        else:

            report = classification_report(
                y_test,
                pred,
                zero_division=0
            )

        print(report)

        with open(
            out / f"{name.lower()}_{model_suffix}_report.txt",
            "w",
            encoding="utf-8"
        ) as f:

            f.write(report)

    # --------------------------------------------------------
    # SAVE SCOREBOARD
    # --------------------------------------------------------

    scoreboard = pd.DataFrame(results)

    scoreboard.to_csv(
        out / f"{model_suffix}_classification_scoreboard.csv",
        index=False
    )

    return scoreboard

# ============================================================
# RUN ALL CLASSICAL MODELS
# ============================================================

def run_all_classical(config=None):

    config = config or load_config()

    # Processed feature dataset
    path = (
        Path(
            config["data"]["processed_dir"]
        )
        / "flowcast_features.csv"
    )

    print(
        f"\nLoading dataset from: {path}"
    )

    df = pd.read_csv(
        path,
        parse_dates=["timestamp"]
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    # ========================================================
    # 1. TRAFFIC VOLUME REGRESSION
    # ========================================================

    print(
        "\n=== TRAFFIC VOLUME REGRESSION ==="
    )

    volume = train_regression_suite(
        df,
        "traffic_volume",
        config
    )

    # ========================================================
    # 2. CONGESTION CLASSIFICATION
    # ========================================================

    print(
        "\n=== CONGESTION CLASSIFICATION ==="
    )

    congestion = train_classification_suite(
        df.dropna(
            subset=[
                "congestion_level"
            ]
        ),

        "congestion_level",

        "congestion",

        config
    )

    # ========================================================
    # 3. ACCIDENT RISK CLASSIFICATION
    # ========================================================

    print(
        "\n=== ACCIDENT RISK CLASSIFICATION ==="
    )

    accident = train_classification_suite(
        df,

        "accident_risk",

        "accident_risk",

        config
    )

    return (
        volume,
        congestion,
        accident
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    results = run_all_classical()

    for result in results:

        print("\n")

        print(result)