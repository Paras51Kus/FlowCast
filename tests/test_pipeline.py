import pandas as pd
from src.features import add_time_features
from src.evaluate import regression_metrics, classification_metrics


def test_time_features():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-01 07:30"])
    })
    out = add_time_features(df)
    assert out.loc[0, "hour"] == 7
    assert out.loc[0, "is_morning_peak"] == 1


def test_regression_metrics():
    result = regression_metrics([100, 200], [110, 190])
    assert "MAE" in result
    assert "RMSE" in result
    assert "MAPE" in result
    assert "R2" in result


def test_classification_metrics():
    result = classification_metrics(
        ["Free-flow", "Heavy"],
        ["Free-flow", "Heavy"]
    )
    assert result["Accuracy"] == 1.0
