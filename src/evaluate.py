import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    mape = (
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        if mask.any() else np.nan
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "R2": float(r2),
    }


def classification_metrics(y_true, y_pred):
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "Recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "F1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "ConfusionMatrix": confusion_matrix(y_true, y_pred).tolist(),
    }
