import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }


def residuals(y_true, y_pred) -> np.ndarray:
    """Beyond the course: residual diagnostics (this + the Q-Q/histogram plots
    built from it in Task 10) check whether the linear model's assumptions
    actually hold - not covered in the course modules, but required to
    validate the regression rather than just report R²."""
    return np.asarray(y_true) - np.asarray(y_pred)


def shap_feature_importance(model, X: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute SHAP value per feature - which inputs move the prediction most, and how much.

    Beyond the course: goes past sklearn's built-in .feature_importances_ to
    give per-account explanations, tying the model to the real adverse-action
    explainability requirement in lending regulation - the project's main
    novelty lever.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    importance = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns, name="mean_abs_shap")
    return importance.sort_values(ascending=False).reset_index().rename(columns={"index": "feature"})
