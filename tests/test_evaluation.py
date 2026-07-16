import numpy as np
import pandas as pd

from credit_line_review.evaluation import regression_metrics, residuals, shap_feature_importance
from credit_line_review.models import fit_random_forest


def test_regression_metrics_perfect_prediction_is_all_zero_error():
    y_true = np.array([100.0, 200.0, 300.0])
    metrics = regression_metrics(y_true, y_true)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0


def test_residuals_computes_true_minus_pred():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([90.0, 210.0])
    assert residuals(y_true, y_pred).tolist() == [10.0, -10.0]


def test_shap_feature_importance_ranks_the_dominant_feature_first():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "dominant": rng.normal(size=200),
        "noise": rng.normal(size=200) * 0.01,
    })
    y = X["dominant"] * 100
    model = fit_random_forest(X, y)
    importance = shap_feature_importance(model, X)
    assert importance.iloc[0]["feature"] == "dominant"
