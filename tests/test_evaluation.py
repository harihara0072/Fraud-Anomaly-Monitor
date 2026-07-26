import numpy as np
import pandas as pd

from credit_line_review.evaluation import (
    bootstrap_r2_confidence_interval,
    grouped_cv_metrics,
    permutation_feature_importance,
    regression_metrics,
    residuals,
    shap_feature_importance,
)
from credit_line_review.models import fit_linear_regression, fit_random_forest


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


def test_permutation_feature_importance_ranks_the_dominant_feature_first():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "dominant": rng.normal(size=200),
        "noise": rng.normal(size=200) * 0.01,
    })
    y = X["dominant"] * 100
    model = fit_random_forest(X, y)
    importance = permutation_feature_importance(model, X, y, n_repeats=5)
    assert importance.iloc[0]["feature"] == "dominant"


def test_grouped_cv_metrics_returns_mean_and_std_per_metric():
    rng = np.random.default_rng(0)
    n_groups = 20
    df = pd.DataFrame({
        "client_id": np.repeat(np.arange(n_groups), 5),
        "x": rng.normal(size=n_groups * 5),
    })
    df["y"] = df["x"] * 2 + rng.normal(scale=0.01, size=n_groups * 5)

    result = grouped_cv_metrics(
        df,
        feature_cols=["x"],
        target_col="y",
        group_col="client_id",
        model_fn=fit_linear_regression,
        n_splits=4,
        transform=None,
    )

    for key in (
        "r2_mean", "r2_std", "mae_mean", "mae_std",
        "rmse_mean", "rmse_std", "mape_mean", "mape_std",
    ):
        assert key in result
    assert result["r2_mean"] > 0.9


def test_bootstrap_r2_confidence_interval_brackets_the_point_estimate():
    y_true = np.array([100.0, 200.0, 300.0, 400.0, 500.0] * 20)
    y_pred = y_true + np.array([5.0, -5.0, 10.0, -10.0, 0.0] * 20)

    ci = bootstrap_r2_confidence_interval(y_true, y_pred, n_bootstrap=200)

    assert ci["r2_lower"] <= ci["r2_mean"] <= ci["r2_upper"]
