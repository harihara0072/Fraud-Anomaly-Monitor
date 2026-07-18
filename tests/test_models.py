import pandas as pd
import pytest

from credit_line_review.models import MeanBaselineRegressor, fit_linear_regression, fit_random_forest


def test_mean_baseline_predicts_training_mean_for_every_row():
    X = pd.DataFrame({"x": [1, 2, 3]})
    y = pd.Series([10.0, 20.0, 30.0])
    model = MeanBaselineRegressor().fit(X, y)
    assert model.predict(X).tolist() == [20.0, 20.0, 20.0]


def test_linear_regression_recovers_exact_linear_relationship():
    X = pd.DataFrame({"x": [1, 2, 3, 4]})
    y = pd.Series([2.0, 4.0, 6.0, 8.0])
    model = fit_linear_regression(X, y)
    assert model.predict(pd.DataFrame({"x": [5]}))[0] == pytest.approx(10.0)


def test_random_forest_fits_without_error_and_predicts_right_shape():
    X = pd.DataFrame({"x": range(20), "y": range(20, 40)})
    y = pd.Series(range(100, 120), dtype=float)
    model = fit_random_forest(X, y)
    predictions = model.predict(X)
    assert len(predictions) == 20
