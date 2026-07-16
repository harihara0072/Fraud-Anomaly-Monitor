import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


class MeanBaselineRegressor:
    """Predicts the training-set mean target for every row - the floor any real model must beat."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MeanBaselineRegressor":
        self.mean_ = y.mean()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_)


def fit_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Course: Module 5 - Training Models & Feature Selection.

    Interpretable baseline - coefficients are defensible to a credit-risk
    reviewer in a way a black-box model isn't.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def fit_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> RandomForestRegressor:
    """Course: Module 7 - Decision Trees & Ensemble Learning.

    Chosen over SVR (Module 6) for non-linear feature interactions without
    kernel tuning, and better scaling to this row count.
    """
    model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=random_state)
    model.fit(X_train, y_train)
    return model
