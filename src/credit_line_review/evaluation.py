import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold


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


def permutation_feature_importance(
    model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 10, random_state: int = 42
) -> pd.DataFrame:
    """Drop in each feature's values (shuffled) and measure how much R2 falls - a
    second, model-agnostic importance measure to sanity-check SHAP's ranking
    against, since SHAP is specific to how a given model was built internally.
    """
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state, scoring="r2"
    )
    importance = pd.Series(result.importances_mean, index=X.columns, name="mean_r2_drop")
    return importance.sort_values(ascending=False).reset_index().rename(columns={"index": "feature"})


def grouped_cv_metrics(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    group_col: str,
    model_fn,
    n_splits: int = 5,
    transform=np.expm1,
) -> dict:
    """Grouped k-fold CV: refit model_fn on each fold's training split, grouped by
    group_col so one group never spans train and test, and report mean +/- std
    of each regression metric across folds instead of a single train/test split's
    point estimate.
    """
    X = df[feature_cols].fillna(0)
    y = df[target_col]
    groups = df[group_col]

    fold_metrics = []
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups=groups):
        model = model_fn(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        actual = y.iloc[test_idx]
        if transform is not None:
            pred = transform(pred)
            actual = transform(actual)
        fold_metrics.append(regression_metrics(actual, pred))

    metrics_df = pd.DataFrame(fold_metrics)
    summary = {}
    for col in metrics_df.columns:
        summary[f"{col}_mean"] = metrics_df[col].mean()
        summary[f"{col}_std"] = metrics_df[col].std()
    return summary


def bootstrap_r2_confidence_interval(
    y_true, y_pred, n_bootstrap: int = 1000, confidence: float = 0.95, random_state: int = 42
) -> dict:
    """Resample the held-out test set with replacement to build a confidence
    interval around R2, showing whether a model's advantage over baseline
    holds up across resamples rather than being an artifact of one split.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    scores = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores[i] = r2_score(y_true[idx], y_pred[idx])

    alpha = (1 - confidence) / 2
    return {
        "r2_mean": float(scores.mean()),
        "r2_lower": float(np.quantile(scores, alpha)),
        "r2_upper": float(np.quantile(scores, 1 - alpha)),
    }
