import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import streamlit as st

from credit_line_review.config import SCORED_TRANSACTIONS_PARQUET

st.set_page_config(page_title="Fraud Anomaly Monitor", layout="wide")


def load_scored_transactions() -> "pd.DataFrame | None":
    if not SCORED_TRANSACTIONS_PARQUET.exists():
        return None
    return pd.read_parquet(SCORED_TRANSACTIONS_PARQUET)


def render_missing_data_notice() -> None:
    st.info(
        "No scored transactions found yet. Run the notebook "
        "(`notebooks/01_fraud_anomaly_detection.ipynb`) through the Model "
        f"Evaluation section to generate `{SCORED_TRANSACTIONS_PARQUET.relative_to(SCORED_TRANSACTIONS_PARQUET.parents[1])}`."
    )


def render_model_comparison_overview(data: "pd.DataFrame | None") -> None:
    st.header("Model Comparison Overview")
    if data is None:
        render_missing_data_notice()
        return
    st.dataframe(data.describe())


def render_threshold_explorer(data: "pd.DataFrame | None") -> None:
    st.header("Threshold Explorer")
    if data is None:
        render_missing_data_notice()
        return
    st.slider("Anomaly threshold", 0.0, 1.0, 0.5)


def render_anomaly_patterns(data: "pd.DataFrame | None") -> None:
    st.header("Anomaly Patterns")
    if data is None:
        render_missing_data_notice()
        return
    st.dataframe(data.head())


def render_case_drilldown(data: "pd.DataFrame | None") -> None:
    st.header("Case Drill-Down")
    if data is None:
        render_missing_data_notice()
        return
    st.dataframe(data)


PAGES = {
    "Model Comparison Overview": render_model_comparison_overview,
    "Threshold Explorer": render_threshold_explorer,
    "Anomaly Patterns": render_anomaly_patterns,
    "Case Drill-Down": render_case_drilldown,
}


def main() -> None:
    st.sidebar.title("Fraud Anomaly Monitor")
    page = st.sidebar.radio("Page", list(PAGES.keys()))
    data = load_scored_transactions()
    PAGES[page](data)


main()
