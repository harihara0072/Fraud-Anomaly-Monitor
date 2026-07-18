import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import plotly.express as px
import streamlit as st

from credit_line_review.config import SCORED_ACCOUNTS_PARQUET

st.set_page_config(page_title="Credit Line Review Assistant", layout="wide")


def load_scored_accounts() -> "pd.DataFrame | None":
    if not SCORED_ACCOUNTS_PARQUET.exists():
        return None
    return pd.read_parquet(SCORED_ACCOUNTS_PARQUET)


def render_missing_data_notice() -> None:
    st.info(
        "No scored accounts found yet. Run the notebook "
        "(`notebooks/01_credit_line_review.ipynb`) through the Model "
        f"Evaluation section to generate `{SCORED_ACCOUNTS_PARQUET.relative_to(SCORED_ACCOUNTS_PARQUET.parents[1])}`."
    )


def render_portfolio_overview(data: "pd.DataFrame | None") -> None:
    st.header("Portfolio Overview")
    if data is None:
        render_missing_data_notice()
        return
    st.dataframe(data.groupby("flag")["residual"].describe())
    modeled = data[~data["is_zero_limit"]]
    fig = px.scatter(modeled, x="credit_limit", y="predicted_limit", color="flag")
    fig.add_shape(
        type="line", x0=0, y0=0, x1=modeled["credit_limit"].max(), y1=modeled["credit_limit"].max()
    )
    st.plotly_chart(fig, width="stretch")


def render_miscalibration_explorer(data: "pd.DataFrame | None") -> None:
    st.header("Limit Miscalibration Explorer")
    if data is None:
        render_missing_data_notice()
        return
    # Closed/zero-limit accounts aren't real miscalibration candidates -
    # exclude them from this explorer (they get their own segment below).
    modeled = data[~data["is_zero_limit"]]
    max_residual = int(modeled["residual"].abs().max())
    threshold = st.slider("Residual threshold ($)", 0, max_residual, min(500, max_residual))
    flagged = modeled[modeled["residual"].abs() > threshold]
    st.metric("Accounts flagged", f"{len(flagged)} ({len(flagged) / len(modeled):.1%})")
    st.dataframe(flagged)

    n_closed = int(data["is_zero_limit"].sum())
    if n_closed:
        st.caption(
            f"{n_closed} additional account(s) have a $0 credit limit "
            "(closed/frozen) and are excluded here — see Segment / Persona View."
        )


def render_segment_view(data: "pd.DataFrame | None") -> None:
    st.header("Segment / Persona View")
    if data is None:
        render_missing_data_notice()
        return
    st.dataframe(data.groupby("cluster")[["credit_limit", "predicted_limit", "residual"]].mean())
    st.subheader("Closed / zero-limit accounts")
    st.dataframe(data[data["is_zero_limit"]])


def render_case_drilldown(data: "pd.DataFrame | None") -> None:
    st.header("Case Drill-Down")
    if data is None:
        render_missing_data_notice()
        return
    account_id = st.selectbox("Account (card id)", data["id_card"])
    st.dataframe(data[data["id_card"] == account_id])


PAGES = {
    "Portfolio Overview": render_portfolio_overview,
    "Limit Miscalibration Explorer": render_miscalibration_explorer,
    "Segment / Persona View": render_segment_view,
    "Case Drill-Down": render_case_drilldown,
}


def main() -> None:
    st.sidebar.title("Credit Line Review Assistant")
    page = st.sidebar.radio("Page", list(PAGES.keys()))
    data = load_scored_accounts()
    PAGES[page](data)


main()
