import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def filter_credit_cards(cards: pd.DataFrame) -> pd.DataFrame:
    """Keep only card_type == 'Credit' rows.

    cards_data.csv assigns a non-null credit_limit to Debit cards too, which
    isn't a meaningful business concept for a debit card - restricting to
    Credit avoids conflating two different card types in the regression target.
    """
    return cards[cards["card_type"] == "Credit"].reset_index(drop=True)


def aggregate_card_behavior(transactions: pd.DataFrame, reference_date: str) -> pd.DataFrame:
    """Aggregate per-transaction rows into one row per card_id of behavioral features.

    Beyond the course: this is RFM (recency/frequency/monetary) feature
    construction, a marketing-analytics technique, not ML curriculum - but
    it's how row-level transactions become card-level model inputs.
    """
    ref = pd.Timestamp(reference_date)
    behavior = transactions.groupby("card_id").agg(
        txn_count=("id", "count"),
        avg_amount=("amount", "mean"),
        median_amount=("amount", "median"),
        std_amount=("amount", "std"),
        distinct_mcc=("mcc", "nunique"),
        distinct_merchant_state=("merchant_state", "nunique"),
        last_txn_date=("date", "max"),
        first_txn_date=("date", "min"),
    ).reset_index()

    behavior["spend_volatility"] = (
        behavior["std_amount"].fillna(0) / behavior["avg_amount"].replace(0, pd.NA)
    ).fillna(0)
    behavior["recency_days"] = (ref - behavior["last_txn_date"]).dt.days
    active_days = (behavior["last_txn_date"] - behavior["first_txn_date"]).dt.days.clip(lower=1)
    behavior["txn_frequency_per_month"] = behavior["txn_count"] / (active_days / 30)

    online_share = (
        transactions.assign(is_online=lambda d: d["use_chip"] == "Online Transaction")
        .groupby("card_id")["is_online"].mean()
        .rename("online_share")
        .reset_index()
    )
    error_rate = (
        transactions.assign(has_error=lambda d: d["errors"].notna())
        .groupby("card_id")["has_error"].mean()
        .rename("error_rate")
        .reset_index()
    )

    behavior = behavior.merge(online_share, on="card_id").merge(error_rate, on="card_id")
    return behavior.drop(columns=["std_amount", "last_txn_date", "first_txn_date"])


def build_modeling_table(
    cards: pd.DataFrame, users: pd.DataFrame, behavior: pd.DataFrame, reference_date: str
) -> pd.DataFrame:
    """Join filtered credit cards + users + behavioral features into one row-per-card modeling table."""
    credit_cards = filter_credit_cards(cards)
    ref = pd.Timestamp(reference_date)
    tenure_months = (
        ref - pd.to_datetime(credit_cards["acct_open_date"], format="%m/%Y")
    ).dt.days / 30.44
    table = credit_cards.assign(tenure_months=tenure_months).merge(
        users, left_on="client_id", right_on="id", suffixes=("_card", "_user")
    )
    table = table.merge(behavior, left_on="id_card", right_on="card_id", how="left").drop(columns=["card_id"])

    # Beyond the course: log1p-transform skewed monetary columns. Financial
    # quantities are realistically log-normal (confirmed in the Task 2 EDA
    # skew check) - untransformed, they'd violate the linear model's
    # roughly-normal-residuals assumption.
    for col in ("yearly_income", "total_debt", "credit_limit"):
        table[f"log_{col}"] = np.log1p(table[col].clip(lower=0))

    # A small slice of Credit-card rows have credit_limit == 0 (confirmed in
    # the Task 2 EDA - these are almost certainly closed/frozen accounts, not
    # a real "what should their limit be" case). They stay in the table for
    # reporting but get flagged here so the notebook can exclude them from
    # the train/test split instead of letting them distort the regression.
    table["is_zero_limit"] = table["credit_limit"] == 0

    return pd.get_dummies(table, columns=["card_brand", "has_chip", "gender"], drop_first=True)


def group_train_test_split(
    df: pd.DataFrame, group_col: str = "client_id", test_size: float = 0.2, random_state: int = 42
):
    """Split df into train/test, keeping all rows for a given group_col value on the same side.

    Beyond the course: standard train/test splitting assumes i.i.d. rows.
    One client can own several cards here, so a plain random split would
    leak client identity across train and test - GroupShuffleSplit avoids that.
    """
    splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df[group_col]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
