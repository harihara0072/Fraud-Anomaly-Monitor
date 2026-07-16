import pandas as pd


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
