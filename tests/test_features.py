import pandas as pd
import pytest

from credit_line_review.features import aggregate_card_behavior, filter_credit_cards


def test_filter_credit_cards_drops_debit_rows():
    cards = pd.DataFrame({
        "id": [1, 2, 3],
        "client_id": [10, 10, 20],
        "card_type": ["Credit", "Debit", "Credit"],
        "credit_limit": [5000.0, 2000.0, 8000.0],
    })
    result = filter_credit_cards(cards)
    assert result["id"].tolist() == [1, 3]


def test_aggregate_card_behavior_computes_expected_features():
    transactions = pd.DataFrame({
        "id": [1, 2, 3],
        "card_id": [100, 100, 100],
        "date": pd.to_datetime(["2019-01-01", "2019-02-01", "2019-03-01"]),
        "amount": [100.0, 200.0, 300.0],
        "mcc": [5411, 5812, 5411],
        "merchant_state": ["ON", "ON", "QC"],
        "use_chip": ["Swipe Transaction", "Online Transaction", "Swipe Transaction"],
        "errors": [None, "Insufficient Balance", None],
    })
    behavior = aggregate_card_behavior(transactions, reference_date="2019-04-01")
    row = behavior.loc[behavior["card_id"] == 100].iloc[0]

    assert row["txn_count"] == 3
    assert row["avg_amount"] == 200.0
    assert row["distinct_mcc"] == 2
    assert row["distinct_merchant_state"] == 2
    assert row["online_share"] == pytest.approx(1 / 3)
    assert row["error_rate"] == pytest.approx(1 / 3)
    assert row["recency_days"] == 31
