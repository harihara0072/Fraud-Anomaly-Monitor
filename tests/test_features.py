import numpy as np
import pandas as pd
import pytest

from credit_line_review.features import (
    aggregate_card_behavior,
    build_modeling_table,
    filter_credit_cards,
    group_train_test_split,
)


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


def test_build_modeling_table_joins_filters_and_log_transforms():
    cards = pd.DataFrame({
        "id": [1, 2],
        "client_id": [10, 10],
        "card_type": ["Credit", "Debit"],
        "credit_limit": [10000.0, 2000.0],
        "card_brand": ["Visa", "Visa"],
        "has_chip": ["YES", "YES"],
        "acct_open_date": ["01/2018", "01/2018"],
    })
    users = pd.DataFrame({
        "id": [10],
        "yearly_income": [60000.0],
        "total_debt": [20000.0],
        "credit_score": [700],
        "current_age": [40],
        "gender": ["Female"],
    })
    behavior = pd.DataFrame({
        "card_id": [1],
        "txn_count": [12],
        "avg_amount": [150.0],
        "online_share": [0.5],
        "error_rate": [0.0],
    })
    table = build_modeling_table(cards, users, behavior, reference_date="2019-01-01")

    assert len(table) == 1  # the Debit card is filtered out
    assert table.loc[0, "log_credit_limit"] == pytest.approx(np.log1p(10000.0))
    assert table.loc[0, "tenure_months"] == pytest.approx(12.0, abs=0.5)
    assert table.loc[0, "txn_count"] == 12
    assert table.loc[0, "is_zero_limit"] == False


def test_build_modeling_table_flags_zero_limit_rows_without_dropping_them():
    cards = pd.DataFrame({
        "id": [1, 2],
        "client_id": [10, 20],
        "card_type": ["Credit", "Credit"],
        "credit_limit": [0.0, 8000.0],
        "card_brand": ["Visa", "Visa"],
        "has_chip": ["YES", "YES"],
        "acct_open_date": ["01/2018", "01/2018"],
    })
    users = pd.DataFrame({
        "id": [10, 20],
        "yearly_income": [60000.0, 70000.0],
        "total_debt": [20000.0, 15000.0],
        "credit_score": [700, 720],
        "current_age": [40, 35],
        "gender": ["Female", "Male"],
    })
    behavior = pd.DataFrame({
        "card_id": [1, 2],
        "txn_count": [12, 20],
        "avg_amount": [150.0, 200.0],
        "online_share": [0.5, 0.3],
        "error_rate": [0.0, 0.0],
    })
    table = build_modeling_table(cards, users, behavior, reference_date="2019-01-01")

    assert len(table) == 2  # both rows kept, unlike the Debit-card filter
    zero_row = table.loc[table["id_card"] == 1].iloc[0]
    nonzero_row = table.loc[table["id_card"] == 2].iloc[0]
    assert zero_row["is_zero_limit"] == True
    assert nonzero_row["is_zero_limit"] == False


def test_group_train_test_split_keeps_client_groups_together():
    df = pd.DataFrame({
        "client_id": [1, 1, 2, 2, 3, 3, 4, 4],
        "value": range(8),
    })
    train, test = group_train_test_split(df, test_size=0.5, random_state=0)
    train_groups = set(train["client_id"])
    test_groups = set(test["client_id"])

    assert train_groups.isdisjoint(test_groups)
    assert len(train) + len(test) == len(df)
