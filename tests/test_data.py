from pathlib import Path

import pandas as pd

from credit_line_review.data import (
    load_cards,
    load_fraud_labels,
    load_mcc_codes,
    load_transactions,
    load_users,
    parse_currency_column,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_currency_column_handles_dollar_signs_and_negatives():
    result = parse_currency_column(pd.Series(["$24295", "$-77.00", "$1,234.56"]))
    assert result.tolist() == [24295.0, -77.0, 1234.56]


def test_load_users_parses_currency_columns():
    df = load_users(FIXTURES / "sample_users.csv")
    assert df["yearly_income"].dtype == float
    assert df.loc[0, "yearly_income"] == 59696.0


def test_load_cards_parses_credit_limit():
    df = load_cards(FIXTURES / "sample_cards.csv")
    assert df["credit_limit"].dtype == float
    assert df.loc[0, "credit_limit"] == 24295.0


def test_load_transactions_parses_amount_and_date():
    df = load_transactions(FIXTURES / "sample_transactions.csv")
    assert df["amount"].dtype == float
    assert df.loc[0, "amount"] == -77.0
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_load_mcc_codes_returns_dict():
    codes = load_mcc_codes(FIXTURES / "sample_mcc_codes.json")
    assert codes["5812"] == "Eating Places and Restaurants"


def test_load_fraud_labels_returns_boolean_series_keyed_by_transaction_id():
    labels = load_fraud_labels(FIXTURES / "sample_fraud_labels.json")
    assert labels.dtype == bool
    assert labels.index.name == "transaction_id"
    assert labels.loc[10649266] == False
    assert labels.loc[99999999] == True
