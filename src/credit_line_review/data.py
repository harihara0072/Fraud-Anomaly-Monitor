import json
from pathlib import Path

import pandas as pd

from fraud_anomaly.config import (
    CARDS_CSV,
    FRAUD_LABELS_JSON,
    MCC_CODES_JSON,
    TRANSACTIONS_CSV,
    USERS_CSV,
)


def parse_currency_column(series: pd.Series) -> pd.Series:
    """Convert a '$1,234.56'-style string column to float."""
    return (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )


def load_users(path: Path = USERS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("per_capita_income", "yearly_income", "total_debt"):
        df[col] = parse_currency_column(df[col])
    return df


def load_cards(path: Path = CARDS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["credit_limit"] = parse_currency_column(df["credit_limit"])
    return df


def load_transactions(path: Path = TRANSACTIONS_CSV, nrows: "int | None" = None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=nrows, parse_dates=["date"])
    df["amount"] = parse_currency_column(df["amount"])
    return df


def load_mcc_codes(path: Path = MCC_CODES_JSON) -> dict:
    with open(path) as f:
        return json.load(f)


def load_fraud_labels(path: Path = FRAUD_LABELS_JSON) -> pd.Series:
    with open(path) as f:
        raw = json.load(f)
    labels = pd.Series(raw["target"], name="is_fraud")
    labels.index = labels.index.astype(int)
    labels.index.name = "transaction_id"
    return labels == "Yes"
