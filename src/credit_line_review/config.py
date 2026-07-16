from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Financial Transactions Dataset"
DASHBOARD_DATA_DIR = PROJECT_ROOT / "dashboard_data"

USERS_CSV = DATA_DIR / "users_data.csv"
CARDS_CSV = DATA_DIR / "cards_data.csv"
TRANSACTIONS_CSV = DATA_DIR / "transactions_data.csv"
MCC_CODES_JSON = DATA_DIR / "mcc_codes.json"
FRAUD_LABELS_JSON = DATA_DIR / "train_fraud_labels.json"

SCORED_TRANSACTIONS_PARQUET = DASHBOARD_DATA_DIR / "scored_transactions.parquet"

# transactions_data.csv spans 2010-01-01 to 2019-10-31 (confirmed via full-file scan);
# use the day after the last transaction as a fixed anchor so recency/tenure
# features are reproducible across notebook re-runs.
REFERENCE_DATE = "2019-11-01"
