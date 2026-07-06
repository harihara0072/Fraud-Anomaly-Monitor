# Machine-Lararning-UofT

SCS 3253 Machine Learning term project — Fraud Anomaly Monitor.

Full design: [`docs/fraud-anomaly-monitor-proposal.md`](docs/fraud-anomaly-monitor-proposal.md)
Assignment brief: `Course content/SCS 3253 Term Project Machine Learning.pdf`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the dataset under `Financial Transactions Dataset/` (gitignored — see the proposal doc for the expected files).

## Running things

- Tests: `pytest`
- Notebook: `jupyter notebook notebooks/01_fraud_anomaly_detection.ipynb`
- Dashboard (optional, after the notebook exports `dashboard_data/scored_transactions.parquet`): `streamlit run dashboard/app.py`