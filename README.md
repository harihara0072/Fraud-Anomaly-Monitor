# Machine-Lararning-UofT

SCS 3253 Machine Learning term project — Credit Line Review Assistant.

Full design: [`docs/credit-line-review-assistant-proposal.md`](docs/credit-line-review-assistant-proposal.md)
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
- Notebook: `jupyter notebook notebooks/01_credit_line_review.ipynb`
- Dashboard (optional, after the notebook exports `dashboard_data/scored_accounts.parquet`): `streamlit run dashboard/app.py`