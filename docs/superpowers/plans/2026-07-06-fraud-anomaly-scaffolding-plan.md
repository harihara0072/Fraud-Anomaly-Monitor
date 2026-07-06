# Fraud Anomaly Monitor — Project Scaffolding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the initial project skeleton for the SCS 3253 term project (design in `docs/fraud-anomaly-monitor-proposal.md`) — directory structure, dependency management, a data-loading module with parsing for the dataset's dollar-string columns, a notebook skeleton matching the required report outline, and a Streamlit dashboard skeleton — so modeling work (Isolation Forest, One-Class SVM, autoencoder) can start directly without any setup friction.

**Architecture:** A `src/fraud_anomaly` Python package holds reusable data-loading/parsing code (imported by both the notebook and the dashboard, avoiding duplicated logic). `notebooks/` holds the graded Jupyter notebook. `dashboard/` holds the optional Streamlit app, which reads a small precomputed Parquet file rather than the raw 1.25GB transactions CSV. `tests/` covers the data module against small synthetic fixtures (not the real multi-GB files).

**Tech Stack:** Python 3.11, pandas, scikit-learn, TensorFlow/Keras, Streamlit, Plotly, PyArrow, pytest.

## Global Constraints

- The raw dataset (`Financial Transactions Dataset/`) must never be committed to git — `transactions_data.csv` is ~1.25GB and `train_fraud_labels.json` is ~159MB.
- Currency columns in the raw CSVs are strings like `"$24295"` or `"$-77.00"` — every loader must convert these to floats.
- Fixtures used in tests are small synthetic samples, never the real data files, so `pytest` runs fast and works even without the dataset present.
- Model training itself (Isolation Forest / One-Class SVM / autoencoder) is out of scope for this plan — this plan only produces the skeleton those models will be written into.

---

## File Structure

```
Machine-Lararning-UofT/
├── .gitignore                                  (new)
├── requirements.txt                            (new)
├── conftest.py                                 (new — makes `fraud_anomaly` importable in tests)
├── src/
│   └── fraud_anomaly/
│       ├── __init__.py                         (new)
│       ├── config.py                           (new — path constants)
│       └── data.py                             (new — loaders + currency parsing)
├── tests/
│   ├── fixtures/
│   │   ├── sample_users.csv                    (new)
│   │   ├── sample_cards.csv                    (new)
│   │   ├── sample_transactions.csv             (new)
│   │   ├── sample_mcc_codes.json                (new)
│   │   └── sample_fraud_labels.json            (new)
│   └── test_data.py                            (new)
├── notebooks/
│   └── 01_fraud_anomaly_detection.ipynb        (new — skeleton matching report outline)
├── dashboard/
│   └── app.py                                  (new — 4-page Streamlit skeleton)
├── dashboard_data/
│   └── .gitkeep                                (new — generated parquet goes here, gitignored)
└── README.md                                   (modified — project overview + setup instructions)
```

---

### Task 1: Dependency management, .gitignore, and directory scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `dashboard_data/.gitkeep`

**Interfaces:**
- Produces: a working Python environment with all libraries later tasks depend on (pandas, scikit-learn, tensorflow, streamlit, plotly, pyarrow, pytest, jupyter, nbformat).

- [ ] **Step 1: Create `requirements.txt`**

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
tensorflow>=2.15
streamlit>=1.32
plotly>=5.18
pyarrow>=14.0
jupyter>=1.0
nbformat>=5.9
pytest>=7.4
```

- [ ] **Step 2: Create `.gitignore`**

```
# Raw dataset — too large for git (transactions_data.csv ~1.25GB, train_fraud_labels.json ~159MB)
Financial Transactions Dataset/

# Generated dashboard data
dashboard_data/*
!dashboard_data/.gitkeep

# Python
__pycache__/
*.pyc
.venv/
venv/
*.egg-info/

# Jupyter
.ipynb_checkpoints/

# Editor/OS
.idea/
.DS_Store
```

- [ ] **Step 3: Create the `dashboard_data/` directory with a placeholder**

```bash
mkdir -p dashboard_data
touch dashboard_data/.gitkeep
```

- [ ] **Step 4: Create a virtual environment and install dependencies**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- [ ] **Step 5: Verify the environment**

Run: `python -c "import pandas, sklearn, tensorflow, streamlit, plotly, pyarrow, pytest; print('ok')"`
Expected: `ok` printed, no `ImportError`

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .gitignore dashboard_data/.gitkeep
git commit -m "chore: scaffold project dependencies and gitignore"
```

---

### Task 2: Data loading module with currency parsing (TDD)

**Files:**
- Create: `conftest.py`
- Create: `src/fraud_anomaly/__init__.py`
- Create: `src/fraud_anomaly/config.py`
- Create: `src/fraud_anomaly/data.py`
- Create: `tests/fixtures/sample_users.csv`
- Create: `tests/fixtures/sample_cards.csv`
- Create: `tests/fixtures/sample_transactions.csv`
- Create: `tests/fixtures/sample_mcc_codes.json`
- Create: `tests/fixtures/sample_fraud_labels.json`
- Create: `tests/test_data.py`

**Interfaces:**
- Consumes: nothing (first code in the package)
- Produces:
  - `fraud_anomaly.data.parse_currency_column(series: pd.Series) -> pd.Series`
  - `fraud_anomaly.data.load_users(path: Path = USERS_CSV) -> pd.DataFrame`
  - `fraud_anomaly.data.load_cards(path: Path = CARDS_CSV) -> pd.DataFrame`
  - `fraud_anomaly.data.load_transactions(path: Path = TRANSACTIONS_CSV, nrows: int | None = None) -> pd.DataFrame`
  - `fraud_anomaly.data.load_mcc_codes(path: Path = MCC_CODES_JSON) -> dict[str, str]`
  - `fraud_anomaly.data.load_fraud_labels(path: Path = FRAUD_LABELS_JSON) -> pd.Series` (boolean, indexed by int `transaction_id`)
  - `fraud_anomaly.config.{DATA_DIR, USERS_CSV, CARDS_CSV, TRANSACTIONS_CSV, MCC_CODES_JSON, FRAUD_LABELS_JSON}` — `Path` constants later tasks (notebook, dashboard) import directly.

- [ ] **Step 1: Create `conftest.py` at the repo root so `fraud_anomaly` is importable in tests**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
```

- [ ] **Step 2: Create `src/fraud_anomaly/__init__.py`**

```python
```

- [ ] **Step 3: Create `src/fraud_anomaly/config.py`**

```python
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
```

- [ ] **Step 4: Create test fixtures**

Create `tests/fixtures/sample_users.csv`:

```
id,current_age,retirement_age,birth_year,birth_month,gender,address,latitude,longitude,per_capita_income,yearly_income,total_debt,credit_score,num_credit_cards
825,53,66,1966,11,Female,462 Rose Lane,34.15,-117.76,$29278,$59696,$127613,787,5
1746,53,68,1966,12,Female,3606 Federal Boulevard,40.76,-73.74,$37891,$77254,$191349,701,5
```

Create `tests/fixtures/sample_cards.csv`:

```
id,client_id,card_brand,card_type,card_number,expires,cvv,has_chip,num_cards_issued,credit_limit,acct_open_date,year_pin_last_changed,card_on_dark_web
4524,825,Visa,Debit,4344676511950444,12/2022,623,YES,2,$24295,09/2002,2008,No
2731,825,Visa,Debit,4956965974959986,12/2020,393,YES,2,$21968,04/2014,2014,No
```

Create `tests/fixtures/sample_transactions.csv`:

```
id,date,client_id,card_id,amount,use_chip,merchant_id,merchant_city,merchant_state,zip,mcc,errors
7475327,2010-01-01 00:01:00,1556,2972,$-77.00,Swipe Transaction,59935,Beulah,ND,58523.0,5499,
7475328,2010-01-01 00:02:00,561,4575,$14.57,Swipe Transaction,67570,Bettendorf,IA,52722.0,5311,
```

Create `tests/fixtures/sample_mcc_codes.json`:

```json
{
    "5812": "Eating Places and Restaurants",
    "5541": "Service Stations",
    "5311": "Department Stores"
}
```

Create `tests/fixtures/sample_fraud_labels.json`:

```json
{"target": {"10649266": "No", "23410063": "No", "99999999": "Yes"}}
```

- [ ] **Step 5: Write the failing tests in `tests/test_data.py`**

```python
from pathlib import Path

import pandas as pd

from fraud_anomaly.data import (
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
```

- [ ] **Step 6: Run the tests to verify they fail**

Run: `pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fraud_anomaly.data'` (or `ImportError`)

- [ ] **Step 7: Implement `src/fraud_anomaly/data.py`**

```python
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
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `pytest tests/test_data.py -v`
Expected: `6 passed`

- [ ] **Step 9: Commit**

```bash
git add conftest.py src/fraud_anomaly/__init__.py src/fraud_anomaly/config.py src/fraud_anomaly/data.py tests/fixtures tests/test_data.py
git commit -m "feat: add data loading module with currency/date parsing"
```

---

### Task 3: Notebook skeleton matching the required report outline

**Files:**
- Create: `scripts/generate_notebook_skeleton.py`
- Create (generated by the script, then committed): `notebooks/01_fraud_anomaly_detection.ipynb`

**Interfaces:**
- Consumes: `fraud_anomaly.config.{USERS_CSV, CARDS_CSV, TRANSACTIONS_CSV, MCC_CODES_JSON, FRAUD_LABELS_JSON}`, `fraud_anomaly.data.{load_users, load_cards, load_transactions, load_mcc_codes, load_fraud_labels}` (from Task 2)
- Produces: `notebooks/01_fraud_anomaly_detection.ipynb` — the skeleton later modeling work (Isolation Forest / One-Class SVM / autoencoder) gets added into, with section headers matching `docs/fraud-anomaly-monitor-proposal.md` Section 4.

- [ ] **Step 1: Create `scripts/generate_notebook_skeleton.py`**

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()

nb["cells"] = [
    nbf.v4.new_markdown_cell(
        "# Fraud Anomaly Monitor — SCS 3253 Term Project\n"
        "\n"
        "Report outline this notebook mirrors: Objective -> Data Preparation -> "
        "Model Design -> Model Evaluation -> Conclusions.\n"
        "See `docs/fraud-anomaly-monitor-proposal.md` for the full design."
    ),
    nbf.v4.new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "sys.path.insert(0, str(Path.cwd().parent / \"src\"))\n"
        "\n"
        "import pandas as pd\n"
        "\n"
        "from fraud_anomaly.data import (\n"
        "    load_cards,\n"
        "    load_fraud_labels,\n"
        "    load_mcc_codes,\n"
        "    load_transactions,\n"
        "    load_users,\n"
        ")"
    ),
    nbf.v4.new_markdown_cell(
        "## 1. Objective\n"
        "\n"
        "We detect anomalous (potentially fraudulent) transactions using "
        "unsupervised anomaly detection, trained without fraud labels, then "
        "evaluate the anomaly signal against held-out ground-truth fraud labels."
    ),
    nbf.v4.new_markdown_cell(
        "## 2. Data Preparation\n"
        "\n"
        "Loading a small sample for the skeleton check below "
        "(`nrows=5000`); the full-scale load happens once modeling starts."
    ),
    nbf.v4.new_code_cell(
        "users = load_users()\n"
        "cards = load_cards()\n"
        "transactions = load_transactions(nrows=5000)\n"
        "mcc_codes = load_mcc_codes()\n"
        "fraud_labels = load_fraud_labels()\n"
        "\n"
        "print(\"users:\", users.shape)\n"
        "print(\"cards:\", cards.shape)\n"
        "print(\"transactions (sample):\", transactions.shape)\n"
        "print(\"mcc codes:\", len(mcc_codes))\n"
        "print(\"fraud labels:\", fraud_labels.shape, \"| fraud rate:\", fraud_labels.mean())"
    ),
    nbf.v4.new_markdown_cell(
        "## 3. Model Design\n"
        "\n"
        "### 3.1 Isolation Forest\n"
        "\n"
        "### 3.2 One-Class SVM\n"
        "\n"
        "### 3.3 Autoencoder"
    ),
    nbf.v4.new_markdown_cell("## 4. Model Evaluation"),
    nbf.v4.new_markdown_cell("## 5. Conclusions"),
]

nbf.write(nb, "notebooks/01_fraud_anomaly_detection.ipynb")
print("wrote notebooks/01_fraud_anomaly_detection.ipynb")
```

- [ ] **Step 2: Run the generator script**

```bash
mkdir -p notebooks
python scripts/generate_notebook_skeleton.py
```

Expected: `wrote notebooks/01_fraud_anomaly_detection.ipynb`

- [ ] **Step 3: Verify the notebook executes top-to-bottom without error**

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_fraud_anomaly_detection.ipynb
```

Expected: exit code `0`, no `Error` in output. (Requires the real dataset files under `Financial Transactions Dataset/` — the loaders read `nrows=5000` from `transactions_data.csv` but the full `users_data.csv`/`cards_data.csv`/label files, so this must be run on a machine with the dataset present.)

- [ ] **Step 4: Commit**

```bash
git add scripts/generate_notebook_skeleton.py notebooks/01_fraud_anomaly_detection.ipynb
git commit -m "feat: add notebook skeleton matching report outline"
```

---

### Task 4: Streamlit dashboard skeleton

**Files:**
- Create: `dashboard/app.py`

**Interfaces:**
- Consumes: `fraud_anomaly.config.SCORED_TRANSACTIONS_PARQUET` (from Task 2)
- Produces: a runnable `streamlit run dashboard/app.py` entry point with the 4 pages from `docs/fraud-anomaly-monitor-proposal.md` Section 5.2, each showing a "no data yet" placeholder until the notebook exports `dashboard_data/scored_transactions.parquet`.

- [ ] **Step 1: Create `dashboard/app.py`**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import streamlit as st

from fraud_anomaly.config import SCORED_TRANSACTIONS_PARQUET

st.set_page_config(page_title="Fraud Anomaly Monitor", layout="wide")


def load_scored_transactions() -> "pd.DataFrame | None":
    if not SCORED_TRANSACTIONS_PARQUET.exists():
        return None
    return pd.read_parquet(SCORED_TRANSACTIONS_PARQUET)


def render_missing_data_notice() -> None:
    st.info(
        "No scored transactions found yet. Run the notebook "
        "(`notebooks/01_fraud_anomaly_detection.ipynb`) through the Model "
        f"Evaluation section to generate `{SCORED_TRANSACTIONS_PARQUET.relative_to(SCORED_TRANSACTIONS_PARQUET.parents[1])}`."
    )


def render_model_comparison_overview(data: "pd.DataFrame | None") -> None:
    st.header("Model Comparison Overview")
    if data is None:
        render_missing_data_notice()
        return
    st.dataframe(data.describe())


def render_threshold_explorer(data: "pd.DataFrame | None") -> None:
    st.header("Threshold Explorer")
    if data is None:
        render_missing_data_notice()
        return
    st.slider("Anomaly threshold", 0.0, 1.0, 0.5)


def render_anomaly_patterns(data: "pd.DataFrame | None") -> None:
    st.header("Anomaly Patterns")
    if data is None:
        render_missing_data_notice()
        return
    st.dataframe(data.head())


def render_case_drilldown(data: "pd.DataFrame | None") -> None:
    st.header("Case Drill-Down")
    if data is None:
        render_missing_data_notice()
        return
    st.dataframe(data)


PAGES = {
    "Model Comparison Overview": render_model_comparison_overview,
    "Threshold Explorer": render_threshold_explorer,
    "Anomaly Patterns": render_anomaly_patterns,
    "Case Drill-Down": render_case_drilldown,
}


def main() -> None:
    st.sidebar.title("Fraud Anomaly Monitor")
    page = st.sidebar.radio("Page", list(PAGES.keys()))
    data = load_scored_transactions()
    PAGES[page](data)


main()
```

- [ ] **Step 2: Verify the app starts without error (headless smoke test)**

```bash
streamlit run dashboard/app.py --server.headless true --server.port 8501 &
SERVER_PID=$!
sleep 3
curl -sf http://localhost:8501 > /dev/null && echo "dashboard responded OK"
kill $SERVER_PID
```

Expected: `dashboard responded OK` printed, no Python traceback in the server output.

- [ ] **Step 3: Commit**

```bash
git add dashboard/app.py
git commit -m "feat: add Streamlit dashboard skeleton with 4 placeholder pages"
```

---

### Task 5: README project overview

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: nothing
- Produces: nothing later tasks depend on — this is documentation only.

- [ ] **Step 1: Replace `README.md` contents**

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add project overview and setup instructions to README"
```

---

## Self-Review Notes

- **Spec coverage:** Task 1 → dependencies/gitignore (proposal Section 5.3 tech stack). Task 2 → data loading needed by Data Preparation section (proposal Section 4.3, Section 5.1). Task 3 → notebook matching the report's required 5-section outline (Section 1) and the 3-model design (Section 4.1). Task 4 → dashboard skeleton matching the 4 pages in proposal Section 5.2. Task 5 → onboarding docs. Model training itself (Isolation Forest/One-Class SVM/autoencoder logic) and the parquet-export step are intentionally out of scope — separate follow-up plan once modeling starts.
- **Placeholder scan:** no TBD/TODO in any step; every step shows complete, runnable code.
- **Type consistency:** `load_transactions`, `load_users`, `load_cards`, `load_mcc_codes`, `load_fraud_labels` signatures in Task 2 match the imports used in Task 3's notebook cell and Task 4's dashboard (`SCORED_TRANSACTIONS_PARQUET`, from the same `config.py`).
