import nbformat as nbf

nb = nbf.v4.new_notebook()

nb["cells"] = [
    nbf.v4.new_markdown_cell(
        "# Credit Line Review Assistant — SCS 3253 Term Project\n"
        "\n"
        "Report outline this notebook mirrors: Objective -> Data Preparation -> "
        "Model Design -> Model Evaluation -> Conclusions.\n"
        "See `docs/credit-line-review-assistant-proposal.md` for the full design."
    ),
    nbf.v4.new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "sys.path.insert(0, str(Path.cwd().parent / \"src\"))\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import plotly.express as px\n"
        "\n"
        "from credit_line_review.data import load_cards, load_transactions, load_users"
    ),
    nbf.v4.new_markdown_cell(
        "## 1. Objective\n"
        "\n"
        "We predict a data-driven recommended credit limit for **existing** "
        "cardholders from their demographics and observed spending behavior, "
        "then flag accounts whose actual limit diverges sharply from that "
        "recommendation. This is a retrospective credit-line-review tool, not "
        "new-account underwriting."
    ),
    nbf.v4.new_markdown_cell("## 2. Data Preparation"),
    nbf.v4.new_markdown_cell("### 2.1 Load raw data"),
    nbf.v4.new_code_cell(
        "users = load_users()\n"
        "cards = load_cards()\n"
        "transactions = load_transactions(nrows=5000)\n"
        "\n"
        "print(\"users:\", users.shape)\n"
        "print(\"cards:\", cards.shape)\n"
        "print(\"transactions (sample):\", transactions.shape)"
    ),
    nbf.v4.new_markdown_cell(
        "### 2.2 Does `card_type` affect whether `credit_limit` is meaningful?\n"
        "\n"
        "A debit card shouldn't carry a real credit limit as a business "
        "concept. Check the split before deciding whether to filter."
    ),
    nbf.v4.new_code_cell(
        "print(cards[\"card_type\"].value_counts())\n"
        "print(cards.groupby(\"card_type\")[\"credit_limit\"].describe())"
    ),
    nbf.v4.new_markdown_cell(
        "If `credit_limit` is populated for Debit cards too, that confirms a "
        "data-quality quirk: restrict modeling to `card_type == \"Credit\"` so "
        "the regression target only reflects an actual credit product."
    ),
    nbf.v4.new_code_cell(
        "credit_cards = cards[cards[\"card_type\"] == \"Credit\"]\n"
        "print(\"credit cards:\", credit_cards.shape, \"of\", cards.shape[0], \"total cards\")"
    ),
    nbf.v4.new_markdown_cell(
        "### 2.3 Distribution of the regression target and key predictors\n"
        "\n"
        "Financial quantities (income, debt, credit limit) are usually "
        "right-skewed / log-normal. Check skew before deciding whether to "
        "log-transform for the linear model."
    ),
    nbf.v4.new_code_cell(
        "monetary_cols = {\n"
        "    \"credit_limit\": credit_cards[\"credit_limit\"],\n"
        "    \"yearly_income\": users[\"yearly_income\"],\n"
        "    \"total_debt\": users[\"total_debt\"],\n"
        "}\n"
        "for name, series in monetary_cols.items():\n"
        "    print(f\"{name}: skew={series.skew():.2f}, mean={series.mean():.0f}, median={series.median():.0f}\")\n"
        "    fig = px.histogram(series, nbins=50, title=f\"Distribution of {name}\")\n"
        "    fig.show()"
    ),
    nbf.v4.new_code_cell(
        "log_limit = np.log1p(credit_cards[\"credit_limit\"])\n"
        "print(f\"log1p(credit_limit): skew={log_limit.skew():.2f}\")\n"
        "fig = px.histogram(log_limit, nbins=50, title=\"Distribution of log1p(credit_limit)\")\n"
        "fig.show()"
    ),
    nbf.v4.new_markdown_cell(
        "A skew close to 0 after `log1p` (vs. the raw skew printed above) "
        "would justify log-transforming `credit_limit`, `yearly_income`, and "
        "`total_debt` before fitting the linear model — but the log1p skew "
        "printed above is large and *negative*, the opposite of what a "
        "successful log-transform of a right-skewed variable should look "
        "like. Investigate why."
    ),
    nbf.v4.new_markdown_cell(
        "### 2.3.1 Why did `log1p(credit_limit)` get *more* skewed, not less?\n"
        "\n"
        "A cluster of `credit_limit == 0` rows would sit far below the rest "
        "of the (log-transformed) distribution, dragging the skew negative. "
        "Check for that directly."
    ),
    nbf.v4.new_code_cell(
        "n_zero = (credit_cards[\"credit_limit\"] == 0).sum()\n"
        "print(f\"Credit-card rows with credit_limit == 0: {n_zero} \"\n"
        "      f\"({n_zero / len(credit_cards):.1%} of {len(credit_cards)} credit cards)\")"
    ),
    nbf.v4.new_markdown_cell(
        "These are almost certainly closed/frozen accounts, not a real "
        "\"what should their limit be\" case — a genuine credit-line-review "
        "candidate has a nonzero limit today. Rather than silently dropping "
        "or blindly modeling them, `build_modeling_table` (Task 4) tags "
        "them with an `is_zero_limit` flag: they stay in the exported "
        "table for reporting, but are excluded from the train/test split "
        "used to fit the regressions, and get their own \"closed accounts\" "
        "segment in the dashboard instead of an under/over-limited flag."
    ),
    nbf.v4.new_code_cell(
        "print(\"credit_score:\", users[\"credit_score\"].describe())\n"
        "fig = px.histogram(users[\"credit_score\"], nbins=40, title=\"Distribution of credit_score\")\n"
        "fig.show()\n"
        "\n"
        "fig = px.histogram(transactions[\"amount\"], nbins=50, title=\"Distribution of transaction amount (sample)\")\n"
        "fig.show()"
    ),
    nbf.v4.new_markdown_cell(
        "### 2.4 Feature engineering & modeling table\n"
        "\n"
        "Aggregate transactions to one row per card, join to cards + users, "
        "log-transform monetary columns, and split by `client_id` so a "
        "client's cards never span train and test — implemented in "
        "`credit_line_review.features` (Tasks 3-4 of the implementation plan) "
        "and wired into this notebook in Task 10."
    ),
    nbf.v4.new_markdown_cell(
        "## 3. Model Design\n"
        "\n"
        "### 3.1 Baseline (predict the training mean)\n"
        "Not a course model - the floor any real model has to beat.\n"
        "\n"
        "### 3.2 Linear/Ridge Regression\n"
        "*Course: Module 5 - Training Models & Feature Selection.* Chosen for "
        "coefficient-level interpretability, which matters for a credit "
        "decision.\n"
        "\n"
        "### 3.3 Random Forest Regression\n"
        "*Course: Module 7 - Decision Trees & Ensemble Learning.* Chosen over "
        "SVR (Module 6) - handles non-linear feature interactions without "
        "kernel tuning and scales better to this row count.\n"
        "\n"
        "### 3.4 KMeans persona clustering + PCA visualization\n"
        "*Course: Module 4 - Clustering (KMeans) and Module 8 - "
        "Dimensionality Reduction (PCA).*\n"
        "\n"
        "### 3.5 Beyond the course\n"
        "Residual diagnostics, log-transforming skewed monetary columns, "
        "grouped train/test splitting, RFM-style behavioral features, and "
        "SHAP feature importance are not course topics - see "
        "`docs/credit-line-review-assistant-proposal.md` §5.2 for why each "
        "one is needed here."
    ),
    nbf.v4.new_markdown_cell(
        "## 4. Model Evaluation\n"
        "\n"
        "*Course: Module 3 - Classification.* The regression-vs-actual "
        "residual, thresholded, becomes a binary \"flag/don't flag\" decision - "
        "precision/recall-style thinking from Module 3 applies to that "
        "decision even though the underlying models are regression/clustering, "
        "not classification."
    ),
    nbf.v4.new_markdown_cell("## 5. Conclusions"),
]

nbf.write(nb, "notebooks/01_credit_line_review.ipynb")
print("wrote notebooks/01_credit_line_review.ipynb")
