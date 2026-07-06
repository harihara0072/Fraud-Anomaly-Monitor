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
