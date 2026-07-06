# Project Proposal: Fraud Anomaly Monitor

**Course:** SCS 3253 – Machine Learning (Term Project)
**Status:** Draft for team discussion — not yet finalized
**Author:** [your name]
**Date:** 2026-07-06

---

## 1. Assignment context (quick recap)

- Select a dataset, train **≥2 alternative models** (regression / classification / clustering) using scikit-learn, TensorFlow, or a custom model, and compare results.
- Deliverables: an **8–10 page report** (Objective → Data Preparation → Model Design → Model Evaluation → Conclusions) + a **Jupyter notebook** (code as appendix) + a **5-slide, 5-minute presentation**.
- Grading (out of 30): 10 pts report/presentation quality, **15 pts correctness & thoroughness of analysis**, 5 pts novelty.
- Full brief: `Course content/SCS 3253 Term Project Machine Learning.pdf`

---

## 2. Dataset

Located in `Financial Transactions Dataset/`:

| File | Contents | Size |
|---|---|---|
| `users_data.csv` | Per-user demographics: age, income, debt, credit score, # cards | ~165 KB |
| `cards_data.csv` | Per-card info: brand, type, credit limit, chip, account open date | ~510 KB |
| `transactions_data.csv` | Individual transactions: date, amount, merchant, city/state/zip, MCC, chip/swipe/online flag, errors | ~1.25 GB |
| `mcc_codes.json` | Merchant Category Code → human-readable category name lookup | ~5 KB |
| `train_fraud_labels.json` | Ground-truth fraud/not-fraud label for a subset of transaction IDs | ~159 MB |

This is a realistic, large-scale synthetic financial transactions dataset (client/card/transaction structure typical of real bank data warehouses), which gives us room for genuinely rigorous data engineering — a plus for the "thoroughness" grading criterion.

---

## 3. Proposed objective / business framing

**Working hypothesis:** In real fraud operations, reliable fraud labels are scarce, delayed, or simply don't exist for most transactions (fraud gets confirmed weeks later, if ever). So instead of "train a classifier on the labels we happen to have," we frame this as:

> Can we detect anomalous, potentially fraudulent transactions using **unsupervised/semi-supervised anomaly detection** — trained without looking at fraud labels at all — and how well does that anomaly signal line up with the ground-truth fraud labels we do have?

We use `train_fraud_labels.json` **only for evaluation**, not training. This is a deliberately more realistic and more novel framing than plain supervised fraud classification (which is the most common student project on this type of dataset), while still giving us a clean way to measure "did it work."

**Business angle for the report:** framed as an early-warning tool for a bank's fraud operations team — flag suspicious transactions for human review before a customer or the bank takes a loss, without depending on lagging fraud-confirmation data.

---

## 4. Machine learning approach

### 4.1 Candidate models (need ≥2, proposing 3 to span more course modules)

| # | Model | Type | Course module it maps to |
|---|---|---|---|
| 1 | **Isolation Forest** | Tree-based ensemble, unsupervised anomaly detection | Module 7 – Decision Trees & Ensemble Learning |
| 2 | **One-Class SVM** | Support vector method, unsupervised anomaly detection | Module 6 – Support Vector Machines |
| 3 | **Autoencoder (neural network)** | Deep learning, reconstruction-error-based anomaly detection | Module 9 – Intro to TensorFlow / Module 10 – Deep Neural Networks |

Using all three (rather than just two) lets us compare a tree-based method, a kernel-based method, and a deep learning method side by side — directly hitting the learning objective "recognize the relative strengths of alternative modelling methods," and gives the report a natural three-way comparison table instead of a single A/B comparison.

### 4.2 Supporting techniques from other modules

- **Module 1 / "End to End Machine Learning Project"** — overall pipeline structure (train/test split, preprocessing pipeline, evaluation discipline).
- **"Introduction to Transactional Machine Learning"** — directly relevant, this module is presumably about exactly this kind of data; should mine it for domain-specific feature ideas (e.g. velocity features, recency features).
- **Module 5 – Training Models & Feature Selection** — feature engineering (e.g., transaction amount z-score per user, time-since-last-transaction, spending velocity, deviation from a card's typical MCC categories) and hyperparameter selection (contamination rate for Isolation Forest, nu for One-Class SVM, latent dimension/threshold for the autoencoder).
- **Module 8 – Dimensionality Reduction** — PCA (or t-SNE/UMAP) to visualize transactions in 2D, colored by anomaly score and by true fraud label, to visually sanity-check that anomalies cluster sensibly.
- **Module 3 – Classification** — even though models are unsupervised, evaluation borrows classification metrics (precision, recall, PR-AUC) against the held-out fraud labels, so this module's evaluation concepts still apply directly.

### 4.3 Evaluation strategy

- Fraud is rare (imbalanced) → use **Precision-Recall curves and PR-AUC**, not plain accuracy or ROC-AUC (ROC is optimistic/misleading under heavy class imbalance — this is exactly the kind of "checking assumptions the model/test requires" the rubric rewards).
- Fraud labels used strictly as a **held-out evaluation set**, never seen during model fitting — needs to be explicit and well-justified in the report's Data Preparation section.
- Compare all three models on the same metrics, plus qualitative comparison (training time, interpretability, sensitivity to hyperparameters).

---

## 5. The application: "Fraud Anomaly Monitor"

**Important scope note:** the course assignment does **not** require an application at all — the only graded deliverables are the Jupyter notebook, the 8-10 page report, and the 5-slide presentation (see Section 1). This dashboard is our own optional addition for the novelty (5 pts) category and for a stronger presentation demo, not something the rubric asks for. We're deliberately scoping it as a lightweight bonus, not a second product to engineer — the plan is Streamlit, not a custom React/FastAPI build, so we don't eat into time budgeted for the actual graded modeling work.

Rather than just producing static charts for the report, the plan is to build a small interactive dashboard. This becomes the natural home for every chart the Model Evaluation section needs anyway (build once, screenshot for the report/slides, optionally demo live in the 5-minute presentation) — and it directly serves the assignment's objective of "explaining technical details to a non-technical audience."

**What kind of application:** A single-page (or few-page) **Streamlit** web app running locally, reading pre-computed model outputs (scores, metrics) from the notebook. Streamlit turns a plain Python script into an interactive browser-based app (sliders, charts, tables) with no separate HTML/CSS/JS or backend API needed — it's the fastest way to get an interactive tool out of a scikit-learn/TensorFlow pipeline with minimal extra code, which matters given this is a side deliverable, not the main deliverable. It won't have the fully custom, branded look of a real commercial product, but a bit of theming/CSS polish gets it most of the way there for a fraction of the effort a custom frontend+backend build would take.

**Who it's for (persona):** A fraud operations manager reviewing flagged transactions and comparing model performance — non-technical audience, so labels, tooltips, and plain-language explanations matter throughout.

### 5.1 Data input / upload flow

The heavy lifting (model training, scoring all transactions) happens offline in the Jupyter notebook, since `transactions_data.csv` is 1.25 GB and not something to fit/score inside a live Streamlit session. The dashboard itself is a **read-only viewer** over precomputed results:

1. Notebook trains all 3 models and scores an evaluation sample (e.g., all transactions with a known fraud label, plus a random sample of unlabeled ones for context).
2. Notebook exports a compact results file — e.g. `dashboard_data/scored_transactions.parquet` — containing: transaction ID, date, amount, merchant city/state, MCC, MCC category name, per-model anomaly score, and true fraud label (where available).
3. Streamlit app loads that single file at startup (`st.cache_data`) — no live upload needed for the course deliverable.
4. *(Stretch goal, optional)* Add a `st.file_uploader` widget so a teammate could drop in a different scored file (e.g., a different sample or time window) without touching code — nice for team iteration, not required for grading.

This keeps the app itself lightweight and fast to load, and avoids ever loading the full 1.25 GB file into a Streamlit session.

### 5.2 Pages / features

**Page 1 — Model Comparison Overview** (default landing page)

```
┌─────────────────────────────────────────────────────────────┐
│  Fraud Anomaly Monitor                     [ i ] About       │
├─────────────────────────────────────────────────────────────┤
│  Model performance (vs. held-out fraud labels)                │
│  ┌───────────────┬───────────┬────────┬────────┬───────────┐ │
│  │ Model         │ Precision │ Recall │  F1    │  PR-AUC   │ │
│  ├───────────────┼───────────┼────────┼────────┼───────────┤ │
│  │ Isolation For.│   0.XX    │  0.XX  │  0.XX  │   0.XX    │ │
│  │ One-Class SVM │   0.XX    │  0.XX  │  0.XX  │   0.XX    │ │
│  │ Autoencoder   │   0.XX    │  0.XX  │  0.XX  │   0.XX    │ │
│  └───────────────┴───────────┴────────┴────────┴───────────┘ │
│                                                                │
│  ┌────────────────────────────┐  ┌──────────────────────────┐│
│  │  Precision-Recall Curves   │  │  Anomaly Score            ││
│  │  (all 3 models overlaid)   │  │  Distribution              ││
│  │        [chart]             │  │  (fraud vs. legit,         ││
│  │                            │  │   per model)  [chart]     ││
│  └────────────────────────────┘  └──────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

- Metrics scorecard table (Precision / Recall / F1 / PR-AUC per model)
- Overlaid Precision-Recall curves for all 3 models
- Anomaly score distribution histograms (fraud vs. legit overlay) — one small chart per model, or a model selector

**Page 2 — Threshold Explorer** (the "wow" interactive page)

```
┌─────────────────────────────────────────────────────────────┐
│  Threshold Explorer            Model: [Isolation Forest ▼]   │
├─────────────────────────────────────────────────────────────┤
│  Anomaly threshold:  ├──────────●───────────┤  0.62           │
│                                                                │
│  ┌───────────────┐   ┌────────────────────────────────────┐  │
│  │ Confusion      │   │  As you drag the slider:            │  │
│  │ Matrix         │   │  Precision: 0.XX   Recall: 0.XX     │  │
│  │  [heatmap]     │   │  Flagged: 1,204 txns (2.3% of total)│  │
│  └───────────────┘   └────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

- A slider to move the anomaly cutoff, with confusion matrix, precision/recall, and "% of transactions flagged" updating live
- Plain-language caption explaining the trade-off ("moving left catches more fraud but flags more legitimate transactions for review")

**Page 3 — Anomaly Patterns**

```
┌─────────────────────────────────────────────────────────────┐
│  Anomaly Patterns                                            │
├─────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────┐  ┌──────────────────────────┐│
│  │ Flagged transactions        │  │ Anomaly rate by merchant  ││
│  │ over time (daily/weekly)    │  │ category (MCC)            ││
│  │       [line chart]          │  │      [bar chart]          ││
│  └────────────────────────────┘  └──────────────────────────┘│
│  ┌───────────────────────────────────────────────────────────┐│
│  │ (Stretch) Geographic hotspot map — flagged txns by state   ││
│  └───────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

- Flagged-transactions-over-time line/area chart (volume vs. anomalies)
- Anomaly rate by MCC category bar chart (using `mcc_codes.json` for readable labels)
- *(Stretch)* choropleth/hotspot map of flagged transactions by `merchant_state`

**Page 4 — Case Drill-Down**

```
┌─────────────────────────────────────────────────────────────┐
│  Case Drill-Down          Filter: [ All Models ▼ ] [Search]  │
├─────────────────────────────────────────────────────────────┤
│  ID       Date        Amount   Merchant       MCC        Score│
│  7482991  2024-03-11  $842.00  Best Buy #221  Electronics 0.94│
│  7482994  2024-03-11  $12.50   Shell Station   Gas         0.88│
│  ...                                                          │
│  (click a row to expand: raw features + which model(s)        │
│   flagged it + true label if known)                            │
└─────────────────────────────────────────────────────────────┘
```

- Sortable/filterable table of top-N flagged transactions (by any model)
- Row expansion showing raw features + per-model scores + true label (if known) — this is the "investigate a case" experience

### 5.3 Tech stack

- **Modeling:** scikit-learn (`IsolationForest`, `OneClassSVM`, `PCA`), TensorFlow/Keras (autoencoder)
- **Dashboard:** Streamlit + Plotly (for interactive charts) or Altair
- **Data interchange notebook → app:** a single Parquet file of scored, sampled transactions (keeps the app fast and avoids loading the full 1.25 GB raw file)

---

## 6. Rough milestones (for team discussion)

1. Data prep: join transactions + users + cards + MCC lookup; engineer behavioral features (velocity, recency, deviation-from-typical-spend, etc.)
2. Train all 3 models unsupervised (no fraud labels used)
3. Evaluate against `train_fraud_labels.json` (PR curves, PR-AUC, confusion matrix at a chosen threshold)
4. Export scored sample → build Streamlit dashboard
5. Write report following the required 5-section outline
6. Build 5-slide presentation (likely: 1 problem framing, 1 approach/models, 1-2 results & dashboard screenshots, 1 conclusions/next steps)

---

## 7. Open questions for the team

- Are we comfortable with the "train without labels, evaluate with labels" framing, or would the team rather do a more conventional supervised comparison as a safer fallback?
- Who owns which model (Isolation Forest / One-Class SVM / Autoencoder) vs. who owns the dashboard?
- Do we want the geographic map / file-uploader stretch goals, or keep scope tight given time constraints?
- Sample size for the dashboard's scored-transactions file — how large can we make it before Streamlit gets sluggish?
- Citation check: need the dataset's original source URL for the report's references (per rubric requirement) — does anyone have the Kaggle/source link this was downloaded from?

---

*This is a discussion draft — nothing here is finalized. Feedback and alternative framings welcome before we lock in the approach.*
