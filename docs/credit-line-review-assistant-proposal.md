# Project Proposal: Credit Line Review Assistant

**Course:** SCS 3253 – Machine Learning (Term Project)
**Status:** Draft for team discussion — supersedes the earlier Fraud Anomaly Monitor proposal
**Date:** 2026-07-15

---

## 1. Assignment context (quick recap)

- Select a dataset, train **≥2 alternative models** (regression / classification / clustering) using scikit-learn, TensorFlow, or a custom model, and compare results.
- Deliverables: an **8–10 page report** (Objective → Data Preparation → Model Design → Model Evaluation → Conclusions) + a **Jupyter notebook** (code as appendix) + a **5-slide, 5-minute presentation**.
- Grading (out of 30): 10 pts report/presentation quality, **15 pts correctness & thoroughness of analysis**, 5 pts novelty.
- Full brief: `Course content/SCS 3253 Term Project Machine Learning.pdf`

---

## 2. Use case (plain English)

Card issuers set a credit limit for each customer, mostly from credit score and income. This project builds a model that also looks at how an **existing** cardholder actually spends — frequency, volatility, online vs. in-person, category diversity — and predicts what their limit *should* be based on that behavior, in addition to their demographics.

We then compare the model's recommendation to the customer's *actual* current limit. A large gap in either direction is actionable:

- **Underlimited** (actual well below recommended) → revenue left on the table, churn risk to a competitor offering more credit.
- **Overlimited** (actual well above recommended) → risk exposure the issuer may not have priced correctly.

**This is retrospective, existing-customer analysis, not new-account underwriting.** It requires transaction history, which only exists once someone already has a card — so it maps to a real, recurring bank process called a "credit line review," not an initial approval decision. We name the project accordingly.

**Falsifiable hypothesis (for the report's Conclusions section):** behavioral spending signal explains credit-limit variance beyond what credit score and income alone capture.

---

## 3. Dataset

Located in `Financial Transactions Dataset/`:

| File | Role | Key columns used |
|---|---|---|
| `cards_data.csv` | Target table | `client_id`, `card_type`, `credit_limit` (**target**), `has_chip`, `acct_open_date`, `card_on_dark_web` |
| `users_data.csv` | Demographic features | `id`→`client_id`, `current_age`, `yearly_income`, `total_debt`, `credit_score`, `num_credit_cards` |
| `transactions_data.csv` | Behavioral features (aggregated to card level) | `client_id`, `card_id`, `amount`, `date`, `use_chip`, `mcc`, `errors` |
| `mcc_codes.json` | Lookup only | MCC → human-readable category name |
| `train_fraud_labels.json` | Optional enrichment | Could contribute "# prior fraud incidents on this account" as a risk feature; not the modeling target |

Modeling grain: **one row per card** (not per transaction) — transactions are aggregated into behavioral features per card before joining to `cards_data`/`users_data`.

---

## 4. Preprocessing

- Parse currency strings (`"$29,278"` → float) in `credit_limit`, `yearly_income`, `total_debt`, `per_capita_income`, `amount`.
- **Data-quality check:** verify whether `credit_limit` is populated sensibly for `card_type == "Debit"` (debit cards shouldn't have a real credit limit as a business concept); if not, restrict modeling to `card_type == "Credit"`. Document as a Data Preparation finding either way.
- Parse `acct_open_date` (MM/YYYY) into account tenure in months from a fixed reference date; parse transaction `date` for recency/velocity features.
- Drop PII/irrelevant fields: `card_number`, `cvv`, `expires`.
- Engineer per-card behavioral features from `transactions_data`: avg/median amount, spend volatility (std/mean), transaction frequency, recency, category diversity (distinct MCCs / entropy), % online vs. chip vs. swipe, decline/error rate, geographic spread.
- Log-transform (`log1p`) skewed monetary columns (income, debt, `credit_limit`) — financial data is realistically log-normal; state this as an explicit assumption check for the regression models.
- One-hot encode `card_brand`, `has_chip`, `gender`.
- **Group-based train/test split by `client_id`** (not a random row split) — a client's multiple cards must not span train and test, or client identity leaks across the split. Use grouped k-fold CV for the final evaluation.

---

## 5. Modeling approach

### 5.1 Models (course-taught concepts, and why chosen over alternatives)

| # | Model | Type | Course module | Why this over alternatives |
|---|---|---|---|---|
| 1 | Linear/Ridge Regression | Regression | Module 5 – Training Models & Feature Selection | Interpretable baseline; credit decisions carry a real transparency expectation (adverse-action reasoning), so a coefficient-level model is defensible in a way a black box isn't |
| 2 | Random Forest / Gradient Boosted Trees | Regression | Module 7 – Decision Trees & Ensemble Learning | Handles non-linear interactions (income × spend-volatility) and mixed feature types without kernel tuning; chosen over SVR (Module 6) because SVR scales worse and needs more careful tuning for a heavy-tailed continuous target |
| 3 | KMeans | Clustering | Module 4 – Clustering | A genuinely different model family from the two regressions above — segments cards into limit-policy personas, satisfying the "compare alternative modelling methods" learning objective with three distinct families, not two similar ones |
| 4 | PCA | Dimensionality reduction | Module 8 | Not used to predict the target — used to visualize the card population in 2D, colored by residual/cluster, as a sanity check that flagged accounts cluster sensibly |
| — | Classification-metric framing | Evaluation | Module 3 – Classification | Reused for the miscalibration-flagging layer: "flagged vs. not" at a residual threshold is a binary decision, so precision/recall-style thinking still applies even without a ground-truth label (see §6) |

### 5.2 Concepts beyond the course (and why they matter)

- **Residual diagnostics** (predicted-vs-actual scatter, residual histogram, Q-Q plot) — needed to check regression assumptions, not just report R².
- **Log-transform / variance stabilization** for skewed monetary data.
- **Group-aware cross-validation** (GroupKFold by `client_id`) — prevents leakage that a default i.i.d. split would miss.
- **SHAP values / permutation importance** — per-account, per-feature explanations, going beyond `.feature_importances_`. This ties the model to a real regulatory concept (adverse-action explainability under lending law) — the strongest novelty lever in the project.
- **RFM-style feature construction** (recency/frequency/monetary) — turns row-level transactions into card-level behavioral features.

---

## 6. Evaluation strategy

"Accuracy" doesn't directly apply to a regression + unsupervised project — the report should say this explicitly, then give concrete substitutes:

- **Baseline comparison**: naive baseline (predict mean, or credit-score-only linear model) vs. Linear Regression vs. Random Forest/GBM, on R², MAE, RMSE, MAPE. Beating the baseline is the actual hypothesis test.
- **Generalization**: compare train vs. held-out (grouped) performance; report grouped k-fold CV mean ± std, not a single split.
- **Residual diagnostics** as a validity metric, not just a chart.
- **Clustering**: silhouette score / inertia elbow to justify k (no ground truth to score against), plus a qualitative "do personas make business sense" check.
- **Flagging layer**: no ground-truth "true miscalibration" label exists, so evaluate as a **business-plausibility simulation** — e.g., "at threshold X, Y% of the portfolio gets flagged, consistent with a typical risk-team review queue capacity" — and say explicitly why a traditional precision/recall metric isn't available here.
- **Stretch**: bootstrap confidence interval or permutation test on R² to show the result isn't a fluke of one split.

---

## 7. Application

Same lightweight pattern as the original fraud proposal (offline notebook does all modeling; Streamlit is a read-only viewer over one precomputed file) — reused because it works within the course's time budget and isn't a graded requirement, just a novelty/presentation aid.

- **Notebook** (`notebooks/01_credit_line_review.ipynb`): loads/joins data, engineers features, trains all models, computes metrics/diagnostics/SHAP, exports `dashboard_data/scored_accounts.parquet` (one row per card: actual limit, predicted limit, residual, flag status, cluster, top feature contributions).
- **Streamlit app** (`dashboard/app.py`), 4 pages:
  1. **Portfolio Overview** — model scorecard (R²/MAE/RMSE/MAPE incl. baseline), actual-vs-predicted scatter with y=x line.
  2. **Limit Miscalibration Explorer** — slider on residual threshold; live count of under/over-limited accounts and % of portfolio flagged.
  3. **Segment/Persona View** — PCA scatter colored by cluster, table of avg income/debt/limit/utilization per persona.
  4. **Case Drill-down** — pick an account, see actual vs. recommended limit and its top SHAP feature contributions.

---

## 8. Rough milestones

See the implementation plan for the detailed step-by-step breakdown. At a high level:

1. EDA — load and visualize raw distributions (income, debt, credit score, credit_limit, transaction amount) to justify preprocessing decisions.
2. Data prep — joins, feature engineering, cleaning, group-based split.
3. Train models (Linear/Ridge, Random Forest/GBM, KMeans, PCA).
4. Evaluate (metrics, residual diagnostics, SHAP, clustering diagnostics, flagging simulation).
5. Export scored dataset → build Streamlit dashboard.
6. Write report (5-section outline) and 5-slide presentation.

---

*This is a discussion draft — nothing here is finalized. Feedback and alternative framings welcome before implementation begins.*
