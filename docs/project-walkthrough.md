# Credit Line Review Assistant — Project Walkthrough

**Course:** SCS 3253 – Machine Learning (Term Project)
**Code:** `src/credit_line_review/`, `notebooks/01_credit_line_review.ipynb`, `dashboard/app.py`

This is a step-by-step guide to what the project does, what data it reads, how that data is cleaned and transformed, what the exploratory analysis found, which models were trained and why, and how the results are evaluated and shipped. Each section explains not just *what* was done but *why that choice over the alternatives*.

---

## 1. The use case

Card issuers set a credit limit for each customer, usually from credit score and income at account opening. This project asks a different question about **existing** cardholders: given how someone actually spends — how often, how much, how volatile their spending is, online vs. in-person, how many merchant categories they touch — what limit does their behavior justify, and how does that compare to what they actually have today?

The output is a per-account recommendation plus a flag:

- **Underlimited** — actual limit well below the model's recommendation → revenue left on the table, churn risk to a competitor offering more credit.
- **Overlimited** — actual limit well above the recommendation → risk exposure that may not be priced correctly.
- **In range** — no action needed.
- **Closed account** — a data-quality category discovered during EDA (see §4), not a modeling outcome.

This is **retrospective, existing-customer analysis** — it needs transaction history, which only exists once someone already has a card. It is not new-account underwriting. That distinction is why the project is named a "credit line *review*," matching a real, recurring bank process rather than an initial approval decision.

**Hypothesis being tested:** behavioral spending signal explains credit-limit variance beyond what credit score and income alone capture. A model that beats a credit-score/income-only baseline supports the hypothesis; a model that doesn't, refutes it. Either result is a valid conclusion — the point of the modeling comparison in §6 is to find out which.

---

## 2. What data is read

All source files live under `Financial Transactions Dataset/` (gitignored — not distributed with the code) and are loaded through `src/credit_line_review/data.py`:

| File | Role in the project | Columns actually used |
|---|---|---|
| `cards_data.csv` | One row per card — supplies the **target** (`credit_limit`) | `id`, `client_id`, `card_type`, `credit_limit`, `card_brand`, `has_chip`, `acct_open_date` |
| `users_data.csv` | One row per customer — demographic features | `id`, `yearly_income`, `total_debt`, `credit_score`, `current_age`, `gender` |
| `transactions_data.csv` | One row per transaction — raw material for behavioral features | `id`, `card_id`, `date`, `amount`, `mcc`, `merchant_state`, `use_chip`, `errors` |
| `mcc_codes.json`, `train_fraud_labels.json` | Not used | These were considered in the original proposal (MCC→category lookup, prior-fraud-count as a risk feature) but weren't needed once category *diversity* — counting distinct MCCs, not naming them — turned out to be enough signal; kept out of scope to stay within the project's time budget |

**Modeling grain: one row per card**, not one row per transaction. `transactions_data.csv` is read as a sample (`nrows=200_000` in the notebook) and aggregated down to one behavioral summary row per `card_id` before anything is joined or modeled — see §4.

Three loader functions (`load_users`, `load_cards`, `load_transactions`) each do one shared cleanup step at read time: the source CSVs store money as strings like `"$29,278"`, so `parse_currency_column` strips `$` and `,` and casts to float for `credit_limit`, `yearly_income`, `total_debt`, `per_capita_income`, and transaction `amount`. `load_transactions` also parses `date` into a real datetime column so later recency/tenure math works.

---

## 3. Preprocessing pipeline

Preprocessing happens in `src/credit_line_review/features.py`, in two stages: per-card behavioral aggregation, then the join into one modeling table.

### 3.1 Filter to real credit products — `filter_credit_cards`

The dataset's `card_type` column contains `"Credit"`, `"Debit"`, and `"Debit (Prepaid)"`. A debit card carrying a `credit_limit` field isn't a meaningful business concept — the EDA (§4) confirmed all three types have a populated, non-null `credit_limit`, so this isn't obviously wrong until you check. Modeling is restricted to `card_type == "Credit"` (2,057 of 6,146 total cards) so the regression target only ever reflects an actual credit product.

### 3.2 Aggregate transactions to card-level behavioral features — `aggregate_card_behavior`

This is **RFM (Recency / Frequency / Monetary) feature construction** — a marketing-analytics technique, not something covered in the course modules, but the mechanism that turns row-level transactions into the one-row-per-card table the models need. For each `card_id`:

- `txn_count`, `avg_amount`, `median_amount` — basic spend volume/level
- `spend_volatility` = std(amount) / mean(amount) — how erratic the spending is, not just how much
- `distinct_mcc`, `distinct_merchant_state` — category and geographic diversity
- `recency_days` — days between the card's last transaction and a fixed reference date
- `txn_frequency_per_month` — transaction count normalized by the card's active window
- `online_share` — fraction of transactions where `use_chip == "Online Transaction"`
- `error_rate` — fraction of transactions with a non-null `errors` value (declines, insufficient balance, etc.)

**Why a fixed reference date, not `pd.Timestamp.now()`:** `transactions_data.csv` spans 2010-01-01 to 2019-10-31 (confirmed by scanning the full file). Using "now" would make `recency_days` — and every notebook re-run's results — silently drift with the calendar. `REFERENCE_DATE = "2019-11-01"` (config.py) is hardcoded to the day after the last real transaction, so recency/tenure features are reproducible no matter when the notebook is executed.

### 3.3 Join and transform — `build_modeling_table`

1. Join filtered credit cards → users (on `client_id`) → behavioral features (on `card_id`).
2. Compute `tenure_months` from `acct_open_date` (stored as `MM/YYYY`) relative to `REFERENCE_DATE`.
3. **Log-transform** (`log1p`) the skewed monetary columns: `yearly_income`, `total_debt`, `credit_limit`. Financial quantities are realistically log-normal — the raw `credit_limit` skew of 3.20 (see §4) would violate the linear model's roughly-normal-residuals assumption if left untransformed.
4. **Flag, don't drop, zero-limit accounts.** EDA turned up 26 Credit-card rows (1.3%) with `credit_limit == 0` — see §4 for how this was found and why it's handled as a flag rather than a filter.
5. One-hot encode the categorical columns: `card_brand`, `has_chip`, `gender`.

### 3.4 Group-aware train/test split — `group_train_test_split`

A plain random row split would let one client's *other* card leak into the opposite split from a card the model is being tested on, since `client_id` can own multiple cards. `group_train_test_split` uses scikit-learn's `GroupShuffleSplit` on `client_id` so every row for a given client lands entirely in train or entirely in test — never split across both. This is a standard-course-adjacent technique (train/test splitting is course material; the *group-aware* variant isn't) that exists specifically because this dataset has a many-cards-per-client structure a plain `train_test_split` would silently violate.

---

## 4. What the exploratory data analysis found

EDA lives in Section 2 of `notebooks/01_credit_line_review.ipynb` and drove three preprocessing decisions above. The actual findings:

**`card_type` vs. `credit_limit`.** All three card types have a populated `credit_limit` — including Debit cards (mean ≈$18,558) and Debit (Prepaid) cards (mean ≈$64). That confirmed the filter in §3.1: without it, the regression target would blend three semantically different quantities.

**Skew of the raw monetary columns**, before any transform:

| Column | Skew | Mean | Median |
|---|---|---|---|
| `credit_limit` | 3.20 | $11,174 | $10,100 |
| `yearly_income` | 3.45 | $45,716 | $40,744 |
| `total_debt` | 1.81 | $63,710 | $58,251 |

All strongly right-skewed, as expected for financial data — a few very high earners/limits stretch the distribution. This is the standard justification for log-transforming before fitting a linear model.

**The log-transform investigation — a real anomaly, not a clean result.** `log1p(credit_limit)` skew came out at **-5.52** — large and *negative*, the opposite of what a successful log-transform of a right-skewed variable should produce. That's the kind of result you don't wave past. Investigating it (Section 2.3.1 of the notebook) found the cause: **26 Credit-card rows (1.3% of 2,057) have `credit_limit == 0`.** `log1p(0) = 0`, and a cluster of zeros sitting far below the $5k–$98k range of every other row drags the whole distribution's skew negative.

These are almost certainly closed or frozen accounts — a $0 limit on a card labeled "Credit" isn't a real "what should this account's limit be" case, so it doesn't belong in the regression target at all. But dropping them silently would also throw away real information about the portfolio (a risk team likely wants to know these exist). The decision made: **flag, don't drop.** `build_modeling_table` adds an `is_zero_limit` boolean column; the modeling pipeline (§5) excludes those rows from training and evaluation, but the export step (§7) still scores and reports them — tagged `closed_account` — so they show up in the dashboard as their own segment instead of either disappearing or corrupting the "underlimited" flag (a $0-vs-predicted-$4,000 residual would otherwise read as an extreme underlimited case).

**`credit_score` and transaction `amount` distributions** were also plotted (Section 2.3) to sanity-check the feature set before modeling — `credit_score` is roughly normal (mean 710, range 480–850), consistent with typical FICO-style scoring; transaction amounts are, unsurprisingly, also right-skewed.

---

## 5. Models used, and why each one over the alternatives

Four model families were trained — three regressions/clustering methods that map directly to course modules, one dimensionality-reduction technique used only for visualization, plus a non-course baseline as the floor everything else has to beat.

### 5.1 Mean baseline (not a course model)

Predicts the training-set mean `log_credit_limit` for every row, regardless of input. This isn't a real model — it's the floor. If a "real" model can't beat guessing the average, its features aren't adding value, and that's an important negative result to be able to state, not something to hide by omitting the comparison.

### 5.2 Linear Regression — *Course: Module 5, Training Models & Feature Selection*

**Why chosen:** interpretability. A credit-limit decision carries a real transparency expectation — in a genuine lending context, adverse-action reasoning requirements mean an issuer needs to be able to say *why* an account was flagged. A linear model's coefficients are directly inspectable in a way a black-box model's aren't. It's included specifically as the interpretable end of the comparison, not because it was expected to win on accuracy.

**Why not Ridge/Lasso instead:** the plan considered Ridge but the feature set here (32 columns after one-hot encoding, ~1,600 training rows) isn't in a regime where multicollinearity or overfitting from a bare `LinearRegression` was a practical concern — added regularization would have been complexity without a demonstrated need.

### 5.3 Random Forest Regression — *Course: Module 7, Decision Trees & Ensemble Learning*

**Why chosen:** the underlying relationship almost certainly isn't linear — e.g., the effect of spend volatility on the "right" limit plausibly depends on income level (an interaction term), and Random Forest captures that kind of non-linear feature interaction without hand-engineering interaction terms.

**Why over SVR (Module 6), the other regression option covered in the course:** SVR scales worse and needs more careful kernel/hyperparameter tuning to handle a heavy-tailed continuous target like this one. Random Forest handles mixed feature types (the one-hot categoricals plus continuous behavioral features) natively and needed no tuning beyond `n_estimators=300, max_depth=8` to already beat the alternatives (see §6) — the practical "does it work well with modest effort" bar SVR would have made harder to clear in the project's time budget.

### 5.4 KMeans persona clustering — *Course: Module 4, Clustering*

**Why chosen:** the assignment requires comparing genuinely *different* model families, not two regressions dressed differently. KMeans segments cards into behavioral "personas" (`fit_kmeans_personas`, 4 clusters, features standardized first via `StandardScaler` since KMeans is distance-based and the feature scales here vary wildly — dollars vs. counts vs. fractions). This is unsupervised, with no target variable at all, which is a structurally distinct task from regression — it satisfies the "compare alternative modeling methods" objective with three real families (linear, tree-ensemble, clustering), not three flavors of regression.

**Caveat, stated honestly rather than hidden:** the silhouette score on the actual data came out at **0.089** — low, meaning the four clusters aren't cleanly separated in this feature space. That's a legitimate finding to report, not a failure to paper over: it suggests card behavior in this dataset doesn't fall into sharply distinct personas, which is itself informative about the population being modeled.

### 5.5 PCA — *Course: Module 8, Dimensionality Reduction*

**Why chosen, and why it's not used to predict anything:** PCA here is purely a visualization aid — projecting the (already-scaled) feature space to 2 components (`project_to_2d`) so the KMeans clusters and residuals can be plotted and eyeballed for whether flagged accounts group up sensibly. It's deliberately kept out of the modeling pipeline itself (no PCA-reduced features feed the regressions) because compressing to 2 components would throw away the exact non-linear interactions Random Forest is there to use.

### 5.6 Techniques used but not taught in the course (and why they were still necessary)

| Technique | Why the course version wasn't enough |
|---|---|
| Residual diagnostics (predicted-vs-actual scatter, residual histogram) | Reporting R² alone doesn't check whether the linear model's assumptions actually hold — needed to validate the regression, not just score it |
| Log-transform of skewed monetary columns | Not a course topic; necessary once §4's EDA showed real financial data doesn't come pre-normalized |
| Grouped train/test split (`GroupShuffleSplit`) | The course's train/test splitting assumes i.i.d. rows; this dataset's one-client-many-cards structure would leak identity across a plain split |
| RFM-style behavioral feature construction | A marketing-analytics technique, not ML curriculum, but the only way to turn per-transaction rows into per-card model inputs |
| SHAP feature importance | Goes past scikit-learn's built-in `.feature_importances_` to give **per-account** explanations — this is the project's main novelty lever, tying the model to the real adverse-action explainability requirement mentioned in §5.2 |

---

## 6. Model evaluation

"Accuracy" doesn't directly apply to a regression + clustering project, so the evaluation uses regression-appropriate metrics computed in `evaluation.py` and reported on the held-out grouped test split (415 rows, from a 2,031-row modeling table that already excludes the 26 closed accounts):

| Model | R² | MAE | RMSE | MAPE |
|---|---|---|---|---|
| Baseline (mean) | -0.071 | $4,051 | $5,773 | 47.4% |
| Linear Regression | -0.082 | $4,071 | $5,803 | 47.6% |
| **Random Forest** | **0.364** | **$3,036** | **$4,448** | **35.1%** |

**Reading this honestly:** Linear Regression actually performs *worse* than the naive mean baseline (negative R² on both). That's a real result, not a bug — it means the linear relationship between the available features and `log_credit_limit`, in the training data used, doesn't capture more signal than just guessing the average; the underlying pattern needs the non-linear interactions Random Forest can model. Random Forest clears the baseline decisively (R² 0.36 vs. -0.07, ~25% lower MAE), which is the concrete evidence for the project's hypothesis in §1: behavioral + demographic features *do* explain real variance in credit limits, but only when the model can capture non-linear structure.

**Residual diagnostics** (predicted-vs-actual scatter with a y=x reference line, plus a residual histogram) are plotted for the winning model to check the errors aren't systematically biased in one direction, rather than trusting a single R² number.

**Clustering** is evaluated by silhouette score (0.089, discussed honestly as a limitation in §5.4) rather than any ground-truth label, since none exists for "correct persona."

**The flagging layer has no ground-truth label either** ("was this account really miscalibrated?" isn't something the dataset records), so it's evaluated as a business-plausibility check instead of precision/recall: at the chosen threshold (1.5× the residual standard deviation), the exported 441-account scored table breaks down as:

| Flag | Count |
|---|---|
| `in_range` | 364 |
| `overlimited` | 38 |
| `closed_account` | 26 |
| `underlimited` | 13 |

That's a plausible risk-review queue size (~12% of the portfolio flagged for one reason or another), which is the kind of sanity check available in place of a formal accuracy metric.

---

## 7. From notebook to application

Consistent with the project's scope, all modeling happens once, offline, in the notebook — the dashboard never trains anything. It's a read-only viewer over one exported file.

**Export (`export.py`, run at the end of the notebook):** `build_scored_accounts` combines the held-out test predictions with the 26 closed accounts (scored too, but for reporting only — not part of any train/test metric) into one table: `id_card, client_id, credit_limit, predicted_limit, residual, cluster, flag, is_zero_limit`. Closed accounts get a sentinel `cluster = -1` (they were excluded from the KMeans fit, so they don't have a real persona) and an unconditional `flag = "closed_account"` override, regardless of what their residual alone would suggest. `export_scored_accounts` writes this to `dashboard_data/scored_accounts.parquet`.

**Dashboard (`dashboard/app.py`, Streamlit, 4 pages), all reading that one Parquet file:**

1. **Portfolio Overview** — the flag-level breakdown table above, plus an actual-vs-predicted scatter with a y=x reference line (closed accounts excluded — they'd distort the axis scale).
2. **Limit Miscalibration Explorer** — a slider on the residual threshold, live count/percentage of accounts flagged at that threshold (closed accounts excluded here too, with a note pointing to where they do show up).
3. **Segment/Persona View** — average limit/prediction/residual per KMeans cluster, plus a dedicated closed-accounts subsection.
4. **Case Drill-Down** — pick one account by `id_card`, see its full scored row.

---

## 8. Where the code lives

| Concern | File |
|---|---|
| Currency/date parsing, raw CSV loading | `src/credit_line_review/data.py` |
| Path/constant configuration (`REFERENCE_DATE`, file paths) | `src/credit_line_review/config.py` |
| Card filtering, RFM aggregation, modeling-table join, grouped split | `src/credit_line_review/features.py` |
| Baseline / Linear / Random Forest model wrappers | `src/credit_line_review/models.py` |
| Regression metrics, residuals, SHAP importance | `src/credit_line_review/evaluation.py` |
| KMeans personas, PCA projection | `src/credit_line_review/clustering.py` |
| Scored-accounts builder and Parquet export | `src/credit_line_review/export.py` |
| End-to-end pipeline (EDA → modeling → export) | `notebooks/01_credit_line_review.ipynb` |
| Read-only viewer over the exported scores | `dashboard/app.py` |
| Unit tests (one file per `src/` module) | `tests/` |

Every function above is unit-tested (22 tests total, `pytest -q`) independent of the notebook, so the modeling logic can be verified without re-running the full data pipeline each time.
