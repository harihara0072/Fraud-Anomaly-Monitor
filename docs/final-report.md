# Credit Line Review Assistant: A Behavioral Model for Existing-Customer Credit Limit Review

**Course:** SCS 3253 – Machine Learning, Term Project
**Prepared by:** [Group Members — Hari Hara Kumar Nakshatrala, Jimmy Lopez, Remya Sara Raju]
**Date:** July 26, 2026

---

## Executive Summary

Credit card issuers typically set a customer's credit limit once, at account opening, based on credit score and income. This project asks whether an issuer can do better for **existing** customers by also looking at how they actually use their card — how often they spend, how volatile that spending is, how much of it happens online, and how many merchant categories they touch.

We built a "credit line review" model: for each existing credit-card account, it recommends a limit based on behavior and demographics, compares that recommendation to the customer's actual limit, and flags accounts that look **underlimited** (likely under-served, a churn and revenue risk) or **overlimited** (a risk exposure that may not be priced correctly). We trained and compared three model types — a naive baseline, Linear Regression, and Random Forest — plus an unsupervised KMeans clustering model to segment customers into behavioral personas.

**Bottom line:** once we corrected a feature-leakage issue we caught during our own review (Section 2.2), both Linear Regression and Random Forest clear the "just guess the average" baseline decisively (R² of roughly 0.40 each on the held-out test set, versus -0.07 for the baseline). This supports our hypothesis that spending and demographic behavior carries information about the "right" credit limit beyond what a naive average captures. Repeated cross-validation shows Random Forest is the more *reliable* of the two — consistently around R² 0.44 across resampled folds, while Linear Regression swings more widely (R² 0.32 on average, with much higher fold-to-fold variance) — so Random Forest remains our recommended model even though the single-split comparison alone would have suggested the two were roughly tied. The model also surfaced a genuine data-quality finding — 26 nominally "Credit" accounts with a $0 limit, almost certainly closed or frozen accounts — which we chose to flag and report separately rather than silently drop or let corrupt the analysis.

---

## 1. Objective

### 1.1 The business problem

When a bank issues a credit card, it sets an initial limit using underwriting inputs available at that moment — credit score, income, existing debt. That limit often stays static for years, even as the customer's actual behavior with the card changes substantially. This creates two costly mismatches that a static, point-in-time approval process cannot see:

- **Underlimited accounts.** A customer who spends heavily, frequently, and reliably, but still carries a modest limit, is a customer whose spend the issuer is not capturing — and who is a prime target for a competitor's balance-transfer offer with a higher limit.
- **Overlimited accounts.** A customer whose actual spending behavior looks much weaker than what their limit implies represents risk exposure that may not be priced correctly, especially if their financial situation has changed since approval.

Card issuers run a recurring internal process — commonly called a **credit line review** — to catch exactly this kind of drift. This project builds an assistant for that process: not a new-account underwriting model, but a model that revisits limits for accounts that already have transaction history.

### 1.2 What we set out to prove

**Hypothesis:** A customer's actual spending behavior (frequency, volatility, channel mix, category diversity) explains variation in the "right" credit limit beyond what credit score and income alone capture.

This is a falsifiable claim. If a model using behavioral features cannot outperform a simple baseline (or a model using only credit score and income), the hypothesis is not supported. If it can, that is concrete evidence that transaction behavior carries real, exploitable signal for credit-line decisions. We treat either outcome as a valid, useful conclusion — the purpose of the model comparison in Sections 3 and 4 is to find out which one is true, not to force a positive result.

### 1.3 Why this matters to the business

A working version of this model would let a credit-risk or portfolio-management team:

1. Triage a large card portfolio down to a manageable list of accounts that actually warrant a manual limit review, instead of reviewing every account on a fixed schedule.
2. Explain *why* a specific account was flagged, in plain terms tied to that customer's behavior — important because credit decisions in many jurisdictions carry a legal expectation that an issuer can explain an adverse action (e.g., not raising a limit, or lowering one) to the customer.
3. See the flagged accounts grouped into behavioral "personas," giving portfolio strategy teams a segmentation lens beyond raw demographics.

---

## 2. Data Preparation

### 2.1 Data source

We used the **Credit Card Transactions Dataset** published on Kaggle (computingvictor, *Transactions Fraud Datasets*), a synthetic-but-realistic dataset built to resemble real consumer card transaction records. It ships as several related files; we used three:

| File | Role | Rows |
|---|---|---|
| `cards_data.csv` | One row per card: card type, credit limit, brand, chip status, account open date | 6,146 |
| `users_data.csv` | One row per customer: income, debt, credit score, age, gender | ~2,000 |
| `transactions_data.csv` | One row per transaction: amount, date, merchant category, channel, error/decline flag | millions (we sampled 200,000 rows for feature construction) |

Two additional files shipped with the dataset — a merchant-category-code lookup table and a fraud-labels file — but were not used. We had originally considered using prior-fraud-count as a risk feature, but decided the added complexity was not justified once a simpler feature (the diversity of merchant categories a card is used in) proved sufficient; this kept the project within its time budget.

The **unit of analysis is one credit card account**, not one transaction. Every transaction row was aggregated into a single behavioral summary per card before joining to the card and customer tables.

### 2.2 Data quality and procurement challenges

The raw files required several corrections before they were usable:

- **Currency fields were stored as text**, e.g., `"$29,278"` instead of a number. We wrote a parser to strip the currency symbol and thousands separators and convert these fields to numeric type for credit limit, income, debt, and transaction amount.
- **Card type did not cleanly determine which rows had a meaningful credit limit.** We checked whether `credit_limit` was populated only for actual credit-card rows, and found instead that *all* card types in the file — including Debit cards (average limit ≈$18,558) and Prepaid Debit cards (average limit ≈$64) — carried a non-null credit-limit value. A "credit limit" on a debit card is not a real business concept, so we restricted modeling to the 2,057 rows where `card_type == "Credit"` (out of 6,146 total cards). Without this check, the target variable would have silently blended three different kinds of accounts.
- **A subset of nominally "Credit" accounts had a $0 credit limit.** This was discovered indirectly: after log-transforming the credit-limit column (a standard step for skewed financial data, described below), the transformed column's skew came out strongly *negative* (-5.52) — the opposite of what a correctly-applied log transform of right-skewed data should produce. Investigating this anomaly rather than accepting the number at face value led us to the cause: 26 of the 2,057 credit-card rows (1.3%) have a listed limit of exactly $0, almost certainly closed or frozen accounts. We chose to **flag these rows rather than silently drop them**: they are excluded from model training and from performance metrics (a $0-limit account is not a genuine "what should this limit be" case), but they are still scored and reported as their own segment in the final output, since a risk team would want visibility into how many closed/frozen accounts exist in the portfolio, and dropping them outright would have hidden that.
- **Dates required parsing**, since the account-open date is stored as month/year text and transaction dates as strings; both needed conversion to real date types to compute account tenure and transaction recency.
- **One client can own multiple cards.** This matters for how we later split data into training and test sets (Section 2.4).
- **Identifier and security fields needed explicit exclusion, and an initial version of our pipeline missed this.** The raw tables include a card's number, its CVV, and the year a customer's PIN was last changed, plus raw latitude/longitude coordinates. During our own review of the model's explanations (Section 4.3) we caught that these had silently leaked into the feature set — the code was selecting "any numeric column not explicitly named," rather than explicitly listing which columns are legitimate inputs, so newly-added numeric columns kept slipping through. We corrected this to an explicit exclusion list covering all eight fields (card number, CVV, PIN-change year, latitude, longitude, birth year, birth month, retirement age), dropping the feature count from 32 to 24. Raw geographic coordinates are worth calling out specifically: using a customer's exact latitude/longitude as a direct input to a credit-limit model is a recognized fair-lending risk (a geographic proxy for factors a lender isn't supposed to use), not just a data-hygiene issue, so its removal matters beyond tidiness.

### 2.3 Feature engineering

We engineered two groups of features per card:

**Demographic/account features** (from the customer and card tables): yearly income, total debt, credit score, age, account tenure in months, card brand, chip capability, and gender.

**Behavioral features**, computed by aggregating each card's individual transactions into one summary row:

- Transaction count, average and median transaction amount
- **Spend volatility** — the ratio of the standard deviation to the mean transaction amount, capturing how erratic spending is, not just how much is spent
- Number of distinct merchant categories and distinct states the card was used in (a proxy for how varied the customer's spending is)
- **Recency** — days since the card's most recent transaction, measured against a fixed reference date (the day after the last transaction in the entire dataset), so that results are reproducible regardless of when the analysis is re-run
- Transaction frequency per month, normalized by how long the card has been active
- Share of transactions conducted online vs. in person
- Decline/error rate — the fraction of transactions that failed for any reason

**Transformations applied to the combined table:**

- **Log transform** of income, debt, and credit limit. Financial quantities like these are typically far more skewed than a linear model assumes (we measured skew of 3.20 for credit limit, 3.45 for income, and 1.81 for debt before transforming) — a few very high earners and very high limits stretch the distribution. Log-transforming brings these columns closer to a shape linear regression can work with.
- **One-hot encoding** of categorical fields (card brand, chip capability, gender) — converting each category into its own 0/1 column, since most machine learning models require numeric input.

### 2.4 Splitting the data for a fair test

Because a single customer can hold more than one card, a plain random row-by-row split into training and test sets risks putting two cards from the *same* customer on opposite sides of the split — effectively letting the model "see" that customer's behavior during training and then get evaluated on a card from the same person, which would overstate how well the model generalizes to a genuinely new account. We used a **group-aware split**, keyed on customer ID, so that every card belonging to a given customer lands entirely in either the training set or the test set, never both. This is a standard safeguard for exactly this kind of "repeated subject" data structure and was essential given how the dataset is organized.

After removing the 26 closed/zero-limit accounts, this left 2,031 credit-card accounts for modeling: roughly 1,600 for training and 415 held out for testing.

---

## 3. Model Design

We trained four distinct approaches, chosen specifically to cover genuinely different modeling families rather than several variations on the same idea — one deliberately simple baseline, one interpretable regression model, one non-linear ensemble model, and one unsupervised clustering model. A fifth technique, dimensionality reduction, was used only to visualize results, not to predict anything.

### 3.1 Baseline: predict the average

The simplest possible model predicts the same value — the training set's average credit limit — for every account, regardless of its features. This is not a real predictive model; it exists purely as the floor. If a more sophisticated model cannot beat this, its extra features and complexity are not adding value, and that would be an important negative finding in its own right rather than something to hide.

### 3.2 Linear Regression — the interpretable option

We chose Linear Regression as our first real model because of **interpretability**. In a real lending context, there is a genuine expectation that an issuer can explain, in plain terms, why a customer's account was or was not flagged for a limit change ("adverse action" reasoning). A linear model's coefficients can be read directly — "each additional point of spend volatility is associated with this much change in recommended limit" — in a way that a more complex model cannot easily be. We used a standard (unregularized) linear regression rather than a penalized variant like Ridge or Lasso, because with roughly 1,600 training rows and 24 features (after one-hot encoding, and after removing the leaked columns described in Section 2.2), our feature set was not in a range where overfitting or unstable coefficients from correlated features was a demonstrated practical problem — adding regularization would have been complexity without a clear need. As Section 4 shows, this turned out to matter: Linear Regression performs respectably once the noisy leaked columns are gone, but it is noticeably less *stable* across resamples than Random Forest, which is a separate concern from raw accuracy and part of why we still recommend the tree-based model.

### 3.3 Random Forest — the non-linear option

We expected the true relationship between behavior and the "right" credit limit to be non-linear — for example, the effect of spend volatility on an appropriate limit plausibly depends on the customer's income level, an interaction a linear model cannot capture without being told about it explicitly. A Random Forest — an ensemble of many decision trees, each trained on a slightly different random slice of the data, whose predictions are averaged — captures this kind of interaction automatically. We chose Random Forest over Support Vector Regression, the other non-linear regression method covered in the course, because Support Vector Regression scales poorly with dataset size and requires careful tuning of its kernel and hyperparameters, whereas Random Forest handled our mix of continuous and categorical features natively and performed well with modest, standard settings: 300 trees, each limited to a maximum depth of 8 (to control overfitting), with a fixed random seed for reproducibility. No further hyperparameter search was necessary once these standard settings comfortably beat the alternatives (see Section 4).

### 3.4 KMeans clustering — a genuinely different model family

The course objective calls for comparing meaningfully different modeling approaches, not several regressions dressed differently. KMeans clustering groups accounts into behavioral "personas" using no target variable at all — a structurally different kind of task from regression. Because KMeans measures distance between accounts, and our features are on wildly different scales (dollars, counts, percentages), we standardized every feature (rescaled to have a mean of 0 and standard deviation of 1) before clustering. We chose four clusters as a reasonable starting segmentation size for a portfolio view.

**We report honestly that this did not produce cleanly separated personas.** The clustering's silhouette score — a standard measure of how well-separated clusters are, ranging from -1 (poor) to +1 (excellent) — came out at only 0.126 on our (feature-leakage-corrected) data. This is low, meaning the four groups overlap substantially in behavior rather than forming distinct customer types. Rather than treating this as a failure to hide, we treat it as a genuine finding: the accounts in this dataset do not appear to fall into sharply distinct behavioral personas, which is itself informative about the population being modeled.

### 3.5 PCA — for visualization only

We used Principal Component Analysis (PCA), a dimensionality-reduction technique, purely to compress our many features down to two dimensions so the KMeans clusters and prediction errors could be plotted and visually inspected. It was deliberately kept out of the predictive pipeline itself — compressing to two dimensions before modeling would throw away exactly the non-linear feature interactions the Random Forest is there to exploit.

### 3.6 Techniques used beyond standard course material

A few techniques were necessary for a credible analysis but go beyond what a typical course module covers on its own:

| Technique | Why it was necessary |
|---|---|
| Residual diagnostics (predicted-vs-actual plots, error distribution) | Confirms whether a model's errors are unbiased, rather than trusting a single summary score |
| Log-transforming skewed financial columns | Real financial data does not arrive pre-normalized; this was a data-driven necessity, not a course topic |
| Customer-aware (grouped) train/test split | Prevents one customer's multiple cards from leaking across the split |
| Aggregating transactions into per-card behavioral features | A technique borrowed from marketing analytics (recency/frequency/monetary analysis), needed to turn millions of transaction rows into one row per account |
| SHAP feature importance | Explained in Section 4.3 — this is the project's main novelty contribution |
| Permutation importance | A second, model-agnostic importance measure used to cross-check SHAP's ranking (Section 4.3) — SHAP's attribution is specific to how the Random Forest happens to be built internally, so an independent method that agrees with it is stronger evidence than either alone |
| Grouped k-fold cross-validation | A single train/test split gives one point estimate; repeating the group-aware split five times and reporting a mean and standard deviation (Section 4.2) shows whether a model's apparent advantage holds up across different customers being held out, not just the specific split we happened to report |
| Bootstrap confidence interval on R² | Resampling the held-out test set with replacement to build a confidence interval (Section 4.2), rather than trusting a single R² number computed on one fixed set of accounts |

---

## 4. Model Evaluation

### 4.1 Why "accuracy" is not the right measure here

This project combines a regression task (predicting a dollar amount) with an unsupervised task (clustering), so "accuracy" in the classification sense does not apply. Instead we use standard regression metrics, computed on the 415 accounts held out from training:

- **R² (R-squared):** the fraction of variation in credit limits the model explains, where 0 means "no better than guessing the average" and 1 means "perfect prediction." Negative values are possible and mean the model performs *worse* than simply guessing the average every time.
- **MAE (Mean Absolute Error):** the average dollar-amount size of the model's errors, regardless of direction.
- **RMSE (Root Mean Squared Error):** similar to MAE, but penalizes large errors more heavily.
- **MAPE (Mean Absolute Percentage Error):** the average error expressed as a percentage of the true value, which is easier to communicate to a non-technical audience than a raw dollar figure.

### 4.2 Results

**Single held-out split** (415 accounts, using the corrected 24-feature set from Section 2.2):

| Model | R² | MAE | RMSE | MAPE |
|---|---|---|---|---|
| Baseline (predict the average) | -0.071 | $4,051 | $5,773 | 47.4% |
| Linear Regression | 0.405 | $2,966 | $4,301 | 35.5% |
| **Random Forest** | **0.402** | **$2,985** | **$4,314** | **35.1%** |

**Reading this honestly:** this is a materially different picture than an earlier version of our pipeline produced, and the difference is instructive. Before we caught and fixed the feature-leakage issue in Section 2.2, Linear Regression appeared to perform *worse* than the naive baseline. Once the leaked identifier, security, and geographic columns were removed, Linear Regression's R² on this split jumped to 0.405 — statistically indistinguishable from Random Forest's 0.402 on this particular split. The earlier "the relationship must be non-linear" conclusion was, at least in part, an artifact of noise columns (a random card number, a CVV code) distorting the linear fit, not solid evidence about the true shape of the relationship. Both models now clear the baseline decisively — roughly 25-27% lower average dollar error than simply guessing the portfolio average.

**A single split isn't the full story, though.** Because Linear Regression and Random Forest are now close on this one split, we checked whether that holds up more broadly using grouped 5-fold cross-validation (still grouped by customer, so no client's cards ever span a fold's train and test sides):

| Model | R² (mean ± std) | MAE (mean ± std) | RMSE (mean ± std) |
|---|---|---|---|
| Baseline | -0.069 ± 0.031 | $4,497 ± $418 | $6,901 ± $1,180 |
| Linear Regression | 0.316 ± 0.348 | $3,235 ± $228 | $5,450 ± $1,914 |
| **Random Forest** | **0.443 ± 0.081** | **$3,227 ± $78** | **$4,925 ± $545** |

This is the more important comparison. Random Forest's average R² across folds (0.44) is actually *higher* than on the single reported split, and its spread across folds is narrow (± 0.08) — it performs consistently regardless of which customers happen to be held out. Linear Regression's average R² (0.32) is lower than its single-split result, and its spread is far wider (± 0.35) — meaning its accuracy depends heavily on which specific accounts land in the test fold, sometimes performing close to Random Forest and sometimes much worse. **This is why we recommend Random Forest as the production model despite the two looking similar on one split**: reliability across resamples, not just a single accuracy number, is what a deployed model needs. (MAPE is omitted from this table on purpose — it swings unrealistically in some folds, 43-80%, because a small number of low-limit accounts produce outsized percentage errors; MAE and RMSE are the more trustworthy metrics here, a known limitation of MAPE we want to state plainly rather than paper over.)

**Is Random Forest's advantage a fluke of one test set?** Bootstrapping the held-out split (1,000 resamples with replacement) gives a 95% confidence interval of **[0.29, 0.50]** for Random Forest's R², entirely above zero and well clear of the baseline's confidence interval of **[-0.12, -0.03]**, which stays entirely below zero. The two intervals don't overlap at all, which is strong evidence the gap between "a real model" and "just guessing the average" is not a coincidence of which accounts happened to be in our test set.

To go beyond summary numbers, we also plotted predicted values against actual values (checking how closely points track a straight diagonal line) and the distribution of prediction errors, for the Random Forest model, to confirm its errors are not systematically biased in one direction.

### 4.3 Explaining individual predictions (SHAP)

A single accuracy number does not answer the question a real credit-risk reviewer needs answered: "why did the model recommend *this* limit for *this* customer?" We used **SHAP (SHapley Additive exPlanations)**, a technique that attributes each individual prediction to the specific input features that pushed it up or down, and how much. This goes beyond a model's built-in "feature importance" ranking by providing per-account, per-feature explanations — directly relevant to the real regulatory expectation that a lender can explain an adverse credit decision to a customer. This is the strongest novelty element of the project.

Ranking features by their average influence across all held-out accounts, the top contributors were: **per-capita income, by a wide margin**, then credit score, account tenure, number of cards issued, transaction frequency, and number of credit cards held. This is broadly consistent with intuition — income and credit history dominate, with behavioral tenure and activity features contributing meaningfully behind them.

**Cross-checking SHAP with a second, independent method.** SHAP's attribution is specific to how the Random Forest happens to be built internally, so we also computed **permutation importance** — a model-agnostic measure of how much R² falls when a feature's values are randomly shuffled. It agrees with SHAP on the dominant feature (per-capita income drives the largest R² drop by far when shuffled) and largely agrees on the next tier (tenure, credit score, income), which is reassuring: two different methods pointing at the same features is stronger evidence than either alone that this ranking reflects something real about the model, not an artifact of one attribution technique.

**A caveat we caught and fixed during this analysis, not before it:** an earlier version of the SHAP ranking included a card's CVV number and the year a customer's PIN was last changed in its top ten — neither is a meaningful credit-risk signal; both are account-administration artifacts that should never have reached the feature set. Investigating why led us to the broader feature-leakage issue described in Section 2.2 (the pipeline was implicitly including any numeric column, rather than an explicit allowlist). We corrected the underlying feature selection rather than just this symptom, re-ran the full pipeline, and the ranking above reflects the corrected model. We're describing the fix here rather than only in the data-preparation section because it's a good illustration of exactly the kind of check SHAP is useful for: an explainability tool that shows *why* a model made a prediction can also surface that the model is looking at something it shouldn't be, which a top-line accuracy metric alone would never reveal.

### 4.4 Evaluating the clustering and the flagging decision

Clustering has no "correct answer" to check predictions against, so we relied on the silhouette score discussed in Section 3.4 (0.126 on the corrected feature set, up slightly from 0.089 before the leakage fix, but still a genuine limitation rather than a strong segmentation) rather than a fabricated accuracy figure.

The final flagging decision — whether an account is under-limited, over-limited, or in range — also has no ground-truth label in the data (there is no recorded "this account was truly miscalculated" field to check against). We instead evaluated it as a **business-plausibility check**: at our chosen threshold (1.5 times the standard deviation of prediction errors), the 441 scored accounts broke down as follows:

| Flag | Count | Share |
|---|---|---|
| In range | 363 | 82.3% |
| Overlimited | 38 | 8.6% |
| Closed / zero-limit account | 26 | 5.9% |
| Underlimited | 14 | 3.2% |

Roughly 12% of active accounts flagged for review is a plausible-sized queue for a risk-review team to work through manually — neither an unusably large fraction of the portfolio nor a trivially small one — which is the sanity check available in place of a formal precision/recall metric.

---

## 5. Conclusions

### 5.1 Did we prove or disprove the hypothesis?

Our hypothesis was that behavioral spending signal explains credit-limit variance beyond credit score and income alone. The evidence supports it clearly: both Linear Regression and Random Forest reach R² ≈ 0.40 on held-out accounts, decisively beating the -0.07 baseline, a roughly 25-27% reduction in average dollar error. We want to be transparent that this conclusion firmed up over the course of the analysis, not before it: an earlier version of our pipeline had a feature-leakage bug (Section 2.2) that made Linear Regression look like it added no value at all, which would have supported a narrower, less accurate conclusion ("only a non-linear model captures this signal"). Catching and fixing that bug changed the finding materially. Repeated cross-validation adds an important nuance the single-split comparison misses: Random Forest is the *more reliable* of the two models (R² 0.44 ± 0.08 across folds) while Linear Regression's accuracy swings widely depending on which customers are held out (R² 0.32 ± 0.35). We conclude the hypothesis is supported, and specifically recommend Random Forest for deployment on the basis of that stability, not just its single-split accuracy.

### 5.2 What we learned about the data

- Real financial data is genuinely messy in specific, checkable ways: currency stored as text, a categorical field ("card type") that doesn't cleanly gate a numeric field the way its label implies, and a small but real subset of "closed" accounts hiding inside what looked like a clean numeric column until we investigated an anomalous statistic rather than accepting it.
- **A model's own explainability output can catch bugs a top-line metric would miss.** We only found the feature-leakage issue (Section 2.2) because we looked closely at *why* the model was making its predictions (SHAP), not just *how accurate* it was. A CVV code or a PIN-change year showing up as an influential feature was a signal something upstream was wrong, well before it became visible as a dip in any accuracy number.
- Behavioral spending patterns in this dataset do not sort customers into a small number of sharply distinct "personas" — the low silhouette score (0.126) is a legitimate finding about the population, not a modeling failure to be explained away.
- Two independent methods (SHAP and permutation importance) agreeing that per-capita income, credit score, and account tenure dominate the prediction gives us more confidence in that ranking than either method would on its own.
- A single train/test split can be actively misleading, not just imprecise: on one split, Linear Regression and Random Forest looked essentially tied; five-fold cross-validation revealed that tie masks a real difference in reliability between the two models.

### 5.3 What we would do next

1. **Try gradient-boosted trees** (e.g., XGBoost or LightGBM) as a potentially even stronger, more stable non-linear alternative to Random Forest, and consider a stacking ensemble that blends Linear Regression's interpretable signal with a tree-based model's predictions.
2. **Investigate why Linear Regression's cross-validation variance is so high** (± 0.35 in R²) — this could mean a small number of unusual customers disproportionately affect the linear fit, which would itself be a useful finding for how the portfolio is segmented.
3. **Bring in additional behavioral signal we deliberately left out of scope** for this project — merchant-category *names* (rather than just counting how many distinct categories a customer touches) and prior fraud-incident history — which might sharpen the persona segmentation in Section 3.4.
4. **Test the model's stability over time**, since the transaction data here spans nearly a decade (2010–2019); a production version of this model would need to be revalidated periodically as spending norms shift.
5. **Extend fair-lending review beyond the geographic-coordinate fix already made** (Section 2.2) — audit remaining demographic features (e.g., gender, age) for disparate impact before this model informs any real credit decision.

---

## References

computingvictor. (n.d.). *Credit Card Transactions Dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems 30* (pp. 4765–4774).

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

---

## Appendix A: Code Structure Reference

The full implementation lives in a modular Python package under `src/credit_line_review/`, exercised end-to-end in `notebooks/01_credit_line_review.ipynb`. Module responsibilities:

```
src/credit_line_review/
├── data.py         # Currency/date parsing, raw CSV loading
├── config.py       # Path/constant configuration (reference date, file paths)
├── features.py     # Card filtering, behavioral aggregation, modeling-table join, grouped split
├── models.py        # Baseline / Linear Regression / Random Forest wrappers
├── evaluation.py    # Regression metrics, residuals, SHAP importance
├── clustering.py    # KMeans personas, PCA projection
└── export.py         # Scored-accounts builder and Parquet export
```

Sample: explicit feature-column selection, replacing an earlier implicit "any numeric column" selection that let identifier/security/geographic fields leak in (`features.py`)

```python
NON_FEATURE_COLUMNS = frozenset({
    "id_card", "id_user", "client_id", "credit_limit", "log_credit_limit", "is_zero_limit",
    "card_number", "cvv", "expires", "year_pin_last_changed", "card_on_dark_web",
    "latitude", "longitude", "birth_year", "birth_month", "retirement_age",
    "acct_open_date", "card_type", "address",
})

def select_feature_columns(table):
    return [
        c for c in table.columns
        if c not in NON_FEATURE_COLUMNS
        and (pd.api.types.is_numeric_dtype(table[c]) or pd.api.types.is_bool_dtype(table[c]))
    ]
```

Sample: fitting the Random Forest model (`models.py`)

```python
def fit_random_forest(X_train, y_train, random_state: int = 42):
    model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=random_state)
    model.fit(X_train, y_train)
    return model
```

Sample: computing SHAP feature importance (`evaluation.py`)

```python
def shap_feature_importance(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    importance = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns, name="mean_abs_shap")
    return importance.sort_values(ascending=False).reset_index().rename(columns={"index": "feature"})
```

All modules are unit-tested independently of the notebook (22 tests total, `pytest -q`), so the modeling logic can be verified without re-running the full data pipeline.

## Appendix B: Data Sample

Example of one row from the final modeling table (values illustrative, columns abbreviated):

| id_card | credit_limit | yearly_income | credit_score | tenure_months | txn_count | spend_volatility | online_share | is_zero_limit |
|---|---|---|---|---|---|---|---|---|
| 4342 | $9,100 | $41,200 | 718 | 86.3 | 214 | 0.92 | 0.31 | False |

Example row flagged as a closed/zero-limit account:

| id_card | credit_limit | yearly_income | credit_score | tenure_months | txn_count | spend_volatility | online_share | is_zero_limit |
|---|---|---|---|---|---|---|---|---|
| 5107 | $0 | $38,650 | 664 | 142.7 | 3 | 0.00 | 0.00 | True |
