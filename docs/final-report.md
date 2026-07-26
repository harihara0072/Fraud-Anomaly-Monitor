# Credit Line Review Assistant: A Behavioral Model for Existing-Customer Credit Limit Review

**Course:** SCS 3253 – Machine Learning, Term Project
**Prepared by:** [Group Members — Hari Hara Kumar Nakshatrala, Jimmy Lopez, Remya Sara Raju]
**Date:** July 26, 2026

---

## Executive Summary

Credit card issuers typically set a customer's credit limit once, at account opening, based on credit score and income. This project asks whether an issuer can do better for **existing** customers by also looking at how they actually use their card — how often they spend, how volatile that spending is, how much of it happens online, and how many merchant categories they touch.

We built a "credit line review" model: for each existing credit-card account, it recommends a limit based on behavior and demographics, compares that recommendation to the customer's actual limit, and flags accounts that look **underlimited** (likely under-served, a churn and revenue risk) or **overlimited** (a risk exposure that may not be priced correctly). We trained and compared three model types — a naive baseline, Linear Regression, and Random Forest — plus an unsupervised KMeans clustering model to segment customers into behavioral personas.

**Bottom line:** the behavioral-plus-demographic Random Forest model explains real variance in credit limits that a linear model and a "just guess the average" baseline both miss (R² of 0.36 vs. -0.07 for the baseline). This supports our hypothesis that spending behavior carries information about the "right" credit limit beyond what credit score and income alone capture. The model also surfaced a genuine data-quality finding — 26 nominally "Credit" accounts with a $0 limit, almost certainly closed or frozen accounts — which we chose to flag and report separately rather than silently drop or let corrupt the analysis.

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

We chose Linear Regression as our first real model because of **interpretability**. In a real lending context, there is a genuine expectation that an issuer can explain, in plain terms, why a customer's account was or was not flagged for a limit change ("adverse action" reasoning). A linear model's coefficients can be read directly — "each additional point of spend volatility is associated with this much change in recommended limit" — in a way that a more complex model cannot easily be. We used a standard (unregularized) linear regression rather than a penalized variant like Ridge or Lasso, because with roughly 1,600 training rows and 32 features (after one-hot encoding), our feature set was not in a range where overfitting or unstable coefficients from correlated features was a demonstrated practical problem — adding regularization would have been complexity without a clear need.

### 3.3 Random Forest — the non-linear option

We expected the true relationship between behavior and the "right" credit limit to be non-linear — for example, the effect of spend volatility on an appropriate limit plausibly depends on the customer's income level, an interaction a linear model cannot capture without being told about it explicitly. A Random Forest — an ensemble of many decision trees, each trained on a slightly different random slice of the data, whose predictions are averaged — captures this kind of interaction automatically. We chose Random Forest over Support Vector Regression, the other non-linear regression method covered in the course, because Support Vector Regression scales poorly with dataset size and requires careful tuning of its kernel and hyperparameters, whereas Random Forest handled our mix of continuous and categorical features natively and performed well with modest, standard settings: 300 trees, each limited to a maximum depth of 8 (to control overfitting), with a fixed random seed for reproducibility. No further hyperparameter search was necessary once these standard settings comfortably beat the alternatives (see Section 4).

### 3.4 KMeans clustering — a genuinely different model family

The course objective calls for comparing meaningfully different modeling approaches, not several regressions dressed differently. KMeans clustering groups accounts into behavioral "personas" using no target variable at all — a structurally different kind of task from regression. Because KMeans measures distance between accounts, and our features are on wildly different scales (dollars, counts, percentages), we standardized every feature (rescaled to have a mean of 0 and standard deviation of 1) before clustering. We chose four clusters as a reasonable starting segmentation size for a portfolio view.

**We report honestly that this did not produce cleanly separated personas.** The clustering's silhouette score — a standard measure of how well-separated clusters are, ranging from -1 (poor) to +1 (excellent) — came out at only 0.089 on our data. This is low, meaning the four groups overlap substantially in behavior rather than forming distinct customer types. Rather than treating this as a failure to hide, we treat it as a genuine finding: the accounts in this dataset do not appear to fall into sharply distinct behavioral personas, which is itself informative about the population being modeled.

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

---

## 4. Model Evaluation

### 4.1 Why "accuracy" is not the right measure here

This project combines a regression task (predicting a dollar amount) with an unsupervised task (clustering), so "accuracy" in the classification sense does not apply. Instead we use standard regression metrics, computed on the 415 accounts held out from training:

- **R² (R-squared):** the fraction of variation in credit limits the model explains, where 0 means "no better than guessing the average" and 1 means "perfect prediction." Negative values are possible and mean the model performs *worse* than simply guessing the average every time.
- **MAE (Mean Absolute Error):** the average dollar-amount size of the model's errors, regardless of direction.
- **RMSE (Root Mean Squared Error):** similar to MAE, but penalizes large errors more heavily.
- **MAPE (Mean Absolute Percentage Error):** the average error expressed as a percentage of the true value, which is easier to communicate to a non-technical audience than a raw dollar figure.

### 4.2 Results

| Model | R² | MAE | RMSE | MAPE |
|---|---|---|---|---|
| Baseline (predict the average) | -0.071 | $4,051 | $5,773 | 47.4% |
| Linear Regression | -0.082 | $4,071 | $5,803 | 47.6% |
| **Random Forest** | **0.364** | **$3,036** | **$4,448** | **35.1%** |

**Reading this honestly:** Linear Regression actually performs slightly *worse* than the naive baseline (both have negative R²). This is a real result, not an error in our code — it means that a straight-line relationship between the available features and credit limit does not capture more signal than simply guessing the portfolio average. The pattern in this data is genuinely non-linear. Random Forest, by contrast, clears the baseline decisively — an R² of 0.36 versus -0.07, and roughly 25% lower average dollar error than the baseline or the linear model. This is the concrete evidence for our hypothesis in Section 1: behavioral and demographic features **do** explain real variance in credit limits, but only a model capable of capturing non-linear structure can access that signal. We did not build a separate ensemble that combines Linear Regression and Random Forest predictions together (e.g., a voting or stacking ensemble); Random Forest is itself an ensemble of individual decision trees, and given how decisively it outperformed the single alternatives, we prioritized explaining and validating that result over adding further model complexity — we note this as a natural next step in Section 5.

To go beyond a single summary number, we also plotted predicted values against actual values (checking how closely points track a straight diagonal line) and the distribution of prediction errors, for the winning Random Forest model, to confirm its errors are not systematically biased in one direction rather than trusting R² alone.

### 4.3 Explaining individual predictions (SHAP)

A single accuracy number does not answer the question a real credit-risk reviewer needs answered: "why did the model recommend *this* limit for *this* customer?" We used **SHAP (SHapley Additive exPlanations)**, a technique that attributes each individual prediction to the specific input features that pushed it up or down, and how much. This goes beyond a model's built-in "feature importance" ranking by providing per-account, per-feature explanations — directly relevant to the real regulatory expectation that a lender can explain an adverse credit decision to a customer. This is the strongest novelty element of the project.

Ranking features by their average influence across all held-out accounts, the top contributors were: per-capita income (by a wide margin), credit score, account tenure, number of cards issued, yearly income, and transaction frequency. This is broadly consistent with intuition — income and credit history dominate, with behavioral tenure and activity features contributing meaningfully behind them.

**An honest caveat on this ranking:** two features that appeared in the top ten — a card's CVV number and the year a customer's PIN was last changed — are not meaningful credit-risk signals; they are account-administration artifacts that should have been excluded from the feature set entirely, and were not. We are reporting this rather than silently correcting it in the write-up, because it is a genuine finding about the current state of the pipeline: these two columns leaked into the model as numeric fields when they should have been filtered out alongside other identifier fields (card number, expiry date). Their SHAP contribution was small relative to income and credit score, so they do not materially change the top-line conclusion, but they should be explicitly excluded before this model is used for anything beyond this course project. We list this as the first item in our recommended next steps (Section 5).

### 4.4 Evaluating the clustering and the flagging decision

Clustering has no "correct answer" to check predictions against, so we relied on the silhouette score discussed in Section 3.4 (0.089 — a genuine limitation, not a strong segmentation) rather than a fabricated accuracy figure.

The final flagging decision — whether an account is under-limited, over-limited, or in range — also has no ground-truth label in the data (there is no recorded "this account was truly miscalculated" field to check against). We instead evaluated it as a **business-plausibility check**: at our chosen threshold (1.5 times the standard deviation of prediction errors), the 441 scored accounts broke down as follows:

| Flag | Count | Share |
|---|---|---|
| In range | 364 | 82.5% |
| Overlimited | 38 | 8.6% |
| Closed / zero-limit account | 26 | 5.9% |
| Underlimited | 13 | 2.9% |

Roughly 12% of active accounts flagged for review is a plausible-sized queue for a risk-review team to work through manually — neither an unusably large fraction of the portfolio nor a trivially small one — which is the sanity check available in place of a formal precision/recall metric.

---

## 5. Conclusions

### 5.1 Did we prove or disprove the hypothesis?

Our hypothesis was that behavioral spending signal explains credit-limit variance beyond credit score and income alone. The evidence supports it, with an important qualifier: **a linear combination** of behavioral and demographic features adds no value over guessing the portfolio average (Linear Regression underperformed the baseline). It is specifically the *non-linear* Random Forest model that unlocks real predictive value (R² = 0.36 vs. -0.07 for the baseline), a roughly 25% reduction in average dollar error versus both simpler alternatives. We conclude the hypothesis is supported, but only when paired with a model flexible enough to capture non-linear interactions between behavior and demographics — a conclusion that itself informs how any future version of this model should be built (Section 5.3).

### 5.2 What we learned about the data

- Real financial data is genuinely messy in specific, checkable ways: currency stored as text, a categorical field ("card type") that doesn't cleanly gate a numeric field the way its label implies, and a small but real subset of "closed" accounts hiding inside what looked like a clean numeric column until we investigated an anomalous statistic rather than accepting it.
- Behavioral spending patterns in this dataset do not sort customers into a small number of sharply distinct "personas" — the low silhouette score is a legitimate finding about the population, not a modeling failure to be explained away.
- Interpretability and predictive power traded off directly in this project: the model we could explain most simply (Linear Regression) was also the one that added the least value, while the model that performed best (Random Forest) required an additional explainability layer (SHAP) to make its reasoning legible to a non-technical reviewer.

### 5.3 What we would do next

1. **Fix the feature-leakage issue identified in Section 4.3.** CVV and PIN-change-year fields should be explicitly excluded from the model's inputs before this analysis is used for anything beyond this course project; this is a data-hygiene fix, not a modeling change, and should be the first thing addressed.
2. **Try gradient-boosted trees** (e.g., XGBoost or LightGBM) as a stronger non-linear alternative to Random Forest, and consider a stacking ensemble that blends Linear Regression's interpretable signal with the tree-based model's non-linear power.
3. **Validate with grouped cross-validation** (multiple train/test splits, not just one) to report a mean and confidence interval for R²/MAE rather than a single point estimate, which would make the "Random Forest beats baseline" conclusion more statistically defensible.
4. **Bring in additional behavioral signal we deliberately left out of scope** for this project — merchant-category *names* (rather than just counting how many distinct categories a customer touches) and prior fraud-incident history — which might sharpen the persona segmentation in Section 3.4.
5. **Test the model's stability over time**, since the transaction data here spans nearly a decade (2010–2019); a production version of this model would need to be revalidated periodically as spending norms shift.

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
