# Customer Churn Prediction — Online Retail (RFM Analysis)

End-to-end data science project: real UK e-commerce transactions →
cleaned dataset → RFM-based customer analytics → time-aware churn
prediction model → deployed inference API.

## Problem
Predict whether a customer will make a repeat purchase within 90 days,
based on their historical RFM (Recency, Frequency, Monetary) behavior —
a classic marketing-analytics problem: which customers are at risk of
churning and should be prioritized for retention outreach.

## Data
**Source:** [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)
(UCI Machine Learning Repository — canonical source, not Kaggle)

- Real transactions from a UK-based online gift retailer, Dec 2010–Dec 2011
- **541,909 raw transactions**, genuinely messy: 1.71% cancelled orders,
  0.46% invalid (≤0) prices, 0.25% negative-quantity non-cancellation
  rows, **24.93% missing CustomerID** (guest purchases), 5,268 duplicate
  rows, inconsistent free-text product descriptions
- Aggregated to **3,370 customers** with a time-based binary `Churned`
  label (see Feature Engineering below)

## Pipeline stages
| Stage | Script | What it does |
|---|---|---|
| 1. Collection | `src/data_collection.py` | Reads UCI Online Retail (manual download — see script docstring for why) |
| 2. Cleaning | `src/data_cleaning.py` | Flags cancellations, invalid transactions, missing customer IDs; dtype optimization |
| 3. EDA | `notebooks/01_eda.ipynb` | Mann-Whitney U tests on RFM vs churn, correlation analysis |
| 4. RFM + target | `src/features.py` | Time-based cutoff, leakage-aware RFM feature construction, churn labeling |
| 5. Modeling | `src/train.py` | Model comparison + calibration (sigmoid, given small calibration set) |
| 5.5 Interpretability | `src/inspect_model.py` | Gini vs permutation importance comparison |
| 6. Deployment | `app/main.py` | FastAPI inference service |

## Data cleaning methodology (Stage 2)
This dataset's messiness is behavioral/transactional rather than the
form-field type seen in the Business project — each issue flagged
explicitly rather than silently dropped:
- **Cancellations** (Invoice prefixed `'C'`): 9,288 rows (1.71%) — real
  business events, not noise
- **Invalid prices** (≤0): 2,517 rows (0.46%)
- **Negative quantity outside cancellations**: 1,336 rows (0.25%) —
  likely manual adjustments/write-offs
- **Missing CustomerID**: 135,080 rows (24.93%) — guest/anonymous
  purchases, excluded from customer-level modeling but retained in the
  cleaned dataset for transparency
- **Duplicates**: 5,268 rows dropped
- **Memory optimization**: 148.8MB → 61.1MB (59% reduction)
- Full report: `data/processed/cleaning_report.md`

## Feature engineering — leakage-aware time-based labeling (Stage 4)
The central design decision of this project: a naive RFM snapshot built
from the *entire* dataset would implicitly know about purchases that
haven't happened yet relative to any individual prediction point — a
common, subtle mistake in churn-modeling tutorials. This project avoids
it with an explicit cutoff structure:
- **Cutoff date**: `(latest date in data) − 90 days` = **2011-09-10**
  (computed dynamically, not hardcoded)
- **Feature window**: all transactions *before* cutoff → RFM features
  computed only from this window
- **Label window**: the 90 days *after* cutoff → `Churned = 1` if no
  valid purchase occurred in this window, else `0`
- Only customers with purchase history *before* the cutoff are included

Result: **3,370 customers, 43.00% churn rate** (1,449 churned / 1,921
retained) — well-balanced compared to Project 1's 15.77% default rate.

## EDA — key findings (Stage 3, see notebook for full detail)
All four hypotheses tested with Mann-Whitney U (non-parametric — RFM
distributions are right-skewed, not normal):
- **Recency** — strongest correlate (r = 0.31 with Churned). Churned
  customers' median recency was 114 days vs. 49 for retained
  (p = 1.24e-74)
- **Frequency** — confirmed (r = -0.25). Retained customers: median 3
  orders vs. 1 for churned (p = 6.62e-117, smallest p-value found)
- **Monetary** — confirmed but weakest RFM component (r = -0.13).
  Retained: £872 median spend vs. £348 for churned (p = 3.72e-98)
- **Cancellation rate** — statistically significant (p = 1.32e-26) but
  **practically negligible**: 2.57% vs 2.65% mean rate between groups.
  Deliberately included as an example of statistical vs. practical
  significance — the same distinction made for Project 1's loan-size
  finding
- **Multicollinearity**: Frequency and UniqueProducts correlate at
  r = 0.70 — documented, not ignored; acceptable for a tree-based model

## Modeling & calibration (Stage 5)
Three-way split (train/calibration/test), same discipline as Project 1:

| Model | ROC-AUC | Brier Score |
|---|---|---|
| Random Forest | 0.738 | 0.209 |
| XGBoost | 0.730 | 0.209 |
| **Random Forest + sigmoid calibration** | **0.738** | **0.205** |

**Calibration method choice — a different lesson than Project 1's**: an
initial run using isotonic regression (the method used in Project 1)
produced an unstable calibrated bin (`predicted=1.000, actual=0.000`) —
a direct consequence of isotonic's non-parametric flexibility overfitting
a calibration set with only ~500 rows. Switching to **sigmoid (Platt
scaling)** — a simpler, 2-parameter method — resolved this, since it's
far more stable with limited calibration data. Project 1 used isotonic
successfully because its calibration set was ~210K rows; the right
calibration method depends on how much calibration data is available,
not just habit.

## Interpretability (Stage 5.5) — Gini vs. permutation importance
Initial (Gini/impurity-based) feature importance ranked `Monetary`
highest — but Gini importance is known to be **biased toward continuous,
high-cardinality features**, independent of true predictive value. Cross-
checked with **permutation importance** (actual AUC drop when a feature
is shuffled):

| Feature | Gini rank | Permutation rank |
|---|---|---|
| Recency | 2nd | **1st** (0.033, ~2x the next feature) |
| Monetary | **1st** | 4th |

Permutation importance confirms `Recency` as the true dominant driver —
consistent with the EDA correlation analysis, and a clear demonstration
of why a single importance method shouldn't be trusted uncritically.

## Deployment (Stage 6)
FastAPI service takes a customer's RFM summary (as would typically be
maintained in a CRM/analytics table) and returns a calibrated churn
probability plus risk tier.

```bash
uvicorn app.main:app --reload --port 8001
# then open http://localhost:8001/docs
```

Or via Docker:
```bash
docker build -t customer-churn-api .
docker run -p 8001:8001 customer-churn-api
```

Example:
```json
POST /predict
{
  "recency": 45, "frequency": 4, "monetary": 850.0,
  "tenure_days": 180, "avg_order_value": 212.5,
  "unique_products": 30, "cancellation_rate": 0.02
}

{ "churn_probability": 0.31, "risk_tier": "Elevated" }
```

## Setup
```bash
pip install -r requirements.txt
# Manually download Online Retail.xlsx (see src/data_collection.py docstring)
python src/data_collection.py
python src/data_cleaning.py
python src/features.py
python src/train.py
python src/inspect_model.py
uvicorn app.main:app --reload --port 8001
```

## Project structure
```
marketing-customer-churn-analysis/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── data_collection.py
│   ├── data_cleaning.py
│   ├── features.py
│   ├── train.py
│   └── inspect_model.py
├── app/
│   ├── main.py
│   └── model/
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

## Author
Vishal Dubey — Statistics educator & researcher (Bayesian modeling, small
area estimation). Same statistical rigor as Project 1 — leakage-aware
labeling, calibration matched to data size, and cross-validated
interpretability — applied to a marketing analytics setting.
"# -marketing-customer-churn-analysis" 
"# -marketing-customer-churn-analysis" 
"# -marketing-customer-churn-analysis" 
