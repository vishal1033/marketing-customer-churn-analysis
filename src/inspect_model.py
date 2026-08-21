"""
inspect_model.py
-----------------
Feature importance inspection — checks whether Recency dominates as EDA
predicted (r = 0.31 with Churned, the strongest pairwise correlation in
notebooks/01_eda.ipynb).

Reports BOTH Gini (impurity-based) and permutation importance, because
Gini importance has a well-known bias: it favors continuous/high-
cardinality features (more possible split points) over features like
Recency, independent of their true predictive contribution. Permutation
importance instead measures the actual drop in model performance when a
feature's values are shuffled — a more trustworthy (if noisier, and
more expensive to compute) signal of what the model actually relies on.
Reporting both and comparing them is itself the point: if they disagree,
that disagreement is informative, not just noise to average away.

Run after train.py:
    python src/inspect_model.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

MODEL_DIR = Path(__file__).resolve().parents[1] / "app" / "model"
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "online_retail_features.csv"
FEATURES = [
    "Recency", "Frequency", "Monetary", "TenureDays",
    "AvgOrderValue", "UniqueProducts", "CancellationRate",
]
TARGET = "Churned"
RANDOM_STATE = 42


def main():
    base_model = joblib.load(MODEL_DIR / "base_model.joblib")
    feature_names = joblib.load(MODEL_DIR / "feature_names.joblib")

    gini_importances = pd.Series(
        base_model.feature_importances_, index=feature_names
    ).sort_values(ascending=False)

    print("=== Gini (impurity-based) importances ===")
    print(gini_importances.to_string())
    print("\nNote: biased toward continuous/high-cardinality features "
          "(e.g. Monetary) regardless of true predictive value.\n")

    # Recreate the same test split used in train.py to compute permutation
    # importance on genuinely held-out data.
    df = pd.read_csv(DATA_PATH, low_memory=False).dropna(subset=FEATURES + [TARGET])
    X = df[FEATURES].values
    y = df[TARGET].values
    _, X_temp, _, y_temp = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
    _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp)

    perm_result = permutation_importance(
        base_model, X_test, y_test, n_repeats=30, random_state=RANDOM_STATE, scoring="roc_auc"
    )
    perm_importances = pd.Series(
        perm_result.importances_mean, index=feature_names
    ).sort_values(ascending=False)

    print("=== Permutation importances (mean AUC drop, 30 repeats) ===")
    print(perm_importances.to_string())


if __name__ == "__main__":
    main()

