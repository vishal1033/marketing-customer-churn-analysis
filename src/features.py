"""
features.py
------------
Stage 4 (combines target definition + feature engineering): builds
per-customer RFM (Recency, Frequency, Monetary) features and a
time-based binary churn label.

LEAKAGE GUARD — this is the central design decision of this script:
  - Pick a CUTOFF DATE = (latest date in the dataset - 90 days)
  - FEATURE WINDOW = all transactions strictly before the cutoff —
    RFM features are computed only from this window, so the model only
    ever sees "the past" relative to the cutoff
  - LABEL WINDOW = the 90 days after the cutoff — a customer is labeled
    Churned=0 if they made at least one valid (non-cancelled) purchase
    in this window, Churned=1 otherwise
  - Only customers who had at least one valid purchase in the FEATURE
    window are included — there's no history to build a prediction from
    for someone who only appears in the label window

This mirrors the leakage-awareness applied in Project 1 (excluding raw
ApprovalFY as a direct feature): here, the risk is using information
from AFTER the point-in-time you're supposedly predicting from. Without
this cutoff structure, an RFM snapshot built from the FULL dataset would
implicitly "know" about purchases that haven't happened yet relative to
any individual prediction point — a subtle but common mistake in churn
modeling tutorials that don't think about this carefully.

Cancelled transactions are excluded from RFM/purchase-counting entirely
(a cancellation isn't a completed purchase), but return behavior is
captured separately as a feature.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

IN_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "online_retail_cleaned.csv"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "online_retail_features.csv"

LABEL_WINDOW_DAYS = 90


def load_and_prepare(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # Exclude guest purchases (no CustomerID) — can't attribute to a
    # customer for customer-level modeling. Documented at cleaning stage.
    customer_col = "Customer ID" if "Customer ID" in df.columns else "CustomerID"
    df = df.rename(columns={customer_col: "CustomerID"})
    df = df[df["CustomerID"].notna()].copy()
    df["CustomerID"] = df["CustomerID"].astype(int)

    price_col = "Price" if "Price" in df.columns else "UnitPrice"
    df["LineTotal"] = df["Quantity"] * df[price_col]

    return df


def compute_cutoff(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    max_date = df["InvoiceDate"].max()
    cutoff = max_date - pd.Timedelta(days=LABEL_WINDOW_DAYS)
    logger.info(f"Data range: {df['InvoiceDate'].min().date()} to {max_date.date()}")
    logger.info(f"Cutoff date: {cutoff.date()} "
                f"(feature window before this, {LABEL_WINDOW_DAYS}-day label window after)")
    return cutoff, max_date


def build_rfm_features(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Computed ONLY on transactions before cutoff — the feature window."""
    feature_window = df[df["InvoiceDate"] < cutoff].copy()
    valid_purchases = feature_window[~feature_window["IsCancellation"]]

    grouped = valid_purchases.groupby("CustomerID")
    rfm = grouped.agg(
        Recency=("InvoiceDate", lambda x: (cutoff - x.max()).days),
        Frequency=("Invoice", "nunique") if "Invoice" in df.columns else ("InvoiceNo", "nunique"),
        Monetary=("LineTotal", "sum"),
        FirstPurchase=("InvoiceDate", "min"),
        LastPurchase=("InvoiceDate", "max"),
        UniqueProducts=("StockCode", "nunique"),
    )
    rfm["TenureDays"] = (rfm["LastPurchase"] - rfm["FirstPurchase"]).dt.days
    rfm["AvgOrderValue"] = rfm["Monetary"] / rfm["Frequency"]

    # Return behavior as a separate feature (cancellations counted here,
    # against total activity including cancellations, for this customer).
    all_customer_activity = feature_window.groupby("CustomerID")
    cancel_counts = all_customer_activity["IsCancellation"].sum()
    total_counts = all_customer_activity["IsCancellation"].count()
    rfm["CancellationRate"] = (cancel_counts / total_counts).reindex(rfm.index).fillna(0)

    rfm = rfm.drop(columns=["FirstPurchase", "LastPurchase"])
    logger.info(f"Customers with purchase history before cutoff: {len(rfm):,}")
    return rfm


def build_churn_label(df: pd.DataFrame, cutoff: pd.Timestamp, rfm_index: pd.Index) -> pd.Series:
    """Label window: (cutoff, cutoff + LABEL_WINDOW_DAYS]. A customer who
    already only has data up to `max_date` = cutoff + LABEL_WINDOW_DAYS
    by construction (see compute_cutoff), so this window is exactly the
    remaining tail of the dataset."""
    label_window = df[(df["InvoiceDate"] >= cutoff) & (~df["IsCancellation"])]
    customers_who_returned = set(label_window["CustomerID"].unique())

    churned = pd.Series(
        [0 if cust_id in customers_who_returned else 1 for cust_id in rfm_index],
        index=rfm_index, name="Churned",
    )
    churn_rate = churned.mean()
    logger.info(f"Churn rate (no purchase in {LABEL_WINDOW_DAYS}-day label window): "
                f"{churn_rate:.4f} ({churn_rate*100:.2f}%)")
    logger.info(f"Class counts:\n{churned.value_counts()}")
    return churned


def main():
    df = load_and_prepare(IN_PATH)
    cutoff, max_date = compute_cutoff(df)

    rfm = build_rfm_features(df, cutoff)
    churned = build_churn_label(df, cutoff, rfm.index)

    result = rfm.join(churned)
    result = result.reset_index()

    result.to_csv(OUT_PATH, index=False)
    logger.info(f"Feature-engineered dataset saved to {OUT_PATH}")
    logger.info(f"Final shape: {result.shape}")


if __name__ == "__main__":
    main()
