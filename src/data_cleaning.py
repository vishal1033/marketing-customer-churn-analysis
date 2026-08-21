"""
data_cleaning.py
-----------------
Stage 2: cleans the raw Online Retail II transactions.

This dataset's messiness is different in character from Project 1's (SBA)
— it's transactional/behavioral messiness rather than form-field
missingness:
  - Cancelled orders (Invoice starts with 'C') — real cancellations, not
    noise, but must be handled deliberately for RFM/purchase modeling
  - Negative Quantity — returns, tied to cancellations
  - Zero or negative Price — invalid/free-sample transactions
  - Missing Customer ID — anonymous/guest purchases; cannot be used for
    customer-level RFM features (no customer to attribute them to), so
    excluded from customer-level analysis but documented, not silently
    dropped without explanation
  - Duplicate rows
  - Inconsistent Description text (free-text product names)

Each is logged and handled explicitly — same philosophy as Project 1's
cleaning stage: no blanket dropna(), every decision justified.
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "online_retail_raw.csv"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    pct_missing = df.isna().mean().sort_values(ascending=False) * 100
    report = pct_missing[pct_missing > 0].to_frame("pct_missing")
    logger.info(f"Columns with missing values:\n{report}")
    return report


def flag_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    """Cancelled orders have an Invoice number prefixed with 'C'. These
    are real business events (a customer returned/cancelled), not data
    errors — flagged explicitly rather than dropped, since a customer's
    cancellation history is itself a potentially predictive feature for
    churn."""
    invoice_col = "Invoice" if "Invoice" in df.columns else "InvoiceNo"
    df["IsCancellation"] = df[invoice_col].astype(str).str.startswith("C")
    n_cancelled = df["IsCancellation"].sum()
    logger.info(f"Flagged {n_cancelled:,} cancellation rows "
                f"({n_cancelled / len(df) * 100:.2f}%)")
    return df


def flag_invalid_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Zero/negative price and negative quantity (outside of flagged
    cancellations) indicate data-entry errors or non-standard transaction
    types (e.g. manual adjustments, samples). Flagged, not silently
    dropped, so downstream analysis can decide whether to include them."""
    price_col = "Price" if "Price" in df.columns else "UnitPrice"
    df["InvalidPrice"] = df[price_col] <= 0
    df["NegativeQuantityNonCancellation"] = (df["Quantity"] < 0) & (~df["IsCancellation"])

    n_invalid_price = df["InvalidPrice"].sum()
    n_neg_qty = df["NegativeQuantityNonCancellation"].sum()
    logger.info(f"Invalid price (<=0): {n_invalid_price:,} rows "
                f"({n_invalid_price / len(df) * 100:.2f}%)")
    logger.info(f"Negative quantity outside cancellations: {n_neg_qty:,} rows "
                f"({n_neg_qty / len(df) * 100:.2f}%) — likely manual "
                f"adjustments/write-offs, not standard sales")
    return df


def flag_missing_customer(df: pd.DataFrame) -> pd.DataFrame:
    """Missing CustomerID = guest/anonymous purchase. These rows are
    legitimate transactions but can't be attributed to a customer, so
    they're excluded from customer-level RFM/churn modeling later —
    documented here rather than silently vanishing at the feature stage."""
    customer_col = "Customer ID" if "Customer ID" in df.columns else "CustomerID"
    df["MissingCustomerID"] = df[customer_col].isna()
    n_missing = df["MissingCustomerID"].sum()
    logger.info(f"Missing CustomerID (guest purchases): {n_missing:,} rows "
                f"({n_missing / len(df) * 100:.2f}%) — will be excluded from "
                f"customer-level modeling, retained here for transparency")
    return df


def standardize_description(df: pd.DataFrame) -> pd.Series:
    """Product descriptions are free text with inconsistent casing/spacing
    — standardize for any downstream product-level grouping."""
    return (
        df["Description"].astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
    )


def parse_invoice_date(df: pd.DataFrame) -> pd.Series:
    date_col = "InvoiceDate"
    return pd.to_datetime(df[date_col], errors="coerce")


def drop_duplicates_report(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Dropped {before - len(df)} duplicate rows")
    return df


def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    before_mb = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["object", "str"]).columns:
        if col != "Description" and df[col].nunique() / max(len(df), 1) < 0.5:
            df[col] = df[col].astype("category")

    after_mb = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"Memory usage: {before_mb:.1f} MB -> {after_mb:.1f} MB "
                f"({(1 - after_mb / before_mb) * 100:.1f}% reduction)")
    return df


def write_cleaning_report(before_shape, after_shape, flags_summary, before_mb, after_mb, out_path: Path):
    lines = [
        "# Data Cleaning Report — Online Retail II\n\n",
        f"- Rows before: {before_shape[0]:,} | after: {after_shape[0]:,}\n",
        f"- Columns before: {before_shape[1]} | after: {after_shape[1]}\n",
        f"- Memory usage: {before_mb:.1f} MB -> {after_mb:.1f} MB "
        f"({(1 - after_mb / before_mb) * 100:.1f}% reduction)\n\n",
        "## Flagged (not dropped) rows\n",
    ]
    for k, v in flags_summary.items():
        lines.append(f"- {k}: {v:,}\n")
    out_path.write_text("".join(lines))
    logger.info(f"Cleaning report written to {out_path}")


def main():
    df = pd.read_csv(RAW_PATH, low_memory=False)
    before_shape = df.shape
    before_mb = df.memory_usage(deep=True).sum() / 1024**2

    missingness_report(df)

    df = flag_cancellations(df)
    df = flag_invalid_transactions(df)
    df = flag_missing_customer(df)
    df["Description"] = standardize_description(df)
    df["InvoiceDate"] = parse_invoice_date(df)
    df = drop_duplicates_report(df)
    df = downcast_dtypes(df)

    after_shape = df.shape
    after_mb = df.memory_usage(deep=True).sum() / 1024**2

    flags_summary = {
        "Cancellations": int(df["IsCancellation"].sum()),
        "Invalid price (<=0)": int(df["InvalidPrice"].sum()),
        "Negative qty outside cancellations": int(df["NegativeQuantityNonCancellation"].sum()),
        "Missing CustomerID (guest purchases)": int(df["MissingCustomerID"].sum()),
    }

    out_path = PROCESSED_DIR / "online_retail_cleaned.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Cleaned dataset saved to {out_path}")

    write_cleaning_report(
        before_shape, after_shape, flags_summary, before_mb, after_mb,
        PROCESSED_DIR / "cleaning_report.md",
    )


if __name__ == "__main__":
    main()
