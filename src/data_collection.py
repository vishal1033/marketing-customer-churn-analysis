"""
data_collection.py
-------------------
Reads the UCI "Online Retail" dataset from a manually-downloaded .xlsx
file in data/raw/.

NOTE: on some networks (this was hit on a locked-down work PC), Python's
HTTP/DNS calls are blocked entirely even though a browser on the same
machine can reach the internet fine — meaning ucimlrepo's API call and
direct requests/urllib downloads both fail with getaddrinfo errors,
while manually downloading the same file through a browser works. This
script sidesteps that: download the file manually, this script just
reads it from disk.

Manual download steps:
  1. Open in your browser: https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx
  2. Save the file as exactly: online_retail_raw.xlsx
  3. Place it in: data/raw/online_retail_raw.xlsx
  4. Run this script — it just reads the local file, no network needed.

Source: https://archive.ics.uci.edu/dataset/352/online+retail
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def main():
    xlsx_path = RAW_DIR / "online_retail_raw.xlsx"
    if not xlsx_path.exists():
        logger.error(
            f"File not found: {xlsx_path}\n"
            f"Download it manually from:\n"
            f"https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx\n"
            f"and save it at exactly that path."
        )
        return

    logger.info(f"Reading {xlsx_path} ...")
    df = pd.read_excel(xlsx_path)
    logger.info(f"Loaded shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")

    out_path = RAW_DIR / "online_retail_raw.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved raw dataset to {out_path}")


if __name__ == "__main__":
    main()
