"""
train.py
--------
Stage 5: model training for customer churn prediction.

Simpler class-balance situation than Project 1 (43% churn rate vs 15.77%
default rate), so aggressive imbalance handling isn't the central story
here — but calibration is still checked explicitly, same discipline as
Project 1, since a marketing team using this to prioritize retention
outreach needs trustworthy probabilities (e.g. to rank customers by
true churn risk, or estimate expected retention-campaign ROI), not just
a good AUC.

Features are all from src/features.py — the RFM feature set discussed
in notebooks/01_eda.ipynb. No categorical encoding needed here (unlike
Project 1) since all features are already numeric.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "online_retail_features.csv"
MODEL_DIR = Path(__file__).resolve().parents[1] / "app" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
FEATURES = [
    "Recency", "Frequency", "Monetary", "TenureDays",
    "AvgOrderValue", "UniqueProducts", "CancellationRate",
]
TARGET = "Churned"


def prepare_data(df: pd.DataFrame):
    df = df.dropna(subset=FEATURES + [TARGET]).copy()
    X = df[FEATURES].values
    y = df[TARGET].values

    # Three-way split, same discipline as Project 1: train (fit) / calib
    # (post-hoc calibration) / test (final untouched evaluation).
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )
    return X_train, X_calib, X_test, y_train, y_calib, y_test


def evaluate(model, X_test, y_test, label: str):
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)

    logger.info(f"\n=== {label} ===")
    logger.info(f"ROC-AUC: {auc:.4f}")
    logger.info(f"Brier score: {brier:.4f}")
    logger.info(f"\n{classification_report(y_test, preds, target_names=['Retained', 'Churned'])}")

    prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10)
    logger.info("Calibration (predicted vs actual, 10 bins):")
    for pt, pp in zip(prob_true, prob_pred):
        logger.info(f"  predicted={pp:.3f}  actual={pt:.3f}")

    return {"auc": auc, "brier": brier}


def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    X_train, X_calib, X_test, y_train, y_calib, y_test = prepare_data(df)

    logger.info(f"Train: {X_train.shape}, Calib: {X_calib.shape}, Test: {X_test.shape}")
    logger.info(f"Train churn rate: {y_train.mean():.4f}, Test churn rate: {y_test.mean():.4f}")

    # --- Model 1: Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    results_rf = evaluate(rf, X_test, y_test, "Random Forest (uncalibrated)")

    # --- Model 2: XGBoost ---
    xgb = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    results_xgb = evaluate(xgb, X_test, y_test, "XGBoost (uncalibrated)")

    best_raw, best_name = (
        (xgb, "xgboost") if results_xgb["auc"] >= results_rf["auc"] else (rf, "random_forest")
    )
    logger.info(f"Best model by AUC: {best_name} — recalibrating on held-out calibration set")

    try:
        from sklearn.frozen import FrozenEstimator
        # Sigmoid (Platt scaling), not isotonic: with only ~500 calibration
        # rows, isotonic regression's non-parametric flexibility overfits
        # sparse bins (observed directly: an initial isotonic run produced
        # a "predicted=1.000, actual=0.000" bin — a clear artifact of too
        # few points in that region, not a real finding). Sigmoid fits
        # just 2 parameters (a logistic curve), far more stable at this
        # sample size, at the cost of assuming a simpler calibration shape.
        # Project 1 used isotonic because its calibration set was ~210K
        # rows — plenty of data to support a flexible fit there.
        calibrated_model = CalibratedClassifierCV(FrozenEstimator(best_raw), method="sigmoid")
    except ImportError:
        calibrated_model = CalibratedClassifierCV(best_raw, method="sigmoid", cv="prefit")
    calibrated_model.fit(X_calib, y_calib)

    results_calibrated = evaluate(calibrated_model, X_test, y_test, f"{best_name} (sigmoid-calibrated)")

    uncalibrated_brier = results_xgb["brier"] if best_name == "xgboost" else results_rf["brier"]
    logger.info(
        f"\nCalibration impact — Brier score: {uncalibrated_brier:.4f} (uncalibrated) -> "
        f"{results_calibrated['brier']:.4f} (calibrated)."
    )

    joblib.dump(calibrated_model, MODEL_DIR / "model.joblib")
    joblib.dump(best_raw, MODEL_DIR / "base_model.joblib")
    joblib.dump(FEATURES, MODEL_DIR / "feature_names.joblib")
    logger.info(f"Model + artifacts saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
