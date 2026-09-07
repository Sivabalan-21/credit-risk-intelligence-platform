"""
Trains the credit-default classifier.

Model choice: LightGBM.
  - Handles the mix of numeric + one-hot categorical features well, trains
    fast even on ~300K rows (fits the assignment's time-box), and gives
    native, well-supported SHAP TreeExplainer integration for Part 4.

Imbalance strategy: `scale_pos_weight` (ratio of negative:positive class),
rather than SMOTE/oversampling. On tabular credit data with hundreds of
features, synthetic oversampling tends to create unrealistic applicant
profiles and mainly slows training down without a reliable AUC gain;
cost-sensitive weighting keeps every training example real while still
correcting the model's bias toward the majority (repaid) class. This
trade-off is documented in the README.
"""
import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from src.data.loader import load_raw_train
from src.data.preprocessor import fit_transform_train, save_preprocessor
from src.ml.evaluate import compute_metrics
from src.utils.config import (
    MODEL_ARTIFACT_PATH, METRICS_PATH, RISK_THRESHOLDS_PATH, TEST_SIZE, RANDOM_STATE,
    RISK_LOW_PERCENTILE, RISK_MEDIUM_PERCENTILE,
)
from src.utils.helpers import save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_model(df=None) -> dict:
    if df is None:
        df = load_raw_train()

    X, y, feature_names, preprocessor, spec = fit_transform_train(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = n_neg / max(n_pos, 1)
    logger.info("Train class balance -> repaid=%d, default=%d, scale_pos_weight=%.2f",
                n_neg, n_pos, scale_pos_weight)

    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(X_train, y_train)

    y_val_proba = model.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(y_val, y_val_proba)
    metrics["n_train"] = int(len(X_train))
    metrics["n_val"] = int(len(X_val))
    metrics["n_features"] = len(feature_names)
    metrics["model"] = "LightGBM (LGBMClassifier)"
    metrics["imbalance_strategy"] = f"scale_pos_weight={scale_pos_weight:.2f}"

    # Risk-band thresholds, derived from percentiles of THIS model's own
    # validation-set output rather than fixed absolute cutoffs.
    #
    # Why: scale_pos_weight fixes ranking (AUC) but systematically inflates
    # predicted probabilities relative to the true ~8% base rate — the raw
    # output is not a calibrated probability. Fixed cutoffs like "Low if
    # < 10%" silently stop being reachable once probabilities are shifted
    # upward this way. Instead, risk bands are defined as relative tiers
    # within the model's own score distribution (bottom 60% = Low, next
    # 30% = Medium, top 10% = High by default) — this is also how risk
    # tiers are commonly defined in practice (score deciles), and it
    # guarantees a populated, usable Low/Medium/High split regardless of
    # calibration.
    risk_low_max = float(np.percentile(y_val_proba, RISK_LOW_PERCENTILE))
    risk_medium_max = float(np.percentile(y_val_proba, RISK_MEDIUM_PERCENTILE))
    risk_thresholds = {
        "risk_low_max": risk_low_max,
        "risk_medium_max": risk_medium_max,
        "low_percentile": RISK_LOW_PERCENTILE,
        "medium_percentile": RISK_MEDIUM_PERCENTILE,
        "note": (
            "Thresholds are percentiles of this model's own validation-set "
            "predicted probabilities, not absolute calibrated probabilities "
            "(scale_pos_weight training inflates raw probabilities — see train.py)."
        ),
    }
    logger.info(
        "Risk band thresholds (data-driven) -> Low < %.4f, Medium < %.4f, High >= %.4f",
        risk_low_max, risk_medium_max, risk_medium_max,
    )

    MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_ARTIFACT_PATH)
    save_preprocessor(preprocessor, spec, feature_names)
    save_json(metrics, METRICS_PATH)
    save_json(risk_thresholds, RISK_THRESHOLDS_PATH)

    logger.info("Model saved -> %s", MODEL_ARTIFACT_PATH)
    return metrics


if __name__ == "__main__":
    m = train_model()
    print("Training complete.")
    print(f"ROC-AUC: {m['roc_auc']}  |  PR-AUC: {m['pr_auc']}")