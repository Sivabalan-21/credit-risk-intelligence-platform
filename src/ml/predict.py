"""Inference: score one or more applicants using the saved model + preprocessor."""
import joblib
import pandas as pd

from src.data.preprocessor import load_preprocessor, transform_new
from src.utils.config import MODEL_ARTIFACT_PATH, RISK_THRESHOLDS_PATH, RISK_LOW_MAX, RISK_MEDIUM_MAX
from src.utils.helpers import risk_band, load_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

_model = None
_preprocessor = None
_spec = None
_feature_names = None
_risk_low_max = RISK_LOW_MAX
_risk_medium_max = RISK_MEDIUM_MAX


def _ensure_loaded():
    global _model, _preprocessor, _spec, _feature_names, _risk_low_max, _risk_medium_max
    if _model is None:
        try:
            _model = joblib.load(MODEL_ARTIFACT_PATH)
            _preprocessor, _spec, _feature_names = load_preprocessor()
        except (AttributeError, ModuleNotFoundError) as e:
            raise RuntimeError(
                "Failed to load the saved model/preprocessor — this almost always means "
                "the .pkl files in models/ were created with a different scikit-learn/"
                "lightgbm version than what's installed now (e.g. trained locally, then "
                "run inside Docker with a different library version). Fix: delete "
                "models/*.pkl and models/*.json, then retrain in the SAME environment "
                "that will run inference — e.g. `docker exec -it <container> python -m "
                "src.ml.train` if running in Docker."
            ) from e
        # Data-driven risk-band thresholds computed at training time (see
        # train.py) — falls back to the fixed config defaults only if a model
        # was trained before this file existed.
        if RISK_THRESHOLDS_PATH.exists():
            thresholds = load_json(RISK_THRESHOLDS_PATH)
            _risk_low_max = thresholds["risk_low_max"]
            _risk_medium_max = thresholds["risk_medium_max"]
            logger.info(
                "Loaded data-driven risk thresholds: Low < %.4f, Medium < %.4f",
                _risk_low_max, _risk_medium_max,
            )
        else:
            logger.warning(
                "No risk_thresholds.json found — using fixed fallback thresholds "
                "(Low < %.2f, Medium < %.2f). Retrain to generate data-driven thresholds.",
                _risk_low_max, _risk_medium_max,
            )
    return _model, _preprocessor, _spec, _feature_names


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a copy of df with probability, risk_band columns appended."""
    model, preprocessor, spec, _ = _ensure_loaded()
    X = transform_new(df, preprocessor, spec)
    proba = model.predict_proba(X)[:, 1]

    out = df.copy()
    out["default_probability"] = proba
    out["risk_band"] = [risk_band(p, _risk_low_max, _risk_medium_max) for p in proba]
    return out


def predict_single(applicant: dict) -> dict:
    """applicant: dict of raw feature_name -> value (same schema as application_train.csv, minus TARGET)."""
    df = pd.DataFrame([applicant])
    result = predict_batch(df)
    return {
        "default_probability": round(float(result["default_probability"].iloc[0]), 4),
        "risk_band": result["risk_band"].iloc[0],
    }


def get_transformed_for_explain(df: pd.DataFrame):
    """Used by explain.py — returns (X_transformed, feature_names, model)."""
    model, preprocessor, spec, feature_names = _ensure_loaded()
    X = transform_new(df, preprocessor, spec)
    return X, feature_names, model