"""
Explainable AI (Part 4): SHAP TreeExplainer over the trained LightGBM model.

Chosen over LIME because SHAP's TreeExplainer is exact (not a local
surrogate approximation) and fast for tree ensembles, and its values are
additive — they sum to the model's raw output, which makes "why did this
applicant score 62% risk" answerable in plain, audit-friendly language.
"""
import numpy as np
import pandas as pd
import shap

from src.ml.predict import get_transformed_for_explain
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _clean_feature_name(name: str) -> str:
    """Strips the sklearn ColumnTransformer 'num__' / 'cat__' prefixes for readability."""
    for prefix in ("num__", "cat__"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name

_explainer = None
_explainer_model_id = None


def _get_explainer(model):
    global _explainer, _explainer_model_id
    if _explainer is None or _explainer_model_id != id(model):
        _explainer = shap.TreeExplainer(model)
        _explainer_model_id = id(model)
    return _explainer


def explain_applicant(applicant: dict, top_n: int = 5) -> dict:
    """Returns the top contributing features (with direction) for a single applicant."""
    df = pd.DataFrame([applicant])
    X, feature_names, model = get_transformed_for_explain(df)

    explainer = _get_explainer(model)
    shap_values = explainer.shap_values(X)

    # LightGBM binary classifier: shap_values may be a list [class0, class1] or a single array
    if isinstance(shap_values, list):
        values = shap_values[1][0]
        base_value = explainer.expected_value[1]
    else:
        values = shap_values[0]
        base_value = explainer.expected_value

    contributions = pd.DataFrame({
        "feature": [_clean_feature_name(f) for f in feature_names],
        "shap_value": values,
        "feature_value": X[0],
    })
    contributions["direction"] = np.where(contributions["shap_value"] > 0, "increases risk", "decreases risk")
    contributions["abs_impact"] = contributions["shap_value"].abs()
    top = contributions.sort_values("abs_impact", ascending=False).head(top_n)

    return {
        "base_value": float(base_value),
        "top_features": top[["feature", "shap_value", "direction"]].to_dict(orient="records"),
    }


def explain_summary_plot_data(X_sample: np.ndarray, feature_names: list, model) -> pd.DataFrame:
    """Global feature importance across a sample of applicants, for the UI's overview chart."""
    explainer = _get_explainer(model)
    shap_values = explainer.shap_values(X_sample)
    values = shap_values[1] if isinstance(shap_values, list) else shap_values
    mean_abs = np.abs(values).mean(axis=0)
    return (
        pd.DataFrame({"feature": [_clean_feature_name(f) for f in feature_names], "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
