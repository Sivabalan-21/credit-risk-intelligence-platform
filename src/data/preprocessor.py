"""
Cleaning, feature engineering, encoding and imputation for the Home Credit
application table. Built as an sklearn ColumnTransformer so the exact same
object (fit on train) is reused unchanged at inference time — no train/serve
skew.
"""
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.utils.config import TARGET_COL, ID_COL, PREPROCESSOR_ARTIFACT_PATH, FEATURE_LIST_PATH
from src.utils.helpers import save_json, load_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

# A few well-known Home Credit data-quality quirks we correct explicitly
# rather than silently imputing over them.
ANOMALOUS_DAYS_EMPLOYED_SENTINEL = 365243  # placeholder for "not employed"


@dataclass
class FeatureSpec:
    numeric_cols: list = field(default_factory=list)
    categorical_cols: list = field(default_factory=list)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-driven feature engineering — kept simple and explainable
    on purpose, since Part 4 (SHAP) needs features a credit analyst can
    reason about, not opaque PCA components."""
    df = df.copy()

    # Fix the known DAYS_EMPLOYED anomaly (365243 = "currently not working")
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == ANOMALOUS_DAYS_EMPLOYED_SENTINEL).astype(int)
        df.loc[df["DAYS_EMPLOYED"] == ANOMALOUS_DAYS_EMPLOYED_SENTINEL, "DAYS_EMPLOYED"] = np.nan

    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    if "DAYS_EMPLOYED" in df.columns:
        df["YEARS_EMPLOYED"] = -df["DAYS_EMPLOYED"] / 365.25

    # Affordability ratios — strong, well-known signal for this dataset
    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(df.columns):
        df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"].replace(0, np.nan)
    if {"YEARS_EMPLOYED", "AGE_YEARS"}.issubset(df.columns):
        df["EMPLOYED_AGE_RATIO"] = df["YEARS_EMPLOYED"] / df["AGE_YEARS"].replace(0, np.nan)

    return df


def infer_feature_spec(df: pd.DataFrame, max_categories: int = 30) -> FeatureSpec:
    """Auto-detects usable numeric / categorical columns, dropping IDs,
    target, and any raw text/free-form columns with too many categories."""
    drop_cols = {TARGET_COL, ID_COL}
    spec = FeatureSpec()
    for col in df.columns:
        if col in drop_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            spec.numeric_cols.append(col)
        else:
            if df[col].nunique(dropna=True) <= max_categories:
                spec.categorical_cols.append(col)
            # else: skip high-cardinality text columns
    return spec


def build_preprocessing_pipeline(spec: FeatureSpec) -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, spec.numeric_cols),
            ("cat", categorical_pipeline, spec.categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def get_output_feature_names(preprocessor: ColumnTransformer) -> list:
    return list(preprocessor.get_feature_names_out())


def fit_transform_train(df: pd.DataFrame):
    """Full train-time pipeline: engineer -> infer spec -> fit -> transform.
    Returns (X_transformed, y, feature_names, fitted_preprocessor, spec)."""
    df = engineer_features(df)
    y = df[TARGET_COL].astype(int)
    spec = infer_feature_spec(df)

    preprocessor = build_preprocessing_pipeline(spec)
    X = preprocessor.fit_transform(df[spec.numeric_cols + spec.categorical_cols])
    feature_names = get_output_feature_names(preprocessor)

    logger.info(
        "Preprocessing fit complete: %d numeric, %d categorical -> %d output features",
        len(spec.numeric_cols), len(spec.categorical_cols), len(feature_names),
    )
    return X, y, feature_names, preprocessor, spec


def transform_new(df: pd.DataFrame, preprocessor: ColumnTransformer, spec: FeatureSpec):
    """Inference-time transform using an already-fitted preprocessor."""
    df = engineer_features(df)
    cols = spec.numeric_cols + spec.categorical_cols
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return preprocessor.transform(df[cols])


def save_preprocessor(preprocessor: ColumnTransformer, spec: FeatureSpec, feature_names: list) -> None:
    joblib.dump({"preprocessor": preprocessor, "spec": spec}, PREPROCESSOR_ARTIFACT_PATH)
    save_json(feature_names, FEATURE_LIST_PATH)
    logger.info("Saved preprocessor -> %s", PREPROCESSOR_ARTIFACT_PATH)


def load_preprocessor():
    bundle = joblib.load(PREPROCESSOR_ARTIFACT_PATH)
    feature_names = load_json(FEATURE_LIST_PATH)
    return bundle["preprocessor"], bundle["spec"], feature_names
