"""
Credit Risk Intelligence Platform — Streamlit UI.

Five sections in one app: EDA, Risk Prediction, Explainability, Business
Rules, and a Talk-to-Data chatbot — demonstrating the full pipeline end to
end for an evaluator with a single `streamlit run app.py`.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from src.data.loader import load_raw_train, build_sqlite_db
from src.data.preprocessor import fit_transform_train
from src.ml.train import train_model
from src.ml.predict import predict_single, get_transformed_for_explain
from src.ml.explain import explain_applicant, explain_summary_plot_data
from src.ml.rules import derive_rules, rules_to_readable
from src.utils.config import MODEL_ARTIFACT_PATH, TARGET_COL, ID_COL, ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY
from src.utils.helpers import load_json
from src.utils.docker_utils import startup_checklist

st.set_page_config(page_title="Credit Risk Intelligence Platform", page_icon="\U0001F4CA", layout="wide")


# --------------------------------------------------------------- caching ---
@st.cache_data(show_spinner=False)
def _load_data():
    return load_raw_train()


@st.cache_resource(show_spinner=False)
def _get_metrics():
    from src.utils.config import METRICS_PATH
    if METRICS_PATH.exists():
        return load_json(METRICS_PATH)
    return None


def _model_ready() -> bool:
    return MODEL_ARTIFACT_PATH.exists()


def _llm_configured() -> bool:
    return bool(ANTHROPIC_API_KEY or OPENAI_API_KEY or GEMINI_API_KEY)


# ----------------------------------------------------------------- header --
st.title("Credit Risk Intelligence Platform")
st.caption("Home Credit Default Risk — EDA, ML scoring, explainability, business rules, and a talk-to-data chatbot.")

checklist = startup_checklist()

if not checklist["dataset_present"]:
    st.error(
        "No dataset found at `data/application_train.csv`. Download it from the "
        "[Home Credit Default Risk Kaggle competition](https://www.kaggle.com/competitions/home-credit-default-risk/data) "
        "and place it in the `data/` folder, then reload this page."
    )
    st.stop()

df = _load_data()

if not checklist["model_trained"]:
    st.warning("No trained model found yet.")
    if st.button("Train model now", type="primary"):
        with st.spinner("Training LightGBM model on the applicant data..."):
            build_sqlite_db(df)
            metrics = train_model(df)
        st.success(f"Model trained — ROC-AUC {metrics['roc_auc']}, PR-AUC {metrics['pr_auc']}")
        st.cache_resource.clear()
        st.rerun()
    st.stop()

tabs = st.tabs(["\U0001F4CA EDA", "\U0001F3AF Risk Prediction", "\U0001F50D Explainability", "\U0001F4CB Business Rules", "\U0001F4AC Talk to Data"])


# ============================================================== EDA TAB ===
with tabs[0]:
    st.subheader("Dataset Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Applicants", f"{len(df):,}")
    c2.metric("Features", f"{df.shape[1] - 2}")
    default_rate = df[TARGET_COL].mean() * 100
    c3.metric("Default rate", f"{default_rate:.1f}%")
    c4.metric("Missing cells", f"{df.isna().sum().sum():,}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Data quality — missing values by column**")
        missing = df.isna().mean().sort_values(ascending=False)
        missing = missing[missing > 0].head(15) * 100
        if len(missing):
            st.bar_chart(missing.rename("% missing"))
        else:
            st.info("No missing values in this sample.")

    with col_b:
        st.markdown("**Feature categorization**")
        numeric_n = df.select_dtypes(include="number").shape[1]
        categorical_n = df.select_dtypes(exclude="number").shape[1]
        st.bar_chart(pd.Series({"Numeric": numeric_n, "Categorical": categorical_n}))

    st.markdown("---")
    st.subheader("Business Insights")

    st.markdown("**1. Default rate by income type**")
    if "NAME_INCOME_TYPE" in df.columns:
        rate_by_income = df.groupby("NAME_INCOME_TYPE")[TARGET_COL].mean().sort_values(ascending=False) * 100
        st.bar_chart(rate_by_income.rename("Default rate %"))

    st.markdown("**2. Default rate by education level**")
    if "NAME_EDUCATION_TYPE" in df.columns:
        rate_by_edu = df.groupby("NAME_EDUCATION_TYPE")[TARGET_COL].mean().sort_values(ascending=False) * 100
        st.bar_chart(rate_by_edu.rename("Default rate %"))

    st.markdown("**3. Income distribution: defaulters vs repaid**")
    if "AMT_INCOME_TOTAL" in df.columns:
        income_by_outcome = df.groupby(TARGET_COL)["AMT_INCOME_TOTAL"].median()
        income_by_outcome.index = ["Repaid", "Default"]
        st.bar_chart(income_by_outcome.rename("Median annual income"))

    st.markdown("**4. Credit-to-income ratio vs default**")
    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        ratio = (df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, pd.NA))
        ratio_by_outcome = ratio.groupby(df[TARGET_COL]).median()
        ratio_by_outcome.index = ["Repaid", "Default"]
        st.bar_chart(ratio_by_outcome.rename("Median credit/income ratio"))

    st.markdown("**5. Applicant age distribution**")
    if "DAYS_BIRTH" in df.columns:
        age = (-df["DAYS_BIRTH"] / 365.25).round().astype(int)
        age_hist = age.value_counts().sort_index()
        st.line_chart(age_hist.rename("Applicant count by age"))

    with st.expander("Raw sample data"):
        st.dataframe(df.head(50))


# ================================================== RISK PREDICTION TAB ===
with tabs[1]:
    st.subheader("Score an applicant")

    mode = st.radio("Input method", ["Pick an existing applicant", "Enter details manually"], horizontal=True)

    applicant_row = None
    if mode == "Pick an existing applicant":
        # A random (seeded, reproducible) sample rather than the first 200
        # rows by ID — application_train.csv is not shuffled by risk profile,
        # so head(200) could easily under-represent whole risk bands.
        sample_ids = df[ID_COL].sample(min(200, len(df)), random_state=42).sort_values().tolist()
        chosen_id = st.selectbox("Applicant ID", sample_ids)
        applicant_row = df[df[ID_COL] == chosen_id].drop(columns=[TARGET_COL]).iloc[0].to_dict()
        st.dataframe(pd.DataFrame([applicant_row]))
    else:
        with st.form("manual_applicant_form"):
            c1, c2, c3 = st.columns(3)
            income = c1.number_input("Annual income", min_value=0.0, value=150000.0, step=1000.0)
            credit = c2.number_input("Credit amount", min_value=0.0, value=500000.0, step=1000.0)
            annuity = c3.number_input("Annuity amount", min_value=0.0, value=25000.0, step=500.0)
            age = c1.number_input("Age (years)", min_value=18, max_value=100, value=35)
            years_employed = c2.number_input("Years employed", min_value=0.0, max_value=60.0, value=5.0)
            contract_type = c3.selectbox("Contract type", ["Cash loans", "Revolving loans"])
            education = c1.selectbox("Education", ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary"])
            income_type = c2.selectbox("Income type", ["Working", "State servant", "Pensioner", "Commercial associate", "Unemployed"])
            family_status = c3.selectbox("Family status", ["Married", "Single / not married", "Civil marriage", "Widow", "Separated"])
            gender = c1.selectbox("Gender", ["M", "F"])
            submitted = st.form_submit_button("Build applicant profile")

        if submitted:
            applicant_row = {
                "AMT_INCOME_TOTAL": income,
                "AMT_CREDIT": credit,
                "AMT_ANNUITY": annuity,
                "DAYS_BIRTH": -int(age * 365.25),
                "DAYS_EMPLOYED": -int(years_employed * 365.25),
                "NAME_CONTRACT_TYPE": contract_type,
                "NAME_EDUCATION_TYPE": education,
                "NAME_INCOME_TYPE": income_type,
                "NAME_FAMILY_STATUS": family_status,
                "CODE_GENDER": gender,
            }
            st.session_state["manual_applicant"] = applicant_row

    if applicant_row is None:
        applicant_row = st.session_state.get("manual_applicant")

    # Detect whether the currently-selected applicant differs from whichever
    # one produced the result sitting in session_state — e.g. the user picked
    # a new applicant ID (or rebuilt a manual profile) but hasn't clicked
    # "Score this applicant" again yet. Without this check, the OLD result
    # stays on screen looking like it belongs to the new selection.
    current_key = json.dumps(applicant_row, sort_keys=True, default=str) if applicant_row else None
    if current_key is not None and current_key != st.session_state.get("scored_applicant_key"):
        st.session_state.pop("last_result", None)

    if applicant_row and st.button("Score this applicant", type="primary"):
        result = predict_single(applicant_row)
        st.session_state["last_applicant"] = applicant_row
        st.session_state["last_result"] = result
        st.session_state["scored_applicant_key"] = current_key

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        band = result["risk_band"]
        color = {"Low": "\U0001F7E2", "Medium": "\U0001F7E1", "High": "\U0001F534"}[band]
        c1, c2 = st.columns(2)
        c1.metric("Default probability", f"{result['default_probability']*100:.1f}%")
        c2.metric("Risk band", f"{color} {band}")
    elif applicant_row:
        st.caption("Selection changed — click **Score this applicant** to see updated results.")

    metrics = _get_metrics()
    if metrics:
        st.markdown("---")
        st.subheader("Model performance (validation set)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ROC-AUC", metrics["roc_auc"])
        m2.metric("PR-AUC", metrics["pr_auc"])
        m3.metric("Recall (defaulters)", metrics["recall_defaulters"])
        m4.metric("Precision (defaulters)", metrics["precision_defaulters"])
        st.caption(f"Model: {metrics.get('model', 'n/a')} · Imbalance strategy: {metrics.get('imbalance_strategy', 'n/a')} "
                   f"· Trained on {metrics.get('n_train', '?')} rows, validated on {metrics.get('n_val', '?')} rows.")


# =================================================== EXPLAINABILITY TAB ===
with tabs[2]:
    st.subheader("Why did the model make this prediction?")
    st.caption("SHAP values — additive feature contributions. Positive = pushes risk up, negative = pushes risk down.")

    applicant_row = st.session_state.get("last_applicant")
    if not applicant_row:
        st.info("Score an applicant in the **Risk Prediction** tab first, then come back here.")
    else:
        with st.spinner("Computing SHAP explanation..."):
            explanation = explain_applicant(applicant_row, top_n=8)
        st.write(f"Base rate (average model output before this applicant's features): `{explanation['base_value']:.3f}`")
        exp_df = pd.DataFrame(explanation["top_features"])
        exp_df["shap_value"] = exp_df["shap_value"].round(3)
        st.dataframe(exp_df, width="stretch")
        st.bar_chart(exp_df.set_index("feature")["shap_value"])

    st.markdown("---")
    st.subheader("Global feature importance (sample of 200 applicants)")
    if st.button("Compute global importance"):
        with st.spinner("Running SHAP over a sample..."):
            sample_df = df.drop(columns=[TARGET_COL]).sample(min(200, len(df)), random_state=42)
            X_sample, feature_names, model = get_transformed_for_explain(sample_df)
            summary = explain_summary_plot_data(X_sample, feature_names, model)
        st.bar_chart(summary.set_index("feature")["mean_abs_shap"])


# ========================================================= RULES TAB =====
with tabs[3]:
    st.subheader("Business-readable decision rules")
    st.caption("A shallow surrogate decision tree trained to mimic the model's decision boundary — reviewable by a credit policy team, independent of the black-box model score.")

    depth = st.slider("Rule tree depth (more depth = more, narrower rules)", 2, 5, 3)
    if st.button("Derive rules", type="primary") or "rules" not in st.session_state:
        with st.spinner("Deriving rules..."):
            X, y, feature_names, _, _ = fit_transform_train(df)
            rules = derive_rules(X, y, feature_names, max_depth=depth)
            st.session_state["rules"] = rules

    rules = st.session_state.get("rules", [])
    if rules:
        readable = rules_to_readable(rules)
        for i, (r, text) in enumerate(zip(rules, readable), 1):
            band = r["risk_label"]
            icon = {"Low": "\U0001F7E2", "Medium": "\U0001F7E1", "High": "\U0001F534"}[band]
            st.markdown(f"{icon} **Rule {i}** — {text}")


# ======================================================= CHATBOT TAB =====
with tabs[4]:
    st.subheader("Ask questions about the applicant data")

    if not _llm_configured():
        st.warning(
            "No LLM API key configured. Set `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` + `LLM_PROVIDER=openai`, "
            "or `GEMINI_API_KEY` + `LLM_PROVIDER=gemini` (Gemini has a free tier) in your `.env` file."
        )
    else:
        from src.talk_to_data.nl_to_sql import TalkToDataSession

        if "chat_session" not in st.session_state:
            st.session_state["chat_session"] = TalkToDataSession()
        if "chat_display" not in st.session_state:
            st.session_state["chat_display"] = []

        st.caption(
            "Try: \"What is the average income of applicants who defaulted?\" · "
            "\"How many applicants are high risk vs low risk by education level?\" · "
            "\"Compare average annuity amount between genders\""
        )

        for turn in st.session_state["chat_display"]:
            with st.chat_message("user"):
                st.write(turn["question"])
            with st.chat_message("assistant"):
                st.write(turn["answer"])
                if turn.get("sql"):
                    with st.expander("SQL used"):
                        st.code(turn["sql"], language="sql")
                if turn.get("dataframe") is not None and not turn["dataframe"].empty:
                    with st.expander("Result data"):
                        st.dataframe(turn["dataframe"])

        question = st.chat_input("Ask a question about the applicant data...")
        if question:
            with st.spinner("Thinking..."):
                result = st.session_state["chat_session"].ask(question)
            st.session_state["chat_display"].append(result)
            st.rerun()
