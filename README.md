# AI-Powered Credit Risk Intelligence Platform

NeoStats AI Engineer Internship — Candidate Assignment.
Home Credit Default Risk dataset → EDA → ML → Explainability → Business Rules → Talk-to-Data → Docker.

Full write-up and screenshots: [`documents/project_presentation.pdf`](documents/project_presentation.pdf).

---

## 1. Architecture overview

             ┌──────────────┐     ┌──────────────┐     ┌────────────────┐     ┌──────────────┐
             │  Data Layer  │ ──▶ │   ML Layer   │ ──▶ │  Talk-to-Data  │     │       UI       │
             │ loader.py    │     │ train.py     │     │  nl_to_sql.py  │ ◀── │   app.py       │
             │ preprocessor │     │ predict.py   │     │  query_runner  │     │  (Streamlit)   │
             │ .py          │     │ evaluate.py  │     │  prompt_       │     │                │
             │ CSV → SQLite │     │ explain.py   │     │  templates.py  │     │  EDA / Predict │
             │ + feature    │     │ rules.py     │     │  LLM → SQL →   │     │  / Explain /   │
             │ pipeline     │     │ (LightGBM +  │     │  validated →   │     │  Rules / Chat  │
             │              │     │  SHAP)       │     │  answer        │     │                │
             └──────────────┘     └──────────────┘     └────────────────┘     └──────────────┘
                      │                    │                     │                     │
                      └────────────────────┴──────────┬──────────┴─────────────────────┘
                                                        ▼
                                          Docker Compose (single service, :8501)
                                 ./data and ./models are host-mounted volumes — never
                                          baked into the image or committed to git


`app.py` is the single entry point that wires all four layers together into one Streamlit app with five tabs: EDA, Risk Prediction, Explainability, Business Rules, and Talk-to-Data.

---

## 2. Scoping decision (read this first)

The Home Credit competition ships **7 relational tables** (`application_train`, `bureau`, `bureau_balance`, `previous_application`, `POS_CASH_balance`, `credit_card_balance`, `installments_payments`). This build works from **`application_train.csv` only** — the applicant-level table.

**Why:** `application_train.csv` already carries demographics, income, credit terms, and several bureau-derived `EXT_SOURCE_*` scores. Reliably joining and aggregating all 7 tables (especially the historical/time-series ones like `bureau_balance` and `installments_payments`) is a multi-day feature-engineering project on its own, and attempting it within this assignment's time-box risked an unreliable end-to-end pipeline — which would have hurt every scoring category, not just ML. Scoping to one clean, well-understood table let every layer (ML, SHAP, rules, chatbot) be built *and verified working end-to-end*.

**Trade-off:** with more time, joining `bureau.csv` (credit bureau history) is very likely the single highest-value addition — it's the classic strong signal in every published Home Credit solution.

---

## 3. Setup & run instructions

### Option A — Docker (recommended, matches submission requirements)

```bash
git clone <your-repo-url>
cd credit_risk_platform

# 1. Get the dataset (NOT included in the repo — see .gitignore)
#    Download application_train.csv from:
#    https://www.kaggle.com/competitions/home-credit-default-risk/data
#    and place it in ./data/application_train.csv

# 2. Configure your LLM key
cp .env.example .env
# edit .env and set GEMINI_API_KEY (free tier, no billing — get one at
# https://aistudio.google.com/apikey), or switch to LLM_PROVIDER=anthropic /
# LLM_PROVIDER=openai if you have billing set up on those instead

# 3. Run
docker-compose up --build
```

Open **http://localhost:8501**. On first load, if no model has been trained yet, click **"Train model now"** in the UI (or run the CLI command below first) — it takes well under a minute on the applicant-level table.

### Option B — Local Python (for development)

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env   # then edit in your API key

# Place application_train.csv in ./data/ (see above), then:
python -m src.data.loader      # builds the SQLite DB for the chatbot
python -m src.ml.train         # trains and saves the model

streamlit run app.py
```

### Verifying it worked

- `models/metrics.json` should exist after training, with `roc_auc` / `pr_auc` populated.
- `data/credit_risk.db` should exist after `python -m src.data.loader`.
- The Streamlit app's **Risk Prediction** tab should show model performance metrics at the bottom once trained.

### Important: always train in the same environment that will run inference

`models/*.pkl` files are Python pickles — they embed references to the exact library versions used to create them. If you train locally (e.g. on Windows, with whatever scikit-learn `pip` resolved) and then run inference **inside Docker** (a different scikit-learn version), loading the pickle can fail with an `AttributeError` about a missing internal class. `requirements.txt` pins exact versions for this reason — but if you ever train outside Docker and run inside Docker (or vice versa), retrain in whichever environment will actually serve predictions:

```bash
# if running via docker-compose, retrain inside the container:
docker exec -it credit_risk_platform python -m src.ml.train
```

> **Note on this repo's committed state:** this codebase was verified end-to-end against the full, real Home Credit dataset (`application_train.csv`, 307,511 rows, 8.1% default rate) — training, SQLite/chatbot, SHAP, and business-rule derivation have all been run against real data, not a synthetic sample. Metrics in §6 are from that real run.

---

## 4. Module-by-module summary

| Module | File(s) | What it does |
|---|---|---|
| Data loading | `src/data/loader.py` | Reads `application_train.csv`, builds a SQLite DB + a business-readable view (`applications_readable`) for the chatbot |
| Preprocessing | `src/data/preprocessor.py` | Fixes the `DAYS_EMPLOYED` sentinel anomaly (365243), engineers `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `CREDIT_TERM`, `AGE_YEARS`, `YEARS_EMPLOYED`, `EMPLOYED_AGE_RATIO`; median/mode imputation; one-hot encoding via a fitted `ColumnTransformer` (identical object reused at inference — no train/serve skew) |
| Training | `src/ml/train.py` | LightGBM with `scale_pos_weight` for class imbalance; train/val split; saves model + preprocessor + metrics artifacts |
| Evaluation | `src/ml/evaluate.py` | ROC-AUC, PR-AUC, confusion matrix, precision/recall for the default class, full classification report |
| Inference | `src/ml/predict.py` | Loads saved artifacts, scores single or batch applicants, maps probability → risk band |
| Explainability | `src/ml/explain.py` | SHAP `TreeExplainer` — per-applicant top-feature contributions and global importance |
| Business rules | `src/ml/rules.py` | Shallow surrogate decision tree → plain IF/THEN rules with coverage % and observed default rate per rule |
| Talk-to-data | `src/talk_to_data/*.py` | LLM → SQL (validated) → execution → LLM → plain-English answer, with short conversation memory |
| UI | `app.py` | Streamlit app tying it all together: EDA / Prediction / Explainability / Rules / Chat |
| Utils | `src/utils/*.py` | Config (env-driven), logging, small helpers, Docker startup checks |

---

## 5. Model selection & class-imbalance strategy

**Model: LightGBM (`LGBMClassifier`).**
- Handles the mix of numeric + one-hot categorical features natively.
- Trains fast even at Home Credit's full ~300K-row scale — important given the assignment's time-box.
- Has fast, exact SHAP `TreeExplainer` support, which Part 4 (explainability) depends on.

**Imbalance strategy: `scale_pos_weight` (cost-sensitive weighting), not SMOTE/oversampling.**
On high-dimensional tabular data with a mix of numeric and one-hot categorical features, synthetic oversampling (SMOTE and variants) tends to generate unrealistic applicant profiles — interpolating between real applicants in one-hot-encoded space doesn't correspond to a real person — and in practice on this kind of dataset the AUC gain over cost-sensitive weighting is marginal at best, while training time grows. `scale_pos_weight = n_negative / n_positive` corrects the model's bias toward the majority (repaid) class while keeping every training example real.

**Evaluation metric choice: ROC-AUC and PR-AUC, *not* accuracy.**
With an ~8% default rate on the real dataset, a model that predicts "no default" for every applicant scores ~92% accuracy while being completely useless. ROC-AUC and PR-AUC (the latter more informative under imbalance) are the metrics reported; recall/precision on the default class specifically are also tracked, since missing a defaulter is typically costlier to a bank than a false alarm.

**Risk banding: data-driven percentile thresholds, not fixed probability cutoffs.**
An earlier version of this pipeline used fixed thresholds (`< 10% → Low`, `10–30% → Medium`, `≥ 30% → High`). In testing against the real dataset, this almost never produced a "Low" band. The cause: `scale_pos_weight` fixes ranking (AUC) but **systematically inflates raw predicted probabilities** relative to the true base rate — verified empirically (a controlled test with an 8% true base rate saw the mean predicted probability jump to ~27% once `scale_pos_weight` was applied). Raw `predict_proba` output from a `scale_pos_weight`-trained model is not a calibrated probability.

The fix: risk bands are computed as **percentiles of the model's own validation-set output** rather than absolute cutoffs — by default, the bottom 60% of applicants (by the model's own score) are Low, the next 30% are Medium, and the top 10% are High. These thresholds are computed once after training (`src/ml/train.py`) and saved to `models/risk_thresholds.json`; `src/ml/predict.py` loads them at inference time, falling back to the fixed `RISK_LOW_MAX`/`RISK_MEDIUM_MAX` env values only if that file doesn't exist yet (i.e. before the first training run). This also mirrors how risk tiers are commonly defined in real credit-risk practice — as relative score deciles within a scored population, not absolute calibrated probabilities.

---

## 6. Evaluation metrics & results

Results below are from a real training run on the full Home Credit dataset.

| Metric | Value |
|---|---|
| ROC-AUC | 0.7689 |
| PR-AUC | 0.2568 |
| Recall (defaulters) | 0.6731 |
| Precision (defaulters) | 0.1771 |
| Model | LightGBM (`LGBMClassifier`) |
| Imbalance strategy | `scale_pos_weight ≈ 11.39` |
| Train / validation rows | 246,008 / 61,503 (of 307,511 total) |
| Engineered feature count | 193 (111 numeric, 15 categorical, after one-hot encoding) |

ROC-AUC 0.7689 is in line with typical single-table (`application_train`-only) solutions publicly discussed for this competition — full-pipeline solutions that join the other 6 tables (bureau history, previous applications, etc.) typically push into the 0.78–0.80 range, consistent with the scoping trade-off in §2.

**Reading precision/recall on the default class correctly:** at the default 0.5 classification threshold, recall of 0.67 means the model catches about two-thirds of actual defaulters, while precision of 0.18 means most applicants it flags at that threshold are false positives. This is expected, not a flaw — with an 8.1% true default rate, any model with reasonable recall will have low precision at a naive threshold. This is exactly why the app's output is a **risk score + band**, not a single yes/no flag: a bank's downstream policy (§8's business rules) decides what to actually do with each band, rather than the model forcing a binary cutoff.

---

## 7. Explainable AI

**Method: SHAP (`TreeExplainer`)**, chosen over LIME because:
- It's **exact** for tree ensembles (not a local linear approximation).
- It's fast enough to compute **per-applicant, on-demand** in the UI rather than needing precomputation.
- SHAP values are **additive** — they sum to the model's raw output, which makes "why is this applicant 62% risk" a directly answerable, audit-friendly question.

The **Explainability** tab shows, for any scored applicant, the top 5–8 features with their SHAP value and direction (`increases risk` / `decreases risk`), plus a global importance chart computed over a 200-applicant sample.

---

## 8. Business rule derivation

A shallow (`max_depth=3` by default, adjustable 2–5 in the UI) `DecisionTreeClassifier` is trained as a **surrogate** on the same transformed features and labels as the production LightGBM model. Its leaves are walked and converted into plain-English `IF ... THEN <risk band>` rules, each annotated with:
- **Coverage** — % of applicants the rule applies to (all rules at one depth sum to 100%).
- **Observed default rate** — the actual default rate among applicants matching that rule.

This is deliberately **not** the same as reading out LightGBM's internal trees (400 trees × depth ~6 has no readable "rule" of its own). A shallow surrogate trained on the same data approximates the model's decision boundary in a form a credit-policy team can review, challenge, and turn into an actual policy — independent of the production model's exact score. Real output from the deployed app (depth 3, on the full dataset):

 Rule 1 — IF EXT_SOURCE_3 ≤ 0.31 AND EXT_SOURCE_2 ≤ 0.38 AND EXT_SOURCE_3 ≤ 0.15
→ High risk (covers 1.14% of applicants, 34.13% observed default rate)

 Rule 8 — IF EXT_SOURCE_3 > 0.31 AND EXT_SOURCE_2 > 0.39 AND EXT_SOURCE_3 > 0.54
→ Low risk (covers 31.5% of applicants, 3.27% observed default rate)


Every one of the 8 derived rules splits on `EXT_SOURCE_2` and `EXT_SOURCE_3` — the two bureau-derived credit scores — which matches the SHAP global feature-importance ranking (§7) exactly. That agreement between two independent explainability methods (a surrogate tree and SHAP) is a good sanity check that both are reflecting genuine model behavior rather than an artifact of either method.

---

## 9. Talk-to-data: prompt engineering & token optimization

**Flow:** question → LLM generates SQL (against a documented schema + 6 few-shot examples) → SQL is validated → executed against SQLite → real result rows → LLM turns them into a plain-English answer.

**Multi-provider by design:** the LLM call is behind a single adapter (`_call_llm` in `nl_to_sql.py`) supporting Anthropic, OpenAI, or **Gemini** — selected via `LLM_PROVIDER` in `.env` with no code change. Gemini is the recommended default: it has a genuine free tier (no billing setup required) via [Google AI Studio](https://aistudio.google.com/apikey), which matters for anyone running this without existing API billing.

**Token optimization:** the LLM is shown a **trimmed, business-readable SQL view** (`applications_readable`, 13 columns) rather than the raw ~19–122-column applicant table. This is the main lever — describing 100+ raw Home Credit columns (many with cryptic names like `FLAG_DOCUMENT_7`) in every prompt would multiply input-token cost for no query-coverage benefit, since almost every plausible business question is answerable from the 13-column view. The raw table is still available as a fallback, documented in the prompt, for the rare question that needs it.

**Hallucination control (several layers):**
1. **SQL validation before execution** (`query_runner.py`) — SELECT-only, single-statement, allow-listed tables, forbidden-keyword regex (`INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/ATTACH/PRAGMA/...`), automatic `LIMIT` cap. A hallucinated table/column name fails validation with a clear message instead of silently returning wrong data. **Verified in testing**: `DROP TABLE`, `UPDATE ... SET`, stacked `SELECT; DROP`, and queries against unknown tables are all correctly rejected pre-execution.
2. **Grounded answer generation** — the answer-synthesis LLM call is only ever shown the actual rows SQLite returned, with an explicit instruction never to invent numbers not present in that data.
3. **Escape hatch** — if a question can't be answered from the documented schema, the model is instructed to output `NO_QUERY` rather than guess, and the UI surfaces a clear "can't answer that" message instead of running a bad query.
4. **Truncation detection and auto-retry (Gemini-specific)** — Gemini reports a `finish_reason` when a response is cut off by its token budget. Left unhandled, this silently returns broken output (e.g. a SQL query missing its `FROM` clause, or an answer cut off mid-sentence) that looks superficially valid. The adapter checks `finish_reason` and automatically retries once with a 4x larger token budget before giving up with a clear error — this was found and fixed during real testing against the deployed app, not designed in from the start.

**Conversation memory:** the last 4 turns (question + SQL) are passed back into the SQL-generation prompt so follow-ups like *"what about for women only?"* resolve against the prior question's context, without letting the conversation history grow unbounded and inflate token cost turn over turn.

**5 required working query patterns** (6 shipped, in `src/talk_to_data/prompt_templates.py`):
1. Average income of applicants who defaulted
2. High-risk vs low-risk counts by education level
3. Top-10 highest-credit defaulters
4. Default rate: under-30 vs over-50 applicants
5. Average annuity by gender
6. Count of applicants with revolving loans

---

## 10. Known limitations & possible improvements

**Limitations:**
- Scoped to `application_train.csv` only — no credit bureau or previous-application history (see §2).
- SHAP explanations are computed fresh on every request rather than cached — fine for the assignment's scale, would need caching/batching for high-traffic production use.
- The talk-to-data chatbot's schema is intentionally narrow (13 columns) — some valid business questions genuinely fall outside it and correctly return "can't answer that."
- No authentication in front of the Streamlit app or the LLM-backed endpoints — fine for an internal evaluator demo, not production-ready as-is.
- Raw model probabilities are uncalibrated (see §5) — addressed via percentile-based risk bands rather than a dedicated calibration model (e.g. `CalibratedClassifierCV`); the latter would be the more "textbook correct" fix but requires a third held-out data split and wasn't worth the added pipeline complexity for this assignment's scope.
- `models/*.pkl` files are Python pickles tied to the exact scikit-learn/LightGBM versions used to create them — training in one environment (e.g. local Windows) and running inference in another (e.g. Docker with a different resolved version) can fail to load with an `AttributeError`. `requirements.txt` pins exact versions to prevent this, and `predict.py` raises a clear, actionable error pointing at the fix if it ever happens.

**With more time, next steps would be:**
- Join `bureau.csv` for credit-bureau history features — the highest-value likely addition.
- Cache SHAP explanations per applicant and add a batch-scoring endpoint.
- Add a confidence/coverage check before trusting the LLM's generated SQL for ambiguous questions.
- Add basic auth and rate-limiting for a shared deployment.
- Track score-distribution drift over time for production monitoring.

---

## 11. Project structure

credit_risk_platform/
├── data/ # Mounted, not committed — see .gitignore
├── documents/
│ └── project_presentation.pdf # Use-case presentation with real app screenshots
├── notebooks/
│ ├── eda.ipynb # Executed EDA notebook
│ └── eda.py # Same notebook as a plain script (jupytext)
├── src/
│ ├── data/ loader.py, preprocessor.py
│ ├── ml/ train.py, predict.py, evaluate.py, explain.py, rules.py
│ ├── talk_to_data/ nl_to_sql.py, query_runner.py, prompt_templates.py
│ └── utils/ logger.py, config.py, helpers.py, docker_utils.py
├── sql/schema.sql # Documents the SQLite schema
├── models/ # Saved artifacts (.pkl/.json) — not committed
├── app.py # Streamlit UI (entry point)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md