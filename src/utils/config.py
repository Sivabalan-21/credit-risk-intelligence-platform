"""
Central configuration for the Credit Risk Intelligence Platform.
Everything that could change between environments (paths, LLM provider,
thresholds) lives here and is overridable via environment variables /.env.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------- paths ----
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))
SQL_DIR = Path(os.getenv("SQL_DIR", BASE_DIR / "sql"))

RAW_TRAIN_FILE = DATA_DIR / os.getenv("RAW_TRAIN_FILENAME", "application_train.csv")
RAW_TEST_FILE = DATA_DIR / os.getenv("RAW_TEST_FILENAME", "application_test.csv")
SQLITE_DB_PATH = DATA_DIR / os.getenv("SQLITE_DB_NAME", "credit_risk.db")

DATASET_URL = os.getenv("DATASET_URL", "")

MODEL_ARTIFACT_PATH = MODELS_DIR / "lgbm_credit_risk.pkl"
PREPROCESSOR_ARTIFACT_PATH = MODELS_DIR / "preprocessor.pkl"
FEATURE_LIST_PATH = MODELS_DIR / "feature_columns.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
RISK_THRESHOLDS_PATH = MODELS_DIR / "risk_thresholds.json"

# ------------------------------------------------------------- ML config ---
TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))

# Risk band thresholds on predicted probability of default.
# NOTE: LightGBM trained with scale_pos_weight (see train.py) produces
# systematically inflated, uncalibrated probabilities — not true
# probabilities. These fixed values are only a fallback for when
# models/risk_thresholds.json doesn't exist yet (i.e. before the first
# training run); after training, risk bands are computed from the
# empirical percentiles of the model's own validation-set output instead
# (see train.py / src/ml/predict.py), which is what actually determines
# the Low/Medium/High split you see in the app.
RISK_LOW_MAX = float(os.getenv("RISK_LOW_MAX", 0.10))
RISK_MEDIUM_MAX = float(os.getenv("RISK_MEDIUM_MAX", 0.30))
# LOW_PERCENTILE / MEDIUM_PERCENTILE: what fraction of applicants (by the
# model's own score) fall into Low / Low+Medium. E.g. defaults below mean
# "bottom 60% of applicants by score = Low, next 30% = Medium, top 10% = High".
RISK_LOW_PERCENTILE = float(os.getenv("RISK_LOW_PERCENTILE", 60))
RISK_MEDIUM_PERCENTILE = float(os.getenv("RISK_MEDIUM_PERCENTILE", 90))

# ------------------------------------------------------------- LLM config --
# Supported providers: "anthropic", "openai", or "gemini" (Gemini has a
# genuine free tier via Google AI Studio — no credit card required, just
# rate-limited — good default if you don't have Anthropic/OpenAI billing set up).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

MAX_SQL_ROWS_RETURNED = int(os.getenv("MAX_SQL_ROWS_RETURNED", 200))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 800))

# ------------------------------------------------------------- misc --------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

for _d in (DATA_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
