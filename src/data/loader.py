"""
Loads the Home Credit application-level data and (re)builds a SQLite
database used by the talk-to-data / NL-to-SQL module.

We deliberately scope to `application_train.csv` (the applicant-level table)
rather than joining all 7 Home Credit tables. That keeps the pipeline
reliable and fast to run within the assignment's time-box, while still
covering demographics, financials, and credit-history-derived flags that
are already present as columns on this table. This scoping decision and
the trade-off are called out in the README.
"""
import sqlite3

import pandas as pd

from src.utils.config import RAW_TRAIN_FILE, RAW_TEST_FILE, SQLITE_DB_PATH, TARGET_COL, ID_COL
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Shrinks numeric dtypes (float64->float32, int64->smallest safe int)
    to cut DataFrame memory footprint. Meaningful on a 307k-row x 122-col
    table on memory-constrained free-tier hosting."""
    before_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    after_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    logger.info("Downcast dtypes: %.1f MB -> %.1f MB", before_mb, after_mb)
    return df


def load_raw_train() -> pd.DataFrame:
    if not RAW_TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {RAW_TRAIN_FILE}. Download application_train.csv from "
            "the Home Credit Default Risk Kaggle competition and place it in ./data"
        )
    logger.info("Loading raw training data from %s", RAW_TRAIN_FILE)
    df = pd.read_csv(RAW_TRAIN_FILE)
    df = _downcast_dtypes(df)
    logger.info("Loaded %d rows, %d columns", *df.shape)
    return df


def load_raw_test() -> pd.DataFrame | None:
    if not RAW_TEST_FILE.exists():
        logger.info("No application_test.csv found — skipping (optional file).")
        return None
    return pd.read_csv(RAW_TEST_FILE)


def build_sqlite_db(df: pd.DataFrame | None = None, table_name: str = "applications") -> str:
    """
    Materializes the applicant table into SQLite so the NL-to-SQL agent can
    run real, validated SQL against it rather than the LLM hallucinating
    over a pandas DataFrame it can't see.
    """
    if df is None:
        df = load_raw_train()

    SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing SQLite DB to %s (table=%s)", SQLITE_DB_PATH, table_name)

    conn = sqlite3.connect(SQLITE_DB_PATH)
    try:
        # chunksize bounds peak memory during the write instead of building
        # the full 307k-row insert payload in memory at once. (No `method=
        # "multi"` here — with 122 columns that would blow past SQLite's
        # 999-variable-per-statement limit at any usable chunk size.)
        df.to_sql(table_name, conn, if_exists="replace", index=False, chunksize=5000)
        # A couple of read-friendly derived columns/views for the chatbot
        conn.execute(f"""
            CREATE VIEW IF NOT EXISTS applications_readable AS
            SELECT
                {ID_COL} AS applicant_id,
                {TARGET_COL} AS defaulted,
                CASE WHEN {TARGET_COL} = 1 THEN 'Default' ELSE 'Repaid' END AS outcome,
                AMT_INCOME_TOTAL AS annual_income,
                AMT_CREDIT AS credit_amount,
                AMT_ANNUITY AS annuity_amount,
                NAME_CONTRACT_TYPE AS contract_type,
                NAME_INCOME_TYPE AS income_type,
                NAME_EDUCATION_TYPE AS education,
                NAME_FAMILY_STATUS AS family_status,
                CODE_GENDER AS gender,
                ROUND(-DAYS_BIRTH / 365.25, 1) AS age_years,
                ROUND(-DAYS_EMPLOYED / 365.25, 1) AS years_employed
            FROM {table_name}
        """)
        conn.commit()
    finally:
        conn.close()

    logger.info("SQLite DB ready with %d rows.", len(df))
    return str(SQLITE_DB_PATH)


if __name__ == "__main__":
    data = load_raw_train()
    build_sqlite_db(data)
