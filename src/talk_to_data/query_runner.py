"""
Executes SQL against the SQLite database, with validation that runs BEFORE
execution — this is the guardrail against LLM hallucination or injection
turning into a destructive or runaway query.
"""
import re
import sqlite3

import pandas as pd

from src.utils.config import SQLITE_DB_PATH, MAX_SQL_ROWS_RETURNED
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Anything beyond a read-only SELECT is refused outright.
_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|"
    r"PRAGMA|VACUUM|TRUNCATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)
_ALLOWED_TABLES = {"applications", "applications_readable"}


class SQLValidationError(ValueError):
    pass


def validate_sql(sql: str) -> str:
    """Raises SQLValidationError if the query is anything but a safe, single
    read-only SELECT against an allow-listed table. Returns the cleaned SQL."""
    cleaned = sql.strip().rstrip(";")

    if not cleaned:
        raise SQLValidationError("Empty SQL query.")

    if ";" in cleaned:
        raise SQLValidationError("Multiple statements are not allowed.")

    if not re.match(r"^\s*SELECT\b", cleaned, re.IGNORECASE):
        raise SQLValidationError("Only SELECT statements are allowed.")

    if _FORBIDDEN_KEYWORDS.search(cleaned):
        raise SQLValidationError("Query contains a forbidden keyword (write/DDL operation).")

    referenced_tables = set(re.findall(r"(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)", cleaned, re.IGNORECASE))
    unknown = referenced_tables - _ALLOWED_TABLES
    if unknown:
        raise SQLValidationError(f"Query references unknown table(s): {unknown}")

    if not re.search(r"\bLIMIT\b", cleaned, re.IGNORECASE):
        cleaned = f"{cleaned} LIMIT {MAX_SQL_ROWS_RETURNED}"

    return cleaned


def run_query(sql: str) -> pd.DataFrame:
    safe_sql = validate_sql(sql)
    logger.info("Executing validated SQL: %s", safe_sql)
    conn = sqlite3.connect(SQLITE_DB_PATH)
    try:
        df = pd.read_sql_query(safe_sql, conn)
    finally:
        conn.close()
    return df
