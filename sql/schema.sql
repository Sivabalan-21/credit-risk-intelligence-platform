-- This file documents the SQLite schema built at runtime by src/data/loader.py
-- (build_sqlite_db). It is NOT executed directly — the actual table is created
-- via pandas.DataFrame.to_sql from application_train.csv, since its 100+ columns
-- are dataset-defined. This file documents the derived, chatbot-facing view.

-- Raw table `applications`: 1 row per applicant, all Home Credit
-- application_train.csv columns (SK_ID_CURR, TARGET, AMT_INCOME_TOTAL, ...).

-- Business-readable view used by the NL-to-SQL chatbot:
CREATE VIEW IF NOT EXISTS applications_readable AS
SELECT
    SK_ID_CURR                                  AS applicant_id,
    TARGET                                       AS defaulted,
    CASE WHEN TARGET = 1 THEN 'Default' ELSE 'Repaid' END AS outcome,
    AMT_INCOME_TOTAL                             AS annual_income,
    AMT_CREDIT                                   AS credit_amount,
    AMT_ANNUITY                                  AS annuity_amount,
    NAME_CONTRACT_TYPE                           AS contract_type,
    NAME_INCOME_TYPE                             AS income_type,
    NAME_EDUCATION_TYPE                          AS education,
    NAME_FAMILY_STATUS                           AS family_status,
    CODE_GENDER                                  AS gender,
    ROUND(-DAYS_BIRTH / 365.25, 1)                AS age_years,
    ROUND(-DAYS_EMPLOYED / 365.25, 1)             AS years_employed
FROM applications;
