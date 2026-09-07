"""
Versioned prompt templates for the talk-to-data agent.

Keeping these in one file (rather than inline strings scattered through
nl_to_sql.py) is what "prompt engineering and token optimization" in the
README refers to: the schema description below is trimmed to only the
columns actually exposed via `applications_readable`, so we're not paying
input-token cost to describe 120 raw Home Credit columns the chatbot never
needs to answer plain-English questions.
"""

PROMPT_VERSION = "v1"

SCHEMA_DESCRIPTION = """
You have access to a SQLite view named `applications_readable` with these columns:
- applicant_id (INTEGER): unique applicant ID
- defaulted (INTEGER): 1 if the applicant defaulted, 0 if repaid
- outcome (TEXT): 'Default' or 'Repaid'
- annual_income (REAL): applicant's total annual income
- credit_amount (REAL): amount of credit/loan
- annuity_amount (REAL): loan annuity payment
- contract_type (TEXT): 'Cash loans' or 'Revolving loans'
- income_type (TEXT): e.g. 'Working', 'Pensioner', 'State servant'
- education (TEXT): highest education level
- family_status (TEXT): marital status
- gender (TEXT): 'M' or 'F'
- age_years (REAL): applicant age in years
- years_employed (REAL): years employed (NULL if not currently employed)

Also available (raw table, use only if the readable view truly can't answer it): `applications`
(full Home Credit application_train schema, 100+ columns, sparse/technical names — avoid unless necessary).
"""

SQL_SYSTEM_PROMPT = f"""You are a SQL generator for a credit-risk analytics chatbot.
{SCHEMA_DESCRIPTION}
Rules:
1. Output ONLY a single valid SQLite SELECT statement. No explanation, no markdown fences, no semicolon.
2. Only query the tables described above. Never write INSERT/UPDATE/DELETE/DROP/ALTER or any other write statement.
3. Prefer `applications_readable` over the raw `applications` table whenever it has the needed columns.
4. Always add a LIMIT clause (<= 200) for any query that could return many rows; omit LIMIT only for single-value aggregate queries.
5. If the question cannot be answered from the schema above, output exactly: NO_QUERY
"""

# Few-shot examples double as the "at least 5 working query patterns" requirement.
FEW_SHOT_EXAMPLES = [
    {
        "question": "What is the average income of applicants who defaulted?",
        "sql": "SELECT AVG(annual_income) AS avg_income_defaulters FROM applications_readable WHERE defaulted = 1",
    },
    {
        "question": "How many applicants are high risk vs low risk by education level?",
        "sql": (
            "SELECT education, outcome, COUNT(*) AS n "
            "FROM applications_readable GROUP BY education, outcome ORDER BY education LIMIT 100"
        ),
    },
    {
        "question": "Show me the 10 applicants with the highest credit amount who defaulted.",
        "sql": (
            "SELECT applicant_id, credit_amount, annual_income, age_years "
            "FROM applications_readable WHERE defaulted = 1 ORDER BY credit_amount DESC LIMIT 10"
        ),
    },
    {
        "question": "What's the default rate for applicants under 30 vs over 50?",
        "sql": (
            "SELECT CASE WHEN age_years < 30 THEN 'under 30' WHEN age_years > 50 THEN 'over 50' "
            "ELSE 'other' END AS age_group, AVG(defaulted) * 100 AS default_rate_pct, COUNT(*) AS n "
            "FROM applications_readable GROUP BY age_group LIMIT 10"
        ),
    },
    {
        "question": "Compare average annuity amount between genders.",
        "sql": (
            "SELECT gender, AVG(annuity_amount) AS avg_annuity, COUNT(*) AS n "
            "FROM applications_readable GROUP BY gender LIMIT 10"
        ),
    },
    {
        "question": "How many applicants have revolving loans?",
        "sql": "SELECT COUNT(*) AS n_revolving FROM applications_readable WHERE contract_type = 'Revolving loans'",
    },
]

ANSWER_SYSTEM_PROMPT = """You are a credit-risk analyst explaining query results to a non-technical
business stakeholder (e.g. a bank branch manager). You will be given the user's original question,
the SQL that was run, and the resulting data (as rows).

Rules:
1. Answer in 2-4 plain-English sentences. No SQL, no markdown tables, no code.
2. State the concrete numbers from the data — don't be vague.
3. If the result set is empty, say so plainly and suggest the question may need rephrasing.
4. Never invent numbers not present in the provided data.
"""


def build_sql_messages(question: str) -> list:
    examples_text = "\n".join(
        f'Q: "{ex["question"]}"\nSQL: {ex["sql"]}' for ex in FEW_SHOT_EXAMPLES
    )
    user_content = f"Examples:\n{examples_text}\n\nNow generate SQL for this question:\nQ: \"{question}\"\nSQL:"
    return [{"role": "user", "content": user_content}]


def build_answer_messages(question: str, sql: str, result_preview: str) -> list:
    user_content = (
        f"Original question: {question}\n\n"
        f"SQL executed: {sql}\n\n"
        f"Result data:\n{result_preview}\n\n"
        "Write the plain-English answer now."
    )
    return [{"role": "user", "content": user_content}]
