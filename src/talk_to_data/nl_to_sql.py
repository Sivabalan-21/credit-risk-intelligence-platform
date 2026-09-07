"""
Talk-to-data orchestrator.

Flow:
    question
        -> LLM generates SQL
        -> SQL validated (query_runner)
        -> executed against SQLite
        -> LLM turns the real result rows into a plain-English answer

Hallucination control:
    - The answer-generation call is only ever shown data that actually came
      back from SQLite, and is explicitly instructed not to invent numbers.
    - SQL is validated against an allow-list before execution.
    - Short conversation memory is passed back to the LLM so follow-up
      questions can resolve correctly without using unbounded history.
"""

import re
import time
from dataclasses import dataclass, field

import pandas as pd

from src.talk_to_data.prompt_templates import (
    SQL_SYSTEM_PROMPT,
    ANSWER_SYSTEM_PROMPT,
    build_sql_messages,
    build_answer_messages,
)

from src.talk_to_data.query_runner import (
    run_query,
    SQLValidationError,
)

from src.utils.config import (
    LLM_PROVIDER,
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_MAX_TOKENS,
)

from src.utils.logger import get_logger


logger = get_logger(__name__)


# Keep only the most recent few turns.
MAX_MEMORY_TURNS = 4


# ============================================================================
# LLM CALL
# ============================================================================

def _call_llm(
    system_prompt: str,
    messages: list,
    max_tokens: int = LLM_MAX_TOKENS,
    max_retries: int = 3,
) -> str:
    """
    Retries _call_llm_once on transient provider errors (rate limits,
    "high demand"/overloaded, timeouts) with exponential backoff, since
    these are usually gone within a few seconds. Non-transient errors
    (bad/missing API key, invalid request) are raised immediately.
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return _call_llm_once(system_prompt, messages, max_tokens)
        except Exception as e:
            last_exc = e
            msg = str(e).lower()
            transient = any(
                token in msg
                for token in (
                    "503", "429", "overloaded", "unavailable",
                    "high demand", "rate limit", "timeout", "resource_exhausted",
                )
            )
            if not transient or attempt == max_retries:
                raise
            wait_s = 2 ** attempt  # 2s, 4s, 8s
            logger.warning(
                "LLM call failed (attempt %d/%d), retrying in %ds: %s",
                attempt, max_retries, wait_s, e,
            )
            time.sleep(wait_s)
    raise last_exc  # pragma: no cover — unreachable, keeps type checkers happy


def _call_llm_once(
    system_prompt: str,
    messages: list,
    max_tokens: int = LLM_MAX_TOKENS,
) -> str:
    """
    Provider-independent LLM adapter.

    Supported providers:
        - anthropic
        - openai
        - gemini

    The provider is selected using LLM_PROVIDER in .env.
    """

    # ------------------------------------------------------------------------
    # Anthropic
    # ------------------------------------------------------------------------

    if LLM_PROVIDER == "anthropic":

        import anthropic

        if not ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file."
            )

        client = anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY
        )

        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=messages,
        )

        if not response.content:
            raise RuntimeError(
                "Anthropic returned an empty response."
            )

        return response.content[0].text.strip()

    # ------------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------------

    elif LLM_PROVIDER == "openai":

        from openai import OpenAI

        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file."
            )

        client = OpenAI(
            api_key=OPENAI_API_KEY
        )

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                }
            ] + messages,
        )

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError(
                "OpenAI returned an empty response."
            )

        return content.strip()

    # ------------------------------------------------------------------------
    # Google Gemini
    # ------------------------------------------------------------------------

    elif LLM_PROVIDER == "gemini":

        from google import genai

        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file."
            )

        client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        # ------------------------------------------------------------
        # Convert our internal messages into one Gemini prompt.
        #
        # This deliberately uses the simplest generate_content()
        # configuration to avoid unsupported arguments with different
        # Gemini model/API versions.
        # ------------------------------------------------------------

        message_parts = []

        for message in messages:

            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "user":
                message_parts.append(
                    f"User:\n{content}"
                )

            elif role == "assistant":
                message_parts.append(
                    f"Assistant:\n{content}"
                )

            else:
                message_parts.append(
                    content
                )

        conversation = "\n\n".join(message_parts)

        full_prompt = (
            f"{system_prompt}\n\n"
            f"{conversation}"
        )

        # ------------------------------------------------------------
        # Gemini generation
        #
        # IMPORTANT:
        # Do not use thinking_config here.
        # Some Gemini model/API combinations reject it with:
        # 400 INVALID_ARGUMENT.
        # ------------------------------------------------------------

        try:

            response = client.models.generate_content(
    model=GEMINI_MODEL,
    contents=full_prompt,
    config={
        "max_output_tokens": max(max_tokens, 1024),
    },
)

        except Exception as e:

            logger.exception(
                "Gemini API call failed"
            )

            raise RuntimeError(
                f"Gemini API request failed: {e}"
            ) from e

        # ------------------------------------------------------------
        # Extract response
        # ------------------------------------------------------------

        text = getattr(response, "text", None)

        if not text:
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        return text.strip()

    # ------------------------------------------------------------------------
    # Unknown provider
    # ------------------------------------------------------------------------

    else:

        raise RuntimeError(
            f"Unknown LLM_PROVIDER: {LLM_PROVIDER}. "
            "Use 'anthropic', 'openai', or 'gemini'."
        )


# ============================================================================
# SQL EXTRACTION
# ============================================================================

def _extract_sql(raw: str) -> str:
    """
    LLMs sometimes wrap SQL in markdown code fences.

    Example:

        ```sql
        SELECT COUNT(*) FROM applicants;
        ```

    becomes:

        SELECT COUNT(*) FROM applicants;
    """

    text = raw.strip()

    fence_match = re.search(
        r"```(?:sql)?\s*(.*?)```",
        text,
        re.IGNORECASE | re.DOTALL,
    )

    if fence_match:
        text = fence_match.group(1).strip()

    return text.strip()


# ============================================================================
# CHAT TURN
# ============================================================================

@dataclass
class ChatTurn:
    """
    Represents one question/answer interaction.
    """

    question: str
    sql: str | None
    answer: str


# ============================================================================
# TALK-TO-DATA SESSION
# ============================================================================

@dataclass
class TalkToDataSession:
    """
    Holds short conversation memory for follow-up questions.
    """

    history: list = field(
        default_factory=list
    )

    # ------------------------------------------------------------------------
    # HISTORY CONTEXT
    # ------------------------------------------------------------------------

    def _history_context(self) -> str:
        """
        Build a short context string from recent conversations.
        """

        if not self.history:
            return ""

        recent = self.history[
            -MAX_MEMORY_TURNS:
        ]

        lines = []

        for turn in recent:

            lines.append(
                f'Previous Q: "{turn.question}" '
                f"-> SQL: {turn.sql}"
            )

        return (
            "Conversation so far "
            "(for context on follow-ups):\n"
            + "\n".join(lines)
            + "\n\n"
        )

    # ------------------------------------------------------------------------
    # ASK
    # ------------------------------------------------------------------------

    def ask(
        self,
        question: str,
    ) -> dict:
        """
        Process a natural-language question.

        Returns:

            {
                "question": question,
                "sql": sql,
                "dataframe": dataframe,
                "answer": answer,
                "error": error
            }
        """

        # ====================================================================
        # STEP 1 — Conversation context
        # ====================================================================

        context = self._history_context()

        sql_messages = build_sql_messages(
            question
        )

        if sql_messages:

            sql_messages[0]["content"] = (
                context
                + sql_messages[0]["content"]
            )

        # ====================================================================
        # STEP 2 — Generate SQL
        # ====================================================================

        try:
            raw_sql = _call_llm(
                SQL_SYSTEM_PROMPT,
                sql_messages,
                max_tokens=400,
            )
        except Exception as e:
            logger.exception("LLM call failed while generating SQL")
            answer = (
                "I couldn't reach the AI model just now (it may be "
                "temporarily overloaded). Please try asking again in "
                "a moment."
            )
            self.history.append(ChatTurn(question, None, answer))
            return {
                "question": question,
                "sql": None,
                "dataframe": None,
                "answer": answer,
                "error": str(e),
            }

        sql = _extract_sql(
            raw_sql
        )

        logger.info(
            "Generated SQL: %s",
            sql,
        )

        # ====================================================================
        # STEP 3 — Handle questions that cannot be answered from the data
        # ====================================================================

        if sql.strip().upper() == "NO_QUERY":

            answer = (
                "I can't answer that from the available applicant data. "
                "Try asking about income, credit amount, age, education, "
                "employment, or default outcomes."
            )

            self.history.append(
                ChatTurn(
                    question,
                    None,
                    answer,
                )
            )

            return {
                "question": question,
                "sql": None,
                "dataframe": None,
                "answer": answer,
                "error": None,
            }

        # ====================================================================
        # STEP 4 — Validate and execute SQL
        # ====================================================================

        try:

            df: pd.DataFrame = run_query(
                sql
            )

        except SQLValidationError as e:

            logger.warning(
                "SQL validation failed: %s | SQL was: %s",
                e,
                sql,
            )

            answer = (
                "I generated a query that didn't pass "
                "safety validation, so I won't run it. "
                "Could you rephrase the question?"
            )

            self.history.append(
                ChatTurn(
                    question,
                    sql,
                    answer,
                )
            )

            return {
                "question": question,
                "sql": sql,
                "dataframe": None,
                "answer": answer,
                "error": str(e),
            }

        except Exception as e:

            logger.exception(
                "SQL execution failed"
            )

            answer = (
                f"That query failed to execute ({e}). "
                "Try rephrasing the question."
            )

            return {
                "question": question,
                "sql": sql,
                "dataframe": None,
                "answer": answer,
                "error": str(e),
            }

        # ====================================================================
        # STEP 5 — Prepare real SQL results for the answer LLM
        # ====================================================================

        if not df.empty:

            result_preview = (
                df.head(20)
                .to_string(index=False)
            )

        else:

            result_preview = (
                "(no rows returned)"
            )

        # ====================================================================
        # STEP 6 — Generate natural-language answer
        # ====================================================================

        answer_messages = build_answer_messages(
            question,
            sql,
            result_preview,
        )

        try:
            answer = _call_llm(
                ANSWER_SYSTEM_PROMPT,
                answer_messages,
                max_tokens=1024,
            )
        except Exception as e:
            logger.exception("LLM call failed while generating the answer")
            answer = (
                "I ran the query successfully, but couldn't reach the AI "
                "model to summarize the result (it may be temporarily "
                "overloaded). Here's the raw result data instead — please "
                "try asking again in a moment for a written answer."
            )
            self.history.append(ChatTurn(question, sql, answer))
            return {
                "question": question,
                "sql": sql,
                "dataframe": df,
                "answer": answer,
                "error": str(e),
            }

        # ====================================================================
        # STEP 7 — Save conversation
        # ====================================================================

        self.history.append(
            ChatTurn(
                question,
                sql,
                answer,
            )
        )

        # ====================================================================
        # STEP 8 — Return result
        # ====================================================================

        return {
            "question": question,
            "sql": sql,
            "dataframe": df,
            "answer": answer,
            "error": None,
        }
