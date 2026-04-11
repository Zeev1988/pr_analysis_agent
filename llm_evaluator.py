"""
Part 3 – LLM Evaluator
Send a FunctionSlice to an LLM and receive a strictly typed security verdict.

Uses `instructor` + `litellm` so any provider works without changing code —
only the model string and the relevant API key need to change.

Configuration (via .env or environment variables):
    LLM_MODEL         – litellm model string, e.g.:
                          openai/gpt-4o-mini
                          anthropic/claude-3-5-sonnet-20241022
                          gemini/gemini-1.5-pro
                        Defaults to "openai/gpt-4o-mini".
    OPENAI_API_KEY    – required for OpenAI models.
    ANTHROPIC_API_KEY – required for Anthropic models.
    GOOGLE_API_KEY    – required for Gemini models.
    LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY – optional; enables tracing.
"""

from __future__ import annotations

import logging
import os
from typing import Literal

import instructor
import litellm
import openai
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from ast_extractor import FunctionSlice, slice_context

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Langfuse observability — optional, gracefully degrades if not installed
# or if LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set.
# v3 API: observe, get_client, and update_current_observation imported from
# root langfuse package.
# ---------------------------------------------------------------------------

try:
    import langfuse
    from langfuse import observe
    _langfuse = langfuse.get_client()
except ImportError:
    _langfuse = None  # type: ignore[assignment]

    def observe(**_kw):  # type: ignore[misc]
        def _dec(fn): return fn
        return _dec

load_dotenv()

_DEFAULT_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")

# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class SecurityVerdict(BaseModel):
    """
    The exact JSON object the LLM must return for every FunctionSlice.

    verdict:            BLOCK if the code contains an exploitable vulnerability,
                        PASS otherwise.
    vulnerability_type: CWE-style label (e.g. "SQL Injection", "Path Traversal").
                        Null when verdict is PASS.
    confidence:         Model's self-reported certainty, 0.0 (low) to 1.0 (high).
    affected_lines:     Subset of the changed lines implicated in the finding.
                        Empty list when verdict is PASS.
    reasoning:          One or two sentences explaining the decision.
    """
    verdict: Literal["BLOCK", "PASS"]
    vulnerability_type: str | None = Field(
        default=None,
        description="CWE-style vulnerability label, or null when verdict is PASS.",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    affected_lines: list[int] = Field(default_factory=list)
    reasoning: str


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a security-focused code reviewer embedded in an automated Pull Request \
analysis pipeline. Your job is to decide whether a changed Python function \
introduces an exploitable security vulnerability.

Rules:
- Focus only on the CHANGED lines listed in the metadata, but consider the full \
  function for context (e.g. to trace untrusted data into a dangerous sink).
- Return BLOCK only when you are confident an attacker could exploit the code \
  as written; return PASS for theoretical risks that require additional \
  preconditions outside this function.
- Keep reasoning to 1-2 sentences."""


def _build_user_message(s: FunctionSlice) -> str:
    changed_str = ", ".join(str(ln) for ln in s.changed_lines)
    sink_flag = "YES" if s.has_sink else "NO"
    return (
        f"File: {s.filename}\n"
        f"Function: {s.function_name}({', '.join(s.params)})\n"
        f"Lines {s.start_line}–{s.end_line} | Changed lines in diff: [{changed_str}]\n"
        f"Known dangerous sink present: {sink_flag}\n\n"
        f"```python\n{s.raw_source}\n```"
    )


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------

# litellm maps provider errors onto openai exception types, so these cover
# rate limits / network blips / server errors across all providers.
_RETRYABLE = (
    openai.RateLimitError,        # 429 – back off and retry
    openai.APIConnectionError,    # network blip
    openai.APITimeoutError,       # request timed out
    openai.InternalServerError,   # 5xx – server-side glitch
)

_api_retry = retry(
    retry=retry_if_exception_type(_RETRYABLE),
    wait=wait_exponential_jitter(initial=1, max=60),
    stop=stop_after_attempt(4),
    reraise=True,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _make_client() -> instructor.Instructor:
    """Return an instructor client backed by litellm.

    litellm routes to the correct provider based on the model string prefix
    (e.g. "anthropic/...", "gemini/..."), reading the appropriate API key
    from the environment automatically.
    """
    return instructor.from_litellm(litellm.completion)


@observe(name="evaluate_slice")
def evaluate_slice(
    function_slice: FunctionSlice,
    *,
    model: str | None = None,
    client: instructor.Instructor | None = None,
) -> SecurityVerdict:
    """
    Send *function_slice* to the LLM and return a validated SecurityVerdict.

    Provider is determined solely by the model string — no code changes needed
    to switch between OpenAI, Anthropic, Gemini, or any litellm-supported model.

    Retries automatically on transient failures (rate limits, network errors,
    5xx responses) with exponential back-off + jitter, up to 4 attempts.
    Permanent failures (bad API key, invalid request) are raised immediately.

    Args:
        function_slice: The FunctionSlice produced by ast_extractor.slice_context.
        model:          litellm model string. Falls back to the LLM_MODEL env var,
                        then "openai/gpt-4o-mini".
        client:         Optional pre-configured instructor client — useful for tests.
    """
    _client = client or _make_client()
    _model = model or _DEFAULT_MODEL
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(function_slice)},
    ]

    @_api_retry
    def _call() -> SecurityVerdict:
        return _client.chat.completions.create(
            model=_model,
            messages=messages,
            response_model=SecurityVerdict,
            temperature=0,
            max_retries=2,  # instructor retries on malformed structured output
        )

    verdict = _call()

    if _langfuse is not None:
        langfuse.update_current_observation(
            metadata={
                "filename": function_slice.filename,
                "function_name": function_slice.function_name,
                "has_sink": function_slice.has_sink,
                "changed_lines": function_slice.changed_lines,
                "model": _model,
                "verdict": verdict.verdict,
                "confidence": verdict.confidence,
                "vulnerability_type": verdict.vulnerability_type,
            },
        )

    return verdict


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    database_py = """\
import os

def get_user(user_id):
    user_id = request.args.get('id')
    query = f'SELECT * FROM users WHERE id = {user_id}'
    cursor.execute(query)
    return cursor.fetchone()
"""

    slices = slice_context("database.py", database_py, changed_lines=[4])

    for s in slices:
        print(f"Evaluating {s.function_name}() …")
        verdict = evaluate_slice(s)
        print(verdict.model_dump_json(indent=2))
