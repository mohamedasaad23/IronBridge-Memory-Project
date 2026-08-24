"""
Shared structured-output LLM call used by router.py and consolidation.py.

Mirrors the pattern already used in agent/client.py's sampling_callback:
real Gemini call if GOOGLE_API_KEY is set, deterministic offline fallback
otherwise, so grading/demo runs are repeatable without a live key.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Type, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

T = TypeVar("T", bound=BaseModel)

MODEL_NAME = "gemini-3.6-flash"


def call_structured(prompt: str, schema: Type[T], offline_fallback: T) -> T:
    """Call the LLM asking for JSON matching `schema`. Falls back to a fixed
    deterministic value if no GOOGLE_API_KEY is configured, so the same
    test suite produces the same routing/consolidation decisions on repeat
    runs — required by the lab's "keep test suites fixed" guardrail.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return offline_fallback

    from google import genai

    client = genai.Client(api_key=api_key)
    full_prompt = (
        f"{prompt}\n\n"
        f"Respond with ONLY a raw JSON object matching this shape "
        f"(no markdown fences, no preamble): {schema.model_json_schema()}"
    )
    resp = client.models.generate_content(model=MODEL_NAME, contents=full_prompt)
    text = resp.text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    try:
        data = json.loads(text)
        return schema.model_validate(data)
    except Exception:
        # A malformed LLM response should never crash the memory pipeline —
        # fall back to the deterministic path and let the caller's own
        # heuristics decide, same posture as an offline run.
        return offline_fallback