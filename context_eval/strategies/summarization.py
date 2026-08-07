"""
Recursive summarization: instead of deleting old history, compact it into
a short summary and continue from there. Costs an extra LLM call per
compaction (or, offline, a deterministic extractive fallback), but can
preserve decisions/findings that a fixed window would drop outright.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from memory._llm import call_structured  # reuse the same Gemini/offline helper
from pydantic import BaseModel, ConfigDict


class Summary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str


COMPACT_PROMPT = """Summarize the conversation turns below in 3-5 sentences.
Preserve: decisions made, unresolved issues, key findings (especially
anything about worker certifications or safety flags).
Discard: redundant tool output, superseded reasoning.

Turns:
{turns}
"""


def _offline_summary(old_messages: list[dict[str, Any]]) -> str:
    """Deterministic extractive fallback: keep any line that looks like a
    finding (contains a safety/certification keyword) verbatim, drop the
    rest. This is intentionally simple — a real recursive summarizer
    would use an LLM every time, which is exactly why this strategy
    costs more output tokens in the comparison table."""
    keep_words = ("expired", "cert", "denied", "reject", "high-risk", "pending_supervisor")
    kept_lines = [
        m["content"][:200]
        for m in old_messages
        if any(w in m["content"].lower() for w in keep_words)
    ]
    if not kept_lines:
        return "Earlier turns contained routine tool checks with no notable findings."
    return "Earlier findings: " + " | ".join(kept_lines[:5])


def apply(
    messages: list[dict[str, Any]], keep_recent: int = 8
) -> list[dict[str, Any]]:
    system = [m for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]

    if len(rest) <= keep_recent:
        return system + rest

    old, recent = rest[:-keep_recent], rest[-keep_recent:]
    turns_text = "\n".join(f"[{m['role']}] {m['content'][:300]}" for m in old)
    decision = call_structured(
        prompt=COMPACT_PROMPT.format(turns=turns_text),
        schema=Summary,
        offline_fallback=Summary(summary=_offline_summary(old)),
    )
    return system + [{"role": "system", "content": f"Earlier context: {decision.summary}"}] + recent