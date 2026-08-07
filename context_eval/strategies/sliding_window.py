"""
Sliding window: the simplest strategy. Keep only the last `keep_turns`
messages (plus the system prompt), discard everything older. No extra
LLM calls, extremely cheap, but anything older than the window is gone
for good — including the critical fact if it fell outside the window.
"""
from __future__ import annotations

from typing import Any


def apply(messages: list[dict[str, Any]], keep_turns: int = 10) -> list[dict[str, Any]]:
    system = [m for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    return system + rest[-keep_turns:]