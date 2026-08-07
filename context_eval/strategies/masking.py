"""
Observation (tool-output) masking: the biggest contributor to context
size is tool output, not dialogue. Instead of dropping history wholesale,
replace OLD tool outputs with a short placeholder while keeping every
non-tool message (user/assistant turns) intact, and keeping the most
recent `keep_recent_tool_outputs` tool results in full.

This is the strategy the lab's own worked example ends up shipping,
because the bloat in a tool-heavy transcript is JSON payloads, not
conversation — masking targets exactly that source of bloat instead of
truncating the timeline itself.
"""
from __future__ import annotations

from typing import Any

PLACEHOLDER = "[tool output omitted — see reasoning above]"


def apply(
    messages: list[dict[str, Any]], keep_recent_tool_outputs: int = 3
) -> list[dict[str, Any]]:
    tool_indices = [i for i, m in enumerate(messages) if m["role"] == "tool"]
    to_mask = set(tool_indices[:-keep_recent_tool_outputs]) if len(
        tool_indices
    ) > keep_recent_tool_outputs else set()

    result = []
    for i, m in enumerate(messages):
        if i in to_mask:
            result.append({**m, "content": PLACEHOLDER})
        else:
            result.append(m)
    return result