"""
Zone-based pruning: rather than one hard cutoff, split history into age
zones and compress each progressively harder the older it is:
  Zone 1 (most recent): kept verbatim.
  Zone 2: tool outputs masked, dialogue kept verbatim.
  Zone 3: everything summarized into one compact line per few turns.
  Zone 4 (oldest): dropped entirely, except lines matching a safety
    keyword, which are kept as a bare fact list — a cheap way to avoid
    losing a critical detail that's aged past the summarized zone.
"""
from __future__ import annotations

from typing import Any

from . import masking

SAFETY_KEYWORDS = ("expired", "cert", "denied", "reject", "high-risk", "pending_supervisor")


def _zone_bounds(n: int) -> tuple[int, int, int]:
    """Return (zone1_start, zone2_start, zone3_start) indices into the
    non-system message list, splitting it into 4 roughly equal zones."""
    z1 = max(n - n // 4, 0)
    z2 = max(n - n // 2, 0)
    z3 = max(n - (3 * n) // 4, 0)
    return z1, z2, z3


def apply(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    system = [m for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    n = len(rest)
    if n == 0:
        return system

    z1_start, z2_start, z3_start = _zone_bounds(n)

    zone4 = rest[:z3_start]          # oldest — heavily pruned
    zone3 = rest[z3_start:z2_start]  # summarized
    zone2 = rest[z2_start:z1_start]  # tool-masked, dialogue kept
    zone1 = rest[z1_start:]          # kept verbatim

    # Zone 4: drop everything except safety-flagged lines, kept as a
    # compact fact list rather than full messages.
    zone4_facts = [
        m["content"][:150]
        for m in zone4
        if any(w in m["content"].lower() for w in SAFETY_KEYWORDS)
    ]
    zone4_out = (
        [{"role": "system", "content": "Oldest-zone flagged facts: " + " | ".join(zone4_facts)}]
        if zone4_facts
        else []
    )

    # Zone 3: one compact line summarizing the whole zone (extractive,
    # same posture as summarization.py's offline fallback — no LLM call
    # needed for this coarse a zone).
    zone3_facts = [
        m["content"][:150]
        for m in zone3
        if any(w in m["content"].lower() for w in SAFETY_KEYWORDS)
    ]
    zone3_out = (
        [{"role": "system", "content": "Mid-zone summary: " + " | ".join(zone3_facts)}]
        if zone3_facts
        else [{"role": "system", "content": f"Mid-zone: {len(zone3)} routine turns, no flags."}]
    )

    # Zone 2: reuse the masking strategy's placeholder logic for tool
    # turns, keep dialogue verbatim.
    zone2_out = masking.apply(zone2, keep_recent_tool_outputs=0)

    return system + zone4_out + zone3_out + zone2_out + zone1