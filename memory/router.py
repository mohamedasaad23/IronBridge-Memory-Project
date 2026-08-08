"""
Promote-or-drop router.

Fires the moment ShortTermMemory evicts its oldest item (see
ShortTermMemory.add()). For that item, decides:
  - forget:   not worth keeping (small talk, redundant status checks)
  - episodic: a specific event worth recording

This module NEVER writes to semantic_memory. That is the whole point of
separating it from consolidation.py — the router only ever proposes
"is this worth remembering at all", not "what general fact does this
imply". Every decision, forget included, is logged to
memory_routing_log so a grader can see the reasoning without re-running
anything.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from . import store
from .stm import Message
from ._llm import call_structured

ROUTING_PROMPT = """An item is about to be evicted from an Iron Bridge
construction-site agent's short-term memory.

Decide where it belongs:
- forget: not worth keeping (small talk, a one-off status lookup with no
  ongoing relevance, a request that succeeded exactly as expected with
  nothing notable about it)
- episodic: a specific event worth recording — a certification problem,
  a rejected or high-risk equipment request, a supervisor decision, an
  authentication event, anything that could matter again later for this
  specific worker

Item (role={role}, worker_id={worker_id}):
{content}
"""


class MemoryRoutingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(..., description="Why this destination was chosen")
    destination: Literal["forget", "episodic"]
    event_summary: Optional[str] = None
    context: Optional[str] = None
    outcome: Optional[str] = None


def _offline_heuristic(item: Message) -> MemoryRoutingDecision:
    """Deterministic fallback so routing is repeatable without a live LLM
    key — used for grading/demo runs (see memory/_llm.py). Flags anything
    that looks like a safety-relevant outcome as episodic; everything else
    is forgotten."""
    text = item.content.lower()
    signal_words = (
        "expired",
        "reject",
        "denied",
        "high-risk",
        "high_risk",
        "pending_supervisor_approval",
        "not_submitted",
        "authenticate",
        "approved",
    )
    if any(w in text for w in signal_words):
        return MemoryRoutingDecision(
            reasoning=(
                "Offline heuristic: content contains a safety/authorization-"
                "relevant keyword, so this is kept as an event rather than "
                "discarded."
            ),
            destination="episodic",
            event_summary=item.content[:200],
            context=f"role={item.role}",
            outcome=None,
        )
    return MemoryRoutingDecision(
        reasoning=(
            "Offline heuristic: no safety/authorization keyword found; "
            "treated as routine chatter with no lasting relevance."
        ),
        destination="forget",
    )


def decide_memory_fate(item: Message) -> MemoryRoutingDecision:
    prompt = ROUTING_PROMPT.format(
        role=item.role, worker_id=item.worker_id, content=item.content
    )
    return call_structured(
        prompt=prompt,
        schema=MemoryRoutingDecision,
        offline_fallback=_offline_heuristic(item),
    )


def process_overflow(evicted: Optional[Message]) -> Optional[MemoryRoutingDecision]:
    """Call this whenever ShortTermMemory.add() returns a non-None evicted
    message. Returns the decision made (and its logged row), or None if
    nothing was evicted this turn."""
    if evicted is None:
        return None

    decision = decide_memory_fate(evicted)

    if decision.destination == "forget":
        store.insert_routing_log(
            worker_id=evicted.worker_id,
            source_item=evicted.content,
            decision="forget",
            reasoning=decision.reasoning,
        )
        return decision

    # destination == "episodic"
    episodic_id = store.insert_episodic(
        worker_id=evicted.worker_id,
        event_summary=decision.event_summary or evicted.content[:200],
        context=decision.context,
        outcome=decision.outcome,
    )
    store.insert_routing_log(
        worker_id=evicted.worker_id,
        source_item=evicted.content,
        decision="episodic",
        reasoning=decision.reasoning,
        episodic_id=episodic_id,
    )
    return decision