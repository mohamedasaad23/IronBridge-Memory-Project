"""
Semantic memory consolidation.

This is a SEPARATE, PERIODIC pass over episodic_memory — never called by
router.py, and never run at write time. Run it on a schedule (cron / a
"consolidate" CLI command) or manually before a demo. It is the ONLY
writer of semantic_memory.

What it has to actually solve (per the lab spec), each mapped to code
below:
  - updates:            a new episode implies a fact already exists     -> _apply
  - versioning:          old row is closed (valid_until), never deleted -> store.close_fact
  - expiration:          facts with a natural expiry (certifications)   -> _check_expirations
  - conflict resolution: two episodes imply contradictory fact values   -> _apply (conflict branch)

Worked conflict this module resolves for real (see demo transcript):
  Worker 2 (Sara Nabil) has an episode "CRANE cert expired on 2025-01-10"
  which consolidates to semantic fact cert_status:CRANE = "invalid
  (expired 2025-01-10)". Later, a new episode records her renewing the
  certification: "CRANE cert renewed, valid_until 2028-01-10". These two
  facts contradict each other for the same worker+fact_key. Consolidation
  does NOT silently overwrite: it closes the old row (valid_until set,
  superseded_by pointing at the new row) and inserts a new version,
  logging the conflict and its resolution to consolidation_log.
"""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from . import store
from ._llm import call_structured

EXTRACTION_PROMPT = """You are consolidating episodic memories into a
general fact about a construction-site worker.

Existing active fact for this worker (may be None if there isn't one yet):
  fact_key: {fact_key}
  current_value: {current_value}

New episode to consider:
  event_summary: {event_summary}
  context: {context}
  outcome: {outcome}

Decide whether this episode should:
- create: introduce a new fact (no current_value exists)
- update: refine the existing fact without contradicting it
- conflict: the episode implies a value that CONTRADICTS current_value
  (e.g. cert was expired, now it's renewed and valid)
- ignore: this episode doesn't actually tell us anything new about this
  fact_key

If action is create/update/conflict, give the new fact_value as a short,
literal string (e.g. "valid_until=2028-01-10", "invalid (expired
2025-01-10)", "repeated attempt to operate CRANE without valid cert").
"""


class ConsolidationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str
    action: Literal["create", "update", "conflict", "ignore"]
    fact_value: Optional[str] = None


# Fact keys this system knows how to derive from episodes. Kept as an
# explicit, small vocabulary rather than free-form keys so semantic
# memory stays queryable instead of turning into unstructured text.
CERT_KEY_PREFIX = "cert_status:"
BEHAVIOR_KEY = "flagged_behavior"


def _offline_extract(
    fact_key: str, current_value: Optional[str], event_summary: str, outcome: Optional[str]
) -> ConsolidationDecision:
    """Deterministic fallback (no live LLM key) covering the two patterns
    this system actually needs: certification status and repeated unsafe
    equipment requests. Kept simple and literal on purpose so the fixed
    demo transcript reproduces the same consolidation result every run."""
    text = f"{event_summary} {outcome or ''}".lower()

    if fact_key.startswith(CERT_KEY_PREFIX):
        if "renew" in text or ("valid" in text and "invalid" not in text and "expired" not in text):
            new_value = "valid (renewed — see source episode)"
            action: Literal["create", "update", "conflict", "ignore"] = (
                "conflict" if current_value and "invalid" in current_value else "update"
            )
            return ConsolidationDecision(
                reasoning=(
                    "Episode reports a renewal/valid certification; this "
                    "contradicts the currently-active 'invalid' fact, so "
                    "the old fact must be closed rather than overwritten."
                    if action == "conflict"
                    else "Episode confirms certification status, no prior fact existed."
                ),
                action=action,
                fact_value=new_value,
            )
        if "expired" in text or "no certification" in text or "not found" in text:
            return ConsolidationDecision(
                reasoning="Episode reports an invalid/expired certification.",
                action="update" if current_value is None else "conflict",
                fact_value="invalid (see source episode)",
            )
        return ConsolidationDecision(reasoning="No new certification signal.", action="ignore")

    if fact_key == BEHAVIOR_KEY:
        if "rejected" in text or "not_submitted" in text or "high-risk" in text:
            return ConsolidationDecision(
                reasoning="Episode shows a rejected/blocked high-risk request worth flagging as a pattern.",
                action="update",
                fact_value="has had a high-risk equipment request rejected or blocked at least once",
            )
        return ConsolidationDecision(reasoning="No flaggable behavior in this episode.", action="ignore")

    return ConsolidationDecision(reasoning="Unrecognized fact_key.", action="ignore")


def _infer_fact_key(episode: dict) -> Optional[str]:
    text = f"{episode['event_summary']} {episode.get('outcome') or ''}".lower()
    if "cert" in text or "crane" in text or "excavator" in text or "scaffold" in text:
        # Best-effort equipment type extraction for the fact key.
        for eq_type in ("CRANE", "EXCAVATOR", "SCAFFOLD", "GENERATOR"):
            if eq_type.lower() in text:
                return f"{CERT_KEY_PREFIX}{eq_type}"
        return f"{CERT_KEY_PREFIX}UNKNOWN"
    if "reject" in text or "high-risk" in text or "not_submitted" in text:
        return BEHAVIOR_KEY
    return None


def _next_version(current: Optional[dict]) -> int:
    return (current["version"] + 1) if current else 1


def _apply(worker_id: int, fact_key: str, episode_id: int, decision: ConsolidationDecision) -> None:
    if decision.action == "ignore":
        return

    current = store.get_active_fact(worker_id, fact_key)

    if decision.action == "create":
        new_id = store.insert_semantic_fact(
            worker_id, fact_key, decision.fact_value or "", version=1,
            source_episode_ids=[episode_id],
        )
        store.insert_consolidation_log(
            worker_id, fact_key, "create", decision.reasoning, new_semantic_id=new_id,
        )
        return

    if decision.action == "update" and current is None:
        # No existing fact to update against — treat as create.
        new_id = store.insert_semantic_fact(
            worker_id, fact_key, decision.fact_value or "", version=1,
            source_episode_ids=[episode_id],
        )
        store.insert_consolidation_log(
            worker_id, fact_key, "create", decision.reasoning, new_semantic_id=new_id,
        )
        return

    if decision.action in ("update", "conflict"):
        # Versioning + conflict resolution: close the old row, never
        # delete it, then insert the new version pointing back at it.
        new_version = _next_version(current)
        new_id = store.insert_semantic_fact(
            worker_id, fact_key, decision.fact_value or "", version=new_version,
            source_episode_ids=[episode_id],
        )
        store.close_fact(current["id"], superseded_by=new_id)
        store.insert_consolidation_log(
            worker_id,
            fact_key,
            action="conflict_resolved" if decision.action == "conflict" else "update",
            reasoning=decision.reasoning,
            old_semantic_id=current["id"],
            new_semantic_id=new_id,
        )


def _check_expirations(worker_id: int) -> None:
    """Certification facts carry a natural expiry independent of any new
    episode — a fact recorded as 'valid until 2025-01-10' should stop
    being reported as active once that date passes, even with no new
    episode telling us so."""
    today = date.today().isoformat()
    for fact in store.get_all_active_facts(worker_id):
        if not fact["fact_key"].startswith(CERT_KEY_PREFIX):
            continue
        if "valid_until=" not in fact["fact_value"]:
            continue
        expiry = fact["fact_value"].split("valid_until=")[-1].strip()
        if expiry < today:
            store.close_fact(fact["id"])
            store.insert_consolidation_log(
                worker_id,
                fact["fact_key"],
                action="expire",
                reasoning=f"Fact's own valid_until={expiry} has passed as of {today}.",
                old_semantic_id=fact["id"],
            )


def consolidate_worker(worker_id: int) -> list[ConsolidationDecision]:
    """Run one consolidation pass for a single worker. Call this
    periodically (a scheduled job) or manually before a demo — never
    from inside the request/response path."""
    decisions: list[ConsolidationDecision] = []
    episodes = store.get_unconsolidated_episodes(worker_id)
    consolidated_ids: list[int] = []

    for ep in episodes:
        fact_key = _infer_fact_key(ep)
        if fact_key is None:
            consolidated_ids.append(ep["id"])
            continue

        current = store.get_active_fact(worker_id, fact_key)
        prompt = EXTRACTION_PROMPT.format(
            fact_key=fact_key,
            current_value=current["fact_value"] if current else None,
            event_summary=ep["event_summary"],
            context=ep["context"],
            outcome=ep["outcome"],
        )
        decision = call_structured(
            prompt=prompt,
            schema=ConsolidationDecision,
            offline_fallback=_offline_extract(
                fact_key,
                current["fact_value"] if current else None,
                ep["event_summary"],
                ep["outcome"],
            ),
        )
        _apply(worker_id, fact_key, ep["id"], decision)
        decisions.append(decision)
        consolidated_ids.append(ep["id"])

    store.mark_episodes_consolidated(consolidated_ids)
    _check_expirations(worker_id)
    return decisions