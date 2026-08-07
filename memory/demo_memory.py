"""
Standalone demo for the memory/ track. Run: python -m memory.demo_memory

Shows, in order:
  1. STM filling up and evicting -> router firing (forget + episodic cases)
  2. Scratchpad surviving a transcript clear
  3. A consolidation pass creating a semantic fact
  4. A SECOND consolidation pass hitting a real conflict (cert renewed
     after being recorded invalid) and resolving it with versioning
  5. Certification expiration firing with no new episode involved

DEMO_AUTO_APPROVE-style determinism: with no GOOGLE_API_KEY set, every
decision below is deterministic (memory/_llm.py's offline fallback),
so this script produces identical output on every run — required for a
reproducible grading demo.
"""
from __future__ import annotations

from . import consolidation, router, store
from .stm import ShortTermMemory

WORKER_ID = 2  # Sara Nabil — has an expired CRANE cert in seed_data.sql


def main() -> None:
    store.ensure_memory_schema()

    print("=== 1. STM fill + promote-or-drop ===")
    # max_turns=2 so the buffer evicts quickly — with only 2 slots, by the
    # time we've added 5 turns, the certification tool-result (turn 3) has
    # been pushed out and reaches the router, not just the small-talk turns.
    stm = ShortTermMemory(max_turns=2)
    stm.update_plan(plan="Help Sara request equipment", subgoal="check certification first")

    turns = [
        ("user", "Hi, what's the weather like at the site today?"),                    # -> forget
        ("assistant", "I don't have weather data, but I can check your equipment access."),  # -> forget
        (
            "tool",
            "check_worker_certification(worker_id=2, equipment_type=CRANE) -> "
            "{'valid': False, 'reason': 'Certification expired on 2025-01-10'}",
        ),                                                                              # -> episodic
        ("user", "Ok, can I still request the mobile crane?"),                          # -> forget
        ("assistant", "Not with that certification — it expired on 2025-01-10."),       # -> forget
    ]
    for role, content in turns:
        evicted = stm.add(role, content, worker_id=WORKER_ID)
        decision = router.process_overflow(evicted)
        if decision:
            print(f"  evicted -> {decision.destination}: {decision.reasoning}")

    print("  scratchpad survives:", stm.get_scratchpad())
    stm.clear_transcript_keep_scratchpad()
    print("  after transcript clear, scratchpad still:", stm.get_scratchpad())
    print()

    print("=== 2. Consolidation pass #1 (expired cert -> semantic fact) ===")
    decisions = consolidation.consolidate_worker(WORKER_ID)
    for d in decisions:
        print(f"  {d.action}: {d.reasoning}")
    print("  active facts:", store.get_all_active_facts(WORKER_ID))
    print()

    print("=== 3. New episode: Sara renews her CRANE cert ===")
    router.store.insert_episodic(
        worker_id=WORKER_ID,
        event_summary="Sara Nabil renewed her CRANE certification.",
        context="Renewal processed at site office.",
        outcome="CRANE certification renewed, valid_until=2028-06-30",
    )
    print("=== 4. Consolidation pass #2 (real conflict resolved) ===")
    decisions = consolidation.consolidate_worker(WORKER_ID)
    for d in decisions:
        print(f"  {d.action}: {d.reasoning}")
    print("  active facts now:", store.get_all_active_facts(WORKER_ID))
    print("  full version history for cert_status:CRANE:")
    for row in store.get_fact_history(WORKER_ID, "cert_status:CRANE"):
        print("   ", row)
    print()

    print("=== 5. Expiration — no new episode, fact expires on its own ===")
    # Manually seed a fact whose own valid_until has already passed, the
    # way a normal consolidation would if the episode had said
    # "valid_until=2025-01-10". This isolates the expiration path from
    # the conflict path above: no contradicting episode is involved here,
    # the fact just goes stale on its own.
    store.insert_semantic_fact(
        worker_id=WORKER_ID,
        fact_key="cert_status:EXCAVATOR",
        fact_value="valid_until=2025-01-10",
        version=1,
        source_episode_ids=[],
    )
    print("  before expiration check:", store.get_active_fact(WORKER_ID, "cert_status:EXCAVATOR"))
    consolidation._check_expirations(WORKER_ID)
    print("  after expiration check:", store.get_active_fact(WORKER_ID, "cert_status:EXCAVATOR"))
    print("  full history:", store.get_fact_history(WORKER_ID, "cert_status:EXCAVATOR"))
    print()

    print("Demo finished. Inspect memory_routing_log / consolidation_log")
    print("tables in db/ironbridge.db for the full audit trail.")


if __name__ == "__main__":
    main()