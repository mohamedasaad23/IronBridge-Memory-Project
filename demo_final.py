#!/usr/bin/env python3
"""
Demo every Final Project concern offline (no API key):

1. Three state graphs start
2. HITL pause on high_risk_dig + cert_coordination
3. Failure ticket on incident_handoff (forced inspect failure)
4. Checkpoint persist + resume after HITL
5. Crash recovery: load latest checkpoint after "kill"
6. Multi-agent routing

Run from ironbridge_final/:  python demo_final.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from platform_db.store import init_db
from platform_db import store
from multi_agent.router import handle_user_message, resume_graph
from state_graph.engine import apply_hitl_to_state
from state_graph.graphs import get_graph


def banner(t: str) -> None:
    print("\n" + "=" * 60)
    print(t)
    print("=" * 60)


def main() -> None:
    init_db()
    banner("1) Multi-agent routing")
    for msg in [
        "What does section 4.2b say about trenches?",
        "Renew crane certification",
        "Approve dig 2.2m trench soil type C",
        "Log crane near-miss incident",
    ]:
        r = handle_user_message(msg, worker_id="W1")
        print(f"  MSG: {msg}")
        print(f"  → agent={r.get('agent')} status={r.get('status')} run={r.get('run_id')}")
        print(f"    {(r.get('reply') or '')[:120]}…")

    banner("2) High-risk dig → HITL supervisor gate")
    r = handle_user_message(
        "Approve dig 2.2m trench soil type C", worker_id="W1", force_agent="high_risk_dig"
    )
    dig_run = r["run_id"]
    print(f"  status={r['status']} node={r.get('node')} run={dig_run}")
    assert r["status"] == "paused_hitl", r
    hitls = store.list_hitl("open")
    print(f"  open HITL tasks: {len(hitls)}")
    task = next(t for t in hitls if t["run_id"] == dig_run)
    print(f"  resolving HITL {task['task_id']} → approved")
    store.resolve_hitl(task["task_id"], "approved", "supervisor OK")
    apply_hitl_to_state(dig_run, "approved", "supervisor OK")
    r2 = resume_graph("high_risk_dig", dig_run)
    print(f"  after resume: status={r2['status']} outcome={r2['data'].get('outcome')}")
    assert r2["status"] == "completed"

    banner("3) Cert coordination → HITL await training results")
    r = handle_user_message(
        "Renew crane certification", worker_id="W2", force_agent="cert_coordination"
    )
    cert_run = r["run_id"]
    print(f"  status={r['status']} run={cert_run}")
    assert r["status"] == "paused_hitl"
    task = next(t for t in store.list_hitl("open") if t["run_id"] == cert_run)
    store.resolve_hitl(task["task_id"], "approved", "results: pass")
    apply_hitl_to_state(cert_run, "approved", "pass")
    # inject results via state
    cp = store.latest_checkpoint(cert_run)
    st = cp["state"]
    st["data"]["training_results"] = "pass"
    store.save_checkpoint(cert_run, st["current_node"], st["step_index"], st)
    r2 = resume_graph("cert_coordination", cert_run)
    print(f"  after resume: status={r2['status']} outcome={r2['data'].get('outcome')}")

    banner("4) Incident handoff → forced failure ticket")
    r = handle_user_message(
        "crane near-miss force fail inspect timeout",
        worker_id="W1",
        force_agent="incident_handoff",
    )
    inc_run = r["run_id"]
    print(f"  status={r['status']} ticket={r['data'].get('failure_ticket')}")
    assert r["status"] == "failed"
    tickets = store.list_tickets("open")
    print(f"  open tickets: {len(tickets)}")
    tck = next(t for t in tickets if t["run_id"] == inc_run)
    # Clear force flag so resume succeeds
    cp = store.latest_checkpoint(inc_run)
    st = cp["state"]
    st["data"]["force_inspect_failure"] = False
    store.save_checkpoint(inc_run, st["current_node"], st["step_index"], st)
    store.resolve_ticket(tck["ticket_id"], "inspection API restored")
    store.update_run_status(inc_run, "running")
    r2 = resume_graph("incident_handoff", inc_run)
    print(f"  after ticket resolve+resume: status={r2['status']} node={r2.get('node')}")
    # may pause on final_report HITL
    if r2["status"] == "paused_hitl":
        task = next(t for t in store.list_hitl("open") if t["run_id"] == inc_run)
        store.resolve_hitl(task["task_id"], "approved", "SO signed")
        apply_hitl_to_state(inc_run, "approved", "SO signed")
        r3 = resume_graph("incident_handoff", inc_run)
        print(f"  after SO HITL: status={r3['status']} outcome={r3['data'].get('outcome')}")

    banner("5) Crash recovery (simulate process kill)")
    g = get_graph("high_risk_dig")
    state = g.run(
        {"depth_m": 2.5, "soil": "C", "equipment_id": "EQ-9902", "equipment_type": "EXCAVATOR"},
        worker_id="W1",
    )
    run_id = state.run_id
    print(f"  run paused/status={state.status} at node={state.current_node}")
    # "kill process" — only DB remains
    cp = store.latest_checkpoint(run_id)
    assert cp is not None
    print(f"  checkpoint saved: {cp['checkpoint_id']} step={cp['step_index']}")
    # "restart" — resume from DB
    if state.status == "paused_hitl":
        task = next(t for t in store.list_hitl("open") if t["run_id"] == run_id)
        store.resolve_hitl(task["task_id"], "approved", "post-crash approve")
        apply_hitl_to_state(run_id, "approved", "post-crash approve")
    recovered = g.resume(run_id)
    print(f"  recovered status={recovered.status} outcome={recovered.data.get('outcome')}")
    assert recovered.status in ("completed", "paused_hitl", "failed")

    banner("6) Admin surfaces (agents / RAG)")
    agents = store.list_agents()
    print(f"  agents registered: {[a['agent_id'] for a in agents]}")
    store.set_agent_tools("memory_rag", ["ask_safety_policy", "list_equipment"])
    print("  tools updated for memory_rag")
    doc_id = store.add_rag_doc("Bonus SOP", "general", "Always wear hard hat on Ironbridge sites.")
    print(f"  RAG doc added: {doc_id}")
    hits = store.search_rag("hard hat")
    print(f"  RAG search hits: {len(hits)}")

    banner("DONE — all Final Project concerns exercised")
    print("Platform: python app_platform/app.py  → http://127.0.0.1:5050")


if __name__ == "__main__":
    main()
