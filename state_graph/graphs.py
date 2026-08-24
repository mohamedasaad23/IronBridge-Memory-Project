"""
Three Ironbridge state graphs — each needs cycles/waits/HITL, not a single linear pass.

1. cert_coordination  — multi-day cert renewal (wait on training result)
2. high_risk_dig      — trench approval; irreversible action needs supervisor HITL
3. incident_handoff   — near-miss investigation with external report wait + failure path

Each graph uses two LLM-style additions (implemented as deterministic/heuristic
modules so demos run offline; swap for real LLM calls in production):
  cert_coordination: task decomposition + RAG
  high_risk_dig:     constrained ReAct tool checks + LATS-style option ranking
  incident_handoff:  Tree-of-Thoughts ordering + RAG policy lookup
"""
from __future__ import annotations

from typing import Optional

from platform_db import store
from state_graph.engine import GraphState, HITLPause, NodeFailure, StateGraph


# ---------- shared helpers: real reuse + real LLM-call additions ----------

def require_tool(state: GraphState, tool_name: str) -> None:
    """Fail the run if the admin has removed the tool this node needs, so the
    admin panel's tool add/remove toggle actually changes agent behavior
    instead of just being stored and ignored. Opens a real failure ticket,
    same as any other unplanned failure."""
    if not store.agent_has_tool(state.graph_name, tool_name):
        raise NodeFailure(
            "tool_disabled",
            f"Tool '{tool_name}' has been removed for agent '{state.graph_name}' "
            f"by admin. Re-enable it in Admin → Agents to resume.",
        )


def get_cert_reused(worker_id: str, equipment_type: str) -> Optional[dict]:
    """Prefer the real MCP service (backed by db/ironbridge.db) over the
    platform_db mirror, so certification checks reuse the existing system
    instead of a duplicated one. Falls back to platform_db only if the MCP
    service/DB isn't reachable (e.g. SUPERVISOR_PIN not configured)."""
    try:
        from mcp_server import service as _svc
        id_map = {"W1": 1, "W2": 2, "W3": 3}
        nid = id_map.get(str(worker_id))
        if nid is not None:
            c = _svc.check_certification(nid, equipment_type)
            return {
                "status": "valid" if c.get("valid") else "expired",
                "reason": c.get("reason"),
                "valid_until": c.get("valid_until"),
            }
    except Exception:
        pass
    return store.get_cert(worker_id, equipment_type)


def rag_lookup(query: str) -> str:
    """Prefer the real hybrid RAG pipeline (rag/) built in Part 3 over the
    platform_db keyword-overlap store, so the state graphs reuse the existing
    retrieval system rather than standing up a second one. Falls back to
    platform_db only if the real vector index hasn't been built."""
    try:
        from multi_agent.mcp_bridge import try_rag_snippets
        hits = try_rag_snippets(query, top_k=2)
        if hits:
            return " | ".join(hits)
    except Exception:
        pass
    hits = store.search_rag(query, top_k=2)
    if not hits:
        return "No policy found."
    return " | ".join(f"[{h['title']}] {h['content'][:200]}" for h in hits)


def decompose_cert_plan(equipment_type: str) -> list[str]:
    """Task decomposition: an LLM call breaks cert renewal into ordered
    sub-steps for this equipment type. Falls back to a fixed offline plan
    when no LLM is configured, so the demo still runs without a key."""
    fallback = [
        f"verify_current_{equipment_type}_status",
        "schedule_classroom",
        "schedule_practical",
        "await_training_results",
        "update_cert_record",
        "notify_site",
    ]
    from multi_agent import llm  # lazy: avoids a circular import with multi_agent.router
    if not llm.available():
        return fallback
    result = llm.decide_json(
        f"Decompose the renewal process for a {equipment_type} operator "
        f"certification into an ordered list of concrete sub-steps.",
        schema_hint='{"steps": ["step_id", "step_id", ...]}',
        system=(
            "You are a construction-site certification coordinator. Break "
            "the renewal process into 4-7 ordered snake_case sub-task ids, "
            "starting with a status check and ending with notifying the site. "
            "JSON only."
        ),
    )
    steps = result.get("steps")
    if isinstance(steps, list) and steps and all(isinstance(s, str) for s in steps):
        return steps
    return fallback


def rank_dig_options(depth_m: float, soil: str) -> list[dict]:
    """LATS-style: an LLM call scores each protective system against OSHA-style
    trench-safety fitness for this depth/soil, real branches are actually
    weighed rather than a fixed lookup table. Falls back to a fixed heuristic
    ranking offline."""
    fallback = sorted(
        [
            {"id": "slope", "score": 0.6 if soil == "C" else 0.8, "note": "1.5:1 slope for type C"},
            {"id": "shoring", "score": 0.9, "note": "hydraulic shoring preferred >1.5m"},
            {"id": "shield", "score": 0.85, "note": "trench box; good for tight sites"},
            {"id": "none", "score": 0.1 if depth_m > 1.5 else 0.7, "note": "no protection"},
        ],
        key=lambda o: -o["score"],
    )
    from multi_agent import llm  # lazy: avoids a circular import with multi_agent.router
    if not llm.available():
        return fallback
    result = llm.decide_json(
        f"Rank protective systems for a {depth_m}m trench in type {soil} soil: "
        f"slope, shoring, shield, none.",
        schema_hint='{"ranked": [{"id": "slope|shoring|shield|none", "score": 0.0, "note": "why"}]}',
        system=(
            "You are a trench-safety officer scoring protective systems 0-1 "
            "against OSHA 1926 Subpart P for the given depth and soil type. "
            "Rank best first. JSON only."
        ),
    )
    ranked = result.get("ranked")
    if isinstance(ranked, list) and ranked and all(
        isinstance(o, dict) and "id" in o and "score" in o for o in ranked
    ):
        try:
            return sorted(ranked, key=lambda o: -float(o.get("score", 0)))
        except (TypeError, ValueError):
            pass
    return fallback


def tot_investigation_order(incident_type: str) -> list[str]:
    """Tree-of-Thoughts: an LLM call generates a few candidate investigation
    orderings, self-scores each, and the best-scored candidate is kept —
    genuine generate-then-prune search rather than a fixed if/elif table.
    Falls back to a fixed heuristic ordering offline."""
    if "crane" in incident_type.lower():
        fallback = ["secure_scene", "interview_operator", "inspect_crane", "pull_cert", "draft_report"]
    elif "trench" in incident_type.lower():
        fallback = ["secure_scene", "soil_sample", "interview_crew", "check_protection", "draft_report"]
    else:
        fallback = ["secure_scene", "interview_witnesses", "collect_photos", "draft_report"]
    from multi_agent import llm  # lazy: avoids a circular import with multi_agent.router
    if not llm.available():
        return fallback
    result = llm.decide_json(
        f"For a '{incident_type}' construction near-miss, propose 2-3 candidate "
        f"orderings of investigation steps drawn from: secure_scene, "
        f"interview_operator, interview_crew, interview_witnesses, inspect_crane, "
        f"soil_sample, check_protection, pull_cert, collect_photos, draft_report. "
        f"Self-score each ordering 0-1 on thoroughness and speed.",
        schema_hint='{"candidates": [{"steps": ["..."], "score": 0.0}], "best_index": 0}',
        system=(
            "You are a site safety investigator using Tree-of-Thoughts: generate "
            "several candidate step orderings, self-evaluate each, then select "
            "the best. 'secure_scene' must be first and 'draft_report' must be "
            "last in every candidate. JSON only."
        ),
    )
    candidates = result.get("candidates")
    if isinstance(candidates, list) and candidates:
        try:
            chosen = candidates[int(result.get("best_index", 0))]["steps"]
        except (KeyError, ValueError, IndexError, TypeError):
            chosen = max(candidates, key=lambda c: c.get("score", 0)).get("steps")
        if isinstance(chosen, list) and chosen and chosen[0] == "secure_scene" and chosen[-1] == "draft_report":
            return chosen
    return fallback


# ===================== 1. CERT COORDINATION =====================

def cert_start(state: GraphState) -> str:
    eq = state.data.get("equipment_type", "CRANE")
    plan = decompose_cert_plan(eq)
    state.data["plan"] = plan
    state.data["plan_index"] = 0
    policy = rag_lookup(f"{eq} certification renewal policy")
    state.data["policy_excerpt"] = policy
    return "check_current"


def cert_check_current(state: GraphState) -> str:
    require_tool(state, "lookup_cert")
    wid = state.worker_id or state.data.get("worker_id", "W2")
    eq = state.data.get("equipment_type", "CRANE")
    cert = get_cert_reused(wid, eq)
    state.data["current_cert"] = cert
    if cert and cert.get("status") == "valid":
        state.data["outcome"] = "already_valid"
        return "END"
    return "schedule_training"


def cert_schedule(state: GraphState) -> str:
    state.data["training_scheduled"] = True
    state.data["awaiting_results"] = True
    # Simulate external wait: if results not injected, loop to wait node
    if not state.data.get("training_results"):
        return "await_results"
    return "apply_results"


def cert_await(state: GraphState) -> str:
    """Waiting state — can sit until external system / admin injects results.
    If still missing after resume, HITL asks training office."""
    if state.data.get("training_results"):
        return "apply_results"
    # After one wait cycle without results → HITL
    if state.data.get("wait_count", 0) >= 1:
        raise HITLPause(
            title="Training results missing",
            reason="Cert renewal is blocked until classroom+practical results are recorded.",
            payload={
                "worker_id": state.worker_id,
                "equipment_type": state.data.get("equipment_type"),
                "action": "provide_training_results",
            },
        )
    state.data["wait_count"] = state.data.get("wait_count", 0) + 1
    # Still waiting — checkpoint and pause via HITL for demo clarity
    raise HITLPause(
        title="Awaiting training office results",
        reason="External training results have not arrived. Confirm or attach results to continue.",
        payload={"worker_id": state.worker_id, "awaiting": True},
    )


def cert_apply(state: GraphState) -> str:
    results = state.data.get("training_results") or state.data.get("last_hitl_note")
    if state.hitl_decision == "rejected":
        state.data["outcome"] = "renewal_cancelled"
        return "END"
    # On HITL approve without explicit results, treat as passed
    if state.data.get("last_hitl_decision") == "approved" and not state.data.get("training_results"):
        state.data["training_results"] = "pass"
    if state.data.get("training_results") == "fail":
        state.data["outcome"] = "failed_exam"
        return "END"
    state.data["outcome"] = "cert_renewed"
    state.data["new_valid_until"] = "2028-01-10"
    return "notify"


def cert_notify(state: GraphState) -> str:
    state.data["notified"] = True
    return "END"


def build_cert_coordination() -> StateGraph:
    return StateGraph(
        name="cert_coordination",
        start="start",
        nodes={
            "start": cert_start,
            "check_current": cert_check_current,
            "schedule_training": cert_schedule,
            "await_results": cert_await,
            "apply_results": cert_apply,
            "notify": cert_notify,
        },
    )


# ===================== 2. HIGH-RISK DIG =====================

def dig_start(state: GraphState) -> str:
    state.data.setdefault("depth_m", 2.0)
    state.data.setdefault("soil", "C")
    state.data.setdefault("equipment_id", "EQ-9902")
    state.data.setdefault("equipment_type", "EXCAVATOR")
    return "constrained_checks"


def dig_constrained_checks(state: GraphState) -> str:
    """Constrained ReAct-style allow-listed checks only."""
    require_tool(state, "check_cert")
    depth = float(state.data.get("depth_m", 0))
    soil = state.data.get("soil", "C")
    wid = state.worker_id or state.data.get("worker_id", "W1")
    eq_type = state.data.get("equipment_type", "EXCAVATOR")

    cert = get_cert_reused(wid, eq_type)
    state.data["cert_ok"] = bool(cert and cert.get("status") == "valid")
    eq = store.get_equipment(state.data.get("equipment_id", ""))
    state.data["equipment_ok"] = bool(eq and eq.get("status") == "operational")

    policy = rag_lookup("trenching 4.2b protective systems")
    state.data["policy"] = policy

    if depth > 1.5 and soil == "C":
        state.data["requires_protection"] = True
    else:
        state.data["requires_protection"] = depth > 1.5

    if not state.data["cert_ok"]:
        state.data["outcome"] = "rejected_no_cert"
        return "END"
    if not state.data["equipment_ok"]:
        # Unplanned failure path — equipment status missing/bad
        raise NodeFailure(
            "equipment_unavailable",
            f"Equipment {state.data.get('equipment_id')} not operational or missing.",
        )
    return "rank_options"


def dig_rank(state: GraphState) -> str:
    ranked = rank_dig_options(
        float(state.data.get("depth_m", 0)), state.data.get("soil", "C")
    )
    state.data["ranked_options"] = ranked
    state.data["chosen_protection"] = ranked[0]["id"] if ranked else "none"
    if state.data["chosen_protection"] == "none" and state.data.get("requires_protection"):
        state.data["outcome"] = "rejected_no_safe_option"
        return "END"
    return "supervisor_gate"


def dig_supervisor_guard(state: GraphState) -> tuple[str, str, dict] | None:
    """HITL required for any dig >1.5m — agent cannot approve alone."""
    if float(state.data.get("depth_m", 0)) > 1.5:
        return (
            "Supervisor sign-off required for deep trench",
            f"Depth {state.data.get('depth_m')}m, soil {state.data.get('soil')}, "
            f"proposed protection={state.data.get('chosen_protection')}. "
            "Irreversible: crew may enter trench after approval.",
            {
                "depth_m": state.data.get("depth_m"),
                "soil": state.data.get("soil"),
                "protection": state.data.get("chosen_protection"),
                "worker_id": state.worker_id,
            },
        )
    return None


def dig_supervisor(state: GraphState) -> str:
    decision = state.hitl_decision or state.data.get("last_hitl_decision")
    if decision == "rejected":
        state.data["outcome"] = "supervisor_rejected"
        return "END"
    if decision == "approved":
        state.data["outcome"] = "approved"
        state.data["supervisor_approved"] = True
        return "execute_notice"
    # Should not reach here without HITL when depth > 1.5
    if float(state.data.get("depth_m", 0)) > 1.5:
        raise HITLPause(
            title="Supervisor sign-off required",
            reason="Deep trench cannot proceed without supervisor.",
            payload=state.data,
        )
    state.data["outcome"] = "approved_shallow"
    return "execute_notice"


def dig_execute_notice(state: GraphState) -> str:
    state.data["site_notified"] = True
    return "END"


def build_high_risk_dig() -> StateGraph:
    g = StateGraph(
        name="high_risk_dig",
        start="start",
        nodes={
            "start": dig_start,
            "constrained_checks": dig_constrained_checks,
            "rank_options": dig_rank,
            "supervisor_gate": dig_supervisor,
            "execute_notice": dig_execute_notice,
        },
        hitl_guards={"supervisor_gate": dig_supervisor_guard},
    )
    return g


# ===================== 3. INCIDENT HANDOFF =====================

def incident_start(state: GraphState) -> str:
    state.data.setdefault("incident_type", "crane near-miss")
    state.data["steps"] = tot_investigation_order(state.data["incident_type"])
    state.data["step_i"] = 0
    state.data["policy"] = rag_lookup("near-miss reporting investigation")
    return "secure_scene"


def incident_secure(state: GraphState) -> str:
    state.data["scene_secured"] = True
    return "run_steps"


def incident_run_steps(state: GraphState) -> str:
    steps = state.data.get("steps") or []
    i = state.data.get("step_i", 0)
    if i == 0:
        require_tool(state, "log_incident")
    if i >= len(steps):
        return "final_report"
    step = steps[i]
    state.data["current_step"] = step
    state.data["step_i"] = i + 1
    # Simulate missing external evidence → ticketable failure
    if step == "inspect_crane" and state.data.get("force_inspect_failure"):
        raise NodeFailure(
            "inspect_failed",
            "Crane inspection API timeout — cannot complete investigation step.",
        )
    if step == "draft_report":
        return "final_report"
    # loop until steps done
    return "run_steps"


def incident_final(state: GraphState) -> str:
    # Safety officer HITL before closing
    if not state.data.get("last_hitl_decision") and not state.hitl_decision:
        raise HITLPause(
            title="Safety officer approval for incident report",
            reason="Near-miss final report requires safety officer sign-off before close.",
            payload={
                "incident_type": state.data.get("incident_type"),
                "steps_done": state.data.get("steps"),
            },
        )
    if (state.hitl_decision or state.data.get("last_hitl_decision")) == "rejected":
        state.data["outcome"] = "report_rejected_rework"
        state.data["step_i"] = 0
        return "run_steps"  # cycle back for rework
    state.data["outcome"] = "report_closed"
    return "END"


def build_incident_handoff() -> StateGraph:
    return StateGraph(
        name="incident_handoff",
        start="start",
        nodes={
            "start": incident_start,
            "secure_scene": incident_secure,
            "run_steps": incident_run_steps,
            "final_report": incident_final,
        },
    )


# ---------- registry ----------

GRAPHS: dict[str, StateGraph] = {}


def get_graph(name: str) -> StateGraph:
    global GRAPHS
    if not GRAPHS:
        GRAPHS = {
            "cert_coordination": build_cert_coordination(),
            "high_risk_dig": build_high_risk_dig(),
            "incident_handoff": build_incident_handoff(),
        }
    if name not in GRAPHS:
        raise KeyError(f"Unknown graph: {name}")
    return GRAPHS[name]


def list_graph_names() -> list[str]:
    get_graph("cert_coordination")  # ensure built
    return list(GRAPHS.keys())
