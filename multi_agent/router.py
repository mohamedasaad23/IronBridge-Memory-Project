"""
Multi-agent supervisor: routes to memory/planning/graphs.
When GOOGLE_API_KEY is set, memory_rag uses Gemini with RAG context + constraints.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from platform_db import store
from multi_agent import llm
from multi_agent import mcp_bridge
from state_graph.graphs import get_graph, list_graph_names


def classify_intent(message: str) -> str:
    m = message.lower()
    if any(w in m for w in ("policy", "manual", "osha", "what does", "section", "4.2b")):
        return "memory_rag"
    if any(w in m for w in ("near-miss", "incident", "investigation", "accident")):
        return "incident_handoff"
    if any(w in m for w in ("approve dig", "dig a", "trench dig", "shoring")) or (
        ("trench" in m or "dig" in m)
        and any(x in m for x in ("approve", "request", "soil", "m "))
    ):
        return "high_risk_dig"
    if any(w in m for w in ("cert", "renew", "training", "licence", "license")):
        return "cert_coordination"
    if any(w in m for w in ("approve", "request equipment", "plan", "schedule board")):
        return "planning"
    # Optional: let Gemini pick agent when key is present
    if llm.available() and len(message) > 20:
        decision = llm.decide_json(
            message,
            schema_hint='{"agent":"memory_rag|planning|cert_coordination|high_risk_dig|incident_handoff","reason":"string"}',
            system="Pick the best Ironbridge agent for this worker message. JSON only.",
        )
        agent = decision.get("agent", "memory_rag")
        if agent in (
            "memory_rag",
            "planning",
            "cert_coordination",
            "high_risk_dig",
            "incident_handoff",
        ):
            return agent
    return "memory_rag"


def memory_rag_reply(
    message: str,
    worker_id: Optional[str] = None,
    history: Optional[list[dict[str, str]]] = None,
) -> dict[str, Any]:
    """Site assistant: RAG + MCP person data injected silently."""
    context_parts = []
    # Admin tool add/remove actually gates behavior here: removing
    # "ask_safety_policy" or "check_certification" for memory_rag stops the
    # assistant from using that context source, not just from listing it.
    if store.agent_has_tool("memory_rag", "ask_safety_policy"):
        # Prefer real RAG index from repo; fall back to platform_db docs
        rag_bits = mcp_bridge.try_rag_snippets(message)
        if rag_bits:
            context_parts.extend(rag_bits)
        else:
            for h in store.search_rag(message, top_k=3):
                context_parts.append(f"POLICY: [{h['title']}] {h['content']}")
    if worker_id and store.agent_has_tool("memory_rag", "check_certification"):
        svc = mcp_bridge.try_service_context(worker_id)
        if svc:
            context_parts.append("PERSON_DATA:\n" + svc)
        else:
            try:
                context_parts.append("PERSON_DATA:\n" + store.mcp_context_for_worker(worker_id))
            except Exception:
                pass
    context = "\n".join(context_parts) if context_parts else ""
    system = (
        "You are the Ironbridge site assistant, talking to ONE specific logged-in worker "
        "(their identity, tasks, attendance, certs and equipment are in PERSON_DATA below). "
        "Use PERSON_DATA and POLICY silently — never mention RAG, embeddings, or internal IDs "
        "unless asked.\n"
        "- If the user asks who they are, or about their name/role/certs/tasks/attendance, "
        "answer directly from PERSON_DATA — do not just re-introduce yourself.\n"
        "- If the user is clarifying or correcting something they just said (e.g. 'no, I meant "
        "me'), re-read the conversation so far and answer the corrected question — never repeat "
        "your previous reply verbatim.\n"
        "- If something truly isn't in PERSON_DATA or POLICY, say plainly you don't have that "
        "and suggest who to ask — don't paper over it with a generic greeting.\n"
        "- Always answer in the same language the user just wrote in (Arabic or English).\n"
        "- Finish every sentence you start. Prefer a short complete answer (2-5 sentences) over "
        "a longer one that risks being cut off."
    )
    answer = llm.chat(message, context=context, system=system, history=history)
    return {
        "agent": "site_assistant",
        "reply": answer,
        "llm": "gemini" if llm.available() else "offline",
    }


def planning_reply(message: str, worker_id: Optional[str] = None) -> dict[str, Any]:
    wid = worker_id or "W1"
    cert = store.get_cert(wid, "EXCAVATOR")
    eq = store.get_equipment("EQ-9902")
    context = f"cert={cert}\nequipment={eq}\nuser_request={message}"

    if llm.available():
        decision = llm.decide_json(
            message,
            schema_hint='{"decision":"APPROVE|REJECT|ESCALATE","reason":"string"}',
            context=context,
            system=(
                "You are Ironbridge planning agent. Only APPROVE if cert status is valid "
                "AND equipment status is operational. Otherwise REJECT or ESCALATE. JSON only."
            ),
        )
        reply = f"Planning (Gemini): {decision.get('decision')} — {decision.get('reason')}"
        return {"agent": "planning", "reply": reply, "decision": decision, "llm": "gemini"}

    steps = [f"1. Cert: {cert}", f"2. Equipment: {eq}"]
    if cert and cert.get("status") == "valid" and eq and eq.get("status") == "operational":
        decision = "APPROVE (use high_risk_dig if depth > 1.5m)"
    else:
        decision = "REJECT or ESCALATE"
    return {
        "agent": "planning",
        "reply": "Planning (offline):\n" + "\n".join(steps) + f"\n\nDecision: {decision}",
        "decision": decision,
        "llm": "offline",
    }


def start_graph(
    graph_name: str,
    worker_id: Optional[str],
    data: Optional[dict] = None,
) -> dict[str, Any]:
    g = get_graph(graph_name)
    state = g.run(data or {}, worker_id=worker_id)
    return {
        "agent": graph_name,
        "run_id": state.run_id,
        "status": state.status,
        "node": state.current_node,
        "data": state.data,
        "history": state.history[-8:],
        "reply": _status_message(state),
        "llm": "gemini" if llm.available() else "offline",
    }


def resume_graph(graph_name: str, run_id: str) -> dict[str, Any]:
    g = get_graph(graph_name)
    state = g.resume(run_id)
    return {
        "agent": graph_name,
        "run_id": state.run_id,
        "status": state.status,
        "node": state.current_node,
        "data": state.data,
        "history": state.history[-8:],
        "reply": _status_message(state),
    }


def _status_message(state) -> str:
    if state.status == "paused_hitl":
        return (
            f"Paused for human approval (HITL). Run `{state.run_id}` is waiting. "
            f"Open Admin → HITL to approve/reject, then resume."
        )
    if state.status == "failed":
        return (
            f"Run failed at node `{state.current_node}`. "
            f"Ticket `{state.data.get('failure_ticket')}` opened. "
            f"Resolve the ticket in Admin, then resume from checkpoint."
        )
    if state.status == "completed":
        return f"Completed. Outcome: {state.data.get('outcome', state.data)}"
    return f"Status={state.status} node={state.current_node}"


def handle_user_message(
    message: str,
    worker_id: Optional[str] = None,
    force_agent: Optional[str] = None,
    history: Optional[list[dict[str, str]]] = None,
) -> dict[str, Any]:
    agents = {a["agent_id"]: a for a in store.list_agents()}

    # Strong workflow keywords always win over the UI dropdown, so
    # "Approve dig..." starts high_risk_dig even if Memory & RAG is selected.
    auto = classify_intent(message)
    graph_names = set(list_graph_names())
    if auto in graph_names:
        intent = auto
    elif force_agent:
        intent = force_agent
    else:
        intent = auto

    if intent in agents and not agents[intent].get("enabled", 1):
        return {
            "agent": intent,
            "reply": f"Agent `{intent}` is disabled by admin.",
            "status": "disabled",
        }

    if intent == "memory_rag":
        return memory_rag_reply(message, worker_id, history=history)
    if intent == "planning":
        return planning_reply(message, worker_id)

    if intent in graph_names:
        data = _parse_graph_input(intent, message, worker_id)
        return start_graph(intent, worker_id, data)

    return memory_rag_reply(message, worker_id)


def _parse_graph_input(graph: str, message: str, worker_id: Optional[str]) -> dict:
    data: dict[str, Any] = {"worker_id": worker_id, "raw": message}
    if graph == "high_risk_dig":
        m = re.search(r"([\d.]+)\s*m", message.lower())
        if m:
            data["depth_m"] = float(m.group(1))
        if "soil" in message.lower():
            for s in ("A", "B", "C"):
                if f"type {s.lower()}" in message.lower() or f"soil {s.lower()}" in message.lower():
                    data["soil"] = s
        data.setdefault("depth_m", 2.0)
        data.setdefault("soil", "C")
        data["equipment_id"] = "EQ-9902"
        data["equipment_type"] = "EXCAVATOR"
    elif graph == "cert_coordination":
        data["equipment_type"] = "CRANE" if "crane" in message.lower() else "EXCAVATOR"
        if "pass" in message.lower():
            data["training_results"] = "pass"
        if "fail" in message.lower() and "failsafe" not in message.lower():
            data["training_results"] = "fail"
    elif graph == "incident_handoff":
        data["incident_type"] = message
        if "timeout" in message.lower() or "force fail" in message.lower():
            data["force_inspect_failure"] = True
    return data
