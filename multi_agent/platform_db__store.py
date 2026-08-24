"""File-backed store (JSON) for graph runs, checkpoints, HITL, tickets, agents, RAG."""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

_LOCK = threading.RLock()
_DATA_DIR = os.environ.get("IRONBRIDGE_DATA", "/tmp/ironbridge_data")
_DB_FILE = os.path.join(_DATA_DIR, "store.json")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uid(prefix: str = "X") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _empty() -> dict:
    return {
        "graph_runs": {},
        "checkpoints": {},
        "hitl_tasks": {},
        "failure_tickets": {},
        "agent_registry": {},
        "rag_documents": {},
        "workers": {},
        "equipment": {},
        "certifications": {},
        "attendance": {},
        "tasks": {},
        "tools_inv": {},
        "requests": {},
        "notifications": {},
    }


def _load() -> dict:
    os.makedirs(_DATA_DIR, exist_ok=True)
    if not os.path.exists(_DB_FILE):
        return _empty()
    with open(_DB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(db: dict) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    tmp = _DB_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)
    os.replace(tmp, _DB_FILE)


def init_db(db_path: str = "") -> None:
    with _LOCK:
        db = _load()
        # Tool names must match multi_agent.tool_registry.ALL_TOOLS so admin
        # toggles and require_tool() checks agree.
        agents = {
            "memory_rag": {
                "agent_id": "memory_rag",
                "name": "Memory & RAG Assistant",
                "description": "Site Q&A, safety manuals, cert recall",
                "enabled": 1,
                "tools": [
                    "ask_safety_policy",
                    "check_worker_certification",
                    "get_equipment_status",
                    "list_equipment",
                ],
            },
            "planning": {
                "agent_id": "planning",
                "name": "Planning / Approval Agent",
                "description": "Multi-step equipment approval",
                "enabled": 1,
                "tools": [
                    "check_worker_certification",
                    "get_equipment_status",
                    "request_equipment",
                    "ask_safety_policy",
                ],
            },
            "cert_coordination": {
                "agent_id": "cert_coordination",
                "name": "Cert Coordination Graph",
                "description": "Multi-visit cert renewal over days",
                "enabled": 1,
                "tools": ["check_worker_certification", "ask_safety_policy"],
            },
            "high_risk_dig": {
                "agent_id": "high_risk_dig",
                "name": "High-Risk Dig Approval",
                "description": "Trench dig with supervisor HITL",
                "enabled": 1,
                "tools": [
                    "check_worker_certification",
                    "get_equipment_status",
                    "ask_safety_policy",
                    "request_equipment",
                ],
            },
            "incident_handoff": {
                "agent_id": "incident_handoff",
                "name": "Incident Investigation Handoff",
                "description": "Near-miss investigation state machine",
                "enabled": 1,
                "tools": [
                    "ask_safety_policy",
                    "check_worker_certification",
                    "get_equipment_status",
                ],
            },
        }
        for k, v in agents.items():
            # Always refresh tool lists to the canonical catalogue (fixes old seeds)
            existing = db["agent_registry"].get(k)
            if existing is None:
                db["agent_registry"][k] = v
            else:
                existing["tools"] = list(v["tools"])
                existing["enabled"] = existing.get("enabled", 1)
        for w in [
            {"id": "W1", "name": "Mohamed Badr", "role_type": "engineer", "role": "Site Engineer", "site_id": "S4", "pin": "1234", "is_admin": 0},
            {"id": "W2", "name": "Sara Nabil", "role_type": "worker", "role": "Crane Operator", "site_id": "S4", "pin": "1111", "is_admin": 0},
            {"id": "W3", "name": "Karim Adel", "role_type": "worker", "role": "Electrician", "site_id": "S4", "pin": "2222", "is_admin": 0},
            {"id": "ADMIN1", "name": "Admin One — Ops", "role_type": "admin", "role": "Administrator", "site_id": "S4", "pin": "9999", "is_admin": 1},
            {"id": "ADMIN2", "name": "Admin Two — Safety", "role_type": "admin", "role": "Administrator", "site_id": "S4", "pin": "8888", "is_admin": 1},
        ]:
            db["workers"].setdefault(w["id"], w)
        # daily tasks per worker
        tasks_seed = [
            {"id": "TK1", "worker_id": "W2", "site_id": "S4", "title": "Inspect CAT 320 hydraulic lines", "priority": "High", "status": "Pending"},
            {"id": "TK2", "worker_id": "W2", "site_id": "S4", "title": "Log steel bar deliveries at Gate 2", "priority": "Normal", "status": "Pending"},
            {"id": "TK3", "worker_id": "W1", "site_id": "S4", "title": "Review trench plan Sector B", "priority": "High", "status": "Pending"},
            {"id": "TK4", "worker_id": "W3", "site_id": "S4", "title": "Fix temporary lighting Bay 3", "priority": "Normal", "status": "Pending"},
        ]
        db.setdefault("tasks", {})
        for tk in tasks_seed:
            db["tasks"].setdefault(tk["id"], tk)
        db.setdefault("tools_inv", {})
        for tool in [
            {"id": "T1", "name": "Impact Driver", "site_id": "S4", "qty": 12},
            {"id": "T2", "name": "Total Station", "site_id": "S4", "qty": 2},
            {"id": "T3", "name": "Welding Set", "site_id": "S4", "qty": 4},
        ]:
            db["tools_inv"].setdefault(tool["id"], tool)
        db.setdefault("requests", {})
        db.setdefault("notifications", {})
        for eq in [
            {"id": "EQ-9902", "name": "CAT 320 Excavator", "site_id": "S4", "status": "operational"},
            {"id": "CR-104", "name": "Liebherr Tower Crane", "site_id": "S4", "status": "maintenance"},
        ]:
            db["equipment"].setdefault(eq["id"], eq)
        db["certifications"].setdefault(
            "W2|CRANE",
            {"worker_id": "W2", "equipment_type": "CRANE", "valid_until": "2025-01-10", "status": "expired"},
        )
        db["certifications"].setdefault(
            "W1|EXCAVATOR",
            {"worker_id": "W1", "equipment_type": "EXCAVATOR", "valid_until": "2028-06-01", "status": "valid"},
        )
        docs = [
            {
                "doc_id": "DOC1",
                "title": "OSHA Trenching 4.2b",
                "topic": "trenching",
                "content": "Section 4.2b: trenches deeper than 1.5m require protective systems (sloping, shoring, or shielding). Soil type C requires 1.5:1 slope. Supervisor sign-off required before entry.",
                "added_at": _now(),
                "active": 1,
            },
            {
                "doc_id": "DOC2",
                "title": "Crane Operator Certification Policy",
                "topic": "certification",
                "content": "Crane operators must hold a valid certification. Expired certs block all crane operations. Renewal requires classroom + practical exam.",
                "added_at": _now(),
                "active": 1,
            },
            {
                "doc_id": "DOC3",
                "title": "Near-Miss Reporting",
                "topic": "incident",
                "content": "Near-miss events must be logged within 2 hours. Investigation assignment is mandatory for crane or trench incidents. Final report requires safety officer approval.",
                "added_at": _now(),
                "active": 1,
            },
        ]
        for d in docs:
            db["rag_documents"].setdefault(d["doc_id"], d)
        _save(db)


def create_run(graph_name: str, worker_id: Optional[str], input_data: dict) -> str:
    with _LOCK:
        db = _load()
        run_id = _uid("run")
        db["graph_runs"][run_id] = {
            "run_id": run_id,
            "graph_name": graph_name,
            "worker_id": worker_id,
            "status": "running",
            "created_at": _now(),
            "updated_at": _now(),
            "input_json": json.dumps(input_data),
            "result_json": None,
        }
        _save(db)
        return run_id


def update_run_status(run_id: str, status: str, result: Optional[dict] = None) -> None:
    with _LOCK:
        db = _load()
        r = db["graph_runs"].get(run_id)
        if not r:
            return
        r["status"] = status
        r["updated_at"] = _now()
        if result is not None:
            r["result_json"] = json.dumps(result)
        _save(db)


def get_run(run_id: str) -> Optional[dict]:
    with _LOCK:
        return _load()["graph_runs"].get(run_id)


def list_runs(limit: int = 50) -> list[dict]:
    with _LOCK:
        runs = list(_load()["graph_runs"].values())
    runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return runs[:limit]


def save_checkpoint(run_id: str, node_id: str, step_index: int, state: dict) -> str:
    with _LOCK:
        db = _load()
        cid = _uid("cp")
        db["checkpoints"][cid] = {
            "checkpoint_id": cid,
            "run_id": run_id,
            "node_id": node_id,
            "step_index": step_index,
            "state_json": json.dumps(state),
            "created_at": _now(),
        }
        _save(db)
        return cid


def latest_checkpoint(run_id: str) -> Optional[dict]:
    with _LOCK:
        cps = [c for c in _load()["checkpoints"].values() if c["run_id"] == run_id]
    if not cps:
        return None
    cps.sort(key=lambda c: c["step_index"])
    d = dict(cps[-1])
    d["state"] = json.loads(d["state_json"])
    return d


def get_checkpoint(checkpoint_id: str) -> Optional[dict]:
    with _LOCK:
        d = _load()["checkpoints"].get(checkpoint_id)
    if not d:
        return None
    out = dict(d)
    out["state"] = json.loads(d["state_json"])
    return out


def open_hitl(run_id: str, node_id: str, title: str, reason: str, payload: dict) -> str:
    with _LOCK:
        db = _load()
        tid = _uid("hitl")
        db["hitl_tasks"][tid] = {
            "task_id": tid,
            "run_id": run_id,
            "node_id": node_id,
            "title": title,
            "reason": reason,
            "payload_json": json.dumps(payload),
            "status": "open",
            "admin_decision": None,
            "admin_note": None,
            "created_at": _now(),
            "resolved_at": None,
        }
        if run_id in db["graph_runs"]:
            db["graph_runs"][run_id]["status"] = "paused_hitl"
            db["graph_runs"][run_id]["updated_at"] = _now()
        _save(db)
        return tid


def resolve_hitl(task_id: str, decision: str, note: str = "") -> Optional[dict]:
    with _LOCK:
        db = _load()
        t = db["hitl_tasks"].get(task_id)
        if not t or t["status"] != "open":
            return None
        t["status"] = decision
        t["admin_decision"] = decision
        t["admin_note"] = note
        t["resolved_at"] = _now()
        _save(db)
        return dict(t)


def list_hitl(status: Optional[str] = "open") -> list[dict]:
    with _LOCK:
        items = list(_load()["hitl_tasks"].values())
    if status:
        items = [i for i in items if i["status"] == status]
    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return items


def get_hitl(task_id: str) -> Optional[dict]:
    with _LOCK:
        return _load()["hitl_tasks"].get(task_id)


def open_ticket(
    run_id: str, node_id: str, error_type: str, error_message: str, checkpoint_id: Optional[str]
) -> str:
    with _LOCK:
        db = _load()
        tid = _uid("tck")
        db["failure_tickets"][tid] = {
            "ticket_id": tid,
            "run_id": run_id,
            "node_id": node_id,
            "error_type": error_type,
            "error_message": error_message,
            "checkpoint_id": checkpoint_id,
            "status": "open",
            "resolution_note": None,
            "created_at": _now(),
            "resolved_at": None,
        }
        if run_id in db["graph_runs"]:
            db["graph_runs"][run_id]["status"] = "failed"
            db["graph_runs"][run_id]["updated_at"] = _now()
        _save(db)
        return tid


def resolve_ticket(ticket_id: str, note: str = "") -> Optional[dict]:
    with _LOCK:
        db = _load()
        t = db["failure_tickets"].get(ticket_id)
        if not t:
            return None
        t["status"] = "resolved"
        t["resolution_note"] = note
        t["resolved_at"] = _now()
        _save(db)
        return dict(t)


def list_tickets(status: Optional[str] = "open") -> list[dict]:
    with _LOCK:
        items = list(_load()["failure_tickets"].values())
    if status:
        items = [i for i in items if i["status"] == status]
    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return items


def list_agents() -> list[dict]:
    with _LOCK:
        return list(_load()["agent_registry"].values())


def agent_tools(agent_id: str) -> list[str]:
    """Current tool allow-list for an agent, as set by the admin panel."""
    with _LOCK:
        a = _load()["agent_registry"].get(agent_id)
        return list(a.get("tools", [])) if a else []


def agent_has_tool(agent_id: str, tool: str) -> bool:
    """Whether `tool` is currently enabled for `agent_id`. Used by graph nodes and
    the chat agent to actually gate behavior on the admin's tool list, not just
    display it."""
    return tool in agent_tools(agent_id)


def set_agent_tools(agent_id: str, tools: list[str]) -> bool:
    """Persist tool allow-list. Creates the agent row if missing so the admin
    panel can register tools without a separate bootstrap."""
    with _LOCK:
        db = _load()
        a = db["agent_registry"].get(agent_id)
        if not a:
            db["agent_registry"][agent_id] = {
                "agent_id": agent_id,
                "name": agent_id,
                "enabled": 1,
                "tools": list(tools),
            }
        else:
            a["tools"] = list(tools)
        _save(db)
        return True


def set_agent_enabled(agent_id: str, enabled: bool) -> bool:
    with _LOCK:
        db = _load()
        a = db["agent_registry"].get(agent_id)
        if not a:
            return False
        a["enabled"] = 1 if enabled else 0
        _save(db)
        return True


def list_rag_docs(active_only: bool = True) -> list[dict]:
    with _LOCK:
        docs = list(_load()["rag_documents"].values())
    if active_only:
        docs = [d for d in docs if d.get("active")]
    docs.sort(key=lambda d: d.get("added_at", ""), reverse=True)
    return docs


def add_rag_doc(title: str, topic: str, content: str) -> str:
    with _LOCK:
        db = _load()
        doc_id = _uid("doc")
        db["rag_documents"][doc_id] = {
            "doc_id": doc_id,
            "title": title,
            "topic": topic,
            "content": content,
            "added_at": _now(),
            "active": 1,
        }
        _save(db)
        return doc_id


def remove_rag_doc(doc_id: str) -> bool:
    with _LOCK:
        db = _load()
        d = db["rag_documents"].get(doc_id)
        if not d:
            return False
        d["active"] = 0
        _save(db)
        return True


def search_rag(query: str, top_k: int = 3) -> list[dict]:
    docs = list_rag_docs(active_only=True)
    q = query.lower().split()
    scored = []
    for d in docs:
        text = (d["title"] + " " + d["content"] + " " + (d.get("topic") or "")).lower()
        score = sum(1 for w in q if w in text)
        if score:
            scored.append((score, d))
    scored.sort(key=lambda t: -t[0])
    return [d for _, d in scored[:top_k]]


def get_cert(worker_id: str, equipment_type: str) -> Optional[dict]:
    with _LOCK:
        return _load()["certifications"].get(f"{worker_id}|{equipment_type}")


def get_equipment(eq_id: str) -> Optional[dict]:
    with _LOCK:
        return _load()["equipment"].get(eq_id)


def get_worker(worker_id: str) -> Optional[dict]:
    with _LOCK:
        return _load()["workers"].get(worker_id)


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _now_time() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def get_attendance(worker_id: str, day: Optional[str] = None) -> Optional[dict]:
    day = day or _today()
    key = f"{worker_id}|{day}"
    with _LOCK:
        return _load().get("attendance", {}).get(key)


def list_attendance(limit: int = 100) -> list[dict]:
    with _LOCK:
        items = list(_load().get("attendance", {}).values())
    items.sort(key=lambda x: x.get("day", "") + x.get("worker_id", ""), reverse=True)
    return items[:limit]


def clock_in(worker_id: str) -> dict:
    """Request check-in: starts as pending_admin (red light) until admin approves (green)."""
    day = _today()
    key = f"{worker_id}|{day}"
    with _LOCK:
        db = _load()
        att = db.setdefault("attendance", {})
        row = att.get(key)
        if row and row.get("admin_status") == "approved" and not row.get("check_out"):
            return dict(row)
        if row and row.get("check_out"):
            return dict(row)
        # new or re-request while pending
        row = {
            "worker_id": worker_id,
            "day": day,
            "check_in": _now_time(),
            "check_out": None,
            "status": "pending_admin",
            "admin_status": "pending",  # pending | approved | rejected
            "light": "red",
            "ai_message": "تم تسجيل طلب الحضور. الأدمن هيراجع طلبك — النور أحمر لحد الموافقة.",
        }
        att[key] = row
        _save(db)
        return dict(row)


def clock_out(worker_id: str) -> dict:
    day = _today()
    key = f"{worker_id}|{day}"
    with _LOCK:
        db = _load()
        att = db.setdefault("attendance", {})
        row = att.get(key)
        if not row or not row.get("check_in"):
            return {"error": "not_checked_in", "worker_id": worker_id, "day": day}
        if row.get("check_out"):
            return dict(row)
        row["check_out"] = _now_time()
        row["status"] = "completed"
        att[key] = row
        _save(db)
        return dict(row)


def attendance_ai_check(worker_id: str) -> dict:
    """Attendance status + light color for UI (red pending / green approved)."""
    day = _today()
    row = get_attendance(worker_id, day)
    worker = get_worker(worker_id) or {"name": worker_id}
    issues = []
    if not row or not row.get("check_in"):
        status = "missing_check_in"
        light = "off"
        ai_message = "لسه مسجلتش حضور النهارده. اضغط Clock In — الطلب هيروح للأدمن."
        issues.append("No check-in today.")
    elif row.get("admin_status") == "pending":
        status = "pending_admin"
        light = "red"
        ai_message = row.get("ai_message") or "طلبك عند الأدمن. النور أحمر لحد ما يوافق."
        issues.append("Waiting for admin approval.")
    elif row.get("admin_status") == "rejected":
        status = "rejected"
        light = "red"
        ai_message = row.get("ai_message") or "الأدمن رفض الحضور."
        issues.append("Rejected by admin.")
    elif row.get("check_out"):
        status = "completed"
        light = "green"
        ai_message = f"الوردية خلصت: دخول {row.get('check_in')} · خروج {row.get('check_out')}"
        issues.append("Shift complete.")
    else:
        status = "on_site"
        light = "green"
        ai_message = row.get("ai_message") or "الأدمن وافق — النور أخضر. أنت مسجل حضور."
        issues.append(f"On site since {row.get('check_in')}.")

    return {
        "worker_id": worker_id,
        "worker_name": worker.get("name"),
        "day": day,
        "record": row,
        "status": status,
        "light": light,
        "ai_message": ai_message,
        "issues": issues,
    }


def approve_attendance(worker_id: str, day: Optional[str] = None, note: str = "") -> Optional[dict]:
    day = day or _today()
    key = f"{worker_id}|{day}"
    with _LOCK:
        db = _load()
        row = db.get("attendance", {}).get(key)
        if not row:
            return None
        row["admin_status"] = "approved"
        row["status"] = "on_site"
        row["light"] = "green"
        row["admin_note"] = note or "approved"
        row["ai_message"] = "الأدمن وافق على حضورك — النور أخضر. يومك يبدأ رسميًا."
        db["attendance"][key] = row
        _save(db)
        return dict(row)


def reject_attendance(worker_id: str, day: Optional[str] = None, note: str = "") -> Optional[dict]:
    day = day or _today()
    key = f"{worker_id}|{day}"
    with _LOCK:
        db = _load()
        row = db.get("attendance", {}).get(key)
        if not row:
            return None
        row["admin_status"] = "rejected"
        row["status"] = "rejected"
        row["light"] = "red"
        row["admin_note"] = note or "rejected"
        row["ai_message"] = "الأدمن رفض طلب الحضور. راجع المكتب."
        db["attendance"][key] = row
        _save(db)
        return dict(row)


def list_pending_attendance() -> list[dict]:
    with _LOCK:
        items = list(_load().get("attendance", {}).values())
    return [i for i in items if i.get("admin_status") == "pending"]


def list_tasks_for(worker_id: str) -> list[dict]:
    with _LOCK:
        tasks = list(_load().get("tasks", {}).values())
    return [x for x in tasks if x.get("worker_id") == worker_id]


def list_tools() -> list[dict]:
    with _LOCK:
        return list(_load().get("tools_inv", {}).values())


def list_equipment_all() -> list[dict]:
    with _LOCK:
        return list(_load().get("equipment", {}).values())


def list_workers_by_role(role_type: Optional[str] = None) -> list[dict]:
    with _LOCK:
        ws = list(_load().get("workers", {}).values())
    if role_type:
        ws = [w for w in ws if w.get("role_type") == role_type]
    return ws


def add_notification(user_id: str, title: str, body: str, kind: str = "info", ref_id: str = "") -> str:
    with _LOCK:
        db = _load()
        nid = _uid("ntf")
        db.setdefault("notifications", {})[nid] = {
            "id": nid,
            "user_id": user_id,
            "title": title,
            "body": body,
            "kind": kind,
            "ref_id": ref_id,
            "read": 0,
            "created_at": _now(),
        }
        _save(db)
        return nid


def list_notifications(user_id: str, unread_only: bool = False) -> list[dict]:
    with _LOCK:
        items = [n for n in _load().get("notifications", {}).values() if n.get("user_id") == user_id]
    if unread_only:
        items = [n for n in items if not n.get("read")]
    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return items


def mark_notification_read(nid: str) -> bool:
    with _LOCK:
        db = _load()
        n = db.get("notifications", {}).get(nid)
        if not n:
            return False
        n["read"] = 1
        _save(db)
        return True


def create_request(
    requester_id: str,
    req_type: str,
    title: str,
    details: dict,
) -> dict:
    """Engineer request — needs BOTH ADMIN1 and ADMIN2 to approve."""
    with _LOCK:
        db = _load()
        rid = _uid("req")
        row = {
            "id": rid,
            "requester_id": requester_id,
            "req_type": req_type,  # workers | tools | equipment
            "title": title,
            "details": details,
            "status": "pending",  # pending | approved | rejected
            "approvals": {"ADMIN1": None, "ADMIN2": None},  # None | approved | rejected
            "created_at": _now(),
            "resolved_at": None,
        }
        db.setdefault("requests", {})[rid] = row
        _save(db)
    # notify both admins
    for aid in ("ADMIN1", "ADMIN2"):
        add_notification(
            aid,
            f"طلب مهندس: {title}",
            f"يحتاج موافقة من الأدمنين. النوع: {req_type}. معرف الطلب: {rid}",
            kind="request",
            ref_id=rid,
        )
    add_notification(
        requester_id,
        "تم إرسال طلبك",
        f"طلب «{title}» بانتظار موافقة الأدمنين (الاتنين).",
        kind="info",
        ref_id=rid,
    )
    return row


def get_request(rid: str) -> Optional[dict]:
    with _LOCK:
        return _load().get("requests", {}).get(rid)


def list_requests(status: Optional[str] = None) -> list[dict]:
    with _LOCK:
        items = list(_load().get("requests", {}).values())
    if status:
        items = [i for i in items if i.get("status") == status]
    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return items


def vote_request(rid: str, admin_id: str, decision: str, note: str = "") -> Optional[dict]:
    """One admin votes. Final status = approved only if BOTH approved; rejected if ANY rejected."""
    if admin_id not in ("ADMIN1", "ADMIN2"):
        return None
    if decision not in ("approved", "rejected"):
        return None
    with _LOCK:
        db = _load()
        row = db.get("requests", {}).get(rid)
        if not row or row.get("status") != "pending":
            return dict(row) if row else None
        row["approvals"][admin_id] = decision
        row.setdefault("notes", {})[admin_id] = note
        a1 = row["approvals"].get("ADMIN1")
        a2 = row["approvals"].get("ADMIN2")
        if a1 == "rejected" or a2 == "rejected":
            row["status"] = "rejected"
            row["resolved_at"] = _now()
        elif a1 == "approved" and a2 == "approved":
            row["status"] = "approved"
            row["resolved_at"] = _now()
        # else still pending
        db["requests"][rid] = row
        _save(db)
        final = dict(row)
    # notifications outside lock
    if final["status"] == "approved":
        add_notification(
            final["requester_id"],
            "تم قبول طلبك",
            f"«{final['title']}» وافق عليه الأدمنين.",
            kind="success",
            ref_id=rid,
        )
        for aid in ("ADMIN1", "ADMIN2"):
            add_notification(aid, "طلب مُعتمد", f"{final['title']} — موافقة مزدوجة مكتملة.", kind="info", ref_id=rid)
    elif final["status"] == "rejected":
        add_notification(
            final["requester_id"],
            "تم رفض طلبك",
            f"«{final['title']}» رُفض (يكفي رفض أدمن واحد).",
            kind="danger",
            ref_id=rid,
        )
    else:
        # still waiting other admin
        other = "ADMIN2" if admin_id == "ADMIN1" else "ADMIN1"
        add_notification(
            other,
            "في انتظار صوتك",
            f"الأدمن الآخر صوّت على «{final['title']}». صوتك مطلوب.",
            kind="request",
            ref_id=rid,
        )
    return final


def mcp_context_for_worker(worker_id: str) -> str:
    """Hidden context for Gemini — like MCP server scoped data (not shown raw to user)."""
    w = get_worker(worker_id) or {}
    tasks = list_tasks_for(worker_id)
    att = attendance_ai_check(worker_id)
    tools = list_tools()
    equipment = list_equipment_all()
    certs = []
    for eq in ("CRANE", "EXCAVATOR"):
        c = get_cert(worker_id, eq)
        if c:
            certs.append(c)
    lines = [
        f"WORKER: id={w.get('id')} name={w.get('name')} role_type={w.get('role_type')} role={w.get('role')} site={w.get('site_id')}",
        f"ATTENDANCE: status={att.get('status')} light={att.get('light')} msg={att.get('ai_message')}",
        "TODAY_TASKS: " + ("; ".join(f"{t['title']} [{t['status']}/{t['priority']}]" for t in tasks) or "none"),
        "CERTS: " + ("; ".join(f"{c['equipment_type']}={c['status']}" for c in certs) or "none"),
        "SITE_TOOLS: " + "; ".join(f"{t['name']} x{t['qty']}" for t in tools),
        "SITE_EQUIPMENT: " + "; ".join(f"{e['name']} ({e['id']})={e['status']}" for e in equipment),
    ]
    return "\n".join(lines)
