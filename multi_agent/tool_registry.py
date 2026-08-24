"""
Shared tool registry used by the platform admin UI and the agent router.

When an admin enables/disables tools for an agent via the platform, the change
is persisted in platform_db *and* reflected here so that:
  - multi_agent.router only offers/uses the enabled tool set
  - mcp_bridge can filter what context/tools it surfaces
  - a live MCP session can be told to refresh (tools/list_changed) when possible

This closes the gap where set_agent_tools only wrote JSON and never affected
runtime behaviour.
"""
from __future__ import annotations

from typing import Optional

from platform_db import store

# Canonical tool catalogue (names match mcp_server tool names where applicable)
ALL_TOOLS: dict[str, str] = {
    "check_worker_certification": "Look up whether a worker is certified for an equipment type",
    "get_equipment_status": "Current status and location of a piece of equipment",
    "ask_safety_policy": "RAG over internal safety manuals (agentic by default)",
    "request_equipment": "Submit an equipment request (may require supervisor)",
    "authenticate_supervisor": "Elevate session role; unlocks approval tool",
    "approve_equipment_request": "Approve/reject a pending request (supervisor only)",
    "generate_site_compliance_report": "Long-running site compliance report with progress",
    "list_equipment": "List equipment units (read-only)",
    "list_workers": "List workers (read-only)",
}

# Default tool sets per agent (used on first boot if registry is empty)
DEFAULT_AGENT_TOOLS: dict[str, list[str]] = {
    "memory_rag": [
        "ask_safety_policy",
        "check_worker_certification",
        "get_equipment_status",
        "list_equipment",
    ],
    "planning": [
        "check_worker_certification",
        "get_equipment_status",
        "request_equipment",
        "ask_safety_policy",
    ],
    "cert_coordination": [
        "check_worker_certification",
        "ask_safety_policy",
    ],
    "high_risk_dig": [
        "check_worker_certification",
        "get_equipment_status",
        "ask_safety_policy",
        "request_equipment",
    ],
    "incident_handoff": [
        "ask_safety_policy",
        "check_worker_certification",
        "get_equipment_status",
    ],
}


def get_enabled_tools(agent_id: str) -> list[str]:
    """Return the currently enabled tool names for an agent."""
    agents = store.list_agents()
    for a in agents:
        if a.get("agent_id") == agent_id:
            tools = a.get("tools") or []
            return [t for t in tools if t in ALL_TOOLS]
    return list(DEFAULT_AGENT_TOOLS.get(agent_id, []))


def set_enabled_tools(agent_id: str, tools: list[str]) -> bool:
    """
    Persist tool allow-list for an agent and return True on success.
    Only names present in ALL_TOOLS are kept (defensive filter).
    """
    cleaned = [t for t in tools if t in ALL_TOOLS]
    ok = store.set_agent_tools(agent_id, cleaned)
    return ok


def tool_allowed(agent_id: str, tool_name: str) -> bool:
    return tool_name in get_enabled_tools(agent_id)


def describe_tools(agent_id: str) -> list[dict]:
    enabled = set(get_enabled_tools(agent_id))
    return [
        {
            "name": name,
            "description": desc,
            "enabled": name in enabled,
        }
        for name, desc in ALL_TOOLS.items()
    ]


def ensure_agent_defaults() -> None:
    """Seed agent_registry entries if missing (idempotent)."""
    existing = {a["agent_id"] for a in store.list_agents()}
    for agent_id, tools in DEFAULT_AGENT_TOOLS.items():
        if agent_id not in existing:
            store.set_agent_tools(agent_id, tools)
            # also ensure a basic registry row exists via set_agent_enabled
            store.set_agent_enabled(agent_id, True)
