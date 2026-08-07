"""
Iron Bridge Construction — MCP Server
Implements all 8 protocol concerns.
"""

from __future__ import annotations
import os
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from mcp.server.fastmcp import FastMCP, Context
from mcp.types import (
    TextContent,
    PromptMessage,
    GetPromptResult,
    SamplingMessage,
    ClientCapabilities,
    ElicitationCapability,
    SamplingCapability,
)

from . import service
from .schemas import (
    CheckWorkerCertInput,
    GetEquipmentStatusInput,
    RequestEquipmentInput,
    AuthenticateSupervisorInput,
    ApproveEquipmentRequestInput,
    GenerateComplianceReportInput,
)

EquipmentType = Literal["CRANE", "EXCAVATOR", "SCAFFOLD", "GENERATOR"]

# =====================================================================
# 3. ELICITATION — response schema
# =====================================================================
# ctx.elicit() requires a Pydantic model class (not a raw JSON-schema dict) —
# the MCP SDK builds the JSON Schema sent to the client from this model and
# validates the client's response against it.
class SupervisorConfirmation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supervisor_confirms: bool = Field(
        ..., description="True only if the supervisor explicitly approves the release"
    )
    safety_notes: str | None = Field(
        default=None, max_length=500, description="Optional safety notes from the supervisor"
    )

mcp = FastMCP(
    "ironbridge-equipment-safety",
    instructions=(
        "Scoped access to Iron Bridge Construction equipment data. "
        "High-risk requests require elicitation + sampling."
    ),
)

# ---------- Session state (demo-local; production must be per-connection) ----------
_session_role: str = "worker"          # worker | supervisor
_supervisor_authenticated: bool = False


# =====================================================================
# 1. CAPABILITY NEGOTIATION  (declared by FastMCP + explicit checks)
# =====================================================================
# FastMCP advertises tools/resources/prompts. We additionally gate
# high-risk paths on client capabilities inside the handlers.


# =====================================================================
# 4. RESOURCES
# =====================================================================
@mcp.resource("construction://policies/lifting-safety")
def lifting_safety_policy() -> str:
    return (
        "# Lifting Safety Policy — Iron Bridge Construction\n\n"
        "1. Only certified operators may request crane equipment.\n"
        "2. Any lift within 15 m of energized power lines requires "
        "explicit supervisor sign-off via elicitation.\n"
        "3. Daily pre-use inspection is mandatory.\n"
        "4. Never exceed rated capacity.\n"
    )

@mcp.resource("construction://policies/electrical-proximity")
def electrical_proximity_policy() -> str:
    return (
        "# Electrical Proximity Policy\n\n"
        "Sites flagged near_power_lines=1 treat every equipment request "
        "as high-risk regardless of equipment type.\n"
        "Minimum clearance: 10 m for cranes, 5 m for excavators.\n"
    )


# =====================================================================
# 5. PROMPTS
# =====================================================================
@mcp.prompt()
def prepare_equipment_receipt(request_id: int) -> GetPromptResult:
    """Reusable template for drafting a receipt explanation."""
    return GetPromptResult(
        description="Draft a clear equipment-receipt note for a given request",
        messages=[
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=(
                        f"Draft a short professional note confirming equipment "
                        f"receipt for request_id={request_id}. Include safety "
                        f"reminders appropriate for the equipment type."
                    ),
                ),
            )
        ],
    )


# =====================================================================
# READ TOOLS (always available)
# =====================================================================
@mcp.tool(
    name="check_worker_certification",
    description="Check whether a worker has a valid certification for an equipment type.",
)
def check_worker_certification(
    worker_id: Annotated[int, Field(ge=1, description="Worker primary key")],
    equipment_type: Annotated[EquipmentType, Field(description="Equipment type to check certification for")],
) -> dict[str, Any]:
    # Defensive validation
    data = CheckWorkerCertInput(worker_id=worker_id, equipment_type=equipment_type)
    worker = service.get_worker(data.worker_id)
    if not worker:
        return {"error": "Worker not found"}
    result = service.check_certification(data.worker_id, data.equipment_type)
    return {"worker": worker["name"], "equipment_type": data.equipment_type, **result}


@mcp.tool(
    name="get_equipment_status",
    description="Return current status and risk flags for a piece of equipment.",
)
def get_equipment_status(
    equipment_id: Annotated[int, Field(ge=1, description="Equipment primary key")],
) -> dict[str, Any]:
    data = GetEquipmentStatusInput(equipment_id=equipment_id)
    eq = service.get_equipment(data.equipment_id)
    if not eq:
        return {"error": "Equipment not found"}
    site = service.get_site(eq["site_id"])
    return {
        "equipment": eq,
        "site": site,
        "is_high_risk": service.is_high_risk(eq, site or {}),
    }

# =====================================================================
# 2. NOTIFICATIONS + 3. ELICITATION + 6. SAMPLING + 8. DEFENSIVE
# =====================================================================
@mcp.tool(
    name="request_equipment",
    description=(
        "Request equipment for a worker at a site. "
        "High-risk items (crane or near power lines) require human confirmation "
        "and a client-model risk summary before the request is written."
    ),
)
async def request_equipment(
    worker_id: Annotated[int, Field(ge=1, description="Requesting worker id")],
    equipment_id: Annotated[int, Field(ge=1, description="Equipment to request")],
    site_id: Annotated[int, Field(ge=1, description="Site where equipment will be used")],
    ctx: Context,
) -> dict[str, Any]:
    # --- 8. Defensive validation (independent of the schema above: re-checks
    # business rules a JSON Schema can't express, e.g. cross-record lookups) ---
    data = RequestEquipmentInput(
        worker_id=worker_id, equipment_id=equipment_id, site_id=site_id
    )

    worker = service.get_worker(data.worker_id)
    if not worker:
        return {"error": "Worker not found", "status": "REJECTED"}

    eq = service.get_equipment(data.equipment_id)
    if not eq:
        return {"error": "Equipment not found", "status": "REJECTED"}
    if eq["status"] != "AVAILABLE":
        return {"error": f"Equipment status is {eq['status']}", "status": "REJECTED"}

    site = service.get_site(data.site_id)
    if not site:
        return {"error": "Site not found", "status": "REJECTED"}

    # Authorization: must have valid cert
    cert = service.check_certification(data.worker_id, eq["type"])
    if not cert["valid"]:
        return {"error": cert["reason"], "status": "REJECTED"}

    high_risk = service.is_high_risk(eq, site)

    if high_risk:
        # --- 1. Capability negotiation ---
        # Check what the client actually declared during initialize before
        # relying on it, instead of finding out mid-call via an exception.
        needs = ClientCapabilities(
            elicitation=ElicitationCapability(), sampling=SamplingCapability()
        )
        if not ctx.session.check_client_capability(needs):
            return {
                "status": "NOT_SUBMITTED",
                "reason": (
                    "Client did not declare elicitation + sampling support at "
                    "initialize — high-risk write blocked before any attempt."
                ),
            }

        # --- 3. Elicitation ---
        try:
            elicit_result = await ctx.elicit(
                message=(
                    f"HIGH-RISK request: {eq['name']} at site '{site['name']}'. "
                    f"Near power lines={bool(site['near_power_lines'])}. "
                    "Supervisor must explicitly confirm."
                ),
                schema=SupervisorConfirmation,
            )
        except Exception:
            return {
                "status": "NOT_SUBMITTED",
                "reason": "Elicitation request failed — high-risk write blocked",
            }

        if elicit_result.action != "accept" or not elicit_result.data or not elicit_result.data.supervisor_confirms:
            return {
                "status": "NOT_SUBMITTED",
                "reason": f"Supervisor did not confirm (action={elicit_result.action})",
            }

        # --- 6. Sampling (ask the client's model for a risk summary — not the
        # server's own model; the client is the one that owns the LLM here) ---
        try:
            sample = await ctx.session.create_message(
                messages=[
                    SamplingMessage(
                        role="user",
                        content=TextContent(
                            type="text",
                            text=(
                                f"Write a 2-sentence safety handover for releasing "
                                f"{eq['name']} (type={eq['type']}) at site "
                                f"'{site['name']}' (near_power_lines="
                                f"{site['near_power_lines']}) to worker "
                                f"{worker['name']}."
                            ),
                        ),
                    )
                ],
                max_tokens=150,
            )
        except Exception:
            return {
                "status": "NOT_SUBMITTED",
                "reason": "Sampling request failed — high-risk write blocked",
            }

        risk_summary = None
        if isinstance(sample.content, TextContent):
            risk_summary = sample.content.text
        else:
            risk_summary = str(sample.content)

        rid = service.create_request(
            data.worker_id,
            data.equipment_id,
            data.site_id,
            status="PENDING_SUPERVISOR_APPROVAL",
            risk_summary=risk_summary,
        )
        return {
            "status": "PENDING_SUPERVISOR_APPROVAL",
            "request_id": rid,
            "risk_summary": risk_summary,
            "supervisor_notes": elicit_result.data.safety_notes,
            "message": "High-risk request recorded; awaiting final supervisor approval tool",
        }

    # Low-risk path — write immediately
    rid = service.create_request(
        data.worker_id, data.equipment_id, data.site_id, status="APPROVED"
    )
    return {"status": "APPROVED", "request_id": rid, "message": "Low-risk request auto-approved"}


@mcp.tool(
    name="authenticate_supervisor",
    description=(
        "Authenticate as a site supervisor. On success the server pushes "
        "tools/list_changed so the approval tool becomes available."
    ),
)
async def authenticate_supervisor(
    worker_id: Annotated[int, Field(ge=1, description="Supervisor worker id")],
    pin: Annotated[str, Field(min_length=4, max_length=8, description="Supervisor PIN")],
    ctx: Context,
) -> dict[str, Any]:
    data = AuthenticateSupervisorInput(worker_id=worker_id, pin=pin)
    if not service.verify_supervisor_pin(data.worker_id, data.pin):
        return {"authenticated": False, "error": "Invalid supervisor credentials"}

    global _session_role, _supervisor_authenticated
    _session_role = "supervisor"
    _supervisor_authenticated = True

    # --- 2. Notifications ---
    # The approval tool genuinely does not exist for this session until now:
    # it is registered here for the first time, then the client is told to
    # re-fetch tools/list. This is a real runtime change to the tool set,
    # not just a notification sent over an unchanged list.
    if mcp._tool_manager.get_tool("approve_equipment_request") is None:
        mcp.add_tool(
            approve_equipment_request,
            name="approve_equipment_request",
            description="Approve or reject a pending equipment request. Supervisor role required.",
        )
        _harden_schema("approve_equipment_request")
    await ctx.session.send_tool_list_changed()

    return {
        "authenticated": True,
        "role": "supervisor",
        "message": "tools/list_changed sent — approve_equipment_request is now registered",
    }


def approve_equipment_request(
    request_id: Annotated[int, Field(ge=1, description="Request to approve or reject")],
    decision: Annotated[Literal["APPROVED", "REJECTED"], Field(description="Final decision")],
    notes: Annotated[str | None, Field(default=None, max_length=500, description="Optional supervisor notes")] = None,
) -> dict[str, Any]:
    # Handler-level authorization: even though this tool only gets registered
    # post-auth, a stale client reference or race could still call it, so we
    # check the session flag independently rather than trusting tool presence.
    if not _supervisor_authenticated:
        return {"error": "Supervisor authentication required", "status": "DENIED"}

    data = ApproveEquipmentRequestInput(
        request_id=request_id, decision=decision, notes=notes
    )
    from . import db as db_module
    existing = db_module.fetch_one(
        "SELECT * FROM equipment_requests WHERE id = ?", (data.request_id,)
    )
    if not existing:
        return {"error": "Request not found"}
    if existing["status"] != "PENDING_SUPERVISOR_APPROVAL":
        return {"error": f"Request is already {existing['status']}"}

    service.update_request_status(data.request_id, data.decision, data.notes)
    return {"request_id": data.request_id, "status": data.decision, "notes": data.notes}


# =====================================================================
# 7. PROGRESS TRACKING
# =====================================================================
@mcp.tool(
    name="generate_site_compliance_report",
    description=(
        "Walk every request at a site, checking certifications and statuses. "
        "Reports intermediate progress (25/50/75/100 %)."
    ),
)
async def generate_site_compliance_report(
    site_id: Annotated[int, Field(ge=1, description="Site to audit")],
    ctx: Context,
) -> dict[str, Any]:
    data = GenerateComplianceReportInput(site_id=site_id)
    site = service.get_site(data.site_id)
    if not site:
        return {"error": "Site not found"}

    requests = service.list_requests_for_site(data.site_id)
    total = max(len(requests), 1)
    findings = []

    for i, req in enumerate(requests, start=1):
        cert = service.check_certification(
            req["worker_id"],
            service.get_equipment(req["equipment_id"])["type"],  # type: ignore[index]
        )
        findings.append(
            {
                "request_id": req["id"],
                "status": req["status"],
                "cert_valid": cert["valid"],
            }
        )
        # Report progress after each item
        pct = int(i / total * 100)
        await ctx.report_progress(progress=pct, total=100, message=f"Audited request {i}/{total}")

    return {
        "site": site["name"],
        "total_requests": len(requests),
        "findings": findings,
        "compliant": all(f["cert_valid"] for f in findings) if findings else True,
    }


# =====================================================================
# 8. DEFENSIVE TOOL DESIGN — enforce additionalProperties: false everywhere
# =====================================================================
# Annotated[..., Field(...)] above gives real per-field constraints (enum,
# minimum, description), but the MCP SDK doesn't set additionalProperties on
# the generated schema by default. We enforce it here so no tool silently
# accepts unexpected extra fields from a model.
def _harden_schema(tool_name: str) -> None:
    tool = mcp._tool_manager.get_tool(tool_name)
    if tool is None:
        return
    # Patch the *advertised* schema (what tools/list shows clients).
    tool.parameters["additionalProperties"] = False
    # Patch the *enforced* schema. FastMCP's own runtime argument model
    # (tool.fn_metadata.arg_model) defaults to extra="ignore" — without this,
    # an unexpected field is silently dropped before it ever reaches the
    # handler, not rejected. Verified: without this rebuild, a bogus extra
    # field passed a real tool call with isError=False.
    arg_model = tool.fn_metadata.arg_model
    arg_model.model_config["extra"] = "forbid"
    arg_model.model_rebuild(force=True)


for _name in (
    "check_worker_certification",
    "get_equipment_status",
    "request_equipment",
    "authenticate_supervisor",
    "generate_site_compliance_report",
):
    _harden_schema(_name)


# =====================================================================
# TRANSPORT ENTRYPOINT
# =====================================================================
if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio").lower()
    if transport == "streamable-http":
        # FastMCP.run() only takes `transport`/`mount_path` — host/port are
        # configured on mcp.settings beforehand, not passed to run() itself.
        mcp.settings.host = os.getenv("MCP_HOST", "127.0.0.1")
        mcp.settings.port = int(os.getenv("MCP_PORT", "8000"))
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")
