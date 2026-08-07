# Iron Bridge Construction — MCP Equipment Safety Server

## Company & Problem

**Iron Bridge Construction** manages heavy equipment (cranes, excavators, scaffolding) across multiple job sites.  
Before this system, site workers used paper checklists. Nothing stopped an uncertified operator from taking a crane, and nothing forced a human supervisor sign-off when the work was near power lines.

**The fix:** an MCP server that gives an LLM *scoped, safe* access to equipment data. The model never talks to the database directly. Every write that carries real risk is gated by capability checks, role changes, elicitation, and sampling.

*(Note: This repository serves as the baseline for the upcoming Memory & RAG integration to handle long-term context and unstructured safety manuals).*

## Protocol Concerns Mapping

| # | Concern | How it appears in this system |
|---|---------|-------------------------------|
| 1 | Capability negotiation | Server checks `ctx.session.check_client_capability(...)` for elicitation+sampling *before* attempting either. |
| 2 | Notifications | Worker starts with read-only tools. `approve_equipment_request` appears dynamically after supervisor auth. |
| 3 | Elicitation | High-risk items pause mid-call to ask a human supervisor for explicit confirmation. |
| 4 | Resources | Safety policies are exposed as resources, not tools. |
| 5 | Prompts | Reusable template `prepare_equipment_receipt`. |
| 6 | Sampling | Server asks the client's model to draft a risk summary before approval. |
| 7 | Progress tracking | `generate_site_compliance_report` streams intermediate progress (25/50/75/100%). |
| 8 | Defensive tool design | Strict JSON Schema, server-side validation, parameterized SQL, and handler-level authorization. |

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # Set SUPERVISOR_PIN (e.g., 4821)
python db/load_data.py      # Initialize the database
python agent/client.py      # Run the automated demo