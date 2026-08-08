# Iron Bridge Construction — MCP Equipment Safety Server

## Company & Problem

**Iron Bridge Construction** manages heavy equipment (cranes, excavators, scaffolding) across multiple job sites.  
Before this system, site workers used paper checklists. Nothing stopped an uncertified operator from taking a crane, and nothing forced a human supervisor sign-off when the work was near power lines.

**The Fix (Phase 1 - MCP Server):** An MCP server that gives an LLM *scoped, safe* access to equipment data. The model never talks to the database directly. Every write that carries real risk is gated by capability checks, role changes, elicitation, and sampling.

**The Memory & Knowledge Gap (Phase 2):**  
Once real usage started, two new problems emerged:
1. **Memory Amnesia:** The agent forgets everything when a session ends. If Sara (Worker 2) is denied a crane today because her certification expired, the agent forgets this by tomorrow and has to re-query and re-discover the same issue, wasting tokens and time.
2. **Ungoverned Knowledge:** The safety policies (OSHA standards and IronBridge internal manuals) are too large (dozens of pages) to expose as static MCP resources. Supervisors constantly ask complex, multi-hop safety questions that require grounded retrieval, not just raw database tool calls.

---

## Part 1: Long-Term Memory Architecture

To solve the amnesia problem, we implemented a complete memory pipeline that visibly extends the existing SQLite database (`ironbridge.db`):

*   **Short-Term Memory & Scratchpad:** A rolling message buffer distinct from the agent's scratchpad. When the transcript is cleared or pruned, the active plan and sub-goals in the scratchpad remain intact.
*   **Promote-or-Drop Router:** When the short-term buffer overflows, the oldest item is evaluated. Routine chatter is sent to `forget`, while safety-relevant events (e.g., expired certs, rejected high-risk requests) are routed to `episodic` memory. **This router never writes to semantic memory.**
*   **Semantic Consolidation Layer:** A separate, periodic pass over episodic memory. It extracts general facts (e.g., `cert_status:CRANE`). It explicitly handles **conflicts and versioning** (e.g., if an episodic memory says a cert is invalid, and a newer one says it was renewed, the old fact is closed with `valid_until` and `superseded_by`, never silently overwritten) and handles natural expirations.

---

## Part 2: Context Window Management

Larkspur's triage calls (or in our case, site compliance checks) generate massive JSON payloads from tool calls (like `generate_site_compliance_report`), burying early critical decisions (like a worker's expired certification) under noise.

We tested all four context management strategies against a deterministic test suite containing 5 variations of heavy tool-noise transcripts to see if the critical certification finding survived.

### Evaluation Table

| Strategy                  | Recall | Avg Input Tokens | Avg Output Tokens | Avg Latency (ms) |
|---------------------------|--------|------------------|-------------------|------------------|
| `sliding_window`          | 0/5    | 917              | 0                 | 0.0              |
| `observation_masking`     | 5/5    | 679              | 0                 | 0.0              |
| `recursive_summarization` | 4/5    | 897              | 111               | 0.1              |
| `zone_based_pruning`      | 5/5    | 2407             | 0                 | 0.1              |

### Chosen Strategy: Observation Masking
**Justification:** We ship **Observation Masking**. It matches Iron Bridge's exact failure mode (the context bloat is JSON tool output, not conversational dialogue). It achieved a perfect 5/5 recall of the buried critical fact at the lowest token cost (679 tokens) and zero latency. Zone-based pruning also achieved 5/5 but consumed nearly 4x the input tokens for no additional benefit. Recursive summarization lost one detail and cost unnecessary output tokens.

---

## Part 3: Retrieval & RAG (In Progress)



---

## Quick Start & Demo Instructions

```bash
# 1. Environment Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # Set SUPERVISOR_PIN and GOOGLE_API_KEY

# 2. Database Initialization
python db/load_data.py      # Initializes tables and seeds baseline data

# 3. Run Memory Demonstrations
python -m memory.demo_memory       # Demonstrates routing, consolidation, and conflict resolution
python -m context_eval.run_eval    # Generates the Context Management comparison table

# 4. Run the Live Memory-Aware Agent
python -m agent.agent_with_memory  # Agent recalls semantic facts before taking live MCP actions