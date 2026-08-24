# Iron Bridge Construction — MCP Equipment Safety Server

## Company & Problem

**Iron Bridge Construction** manages heavy equipment (cranes, excavators, scaffolding) across multiple job sites.  
Before this system, site workers used paper checklists. Nothing stopped an uncertified operator from taking a crane, and nothing forced a human supervisor sign-off when the work was near power lines.

**The fix:** an MCP server that gives an LLM *scoped, safe* access to equipment data. The model never talks to the database directly. Every write that carries real risk is gated by capability checks, role changes, elicitation, and sampling.

*(Note: This repository serves as the baseline for the upcoming Memory & RAG integration to handle long-term context and unstructured safety manuals).*

## Part 1: Memory (Short-Term, Episodic, Semantic)

**The problem:** a live agent session on a site is chatty — status checks,
small talk, tool results — but only some of it matters *beyond* the current
conversation. Nothing in the baseline system distinguished a one-off
question ("what's the weather?") from a fact that should still be true
next week ("worker 2's crane cert is invalid"). Without that distinction,
either everything gets kept (context bloat, stale facts never resolved)
or everything gets dropped (the agent re-asks/re-derives things it was
already told).

**Short-term memory** (`memory/stm.py`): a fixed-size turn buffer
(`ShortTermMemory`) plus a scratchpad (`plan`/`subgoal`/`working_vars`)
that survives independently of the turn buffer, so a transcript clear
doesn't lose the agent's current goal.

**Promote-or-drop routing** (`memory/router.py`): fires the moment the
buffer evicts its oldest item. Each evicted item is routed to either
`forget` (small talk, redundant status checks — logged but discarded) or
`episodic` (a specific event worth recording). The router **never** writes
semantic facts directly — that's a deliberate separation from
consolidation, so "is this worth keeping at all" and "what general fact
does this imply" stay independent decisions. Every routing decision,
`forget` included, is logged to `memory_routing_log` so the reasoning is
inspectable without re-running anything.

**Consolidation** (`memory/consolidation.py`): a periodic pass over
unconsolidated episodes that extracts or updates semantic facts
(`cert_status:CRANE`, etc.), with real version history — `valid_from` /
`valid_until` / `superseded_by` — rather than overwriting in place. A
second consolidation pass that contradicts an active fact (e.g. a cert
renewal after an earlier "invalid" fact) closes the old fact and creates
a new version instead of silently replacing it, so the full history stays
queryable.

**Demo:** `python -m memory.demo_memory` runs all of the above end to end
against a deterministic offline heuristic (no `GOOGLE_API_KEY` required)
— buffer eviction → promote-or-drop → a consolidation pass creating a
fact → a second pass resolving a real conflict → a fact expiring with no
new episode involved.

## Part 2: Context Window Management

**The problem:** a long session eventually exceeds any practical context
budget. The four strategies below trade off recall (does the critical
fact survive?) against token cost differently, and the lab requires
picking one with numbers, not intuition.

| Strategy | Recall | Avg Input Tokens | Avg Output Tokens | Avg Latency (ms) |
|---|---|---|---|---|
| `sliding_window` | 0/10 | 965 | 0 | ~0.0 |
| `observation_masking` | **10/10** | 696 | 0 | ~0.0 |
| `recursive_summarization` | 4/10 | 906 | 120 | ~0.1 |
| `zone_based_pruning` | 7/10 | 2544 | 0 | ~0.0 |

Run with `python -m context_eval.run_eval` against the fixed 10-transcript
suite in `context_eval/test_transcripts/generate.py` — 6 certification-status
variants (varying depth and where the fact lands) plus 4 equipment-status
variants (a distinct scenario, so the comparison isn't tuned to one
critical-fact string).

### Chosen Strategy: Observation Masking

`sliding_window` loses the critical fact outright (0/10) — a fixed
recency window has no way to protect an older fact that's still relevant.
`zone_based_pruning` looked tied with masking on the original smaller
suite, but against the more varied set it drops to 7/10 — it keeps whole
zones rather than the specific tool output that matters, so it's less
reliable exactly where the fact type or position varies — at more than
3.5x the token cost of masking (2544 vs 696) for worse recall.
`recursive_summarization` trades a real generation cost (120 output
tokens/call, plus an LLM round-trip) for only 4/10, i.e. strictly worse
on both recall and cost here. **Observation masking wins outright**: it
targets the actual bloat source (old tool outputs) while leaving
everything else — including whichever fact matters — intact, at the
lowest token cost of any strategy, and it's the only one that held 100%
recall once the suite stopped being certification-only.

## Part 3: Retrieval & RAG

To close the "Ungoverned Knowledge" gap from the problem statement, we built
a retrieval pipeline over 5 IronBridge internal safety manuals
(`rag/corpus/`, 24 section-level chunks), extending the same `ironbridge.db`
used by Parts 1/2 (`db/rag_schema.sql` adds `rag_chunks` and `self_rag_log` —
no parallel database).

**Vector store:** real HNSW ANN index (`hnswlib`) with a SQL metadata index
on `topic`, used to pre-filter candidates *before* similarity scoring.
Falls back to a deterministic brute-force cosine index if `hnswlib` isn't
installed, so the demo is reproducible either way.

**Retrieval architectures:**
- **Naive RAG** — embed query → vector search → generate.
- **Hybrid Search** — vector + BM25 (`rank_bm25`, with a pure-Python
  fallback) fused via Reciprocal Rank Fusion. BM25 exists specifically
  because embeddings don't represent short identifiers like `4.2b`
  distinctively.
- **Agentic RAG** — multi-hop: a first hybrid-search hop, then a follow-up
  hop straight through the vector store's metadata pre-filter for any
  topic implied by the query but missing from hop 1's results, rather
  than diluting the query text with extra topic keywords before
  re-embedding, which retrieves the wrong chunks.

**Self-RAG-style verification** (`rag/self_rag.py`): every RAG answer and
every recalled semantic-memory fact is checked for **relevance** (does the
context actually address the query?) and **support** (does the answer
follow strictly from it?), logged to `self_rag_log`. An off-topic query or
an ungrounded answer is flagged rather than returned silently — wired into
both `ask_safety_policy` (`mcp_server/server.py`) and the semantic-fact
recall step in `agent/agent_with_memory.py`.

### Retrieval Comparison Table

Fixed 12-question set (`retrieval_eval/test_questions.py`), 3 categories:
general (favors naive), citation-heavy (favors hybrid — e.g. "what does
section 4.2b say"), decomposition (favors agentic — needs 2 topics
combined). Run with `python -m retrieval_eval.run_eval`.

| Architecture | Accuracy | Avg Tokens | Avg Latency (ms) | general | citation | decomposition |
|---|---|---|---|---|---|---|
| `naive_rag` | 4/12 | 190.4 | ~0.2 | 3/4 | 1/4 | 0/4 |
| `hybrid_search` | 8/12 | 178.8 | ~1.0 | 4/4 | 2/4 | 2/4 |
| `agentic_rag` | **10/12** | 229.7 | ~1.1 | 4/4 | 3/4 | **3/4** |

*(These numbers reproduce exactly, run after run — verified across 5+ back-to-back
executions of `python -m retrieval_eval.run_eval` on a freshly-ingested index. An
earlier version of this table was affected by a non-determinism bug in the hybrid
RRF fusion step — a Python `set` iterated in hash order, which shuffled tie-breaking
between runs and made `agentic_rag`'s accuracy flip between 10/12 and 11/12
depending on process hash-seed. `rag/hybrid_rag.py` now sorts candidates and
tie-breaks the RRF sort explicitly by `(-score, chunk_id)`, so the table above is
now stable.)*

### Chosen Strategy: Agentic RAG (primary; Hybrid as fallback)

Agentic RAG wins outright on every category and roughly doubles naive
RAG's overall accuracy, at a modest token/latency cost (~35% more tokens
than hybrid). Its edge is concentrated where the corpus is genuinely
hard: decomposition questions that need a second topic pulled in
specifically, which naive RAG almost never manages (0/4) and hybrid only
partially closes (2/4) — agentic RAG's metadata-pre-filtered follow-up
hop gets 3/4. We ship `ask_safety_policy` on agentic RAG by default, with
hybrid available as an explicit `mode="hybrid"` fallback for callers who
want lower, more predictable latency and don't need multi-hop.

The remaining 2 misses (1 citation-heavy, 1 decomposition) are a known
limitation worth investigating further rather than one papered over —
likely BM25 sharing common section numbers across manuals on a corpus
this small, and/or a residual case in the multi-hop topic-detection
keyword list. Not blocking for this delivery, but worth a follow-up pass.

## Part 4: Decomposition & Planning (Safety Equipment Approval Agent)

**The problem:** the memory/RAG agent above only ever handles one tool call
or one LLM turn at a time. A real recurring request doesn't fit that shape:
an engineer requests an excavator to dig a trench that *might* be unstable.
Approving it safely means checking trench depth, soil stability, whether
OSHA requires shoring, whether the engineer is certified, whether the
equipment is actually available, and whether the risk level requires a
supervisor sign-off — in an order that isn't fixed, on facts that can
contradict each other mid-decision. A wrong approval (an excavator in an
unstable trench) or a wrong rejection (a request that would have cleared
with a different check order) both cost something real. This is a planning
problem, not a memory problem, so it gets its own agent:
**`agent/planning_agent.py`** — separate from `agent/agent_with_memory.py`,
reusing the same `mcp_server/` and `db/`, built on top of a fork of
[`AmrSheta22/task_decomposition_and_planning`](https://github.com/AmrSheta22/task_decomposition_and_planning)
inside `planning/`.

Work was split two ways so both people could build against the same real
request type without colliding on files:

| Concern | Owner | File |
|---|---|---|
| Decomposition-first (whole DAG generated up front, executed in topological order) | A | `planning/algorithms/decomposition.py` |
| Dynamic/interleaved decomposition (next step generated after observing the last result) | B | `planning/algorithms/dynamic_decomposition.py` |
| Plan-and-Solve (deterministic sub-tasks: certification + equipment availability) | A | `planning/algorithms/plan_and_solve.py` |
| Tree of Thoughts (ordering the depth/soil/cert/availability checks) | B | `planning/algorithms/tree_of_thoughts.py` |
| LATS (the final APPROVE / REJECT / ESCALATE decision) | A | `planning/algorithms/lats.py` |
| Grounded `EnvironmentFeedback` (real DB/OSHA checks, replacing the toolkit's random default) | B | `planning/algorithms/environment.py` |
| Self-Refine (the message sent back to the engineer) | A | `planning/algorithms/self_refine.py` |
| Reflexion (the final decision, retried across trials with a carried reflection) | B | `planning/algorithms/reflexion.py` |
| Routing + orchestration, DAG cycle check, evaluation harness | both | `agent/planning_agent.py`, `planning/models.py`, `planning_eval/` |

**Routing (locatable in `agent/planning_agent.py` as `SUB_TASK_ROUTING`):**
the two deterministic DB lookups go to Plan-and-Solve, check ordering goes
to Tree of Thoughts, and the final decision — the single most expensive
node to get wrong — goes to LATS, scored by the real `GroundedEnvironment`
instead of the model's own opinion of itself. `planning/models.py`'s `Plan`
model enforces acyclicity at construction time (`validate_dag`, backed by
`networkx.is_directed_acyclic_graph`) — a plan that could deadlock is
rejected before it ever runs, not caught at execution time.

**Grounding:** every deterministic check calls `mcp_server/service.py`
directly against `db/ironbridge.db` — e.g. `check_certification` correctly
flags worker 2 (Sara Nabil)'s expired CRANE certification straight from the
seed data, not from an LLM guess. `GroundedEnvironment.evaluate()` (Owner
B) replaces the toolkit's randomized `environment.py` default with real
OSHA/DB checks, and LATS/Reflexion consume that evaluator, not a separate
self-opinion.

**Comparison table:** `planning_eval/run_eval.py` is the fixed test suite
and harness (dynamic-favored, lookahead-favored, and simple-deterministic
cases). Running it end to end for all eight methods (decomposition-first
vs. dynamic, PS vs. ToT vs. LATS, Self-Refine vs. Reflexion) requires a
`GOOGLE_API_KEY`/`GEMINI_API_KEY` — the full accuracy/calls/tokens/latency
table and the per-sub-task justification belong here once that run is
captured; this repo intentionally does not ship fabricated numbers.

**Demo:** `python -m agent.planning_agent` runs one full request
(decomposition-first DAG → Plan-and-Solve on the deterministic checks →
LATS on the final decision, grounded via `GroundedEnvironment` → Self-Refine
on the engineer-facing message) end to end and prints each stage with its
routing label. Requires a `GOOGLE_API_KEY` in `.env`.

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

## Bugfix Notes (post-review pass)

A full review of this repo turned up and fixed six issues before any
Graph RAG bonus work was started:

1. **`.env.example` broke the "offline, no API key" guarantee.** It shipped
   `GOOGLE_API_KEY=your-google-api-key-here`; copying it verbatim (as the
   Quick Start instructs) made every `if not api_key` check pass truthy,
   so `memory/_llm.py`, `rag/embeddings.py`, and `rag/self_rag.py` all
   attempted a real Gemini call and crashed instead of using the
   deterministic offline fallback. Fixed by commenting the line out.
2. **`VectorStore.build_ann_index()` never set `self._ids`** on a
   freshly-built in-memory index — only the disk-reload path did. This
   crashed `rag/demo_rag.py` and `demo_all.py` (both ingest then search
   in the same process). Fixed by setting `self._ids` right after the
   index is built, not only when reloading from disk.
3. **Non-deterministic retrieval eval.** `hybrid_rag.py`'s RRF fusion
   iterated a Python `set`, whose string-hash iteration order is
   randomized per process — tied chunks got different tie-break order on
   different runs, and `agentic_rag`'s accuracy on the fixed 12-question
   set flipped between 10/12 and 11/12 run to run, which directly
   violates the "keep test suites fixed" guardrail. Fixed by sorting the
   candidate ids and tie-breaking the RRF sort explicitly on
   `(-score, chunk_id)`. Verified stable across 5+ consecutive runs.
4. **Dangling doc reference** to a nonexistent `patches/agent_integration_patch.md`
   in `rag/self_rag.py`'s docstring — pointed at the real integration
   point (`agent/agent_with_memory.py`) instead.
5. **MCP stdio protocol corruption.** `mcp_server/server.py` calls
   `rag.ingest.run()` at import time to build the RAG index on boot; that
   function was printing progress lines to stdout — the same stream
   carrying JSON-RPC messages on the stdio transport — which corrupted
   every live agent session (`Failed to parse JSONRPC message from
   server`, visible in `agent/agent_with_memory.py` runs). Fixed by
   routing `rag/ingest.py`'s progress logging to stderr.
6. **Debug leftovers** in `agent/client.py` printed `SUPERVISOR_PIN` and
   the working directory to stdout on every run. Removed.

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # Set SUPERVISOR_PIN (e.g., 4821)
python db/load_data.py      # Initialize the database — run this FIRST; every
                             # demo/eval below depends on the base schema
                             # (workers, sites, equipment) existing.
python agent/client.py      # Run the automated demo

# Part 1 (memory) — no GOOGLE_API_KEY required, runs fully offline/deterministic
python -m memory.demo_memory      # buffer eviction, promote-or-drop, consolidation

# Part 2 (context management) — same offline guarantee
python -m context_eval.run_eval   # reproduces the comparison table above

# Part 3 (RAG) — no GOOGLE_API_KEY required, runs fully offline/deterministic
python -m rag.ingest              # chunk -> embed -> build vector index
python -m rag.demo_rag            # standalone demo, mirrors memory/demo_memory.py
python -m retrieval_eval.run_eval # reproduces the comparison table above

# Part 4 (Decomposition & Planning) — requires GOOGLE_API_KEY (LLM-driven planning/search)
python -m agent.planning_agent    # one full request: DAG -> Plan-and-Solve -> LATS -> Self-Refine
python -m planning_eval.run_eval  # fixed test suite, drives the Part 4 comparison table

# All of the above in one pass, one transcript:
python -m demo_all
```
---

## Part 5: Final Project — State Graphs, Platform, Dual-Admin, Multi-Agent

### What was added on top of MCP + Memory/RAG + Planning

| Folder | Role |
|--------|------|
| `state_graph/` | 3 cyclic graphs: cert coordination, high-risk dig, incident handoff + checkpoint / HITL / failure tickets |
| `platform_db/` | Product data: attendance lights, dual-admin requests, notifications, roles (worker/engineer/admin) |
| `app_platform/` | Flask website — worker / engineer / dual admin UI |
| `multi_agent/` | Router + Gemini client + **mcp_bridge** (reuses `mcp_server.service` + `rag` when available) |
| `demo_final.py` | Offline proof of graphs, HITL, tickets, crash-resume |

### Three roles

| Role | Login | Capabilities |
|------|-------|----------------|
| Worker | W2 / `1111`, W3 / `2222` | Attendance (red→green after admin), today's tasks via assistant |
| Engineer | W1 / `1234` | Same + request workers/tools/equipment (**both ADMIN1 and ADMIN2 must approve**) |
| Admin | ADMIN1 / `9999`, ADMIN2 / `8888` | Attendance approve, dual vote on engineer requests, HITL, tickets |

### Run the product surface

```bash
pip install -r requirements.txt
# set GOOGLE_API_KEY and SUPERVISOR_PIN in .env
python app_platform/app.py
# → http://127.0.0.1:5050
```

Offline graph demo (no key required for core paths):

```bash
python demo_final.py
```

### Hidden RAG / memory

Chat injects `PERSON_DATA` from `mcp_bridge.try_service_context` (real DB via `mcp_server.service` when `SUPERVISOR_PIN` + DB work) and policy snippets from `rag/` when the vector index exists; otherwise `platform_db` seed. The UI never lists "RAG sources" — the assistant answers in plain language.

### Dual-admin rule

`platform_db.store.vote_request`: status becomes `approved` only if **ADMIN1 and ADMIN2** both approve; **any** reject → `rejected`. Notifications fire on every vote.

### Checkpoint / HITL / tickets

See `state_graph/engine.py`. Admin resolves HITL and failure tickets in the platform; runs resume from the latest checkpoint (not from scratch).

### Wiring into prior labs

- Does **not** replace `agent/agent_with_memory.py` or `agent/planning_agent.py`
- Graphs are a **new** agent family beside them
- `multi_agent/mcp_bridge.py` is the deliberate reuse point for `mcp_server/` and `rag/`

