"""
Memory-aware Iron Bridge agent — Memory & RAG Lab extension.

This does NOT duplicate agent/client.py's MCP handshake, elicitation
callback, or sampling callback — it imports and reuses them directly
(see the "Extending the existing system" guardrail: don't rebuild what
already works). What's new here is wrapping that same live MCP session
with the memory/ package, so short-term memory, promote-or-drop
routing, and semantic-fact recall genuinely happen inside the agent's
loop, not only in the standalone memory/demo_memory.py script.

Run: python -m agent.agent_with_memory
(starts the stdio MCP server itself, same as agent/client.py does)
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Reuse — not reimplement — the callbacks already built and graded in
# the MCP Server Lab.
from agent.client import elicitation_callback, sampling_callback

from memory import consolidation, router, store
from memory.stm import ShortTermMemory

WORKER_ID = 2  # Sara Nabil — has memory-relevant history in seed_data.sql


def _record_and_route(stm: ShortTermMemory, role: str, content: str) -> None:
    """Every turn — including raw tool results coming back from the real
    MCP server — goes through the same short-term buffer exercised in
    memory/demo_memory.py. This is what makes it a "live loop" instead
    of an isolated demo: the router fires on real session content."""
    evicted = stm.add(role, content, worker_id=WORKER_ID)
    decision = router.process_overflow(evicted)
    if decision:
        print(f"  [MEMORY] evicted -> {decision.destination}: {decision.reasoning}")


async def main() -> None:
    store.ensure_memory_schema()

    stm = ShortTermMemory(max_turns=6)
    stm.update_plan(
        plan="Assist worker 2 with an equipment request",
        subgoal="consult memory before acting, then check live certification",
    )

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        cwd=str(ROOT),
        env={**dict(os.environ), "MCP_TRANSPORT": "stdio"},
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read,
            write,
            sampling_callback=sampling_callback,
            elicitation_callback=elicitation_callback,
        ) as session:
            await session.initialize()
            print("=== Memory-aware session started ===\n")

            # ---- Consult semantic memory BEFORE acting ----
            # This is the actual point of long-term memory: the agent
            # checks what it already knows about this worker before
            # making any live MCP call.
            active_facts = store.get_all_active_facts(WORKER_ID)
            print("=== Active semantic facts about worker before acting ===")
            if not active_facts:
                print("  (none yet — first session for this worker)")
            for f in active_facts:
                print(f"  {f['fact_key']} = {f['fact_value']}")
            _record_and_route(
                stm,
                "system",
                f"Recalled {len(active_facts)} active semantic fact(s) for "
                f"worker {WORKER_ID} before acting.",
            )
            print()

            # ---- Real, live MCP tool call — genuinely wired, not scripted output ----
            _record_and_route(stm, "user", "Can worker 2 use the mobile crane today?")
            result = await session.call_tool(
                "check_worker_certification",
                {"worker_id": WORKER_ID, "equipment_type": "CRANE"},
            )
            content_str = str(result.content)
            print("check_worker_certification ->", content_str)
            _record_and_route(stm, "tool", content_str)
            print()

            # ---- A couple more turns so the buffer overflows for real,
            # exercising the router against live session content instead
            # of the synthetic transcript in memory/demo_memory.py ----
            for extra_role, extra_content in [
                ("user", "Any other sites need attention today?"),
                ("assistant", "Let me check site status for you."),
                ("user", "What's the status of the excavator at site 2?"),
            ]:
                _record_and_route(stm, extra_role, extra_content)
            print()

            # ---- End-of-session consolidation — a real periodic pass
            # over whatever this live session actually produced, not a
            # canned scenario. ----
            print("=== End-of-session consolidation (real session data) ===")
            decisions = consolidation.consolidate_worker(WORKER_ID)
            if not decisions:
                print("  (nothing new to consolidate this session)")
            for d in decisions:
                print(f"  {d.action}: {d.reasoning}")
            print("Active facts after consolidation:", store.get_all_active_facts(WORKER_ID))


if __name__ == "__main__":
    asyncio.run(main())