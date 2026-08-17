"""Safety Equipment Approval Agent — Iron Bridge Construction, Part 4 (Decomposition & Planning).

This agent is separate from agent/agent_with_memory.py (the memory/RAG agent) and does
not touch its code path. It reuses the same mcp_server/ and db/ as the read layer, with
planning/ (forked + adapted from AmrSheta22/task_decomposition_and_planning) as the
implementation layer.

ROUTING LOGIC (locatable concern — see SUB_TASK_ROUTING below):
    check_certification, check_equipment_availability  -> Plan-and-Solve (deterministic,
                                                             single DB lookup, no branching)
    check_site_hazards ordering                         -> Tree of Thoughts (several valid
                                                             orderings, self-evaluated search)
    final_decision (APPROVE / REJECT / ESCALATE)         -> LATS (highest cost of a wrong
                                                             answer, grounded external
                                                             feedback via environment.py)
    engineer-facing response                             -> Self-Refine (cheap to redo,
                                                             one draft/critique/revision pass)

Run: python -m agent.planning_agent
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planning.algorithms import (  # noqa: E402
    decompose_goal,
    execute_plan,
    plan_and_solve,
    lats,
    GroundedEnvironment,
    reflect_and_refine,
)

# Locatable concern: which sub-task type routes to which planning method, and why.
SUB_TASK_ROUTING = {
    "check_certification": "plan_and_solve",
    "check_equipment_availability": "plan_and_solve",
    "check_site_hazards": "tree_of_thoughts",  # see planning/algorithms/tree_of_thoughts.py
    "final_decision": "lats",
}


@dataclass
class ApprovalResult:
    goal: str
    dag_outputs: dict[str, str]
    ps_certification: str
    ps_availability: str
    lats_decision: str
    lats_score: float
    engineer_message: str
    self_refine_critique: str


def run_safety_approval(
    worker_id: int,
    equipment_id: int,
    site_id: int,
    llm,
    lats_iterations: int = 2,
) -> ApprovalResult:
    """Decompose an equipment request into a DAG (decomposition-first), ground the
    deterministic checks via Plan-and-Solve, resolve the final decision via LATS against
    the real GroundedEnvironment, then Self-Refine the message sent back to the engineer.
    """
    goal = (
        f"Worker {worker_id} has requested equipment {equipment_id} at site {site_id}. "
        "Decide whether to APPROVE, REJECT, or ESCALATE the request."
    )
    request = {"worker_id": worker_id, "equipment_id": equipment_id, "site_id": site_id}

    # 1. Decomposition-first: whole plan generated up front, executed in topological order.
    plan = decompose_goal(goal, llm)
    dag_outputs = execute_plan(plan, llm, request)

    # 2. Plan-and-Solve re-run standalone for the two deterministic checks, so their
    #    grounded facts are also available in isolation (used by the eval harness to
    #    compare PS in isolation against the DAG's inline grounded execution).
    ps_result = plan_and_solve(worker_id, equipment_id, llm)

    # 3. LATS for the highest-stakes node: the final decision, scored by the real
    #    GroundedEnvironment (mcp_server/service.py), not the model's own opinion.
    dag_context = "\n".join(f"{task_id}: {output}" for task_id, output in dag_outputs.items())
    lats_task = f"{goal}\n\nDAG findings so far:\n{dag_context}\n\nPlan-and-Solve facts:\n{ps_result.solution}"
    env = GroundedEnvironment()
    lats_result = lats(lats_task, llm, env, iterations=lats_iterations)

    # 4. Self-Refine the engineer-facing message, never contradicting the LATS decision.
    draft = dag_outputs.get("final_decision", lats_result.output)
    refined = reflect_and_refine(lats_result.output, draft, llm)

    return ApprovalResult(
        goal=goal,
        dag_outputs=dag_outputs,
        ps_certification=ps_result.grounded_facts[0] if ps_result.grounded_facts else "",
        ps_availability=ps_result.grounded_facts[1] if len(ps_result.grounded_facts) > 1 else "",
        lats_decision=lats_result.output,
        lats_score=lats_result.best_score,
        engineer_message=refined.revised,
        self_refine_critique=refined.critique,
    )


def _build_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY (or GEMINI_API_KEY) in your .env before running this agent.")
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key, temperature=0.1)


if __name__ == "__main__":
    llm = _build_llm()
    result = run_safety_approval(worker_id=2, equipment_id=2, site_id=1, llm=llm)
    print("GOAL:", result.goal)
    print("\nDAG OUTPUTS:")
    for task_id, output in result.dag_outputs.items():
        print(f"  [{SUB_TASK_ROUTING.get(task_id, 'reasoning')}] {task_id}: {output}")
    print("\nLATS DECISION:", result.lats_decision, f"(score={result.lats_score:.2f})")
    print("\nENGINEER MESSAGE:\n", result.engineer_message)
