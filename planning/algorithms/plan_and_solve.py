from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from mcp_server import service


@dataclass
class PlanAndSolveResult:
    plan: str
    grounded_facts: list[str]
    solution: str


def plan_and_solve(worker_id: int, equipment_id: int, llm: BaseChatModel) -> PlanAndSolveResult:
    """Plan-and-Solve, routed to the two deterministic sub-tasks in the DAG: certification
    and equipment availability. Both are single DB lookups with no real branching or
    failure-recovery need, so a single explicit plan phase followed by one grounded solve
    phase is the right shape here — Tree of Thoughts' search and LATS' external-feedback
    loop would be pure overhead on a sub-task shaped like this.
    """
    plan_response = llm.invoke([
        ("system", "You use Plan-and-Solve prompting for a construction safety check. Clearly separate PLAN from ACTIONS."),
        ("human", f"""A worker (id={worker_id}) has requested equipment (id={equipment_id}).
Before this request can proceed, two deterministic facts must be established.
State a short PLAN naming exactly which two database checks you will run and in what order."""),
    ], temperature=0.1)
    plan_text = plan_response.content
    if not isinstance(plan_text, str) or not plan_text.strip():
        raise RuntimeError("The chat model returned an empty or unsupported response")
    plan_text = plan_text.strip()

    # Solve: grounded DB calls, not model guesses.
    equipment = service.get_equipment(equipment_id)
    facts: list[str] = []
    if not equipment:
        facts.append(f"No equipment found with id {equipment_id}.")
    else:
        cert = service.check_certification(worker_id, equipment["type"])
        facts.append(
            f"Certification: {'VALID' if cert['valid'] else 'INVALID'} for worker {worker_id} "
            f"on {equipment['type']}"
            + (f" until {cert['valid_until']}." if cert["valid"] else f" — {cert['reason']}.")
        )
        facts.append(
            f"Availability: {equipment['name']} is "
            + ("AVAILABLE." if equipment["status"] == "AVAILABLE" else f"NOT available (status={equipment['status']}).")
        )

    solve_response = llm.invoke([
        ("system", "Summarize grounded database facts for a construction safety decision. Do not invent facts beyond what is given."),
        ("human", f"""Plan:
{plan_text}

Grounded facts from the database:
{chr(10).join('- ' + fact for fact in facts)}

Write one short paragraph combining these into a solved sub-result the downstream decision
step can use directly."""),
    ], temperature=0.1)
    solution = solve_response.content
    if not isinstance(solution, str) or not solution.strip():
        raise RuntimeError("The chat model returned an empty or unsupported response")

    return PlanAndSolveResult(plan=plan_text, grounded_facts=facts, solution=solution.strip())
