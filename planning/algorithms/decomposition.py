from __future__ import annotations

import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict

from ..models import Plan

# Inject root to access mcp_server modules (mirrors planning/algorithms/environment.py)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from mcp_server import service


PLANNER_SYSTEM = """You are the decomposition-first planner for Iron Bridge Construction's \
Safety Equipment Approval Agent. A worker has requested a piece of heavy equipment for a job \
site and the whole plan must be produced up front, in one shot, before anything is executed.

Prefer these canonical task ids whenever they apply, so the executor can run them against the \
real equipment database instead of guessing:
- check_certification: is the requesting worker certified for this equipment type?
- check_equipment_availability: is the specific equipment unit AVAILABLE right now?
- check_site_hazards: does the site have hazards (power lines, unstable/Type C soil, trench \
depth) that change what's required before approval?
- check_supervisor_signoff: does this request's risk level require an explicit supervisor \
sign-off before it can proceed?
- final_decision: exactly one terminal task, depending on every check that actually ran, that \
states APPROVE, REJECT, or ESCALATE with a one-line reason.

Independent checks should not depend on each other. The plan must end with exactly one \
final_decision task depending on every necessary branch."""


class PlannedTask(BaseModel):
    """Wire schema; richer semantic constraints are applied by the Task domain model."""

    model_config = ConfigDict(extra="forbid")

    id: str
    instruction: str
    depends_on: list[str]


class GeneratedPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str
    tasks: list[PlannedTask]


def decompose_goal(goal: str, llm: BaseChatModel) -> Plan:
    """Generate the whole DAG up front, in one shot. Acyclicity and dependency validity
    are enforced at construction time by Plan.validate_dag (see planning/models.py) —
    a plan that could deadlock never leaves this function."""
    generated = llm.with_structured_output(
        GeneratedPlan,
        method="json_schema",
    ).invoke([
        ("system", PLANNER_SYSTEM),
        ("human", f"""Decompose this equipment request into 3-6 tasks: {goal!r}
Use the canonical task ids listed above wherever they apply. Preserve the supplied goal exactly
in the plan's goal field."""),
    ], temperature=0.1)
    # The caller's goal remains authoritative even if the model paraphrases it.
    payload = generated.model_dump()
    payload["goal"] = goal
    return Plan.model_validate(payload)


# Task ids answered directly from the real database instead of an LLM guess.
GROUNDED_TASK_IDS = {"check_certification", "check_equipment_availability"}


def _run_grounded_task(task_id: str, request: dict) -> str:
    """Execute a canonical task against mcp_server/service.py — no LLM in the loop."""
    equipment_id = request["equipment_id"]
    worker_id = request["worker_id"]

    equipment = service.get_equipment(equipment_id)
    if not equipment:
        return f"DB lookup failed: no equipment with id {equipment_id}."

    if task_id == "check_certification":
        result = service.check_certification(worker_id, equipment["type"])
        if result["valid"]:
            return (
                f"Certification VALID for worker {worker_id} on {equipment['type']} "
                f"(until {result['valid_until']})."
            )
        return f"Certification INVALID for worker {worker_id} on {equipment['type']}: {result['reason']}."

    if task_id == "check_equipment_availability":
        if equipment["status"] == "AVAILABLE":
            return f"Equipment {equipment['name']} ({equipment['type']}) is AVAILABLE."
        return (
            f"Equipment {equipment['name']} ({equipment['type']}) is NOT available: "
            f"status={equipment['status']}."
        )

    raise KeyError(task_id)


def execute_plan(
    plan: Plan,
    llm: BaseChatModel,
    request: dict,
    max_workers: int = 4,
) -> dict[str, str]:
    """Execute the DAG in topological batches, once, with no ability to react to what a
    step returns (that reactivity is dynamic_decomposition's job — see
    planning/algorithms/dynamic_decomposition.py). Canonical deterministic task ids are
    answered from the real database; every other task is reasoned about by the LLM using
    prior outputs as context.

    `request` must contain `worker_id`, `equipment_id`, and `site_id`.
    """
    outputs: dict[str, str] = {}
    for batch in plan.execution_batches():
        grounded_ids = [task_id for task_id in batch if task_id in GROUNDED_TASK_IDS]
        reasoning_ids = [task_id for task_id in batch if task_id not in GROUNDED_TASK_IDS]

        for task_id in grounded_ids:
            outputs[task_id] = _run_grounded_task(task_id, request)

        if reasoning_ids:
            prompts: dict[str, str] = {}
            for task_id in reasoning_ids:
                task = plan.task(task_id)
                context = "\n\n".join(
                    f"OUTPUT FROM {dependency}:\n{outputs[dependency]}"
                    for dependency in task.depends_on
                    if dependency in outputs
                ) or "No prerequisite outputs."
                prompts[task_id] = f"""Overall goal: {plan.goal}
Current task: {task.instruction}
Prerequisite outputs:
{context}
Complete only the current task. Be concrete and concise. If this is the final_decision
task, state APPROVE, REJECT, or ESCALATE on the first line, then one line of reasoning."""
            with ThreadPoolExecutor(max_workers=min(max_workers, len(reasoning_ids))) as pool:
                futures = {
                    pool.submit(
                        llm.invoke,
                        [
                            ("system", "You execute one node in a validated safety-approval DAG for Iron Bridge Construction."),
                            ("human", prompt),
                        ],
                        temperature=0.2,
                    ): task_id
                    for task_id, prompt in prompts.items()
                }
                for future in as_completed(futures):
                    content = future.result().content
                    if not isinstance(content, str) or not content.strip():
                        raise RuntimeError("The chat model returned an empty or unsupported response")
                    outputs[futures[future]] = content.strip()
    return outputs


def final_output(plan: Plan, outputs: dict[str, str]) -> str:
    terminals = plan.terminal_tasks()
    if len(terminals) != 1:
        raise ValueError(f"Expected exactly one terminal decision task, found {terminals}")
    return outputs[terminals[0]]
