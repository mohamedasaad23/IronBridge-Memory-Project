from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

RUBRIC = (
    "1. States the decision (APPROVE / REJECT / ESCALATE) unambiguously.\n"
    "2. States the specific reason(s) behind the decision.\n"
    "3. States the engineer's concrete next step.\n"
    "4. Is consistent with the grounded LATS decision it is based on — it must not soften, "
    "reverse, or omit that decision."
)


def deterministic_checks(lats_decision: str, draft: str) -> list[str]:
    """Grounded checks a critic model shouldn't need to guess at."""
    issues: list[str] = []
    draft_lower = draft.lower()
    decision_word = next(
        (word for word in ("approve", "reject", "escalate") if word in lats_decision.lower()),
        None,
    )
    if decision_word and decision_word not in draft_lower:
        issues.append(f"The grounded decision was '{decision_word.upper()}' but the draft never says so.")
    if not re.search(r"next step|please|contact|submit|reschedule|resubmit|supervisor", draft_lower):
        issues.append("The draft does not name a concrete next step for the engineer.")
    if len(draft.split()) < 15:
        issues.append("The draft is too short to contain both a reason and a next step.")
    return issues


@dataclass
class ReflectionResult:
    draft: str
    critique: str
    revised: str
    grounded_issues: list[str]


def reflect_and_refine(lats_decision: str, draft: str, llm: BaseChatModel) -> ReflectionResult:
    """Self-Refine for the engineer-facing message: one draft, one critique against the
    explicit rubric above, one revision. This output is cheap to redo — unlike the final
    decision itself, a bad message just gets rewritten, so a single Self-Refine pass is
    enough and Reflexion's multi-trial memory would be unnecessary cost here.
    """
    grounded = deterministic_checks(lats_decision, draft)
    grounded_report = "\n".join(f"- {issue}" for issue in grounded) or "- Deterministic checks passed."

    critique_response = llm.invoke([
        ("system", "You are a separate critic reviewing a safety-decision message to a construction worker. Judge strictly against the rubric; do not rewrite the draft."),
        ("human", f"""Grounded decision from LATS: {lats_decision}

Rubric:
{RUBRIC}

External deterministic checks:
{grounded_report}

Draft message to the engineer:
{draft}

List concrete issues. If there are none, respond exactly PASS."""),
    ], temperature=0.2)
    critique = critique_response.content
    if not isinstance(critique, str) or not critique.strip():
        raise RuntimeError("The chat model returned an empty or unsupported response")
    critique = critique.strip()

    if critique.upper() == "PASS" and not grounded:
        revised = draft
    else:
        response = llm.invoke([
            ("system", "Revise a safety-decision message using both external checks and an independent critique. Never contradict the grounded decision."),
            ("human", f"Grounded decision: {lats_decision}\n\nDraft:\n{draft}\n\nGrounded checks:\n{grounded_report}\n\nCritique:\n{critique}\n\nReturn only the improved message to the engineer."),
        ], temperature=0.2)
        revised_content = response.content
        if not isinstance(revised_content, str) or not revised_content.strip():
            raise RuntimeError("The chat model returned an empty or unsupported response")
        revised = revised_content.strip()

    return ReflectionResult(draft, critique, revised, grounded)
