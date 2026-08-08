"""
Generates fixed, tool-output-heavy test transcripts for the context
management evaluation. Kept deterministic (no randomness) so re-running
`run_eval.py` always scores the same transcripts — required by the lab's
"keep test suites fixed once you start evaluating" guardrail.

Each transcript buries ONE critical fact early on (e.g. a worker's
certification status) under many turns of large, realistic tool-call
JSON output, then asks a final question that can only be answered
correctly if that early fact survived whatever pruning strategy was
applied. This mirrors the lab's own worked example (the allergy detail
buried under 30+ tool calls).
"""
from __future__ import annotations

import json
from typing import Any


def _fake_compliance_blob(site_id: int, turn: int) -> str:
    """A large, realistic-looking tool result — the actual noise source.
    Modeled on generate_site_compliance_report's real output shape."""
    findings = [
        {
            "request_id": 100 + turn * 3 + i,
            "status": "APPROVED" if i % 2 == 0 else "PENDING_SUPERVISOR_APPROVAL",
            "cert_valid": True,
            "equipment": ["Tower Crane TC-40", "Excavator EX-220", "Scaffold Set S-12"][i % 3],
            "site_id": site_id,
            "notes": "Routine daily inspection completed, no anomalies noted.",
        }
        for i in range(6)
    ]
    return json.dumps(
        {"site": f"Site {site_id}", "turn": turn, "findings": findings, "compliant": True}
    )


def build_certification_transcript(
    critical_turn: int = 3,
    total_turns: int = 40,
    final_question_turn: int | None = None,
) -> dict[str, Any]:
    """Critical fact: worker 2's CRANE certification expired 2025-01-10,
    stated once at `critical_turn`. Buried under `total_turns` of tool
    noise. The final turn asks the question a correct system must answer
    using that buried fact.

    Returns {"messages": [...], "critical_marker": str, "expected_answer_contains": str}
    """
    final_question_turn = final_question_turn or total_turns
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": "You are the Iron Bridge equipment safety assistant.",
        }
    ]

    critical_marker = "CERT-EXPIRED-WORKER-2-CRANE-2025-01-10"

    for turn in range(1, total_turns + 1):
        if turn == critical_turn:
            messages.append(
                {
                    "role": "tool",
                    "content": (
                        f"check_worker_certification(worker_id=2, equipment_type=CRANE) -> "
                        f"{{'valid': False, 'reason': 'Certification expired on 2025-01-10'}} "
                        f"[[{critical_marker}]]"
                    ),
                }
            )
            # A real agent reads the tool result and states the finding in
            # its own reply — the fact doesn't only live inside raw JSON.
            # This is what lets masking (which only touches role=="tool")
            # legitimately preserve it while still clearing out bulky
            # tool payloads; it is not a free pass, sliding-window and a
            # naive summarizer can still lose this line if it falls
            # outside their kept window/summary.
            messages.append(
                {
                    "role": "assistant",
                    "content": "Noted — worker 2's CRANE certification expired on "
                    "2025-01-10, flagging before any crane request is approved.",
                }
            )
        elif turn == final_question_turn:
            messages.append(
                {
                    "role": "user",
                    "content": "Before we approve worker 2 for the mobile crane today, "
                    "any certification concerns I should know about?",
                }
            )
        else:
            # Filler: realistic tool-heavy noise, alternating user chatter
            # and large compliance-report-shaped tool output.
            if turn % 4 == 0:
                messages.append(
                    {"role": "user", "content": f"Any updates on site {1 + (turn % 3)}?"}
                )
            else:
                messages.append(
                    {
                        "role": "tool",
                        "content": _fake_compliance_blob(site_id=1 + (turn % 3), turn=turn),
                    }
                )

    return {
        "messages": messages,
        "critical_marker": critical_marker,
        "expected_answer_contains": "expired",
    }


def build_equipment_status_transcript(
    critical_turn: int = 4,
    total_turns: int = 30,
    final_question_turn: int | None = None,
) -> dict[str, Any]:
    """Second scenario type — deliberately NOT a certification fact, to
    check strategies generalize rather than being tuned to the one
    critical_marker string in build_certification_transcript.

    Critical fact: Tower Crane TC-40 at site 2 was pulled OUT_OF_SERVICE
    for a hydraulic fault, stated once at `critical_turn`. Buried under
    `total_turns` of unrelated site-chatter noise. The final turn asks
    the question a correct system must answer using that buried fact.
    """
    final_question_turn = final_question_turn or total_turns
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": "You are the Iron Bridge equipment safety assistant.",
        }
    ]

    critical_marker = "EQUIP-TC40-OUT-OF-SERVICE-HYDRAULIC-FAULT"

    for turn in range(1, total_turns + 1):
        if turn == critical_turn:
            messages.append(
                {
                    "role": "tool",
                    "content": (
                        "get_equipment_status(equipment_id=1) -> "
                        "{'equipment': 'Tower Crane TC-40', 'status': 'OUT_OF_SERVICE', "
                        "'reason': 'hydraulic fault reported by site 2'} "
                        f"[[{critical_marker}]]"
                    ),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": "Noted — Tower Crane TC-40 is out of service for a "
                    "hydraulic fault, so it can't be assigned to any new request.",
                }
            )
        elif turn == final_question_turn:
            messages.append(
                {
                    "role": "user",
                    "content": "Is the Tower Crane TC-40 available to assign right now, "
                    "or is there something blocking it?",
                }
            )
        else:
            if turn % 5 == 0:
                messages.append(
                    {"role": "user", "content": f"Any new requests at site {1 + (turn % 3)}?"}
                )
            else:
                messages.append(
                    {
                        "role": "tool",
                        "content": _fake_compliance_blob(site_id=1 + (turn % 3), turn=turn),
                    }
                )

    return {
        "messages": messages,
        "critical_marker": critical_marker,
        "expected_answer_contains": "out of service",
    }


# Fixed suite — 10 variations across 2 scenario types (certification status,
# equipment status) and a range of depths/positions, so the comparison
# isn't tuned to one critical-fact string. Per the lab's "keep test suites
# fixed once you start evaluating" guardrail, this set is now locked.
def build_test_suite() -> list[dict[str, Any]]:
    return [
        build_certification_transcript(critical_turn=3, total_turns=20),
        build_certification_transcript(critical_turn=3, total_turns=40),
        build_certification_transcript(critical_turn=5, total_turns=40),
        build_certification_transcript(critical_turn=2, total_turns=60),
        build_certification_transcript(critical_turn=10, total_turns=60),
        build_certification_transcript(critical_turn=25, total_turns=60),
        build_equipment_status_transcript(critical_turn=4, total_turns=30),
        build_equipment_status_transcript(critical_turn=8, total_turns=45),
        build_equipment_status_transcript(critical_turn=2, total_turns=50),
        build_equipment_status_transcript(critical_turn=35, total_turns=50),
    ]