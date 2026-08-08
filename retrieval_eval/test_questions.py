"""
retrieval_eval/test_questions.py

Fixed, locked evaluation set (never edit once scoring starts — same rule
as context_eval/'s test suite). 12 questions across 3 categories:

  general        - single-topic factual question, favors naive_rag
  citation       - asks for a specific numbered section, favors hybrid
                   (BM25 catches the literal section-id token)
  decomposition  - needs chunks from 2 different topics combined,
                   favors agentic_rag's multi-hop topic pre-filter

Each question lists the expected_chunk_ids that MUST all appear in the
top_k retrieved set for the answer to count as correct (see
retrieval_eval/run_eval.py::score_question).
"""

from dataclasses import dataclass
from typing import List


@dataclass
class EvalQuestion:
    qid: str
    category: str  # "general" | "citation" | "decomposition"
    query: str
    expected_chunk_ids: List[str]


QUESTIONS: List[EvalQuestion] = [
    # --- general (4) ---
    EvalQuestion("G1", "general", "How long before refueling must a generator cool down?",
                 ["MAN-7:3"]),
    EvalQuestion("G2", "general", "What oxygen level prohibits entry into an excavation?",
                 ["MAN-5:5"]),
    EvalQuestion("G3", "general", "How much load capacity can a crane operator exceed without a variance?",
                 ["MAN-3:3"]),
    EvalQuestion("G4", "general", "How often must a competent person inspect a fall arrest harness?",
                 ["MAN-6:3"]),

    # --- citation-heavy (4) ---
    EvalQuestion("C1", "citation", "What does section 4.2b say?",
                 ["MAN-4:4.2b"]),
    EvalQuestion("C2", "citation", "What is covered in MAN-5 section 5?",
                 ["MAN-5:5"]),
    EvalQuestion("C3", "citation", "Summarize crane operations section 4.",
                 ["MAN-3:4"]),
    EvalQuestion("C4", "citation", "What does electrical safety section 4.1 require?",
                 ["MAN-4:4.1"]),

    # --- decomposition (4, needs 2 topics combined) ---
    EvalQuestion(
        "D1", "decomposition",
        "What protective system does excavation require for a trench, and what "
        "lockout/tagout procedure does electrical work require, when both happen "
        "in the same trench?",
        ["MAN-5:3", "MAN-4:2"],
    ),
    EvalQuestion(
        "D2", "decomposition",
        "What harness inspection does fall protection require, and what fuel storage "
        "rule applies to a generator, when a crew does roof work with a generator "
        "on the roof?",
        ["MAN-6:3", "MAN-7:2"],
    ),
    EvalQuestion(
        "D3", "decomposition",
        "Before a multi-crane lift near an excavation, what do the load limit rules and the "
        "utility locate rules each require?",
        ["MAN-3:3", "MAN-5:4"],
    ),
    EvalQuestion(
        "D4", "decomposition",
        "What does the lockout/tagout rule require, and what does the harness "
        "inspection rule require about removing damaged equipment from service, "
        "after an incident?",
        ["MAN-4:2", "MAN-6:3"],
    ),
]
