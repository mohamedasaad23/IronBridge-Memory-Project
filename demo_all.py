"""
demo_all.py

Single cohesive transcript exercising every concern in the Memory & RAG
lab, in one run: short-term memory promote-or-drop, consolidation
(including a real conflict resolution), all four context-management
strategies against the fixed test suite, all three RAG architectures
answering questions, and Self-RAG verification both accepting and
flagging a result.

This does not reimplement any of the four underlying demo/eval scripts —
it imports and calls their existing entry points and print helpers in
sequence, so there is exactly one place each concern's logic lives.

Prerequisite: python db/load_data.py (base schema must exist first —
see README Quick Start).

Run: python -m demo_all
"""
from __future__ import annotations


def _banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    _banner("PART 1 -- MEMORY (short-term buffer, promote-or-drop, consolidation)")
    from memory.demo_memory import main as memory_demo
    memory_demo()

    _banner("PART 2 -- CONTEXT WINDOW MANAGEMENT (4-strategy comparison)")
    from context_eval.run_eval import run as context_eval_run, print_table as context_print_table
    context_print_table(context_eval_run())

    _banner("PART 3 -- RAG (naive / hybrid / agentic, Self-RAG verification)")
    from rag.demo_rag import main as rag_demo
    rag_demo()

    _banner("PART 3 -- RETRIEVAL ARCHITECTURE COMPARISON (12-question eval)")
    from retrieval_eval.run_eval import run_eval as retrieval_eval_run, print_report as retrieval_print_report
    retrieval_print_report(retrieval_eval_run())

    _banner("DONE -- every concern above has now fired at least once in this run.")


if __name__ == "__main__":
    main()
