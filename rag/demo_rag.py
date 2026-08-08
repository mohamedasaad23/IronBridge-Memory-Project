"""
rag/demo_rag.py

Standalone demo script, mirrors memory/demo_memory.py's role for Part 1.
Run with: python -m rag.demo_rag

Demonstrates, in order:
  1. Ingest (idempotent — safe even if already ingested)
  2. Naive RAG on a general question
  3. Hybrid Search on a citation-heavy question ("what does section 4.2b say")
  4. Agentic RAG on a decomposition question (needs 2 topics)
  5. Self-RAG verification catching an ungrounded / off-topic query
  6. Self-RAG verification applied to a recalled semantic-memory fact
"""

from __future__ import annotations

from rag import agentic_rag, hybrid_rag, ingest, naive_rag, self_rag
from rag.vector_store import VectorStore


def _print_result(label: str, result) -> None:
    print(f"\n--- {label} ({result.architecture}) ---")
    print("Answer:", result.answer)
    print("Retrieved chunks:", [c.chunk_id for c in result.chunks])


def main() -> None:
    print("=== 1. Ingest ===")
    store = ingest.run()

    print("\n=== 2. Naive RAG — general question ===")
    q1 = "How long before refueling must a generator cool down?"
    r1 = naive_rag.answer(store, q1)
    _print_result(q1, r1)

    print("\n=== 3. Hybrid Search — citation-heavy question ===")
    q2 = "What does section 4.2b say?"
    r2 = hybrid_rag.answer(store, q2)
    _print_result(q2, r2)

    print("\n=== 4. Agentic RAG — decomposition question ===")
    q3 = (
        "What protective system does excavation require for a trench, and what "
        "lockout/tagout procedure does electrical work require, when both happen "
        "in the same trench?"
    )
    r3 = agentic_rag.answer(store, q3)
    _print_result(q3, r3)
    print(f"Hops taken: {r3.hops}")

    print("\n=== 5. Self-RAG verification — RAG answer ===")
    v1 = self_rag.check_rag_answer(store.conn, q3, r3.chunks, r3.answer)
    print(f"relevance_pass={v1.relevance_pass} ({v1.relevance_reason})")
    print(f"support_pass={v1.support_pass} ({v1.support_reason})")
    print(f"final_action={v1.final_action}")

    print("\n=== 5b. Self-RAG verification — an unrelated query (should NOT pass relevance) ===")
    off_topic_query = "What is the office dress code for administrative staff?"
    off_chunks = hybrid_rag.answer(store, off_topic_query).chunks
    v2 = self_rag.check_rag_answer(
        store.conn, off_topic_query, off_chunks,
        "The office dress code requires business casual attire.",
    )
    print(f"relevance_pass={v2.relevance_pass} ({v2.relevance_reason})")
    print(f"final_action={v2.final_action}")

    print("\n=== 6. Self-RAG verification — recalled semantic-memory fact ===")
    recalled_fact = "cert_status:CRANE for worker Sara = expired 2025-06-01, valid_until 2025-06-01"
    recall_query = "Is Sara certified to operate the crane today?"
    v3 = self_rag.check_semantic_recall(store.conn, recall_query, recalled_fact)
    print(f"relevance_pass={v3.relevance_pass}, support_pass={v3.support_pass}, "
          f"final_action={v3.final_action}")

    print("\nAll self_rag_log rows:")
    for row in store.conn.execute(
        "SELECT log_id, source_type, final_action FROM self_rag_log ORDER BY log_id"
    ):
        print(" ", dict(row))


if __name__ == "__main__":
    main()
