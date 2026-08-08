"""
retrieval_eval/run_eval.py

Runs all 3 retrieval architectures against the fixed 12-question set,
scores accuracy (all expected_chunk_ids retrieved among top_k), and
reports approx tokens and latency.

Run with: python -m retrieval_eval.run_eval
"""

from __future__ import annotations

import time
from typing import Callable, List

from rag import agentic_rag, hybrid_rag, naive_rag
from rag.vector_store import VectorStore
from retrieval_eval.test_questions import QUESTIONS, EvalQuestion

ARCHS: List[tuple] = [
    ("naive_rag", naive_rag.answer),
    ("hybrid_search", hybrid_rag.answer),
    ("agentic_rag", agentic_rag.answer),
]


def _approx_tokens(text: str) -> int:
    # Simple deterministic approximation (whitespace-split word count),
    # good enough for relative comparison across architectures without
    # requiring a real tokenizer dependency.
    return len(text.split())


def score_question(q: EvalQuestion, retrieved_ids: List[str]) -> bool:
    return all(cid in retrieved_ids for cid in q.expected_chunk_ids)


def run_eval() -> dict:
    store = VectorStore()
    results = {}

    for name, fn in ARCHS:
        by_category = {"general": [0, 0], "citation": [0, 0], "decomposition": [0, 0]}
        correct = 0
        total_tokens = 0
        total_latency_ms = 0.0

        for q in QUESTIONS:
            start = time.perf_counter()
            rag_result = fn(store, q.query)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            retrieved_ids = [c.chunk_id for c in rag_result.chunks]
            ok = score_question(q, retrieved_ids)
            correct += int(ok)
            by_category[q.category][1] += 1
            by_category[q.category][0] += int(ok)

            total_tokens += _approx_tokens(rag_result.answer) + sum(
                _approx_tokens(c.text) for c in rag_result.chunks
            )
            total_latency_ms += elapsed_ms

        n = len(QUESTIONS)
        results[name] = {
            "accuracy": f"{correct}/{n}",
            "avg_tokens": round(total_tokens / n, 1),
            "avg_latency_ms": round(total_latency_ms / n, 2),
            "by_category": {
                cat: f"{c}/{t}" for cat, (c, t) in by_category.items()
            },
        }
    return results


def print_report(results: dict) -> None:
    print(f"{'Architecture':<15} {'Accuracy':<10} {'AvgTokens':<11} {'AvgLatency(ms)':<15} By category")
    for name, r in results.items():
        cats = ", ".join(f"{k} {v}" for k, v in r["by_category"].items())
        print(f"{name:<15} {r['accuracy']:<10} {r['avg_tokens']:<11} {r['avg_latency_ms']:<15} {cats}")


if __name__ == "__main__":
    report = run_eval()
    print_report(report)
