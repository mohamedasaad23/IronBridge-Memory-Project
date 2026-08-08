"""
rag/naive_rag.py

Baseline architecture: embed query -> vector search -> generate.
No BM25, no metadata pre-filter, no multi-hop. Strong on general
questions, weaker on citation-heavy and decomposition questions (see
retrieval_eval/run_eval.py for the comparison).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from rag.embeddings import embed
from rag.generation import generate_answer
from rag.vector_store import SearchResult, VectorStore


@dataclass
class RagResult:
    answer: str
    chunks: List[SearchResult]
    architecture: str = "naive_rag"


def answer(store: VectorStore, query: str, top_k: int = 3) -> RagResult:
    query_vec = embed(query)
    chunks = store.search(query_vec, top_k=top_k)
    ans = generate_answer(query, chunks)
    return RagResult(answer=ans, chunks=chunks, architecture="naive_rag")
