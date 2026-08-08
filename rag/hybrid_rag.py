"""
rag/hybrid_rag.py

Vector search + BM25 keyword search, fused via Reciprocal Rank Fusion
(RRF). This is what makes citation-heavy queries reliable: BM25 catches
the literal section-id token ("4.2b") that the embedding fallback can't
represent distinctively (see rag/bm25_index.py docstring).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from rag.bm25_index import BM25Index
from rag.embeddings import embed
from rag.generation import generate_answer
from rag.vector_store import SearchResult, VectorStore

RRF_K = 60  # standard RRF damping constant


@dataclass
class RagResult:
    answer: str
    chunks: List[SearchResult]
    architecture: str = "hybrid_search"


def _build_bm25(store: VectorStore) -> BM25Index:
    all_chunks = store.get_all_chunks()
    # Prefix section_id into the indexed text so exact citation tokens
    # like "4.2b" are matchable even though they may not appear verbatim
    # in the body prose.
    texts = [f"{c.section_id} {c.heading} {c.text}" for c in all_chunks]
    ids = [c.chunk_id for c in all_chunks]
    return BM25Index(ids, texts)


def answer(store: VectorStore, query: str, top_k: int = 3, fetch_k: int = 8) -> RagResult:
    query_vec = embed(query)
    vector_hits = store.search(query_vec, top_k=fetch_k)
    vector_rank = {r.chunk_id: i for i, r in enumerate(vector_hits)}

    bm25 = _build_bm25(store)
    # Only rank chunks with a genuine positive BM25 score. Without this
    # filter, the many chunks tied at score 0.0 still receive sequential
    # ranks from sorted(), so RRF (which is rank-based, not score-based)
    # lets a zero-relevance chunk that merely also appears in the vector
    # hits outrank a chunk with a single, overwhelming exact keyword match
    # (e.g. an exact "4.2b" section-id hit).
    bm25_scores = sorted(
        [(cid, s) for cid, s in bm25.score(query) if s > 0.0],
        key=lambda t: t[1], reverse=True,
    )[:fetch_k]
    bm25_rank = {cid: i for i, (cid, _score) in enumerate(bm25_scores)}

    # Sorted, not a bare set: Python randomizes string-hash order per
    # process, so iterating a set here made tie-breaking (and therefore
    # the eval's top_k cutoff) different on every run — the same fixed
    # question set could score differently run to run. Deterministic
    # ordering going into the sort, plus an explicit (score, chunk_id)
    # tie-break on the sort itself, makes this reproducible.
    all_ids = sorted(set(vector_rank) | set(bm25_rank))
    fused = []
    for cid in all_ids:
        rrf = 0.0
        if cid in vector_rank:
            rrf += 1.0 / (RRF_K + vector_rank[cid] + 1)
        if cid in bm25_rank:
            rrf += 1.0 / (RRF_K + bm25_rank[cid] + 1)
        fused.append((cid, rrf))
    fused.sort(key=lambda t: (-t[1], t[0]))
    top_ids = [cid for cid, _ in fused[:top_k]]

    chunk_by_id = {c.chunk_id: c for c in store.get_all_chunks()}
    top_chunks = [chunk_by_id[cid] for cid in top_ids if cid in chunk_by_id]

    ans = generate_answer(query, top_chunks)
    return RagResult(answer=ans, chunks=top_chunks, architecture="hybrid_search")
