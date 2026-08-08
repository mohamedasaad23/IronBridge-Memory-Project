"""
rag/agentic_rag.py

Multi-hop reasoning loop for decomposition-style questions that need two
topics combined (e.g. "what do I need for both excavation work and
electrical work on the same trench").

Hop 1: normal hybrid retrieval (rag/hybrid_rag.py) on the raw query.
Hop 2+: for any topic implied by the query but NOT yet represented among
the hop-1 chunks, issue a follow-up retrieval that goes straight to
VectorStore.search(..., topic_filter=<topic>) — i.e. the metadata
pre-filter, not a diluted "query + topic keyword" embedding. This is the
fix noted in the session log: an earlier version tried to steer the
second hop by appending topic words onto the query text before
re-embedding, which diluted the query vector and often still missed the
second topic. Going straight to the metadata pre-filter instead
guarantees the second topic's chunks are actually retrieved.

Topic detection is a simple keyword match against store.topics()
(stdlib only, deterministic) — good enough for a fixed 12-question eval
set of known domain topics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from rag.embeddings import embed
from rag.generation import generate_answer
from rag.hybrid_rag import answer as hybrid_answer
from rag.vector_store import SearchResult, VectorStore

MAX_HOPS = 3
PER_HOP_K = 2

# maps topic -> a few surface keywords, used only for deciding which
# *additional* topic to pre-filter on for a follow-up hop.
_TOPIC_KEYWORDS = {
    "crane_operations": ["crane", "rigging", "lift", "boom", "load chart"],
    "electrical_safety": ["electrical", "lockout", "tagout", "voltage", "arc flash", "ground"],
    "excavation": ["excavation", "trench", "trenching", "dig", "soil", "shoring"],
    "fall_protection_ppe": ["fall protection", "harness", "anchor", "ppe", "lanyard"],
    "generator_fuel": ["generator", "fuel", "diesel", "refuel", "spill"],
}


@dataclass
class RagResult:
    answer: str
    chunks: List[SearchResult]
    architecture: str = "agentic_rag"
    hops: int = 1


def _topics_implied_by_query(query: str, known_topics: List[str]) -> List[str]:
    q = query.lower()
    implied = []
    for topic in known_topics:
        for kw in _TOPIC_KEYWORDS.get(topic, [topic.replace("_", " ")]):
            if kw in q:
                implied.append(topic)
                break
    return implied


def answer(store: VectorStore, query: str, top_k: int = 4) -> RagResult:
    hop1 = hybrid_answer(store, query, top_k=top_k)
    all_chunks: List[SearchResult] = list(hop1.chunks)
    seen_ids = {c.chunk_id for c in all_chunks}
    hops = 1

    known_topics = store.topics()
    implied_topics = _topics_implied_by_query(query, known_topics)
    covered_topics = {c.topic for c in all_chunks}
    missing_topics = [t for t in implied_topics if t not in covered_topics]

    query_vec = embed(query)
    for topic in missing_topics:
        if hops >= MAX_HOPS:
            break
        hops += 1
        # Follow-up hop: go straight to the metadata pre-filter for the
        # missing topic, rather than re-embedding a topic-diluted query.
        follow_up = store.search(query_vec, top_k=PER_HOP_K, topic_filter=topic)
        for c in follow_up:
            if c.chunk_id not in seen_ids:
                all_chunks.append(c)
                seen_ids.add(c.chunk_id)

    ans = generate_answer(query, all_chunks)
    return RagResult(answer=ans, chunks=all_chunks, architecture="agentic_rag", hops=hops)
