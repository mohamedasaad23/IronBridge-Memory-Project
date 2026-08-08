"""
rag/bm25_index.py

Keyword scorer, used by hybrid_rag.py alongside vector search.

Why this exists at all: embeddings.py's hashed bag-of-words fallback (and
even real Gemini embeddings) don't represent short structured identifiers
like "4.2b" distinctively — a query like "what does section 4.2b say"
scores every electrical_safety chunk almost identically on cosine
similarity, because the section number contributes almost nothing to the
embedding's direction. BM25's exact-token matching picks the literal
"4.2b" token match instead, which is exactly what a citation-heavy query
needs.

Uses rank_bm25 if installed; otherwise a small deterministic BM25
implementation (same fallback philosophy as the rest of rag/).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    from rank_bm25 import BM25Okapi  # type: ignore
    _HAS_RANK_BM25 = True
except ImportError:
    _HAS_RANK_BM25 = False

# Dotted section identifiers like "4.1" or "4.2b" must tokenize as ONE
# token, or they collide with unrelated chunks that merely contain the
# digit "4" (e.g. doc id MAN-4). The dotted-id alternative is tried first
# so it wins over the plain-digit alternative at the same position.
_TOKEN_RE = re.compile(r"\d+\.\d+[a-z]?|[a-z0-9']+")

# Common query scaffolding words ("what does X say", "summarize Y section")
# carry almost no discriminating signal on a corpus this small, but their
# low document frequency across only 5 files still gives them a nonzero
# IDF that can outweigh a single exact section-id match. Stripping them
# lets the actual content/identifier tokens decide the ranking.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "what", "does", "do", "did", "how", "when", "where", "which", "who",
    "say", "says", "said", "summarize", "summarise", "require", "requires",
    "required", "requiring", "covered", "cover", "covers", "about", "for",
    "to", "of", "in", "on", "and", "or", "if", "both", "each", "also",
    "must", "need", "needs", "applies", "apply", "section", "manual",
}


def _tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


@dataclass
class _SimpleBM25:
    """Minimal, dependency-free BM25 (Okapi variant), k1=1.5, b=0.75."""

    corpus_tokens: List[List[str]]
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self):
        self.doc_lens = [len(doc) for doc in self.corpus_tokens]
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0.0
        self.df: Dict[str, int] = {}
        for doc in self.corpus_tokens:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1
        n = len(self.corpus_tokens)
        self.idf: Dict[str, float] = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in self.df.items()
        }

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = []
        for doc, dl in zip(self.corpus_tokens, self.doc_lens):
            tf: Dict[str, int] = {}
            for t in doc:
                tf[t] = tf.get(t, 0) + 1
            score = 0.0
            for q in query_tokens:
                if q not in tf:
                    continue
                idf = self.idf.get(q, 0.0)
                freq = tf[q]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                score += idf * (freq * (self.k1 + 1)) / (denom or 1)
            scores.append(score)
        return scores


class BM25Index:
    def __init__(self, chunk_ids: List[str], texts: List[str]):
        self.chunk_ids = chunk_ids
        tokenized = [_tokenize(t) for t in texts]
        # Boost exact section-id matches: append the raw section token
        # (already part of `text` via chunking.py's heading line for MAN-4
        # style docs, but section_id itself may not appear in body text,
        # so callers pass it in via texts already prefixed — see hybrid_rag.py).
        self._impl = BM25Okapi(tokenized) if _HAS_RANK_BM25 else _SimpleBM25(tokenized)

    def backend(self) -> str:
        return "rank_bm25" if _HAS_RANK_BM25 else "simple-bm25-fallback"

    def score(self, query: str) -> List[Tuple[str, float]]:
        q_tokens = _tokenize(query)
        scores = self._impl.get_scores(q_tokens)
        return list(zip(self.chunk_ids, scores))
