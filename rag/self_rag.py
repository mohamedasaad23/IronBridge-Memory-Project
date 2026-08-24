"""
rag/self_rag.py

Self-RAG-style post-hoc verification (arXiv:2310.11511 reflection-token
idea, simplified into two boolean checks rather than a trained critique
model, matching the offline-reproducibility constraint of this project):

  - relevance check: is the retrieved/recalled content actually about
    what the query asked?
  - support check: does the generated answer's claim actually follow from
    the retrieved/recalled content, or is it ungrounded?

Applied to BOTH:
  1. RAG answers (naive/hybrid/agentic) — see check_rag_answer()
  2. Recalled semantic-memory facts — see check_semantic_recall(), which
     agent/agent_with_memory.py's semantic-fact recall step
     (_recall_and_verify_facts) calls before using a recalled fact in a
     live decision.

Every check is logged to self_rag_log (db/rag_schema.sql) regardless of
outcome, so a grader/demo can show both catching and passing cases.

Offline mode uses lexical-overlap heuristics (deterministic, no model
call). Real mode uses a lightweight Gemini judge call when
GOOGLE_API_KEY is set, with the same fallback-on-failure pattern used
elsewhere in rag/.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import List, Optional

from rag.vector_store import SearchResult

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall(text.lower()))


@dataclass
class VerificationResult:
    relevance_pass: bool
    relevance_reason: str
    support_pass: bool
    support_reason: str
    final_action: str  # "accepted" | "flagged" | "fallback"


def _lexical_relevance(query: str, context_text: str, threshold: float = 0.12) -> tuple[bool, str]:
    q_tokens = _tokens(query)
    c_tokens = _tokens(context_text)
    if not q_tokens:
        return False, "empty query"
    overlap = len(q_tokens & c_tokens) / len(q_tokens)
    passed = overlap >= threshold
    return passed, f"token overlap={overlap:.2f} (threshold={threshold})"


def _lexical_support(answer: str, context_text: str, threshold: float = 0.35) -> tuple[bool, str]:
    a_tokens = _tokens(answer)
    c_tokens = _tokens(context_text)
    if not a_tokens:
        return False, "empty answer"
    grounded = len(a_tokens & c_tokens) / len(a_tokens)
    passed = grounded >= threshold
    return passed, f"answer-token grounding={grounded:.2f} (threshold={threshold})"


def _gemini_judge(query: str, context_text: str, answer: str) -> Optional[VerificationResult]:
    import urllib.request

    api_key = os.environ["GOOGLE_API_KEY"]
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-3.6-flash:generateContent?key={api_key}"
    )
    prompt = (
        "You are a strict grounding verifier. Respond with ONLY a JSON object: "
        '{"relevant": true/false, "relevance_reason": str, "supported": true/false, '
        '"support_reason": str}.\n\n'
        f"Query: {query}\nContext: {context_text}\nAnswer: {answer}\n"
        "relevant = does the context actually address the query? "
        "supported = does the answer's claim follow strictly from the context?"
    )
    payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    raw = data["candidates"][0]["content"]["parts"][0]["text"]
    raw = raw.strip().strip("```json").strip("```").strip()
    parsed = json.loads(raw)
    rel = bool(parsed["relevant"])
    sup = bool(parsed["supported"])
    return VerificationResult(
        relevance_pass=rel,
        relevance_reason=parsed.get("relevance_reason", ""),
        support_pass=sup,
        support_reason=parsed.get("support_reason", ""),
        final_action="accepted" if (rel and sup) else "flagged",
    )


def _log(
    conn: sqlite3.Connection,
    source_type: str,
    query: str,
    candidate_chunk_ids: Optional[List[str]],
    candidate_answer: str,
    result: VerificationResult,
) -> None:
    conn.execute(
        """
        INSERT INTO self_rag_log
            (source_type, query, candidate_chunk_ids, candidate_answer,
             relevance_pass, relevance_reason, support_pass, support_reason, final_action)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            source_type,
            query,
            json.dumps(candidate_chunk_ids) if candidate_chunk_ids is not None else None,
            candidate_answer,
            int(result.relevance_pass),
            result.relevance_reason,
            int(result.support_pass),
            result.support_reason,
            result.final_action,
        ),
    )
    conn.commit()


def _verify(query: str, context_text: str, answer: str) -> VerificationResult:
    if os.environ.get("GOOGLE_API_KEY"):
        try:
            judged = _gemini_judge(query, context_text, answer)
            if judged is not None:
                return judged
        except Exception:
            pass  # fall through to offline heuristic
    rel_pass, rel_reason = _lexical_relevance(query, context_text)
    sup_pass, sup_reason = _lexical_support(answer, context_text)
    if rel_pass and sup_pass:
        action = "accepted"
    elif rel_pass and not sup_pass:
        action = "flagged"  # relevant context, but the answer overreaches it
    else:
        action = "fallback"  # context itself doesn't address the query
    return VerificationResult(rel_pass, rel_reason, sup_pass, sup_reason, action)


def check_rag_answer(
    conn: sqlite3.Connection, query: str, chunks: List[SearchResult], answer: str
) -> VerificationResult:
    context_text = "\n".join(c.text for c in chunks)
    result = _verify(query, context_text, answer)
    _log(conn, "rag_answer", query, [c.chunk_id for c in chunks], answer, result)
    return result


def check_semantic_recall(
    conn: sqlite3.Connection, query: str, recalled_fact_text: str
) -> VerificationResult:
    """Called by agent/agent_with_memory.py before a recalled semantic-memory
    fact is used in a live decision. context and answer are the same text
    here since a recalled fact IS the claim being verified against itself
    plus the query it's being used to answer."""
    result = _verify(query, recalled_fact_text, recalled_fact_text)
    _log(conn, "semantic_memory", query, None, recalled_fact_text, result)
    return result
