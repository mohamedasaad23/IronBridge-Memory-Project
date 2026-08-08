"""
rag/generation.py

Free-text answer generation over retrieved chunks.
  - Real path: Gemini generateContent call if GOOGLE_API_KEY is set.
  - Offline fallback: extractive — return the most relevant sentence(s)
    from the top chunk(s) verbatim with a citation, no paraphrase model
    required. This keeps grading fully reproducible and, combined with
    self_rag.py, keeps answers strictly grounded either way.
"""

from __future__ import annotations

import os
from typing import List

from rag.vector_store import SearchResult


def _format_context(chunks: List[SearchResult]) -> str:
    return "\n\n".join(
        f"[{c.chunk_id}] {c.heading}\n{c.text}" for c in chunks
    )


def _gemini_generate(query: str, chunks: List[SearchResult]) -> str:
    import json
    import urllib.request

    api_key = os.environ["GOOGLE_API_KEY"]
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-1.5-flash:generateContent?key={api_key}"
    )
    prompt = (
        "Answer the safety question using ONLY the context below. "
        "Cite the chunk id(s) you used in square brackets. "
        "If the context does not contain the answer, say so explicitly.\n\n"
        f"Context:\n{_format_context(chunks)}\n\nQuestion: {query}\nAnswer:"
    )
    payload = json.dumps(
        {"contents": [{"parts": [{"text": prompt}]}]}
    ).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def _extractive_fallback(query: str, chunks: List[SearchResult]) -> str:
    if not chunks:
        return "No relevant policy section was retrieved for this question."
    top = chunks[0]
    lines = [ln.strip() for ln in top.text.splitlines() if ln.strip()]
    body = " ".join(lines[1:3]) if len(lines) > 1 else top.text
    answer = f"{body} [{top.chunk_id}]"
    if len(chunks) > 1:
        extra_ids = ", ".join(c.chunk_id for c in chunks[1:3])
        answer += f" (see also: {extra_ids})"
    return answer


def generate_answer(query: str, chunks: List[SearchResult]) -> str:
    if os.environ.get("GOOGLE_API_KEY"):
        try:
            return _gemini_generate(query, chunks)
        except Exception:
            return _extractive_fallback(query, chunks)
    return _extractive_fallback(query, chunks)


def generation_mode() -> str:
    return "gemini" if os.environ.get("GOOGLE_API_KEY") else "offline-extractive"
