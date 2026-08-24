"""
Bridge platform chat → existing Ironbridge MCP / service layer when available.

Falls back to platform_db seed data if:
  - SUPERVISOR_PIN / DB not configured
  - mcp_server imports fail

This keeps the product UI working offline while satisfying "reuse mcp_server/db".
"""
from __future__ import annotations

from typing import Any, Optional


def try_service_context(worker_key: str) -> Optional[str]:
    """Build PERSON_DATA string from real mcp_server.service if possible."""
    try:
        from mcp_server import service
        # Map platform string ids → numeric seed ids used in ironbridge.db
        id_map = {"W1": 1, "W2": 2, "W3": 3, "ADMIN1": 99, "ADMIN2": 98}
        wid = id_map.get(worker_key)
        if wid is None:
            try:
                wid = int(worker_key)
            except ValueError:
                return None
        worker = service.get_worker(wid)
        if not worker:
            return None
        lines = [f"WORKER: {dict(worker)}"]
        for eq_type in ("CRANE", "EXCAVATOR", "SCAFFOLD"):
            try:
                cert = service.check_certification(wid, eq_type)
                lines.append(f"CERT {eq_type}: {cert}")
            except Exception:
                pass
        return "\n".join(lines)
    except Exception as e:
        return None  # silent fallback


def try_rag_snippets(query: str, top_k: int = 3) -> list[str]:
    """Use existing rag pipeline if index exists; else empty."""
    try:
        from rag.vector_store import VectorStore
        from rag.embeddings import embed_query
        vs = VectorStore()
        vec = embed_query(query)
        results = vs.search(vec, top_k=top_k)
        out = []
        for r in results:
            out.append(f"POLICY: [{getattr(r,'heading',None) or r.chunk_id}] {r.text[:300]}")
        return out
    except Exception:
        return []
