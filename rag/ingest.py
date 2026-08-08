"""
rag/ingest.py

Idempotent, rerunnable pipeline: chunk corpus -> embed -> upsert into
rag_chunks (existing ironbridge.db) -> (re)build the ANN index.

Run with: python -m rag.ingest
"""

from __future__ import annotations

import sys

from rag.chunking import chunk_corpus
from rag.embeddings import embed, embedding_mode
from rag.vector_store import VectorStore


def run(verbose: bool = True) -> VectorStore:
    """Chunk -> embed -> upsert -> build ANN index.

    Progress lines go to stderr, not stdout. mcp_server/server.py calls
    this at import time on the *stdio* MCP transport, where stdout is the
    JSON-RPC wire — anything printed there corrupts the protocol stream
    (the client tries to parse each line as a JSON-RPC message and fails).
    stderr is safe for both the CLI (`python -m rag.ingest`, where stderr
    shows in the same terminal) and the server (where stderr is just logs).
    """
    def _log(msg: str) -> None:
        if verbose:
            print(msg, file=sys.stderr)

    chunks = chunk_corpus()
    _log(f"Chunked {len(chunks)} sections from rag/corpus/*.md")

    store = VectorStore()
    pairs = [(c, embed(c.text)) for c in chunks]
    store.upsert_chunks(pairs)
    dim = len(pairs[0][1]) if pairs else 0
    _log(f"Embedded {len(pairs)} chunks (dim={dim}, mode={embedding_mode()})")

    store.build_ann_index()
    _log(
        f"Vector store built: rag/index/ ({store.ann_backend()}) + "
        f"db/ironbridge.db:rag_chunks"
    )
    return store


if __name__ == "__main__":
    run()
