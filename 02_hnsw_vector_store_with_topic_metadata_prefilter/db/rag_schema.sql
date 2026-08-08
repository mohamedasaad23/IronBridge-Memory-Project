-- db/rag_schema.sql
-- Extends the EXISTING ironbridge.db with RAG tables. Never creates a
-- parallel database — this is executed against the same connection that
-- memory/ and context_eval/ already use.

CREATE TABLE IF NOT EXISTS rag_chunks (
    chunk_id      TEXT PRIMARY KEY,       -- "{doc_id}:{section_id}", e.g. "MAN-4:4.2b"
    doc_id        TEXT NOT NULL,          -- e.g. "MAN-4"
    topic         TEXT NOT NULL,          -- e.g. "electrical_safety"
    section_id    TEXT NOT NULL,          -- e.g. "4.2b"
    heading       TEXT NOT NULL,
    last_reviewed TEXT NOT NULL,
    text          TEXT NOT NULL,          -- heading + body, used for embedding/BM25
    source_file   TEXT NOT NULL,
    embedding     TEXT NOT NULL           -- JSON-encoded float vector (dim=256)
);

-- Metadata index used for pre-filtering candidates by topic BEFORE
-- similarity scoring (see rag/vector_store.py::search, topic_filter).
CREATE INDEX IF NOT EXISTS idx_rag_chunks_topic ON rag_chunks (topic);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_doc_id ON rag_chunks (doc_id);

-- Self-RAG-style verification log. Applied to BOTH RAG answers and
-- recalled semantic-memory facts (see rag/self_rag.py and the
-- agent/agent_with_memory.py integration patch).
CREATE TABLE IF NOT EXISTS self_rag_log (
    log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    source_type     TEXT NOT NULL CHECK (source_type IN ('rag_answer', 'semantic_memory')),
    query           TEXT NOT NULL,
    candidate_chunk_ids TEXT,             -- JSON list, null for semantic_memory checks
    candidate_answer TEXT NOT NULL,
    relevance_pass  INTEGER NOT NULL,     -- 0/1
    relevance_reason TEXT,
    support_pass    INTEGER NOT NULL,     -- 0/1
    support_reason  TEXT,
    final_action    TEXT NOT NULL CHECK (final_action IN ('accepted', 'flagged', 'fallback'))
);

CREATE INDEX IF NOT EXISTS idx_self_rag_log_source_type ON self_rag_log (source_type);
