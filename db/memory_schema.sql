PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS episodic_memory (
    id              INTEGER PRIMARY KEY,
    worker_id       INTEGER REFERENCES workers(id),
    event_summary   TEXT NOT NULL,          -- what happened
    context         TEXT,                   -- surrounding situation
    outcome         TEXT,                   -- result / decision made
    consolidated    INTEGER NOT NULL DEFAULT 0,  -- 0/1, set by consolidation pass
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS semantic_memory (
    id              INTEGER PRIMARY KEY,
    worker_id       INTEGER REFERENCES workers(id),
    fact_key        TEXT NOT NULL,          -- e.g. "cert_status:CRANE"
    fact_value      TEXT NOT NULL,          -- e.g. "valid_until=2027-06-30"
    version         INTEGER NOT NULL DEFAULT 1,
    valid_from      TEXT NOT NULL DEFAULT (datetime('now')),
    valid_until     TEXT,                   -- NULL = currently active
    superseded_by   INTEGER REFERENCES semantic_memory(id),
    source_episode_ids TEXT,                -- JSON list of episodic_memory.id that produced this fact
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_semantic_active
    ON semantic_memory(worker_id, fact_key, valid_until);

CREATE TABLE IF NOT EXISTS memory_routing_log (
    id              INTEGER PRIMARY KEY,
    worker_id       INTEGER REFERENCES workers(id),
    source_item     TEXT NOT NULL,          -- the STM item being evicted
    decision        TEXT NOT NULL,          -- forget | episodic
    reasoning       TEXT NOT NULL,
    episodic_id     INTEGER REFERENCES episodic_memory(id),  -- set if decision=episodic
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS consolidation_log (
    id              INTEGER PRIMARY KEY,
    worker_id       INTEGER REFERENCES workers(id),
    fact_key        TEXT NOT NULL,
    action          TEXT NOT NULL,          -- create | update | expire | conflict_resolved
    old_semantic_id INTEGER REFERENCES semantic_memory(id),
    new_semantic_id INTEGER REFERENCES semantic_memory(id),
    reasoning       TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);