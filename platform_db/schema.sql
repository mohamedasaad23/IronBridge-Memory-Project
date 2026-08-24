-- Ironbridge Final Project schema (extends existing ironbridge.db concepts)
-- Checkpointing, HITL tasks, failure tickets, agent registry, RAG docs

CREATE TABLE IF NOT EXISTS graph_runs (
    run_id TEXT PRIMARY KEY,
    graph_name TEXT NOT NULL,
    worker_id TEXT,
    status TEXT NOT NULL DEFAULT 'running', -- running | paused_hitl | failed | completed | cancelled
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    input_json TEXT,
    result_json TEXT
);

CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    step_index INTEGER NOT NULL,
    state_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES graph_runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_run ON checkpoints(run_id, step_index);

CREATE TABLE IF NOT EXISTS hitl_tasks (
    task_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    title TEXT NOT NULL,
    reason TEXT NOT NULL,
    payload_json TEXT,
    status TEXT NOT NULL DEFAULT 'open', -- open | approved | rejected | cancelled
    admin_decision TEXT,
    admin_note TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    FOREIGN KEY (run_id) REFERENCES graph_runs(run_id)
);

CREATE TABLE IF NOT EXISTS failure_tickets (
    ticket_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    error_type TEXT NOT NULL,
    error_message TEXT NOT NULL,
    checkpoint_id TEXT,
    status TEXT NOT NULL DEFAULT 'open', -- open | investigating | resolved
    resolution_note TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    FOREIGN KEY (run_id) REFERENCES graph_runs(run_id),
    FOREIGN KEY (checkpoint_id) REFERENCES checkpoints(checkpoint_id)
);

CREATE TABLE IF NOT EXISTS agent_registry (
    agent_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    tools_json TEXT NOT NULL DEFAULT '[]' -- allow-list of tool names
);

CREATE TABLE IF NOT EXISTS rag_documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    topic TEXT,
    content TEXT NOT NULL,
    added_at TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS workers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    role TEXT,
    site_id TEXT,
    pin TEXT,
    is_admin INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS equipment (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    site_id TEXT,
    status TEXT DEFAULT 'operational'
);

CREATE TABLE IF NOT EXISTS certifications (
    worker_id TEXT NOT NULL,
    equipment_type TEXT NOT NULL,
    valid_until TEXT,
    status TEXT DEFAULT 'valid',
    PRIMARY KEY (worker_id, equipment_type)
);
