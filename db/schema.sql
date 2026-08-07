-- Iron Bridge Construction — Equipment Safety Schema
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sites (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    near_power_lines INTEGER NOT NULL DEFAULT 0  -- 0/1
);

CREATE TABLE IF NOT EXISTS workers (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    role        TEXT NOT NULL DEFAULT 'worker'   -- worker | supervisor
);

CREATE TABLE IF NOT EXISTS equipment (
    id          INTEGER PRIMARY KEY,
    site_id     INTEGER NOT NULL REFERENCES sites(id),
    name        TEXT NOT NULL,
    type        TEXT NOT NULL,                   -- CRANE | EXCAVATOR | SCAFFOLD | GENERATOR
    high_risk   INTEGER NOT NULL DEFAULT 0,      -- 0/1
    status      TEXT NOT NULL DEFAULT 'AVAILABLE' -- AVAILABLE | IN_USE | MAINTENANCE
);

CREATE TABLE IF NOT EXISTS certifications (
    id              INTEGER PRIMARY KEY,
    worker_id       INTEGER NOT NULL REFERENCES workers(id),
    equipment_type  TEXT NOT NULL,
    valid_until     TEXT NOT NULL,               -- ISO date YYYY-MM-DD
    UNIQUE(worker_id, equipment_type)
);

CREATE TABLE IF NOT EXISTS equipment_requests (
    id              INTEGER PRIMARY KEY,
    worker_id       INTEGER NOT NULL REFERENCES workers(id),
    equipment_id    INTEGER NOT NULL REFERENCES equipment(id),
    site_id         INTEGER NOT NULL REFERENCES sites(id),
    status          TEXT NOT NULL DEFAULT 'PENDING_SUPERVISOR_APPROVAL',
    -- PENDING_SUPERVISOR_APPROVAL | APPROVED | REJECTED | NOT_SUBMITTED
    risk_summary    TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY,
    request_id  INTEGER REFERENCES equipment_requests(id),
    action      TEXT NOT NULL,
    detail      TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
