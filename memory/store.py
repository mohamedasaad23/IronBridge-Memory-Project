"""
Persistence layer for long-term memory.

Reuses the SAME ironbridge.db the MCP server already writes to (see
mcp_server/db.py) — this is an extension of the existing database, not
a parallel one. memory_schema.sql just adds new tables to it.

Write ownership is enforced here, not just by convention:
  - insert_episodic()      -> called only from memory/router.py
  - insert_routing_log()   -> called only from memory/router.py
  - upsert_semantic_fact() -> called only from memory/consolidation.py
  - insert_consolidation_log() -> called only from memory/consolidation.py
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

DB_PATH = Path(__file__).resolve().parent.parent / "db" / "ironbridge.db"
MEMORY_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "memory_schema.sql"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def ensure_memory_schema() -> None:
    """Idempotent — safe to call on every startup. Applies memory_schema.sql
    (CREATE TABLE IF NOT EXISTS ...) against the existing ironbridge.db."""
    with get_conn() as conn:
        conn.executescript(MEMORY_SCHEMA_PATH.read_text(encoding="utf-8"))
        conn.commit()


# =====================================================================
# EPISODIC MEMORY  (written only by memory/router.py)
# =====================================================================
def insert_episodic(
    worker_id: Optional[int],
    event_summary: str,
    context: Optional[str],
    outcome: Optional[str],
) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO episodic_memory (worker_id, event_summary, context, outcome)
               VALUES (?, ?, ?, ?)""",
            (worker_id, event_summary, context, outcome),
        )
        conn.commit()
        return cur.lastrowid or 0


def get_unconsolidated_episodes(worker_id: int) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM episodic_memory
               WHERE worker_id = ? AND consolidated = 0
               ORDER BY created_at""",
            (worker_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def mark_episodes_consolidated(episode_ids: list[int]) -> None:
    if not episode_ids:
        return
    with get_conn() as conn:
        qmarks = ",".join("?" * len(episode_ids))
        conn.execute(
            f"UPDATE episodic_memory SET consolidated = 1 WHERE id IN ({qmarks})",
            episode_ids,
        )
        conn.commit()


# =====================================================================
# ROUTING LOG  (written only by memory/router.py)
# =====================================================================
def insert_routing_log(
    worker_id: Optional[int],
    source_item: str,
    decision: str,
    reasoning: str,
    episodic_id: Optional[int] = None,
) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO memory_routing_log
               (worker_id, source_item, decision, reasoning, episodic_id)
               VALUES (?, ?, ?, ?, ?)""",
            (worker_id, source_item, decision, reasoning, episodic_id),
        )
        conn.commit()
        return cur.lastrowid or 0


# =====================================================================
# SEMANTIC MEMORY  (written only by memory/consolidation.py)
# =====================================================================
def get_active_fact(worker_id: int, fact_key: str) -> Optional[dict[str, Any]]:
    """The one currently-valid row for (worker_id, fact_key), if any."""
    with get_conn() as conn:
        row = conn.execute(
            """SELECT * FROM semantic_memory
               WHERE worker_id = ? AND fact_key = ? AND valid_until IS NULL
               ORDER BY version DESC LIMIT 1""",
            (worker_id, fact_key),
        ).fetchone()
        return dict(row) if row else None


def get_fact_history(worker_id: int, fact_key: str) -> list[dict[str, Any]]:
    """Full version history — old facts are never deleted, only closed out."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM semantic_memory
               WHERE worker_id = ? AND fact_key = ?
               ORDER BY version""",
            (worker_id, fact_key),
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_active_facts(worker_id: int) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM semantic_memory
               WHERE worker_id = ? AND valid_until IS NULL
               ORDER BY fact_key""",
            (worker_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def close_fact(semantic_id: int, superseded_by: Optional[int] = None) -> None:
    """Ends a fact's validity window. Never deletes the row — this is the
    versioning guarantee: an old fact is closed, not erased."""
    with get_conn() as conn:
        conn.execute(
            """UPDATE semantic_memory
               SET valid_until = datetime('now'), superseded_by = ?
               WHERE id = ?""",
            (superseded_by, semantic_id),
        )
        conn.commit()


def insert_semantic_fact(
    worker_id: int,
    fact_key: str,
    fact_value: str,
    version: int,
    source_episode_ids: list[int],
) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO semantic_memory
               (worker_id, fact_key, fact_value, version, source_episode_ids)
               VALUES (?, ?, ?, ?, ?)""",
            (worker_id, fact_key, fact_value, version, json.dumps(source_episode_ids)),
        )
        conn.commit()
        return cur.lastrowid or 0


def insert_consolidation_log(
    worker_id: int,
    fact_key: str,
    action: str,
    reasoning: str,
    old_semantic_id: Optional[int] = None,
    new_semantic_id: Optional[int] = None,
) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO consolidation_log
               (worker_id, fact_key, action, old_semantic_id, new_semantic_id, reasoning)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (worker_id, fact_key, action, old_semantic_id, new_semantic_id, reasoning),
        )
        conn.commit()
        return cur.lastrowid or 0