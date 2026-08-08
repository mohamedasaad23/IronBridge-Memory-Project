"""Database helpers — parameterized queries only."""

from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Any, Optional

DB_PATH = Path(__file__).resolve().parent.parent / "db" / "ironbridge.db"

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def fetch_one(sql: str, params: tuple = ()) -> Optional[dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(sql, params).fetchone()
        return dict(row) if row else None

def fetch_all(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

def execute(sql: str, params: tuple = ()) -> int:
    with get_conn() as conn:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur.lastrowid or 0
