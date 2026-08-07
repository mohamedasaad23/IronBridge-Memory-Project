"""Initialize the SQLite database from schema.sql + seed_data.sql."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "ironbridge.db"
SCHEMA = Path(__file__).parent / "schema.sql"
SEED = Path(__file__).parent / "seed_data.sql"

def main() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    conn.executescript(SEED.read_text(encoding="utf-8"))
    conn.commit()
    conn.close()
    print(f"Database ready: {DB_PATH}")

if __name__ == "__main__":
    main()