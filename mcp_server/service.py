"""Business logic — authorization, certification checks, request creation."""

from __future__ import annotations
import os
from datetime import date
from typing import Any, Optional
from . import db

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
# Never hard-code credentials in source. Set SUPERVISOR_PIN in your local
# .env file (see .env.example) — .env is already in .gitignore.
SUPERVISOR_PIN = os.environ.get("SUPERVISOR_PIN")
if not SUPERVISOR_PIN:
    raise RuntimeError(
        "SUPERVISOR_PIN is not set. Copy .env.example to .env and set a PIN "
        "before starting the server."
    )

def get_worker(worker_id: int) -> Optional[dict[str, Any]]:
    return db.fetch_one("SELECT * FROM workers WHERE id = ?", (worker_id,))

def get_equipment(equipment_id: int) -> Optional[dict[str, Any]]:
    return db.fetch_one("SELECT * FROM equipment WHERE id = ?", (equipment_id,))

def get_site(site_id: int) -> Optional[dict[str, Any]]:
    return db.fetch_one("SELECT * FROM sites WHERE id = ?", (site_id,))

def check_certification(worker_id: int, equipment_type: str) -> dict[str, Any]:
    row = db.fetch_one(
        "SELECT * FROM certifications WHERE worker_id = ? AND equipment_type = ?",
        (worker_id, equipment_type),
    )
    if not row:
        return {"valid": False, "reason": "No certification found for this type"}
    if row["valid_until"] < date.today().isoformat():
        return {"valid": False, "reason": f"Certification expired on {row['valid_until']}"}
    return {"valid": True, "valid_until": row["valid_until"]}

def is_high_risk(equipment: dict, site: dict) -> bool:
    return bool(equipment.get("high_risk")) or bool(site.get("near_power_lines"))

def create_request(
    worker_id: int,
    equipment_id: int,
    site_id: int,
    status: str = "PENDING_SUPERVISOR_APPROVAL",
    risk_summary: Optional[str] = None,
) -> int:
    rid = db.execute(
        """INSERT INTO equipment_requests
           (worker_id, equipment_id, site_id, status, risk_summary)
           VALUES (?, ?, ?, ?, ?)""",
        (worker_id, equipment_id, site_id, status, risk_summary),
    )
    db.execute(
        "INSERT INTO audit_log (request_id, action, detail) VALUES (?, ?, ?)",
        (rid, "CREATED", f"status={status}"),
    )
    return rid

def update_request_status(request_id: int, status: str, notes: Optional[str] = None) -> None:
    db.execute(
        "UPDATE equipment_requests SET status = ? WHERE id = ?",
        (status, request_id),
    )
    db.execute(
        "INSERT INTO audit_log (request_id, action, detail) VALUES (?, ?, ?)",
        (request_id, status, notes or ""),
    )

def list_requests_for_site(site_id: int) -> list[dict[str, Any]]:
    return db.fetch_all(
        "SELECT * FROM equipment_requests WHERE site_id = ? ORDER BY id",
        (site_id,),
    )

def verify_supervisor_pin(worker_id: int, pin: str) -> bool:
    worker = get_worker(worker_id)
    if not worker or worker["role"] != "supervisor":
        return False
    return pin == SUPERVISOR_PIN
