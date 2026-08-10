"""
incident_manager.py
-------------------
Incident lifecycle manager. Uses __file__-based absolute path so
incident_log.csv is always in the project folder on Windows/Mac/Linux.
"""

import csv
import os
from datetime import datetime

# Absolute path — works on Windows regardless of where terminal was launched
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IR_FILE   = os.path.join(_BASE_DIR, "incident_log.csv")

HEADERS = [
    "incident_id",
    "opened_at",
    "severity",
    "root_cause",
    "status",
    "assigned_to",
    "sla_minutes",
    "resolved_at",
    "resolution_note",
    "escalated_to",
]

SLA_MAP = {"CRITICAL": 15, "HIGH": 60, "MEDIUM": 240}


# ── Public API ────────────────────────────────────────────────────────────────

def open_incident(severity: str, root_cause: str, operator: str) -> str:
    incident_id = f"INC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    row = {
        "incident_id":     incident_id,
        "opened_at":       _now(),
        "severity":        severity.upper(),
        "root_cause":      root_cause,
        "status":          "OPEN",
        "assigned_to":     operator,
        "sla_minutes":     SLA_MAP.get(severity.upper(), 60),
        "resolved_at":     "",
        "resolution_note": "",
        "escalated_to":    "",
    }
    _append_row(row)
    return incident_id


def update_status(incident_id: str, new_status: str, operator: str = "") -> bool:
    return _patch(incident_id, {"status": new_status, "assigned_to": operator or ""})


def resolve_incident(incident_id: str, note: str, operator: str = "") -> bool:
    return _patch(incident_id, {
        "status":          "RESOLVED",
        "resolved_at":     _now(),
        "resolution_note": note,
        "assigned_to":     operator or "",
    })


def escalate_incident(incident_id: str, escalate_to: str) -> bool:
    return _patch(incident_id, {
        "status":       "ESCALATED",
        "escalated_to": escalate_to,
    })


def get_all_incidents() -> list:
    return list(reversed(_read_all()))


def get_open_incidents() -> list:
    return [i for i in _read_all() if i["status"] in ("OPEN", "IN_PROGRESS", "ESCALATED")]


def get_sla_status(incident: dict) -> tuple:
    try:
        opened = datetime.strptime(incident["opened_at"], "%Y-%m-%d %H:%M:%S")
    except (ValueError, KeyError):
        return 0.0, "OK"

    if incident["status"] == "RESOLVED":
        try:
            resolved = datetime.strptime(incident["resolved_at"], "%Y-%m-%d %H:%M:%S")
            elapsed  = (resolved - opened).total_seconds() / 60.0
        except (ValueError, KeyError):
            elapsed = 0.0
    else:
        elapsed = (datetime.now() - opened).total_seconds() / 60.0

    sla = float(incident.get("sla_minutes") or 60)
    if elapsed >= sla:
        label = "BREACHED"
    elif elapsed >= sla * 0.8:
        label = "WARNING"
    else:
        label = "OK"

    return round(elapsed, 1), label


# ── Internal helpers ──────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _append_row(row: dict) -> None:
    file_exists = os.path.exists(IR_FILE)
    with open(IR_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def _read_all() -> list:
    if not os.path.exists(IR_FILE):
        return []
    with open(IR_FILE, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_all(rows: list) -> None:
    with open(IR_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def _patch(incident_id: str, updates: dict) -> bool:
    rows  = _read_all()
    found = False
    for row in rows:
        if row.get("incident_id") == incident_id:
            row.update(updates)
            found = True
            break
    if found:
        _write_all(rows)
    return found
