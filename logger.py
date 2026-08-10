"""
logger.py  —  Audit event logger
---------------------------------
Uses __file__-based absolute path so audit_log.csv is always written
to the project folder regardless of where the terminal was launched from.
Works on Windows, Mac, and Linux.
"""

import csv
import os
from datetime import datetime

# Always write next to this file, not relative to cwd
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE  = os.path.join(_BASE_DIR, "audit_log.csv")


def log_event(row, decision: str, actions) -> None:
    """
    Append one audit record to LOG_FILE.

    Parameters
    ----------
    row      : pandas Series — latest telemetry row
    decision : str — 'APPROVED-RL' | 'DENIED' | 'AUTO-HEALING' | 'CRITICAL' etc.
    actions  : list[str] or str — corrective action(s) taken
    """
    if isinstance(actions, (list, tuple)):
        actions_str = "; ".join(str(a) for a in actions)
    else:
        actions_str = str(actions)

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            round(float(row["latency"]),     4),
            round(float(row["packet_loss"]), 4),
            round(float(row["throughput"]),  4),
            decision,
            actions_str,
        ])
        f.flush()
        os.fsync(f.fileno())
