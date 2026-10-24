"""SQLite audit trail: every proposal and the Skeptic's verdict, with the reason."""
import sqlite3
from pathlib import Path

from .models import Proposal

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL,
    symbol  TEXT NOT NULL,
    side    TEXT NOT NULL,
    entry   REAL NOT NULL,
    stop    REAL NOT NULL,
    target  REAL NOT NULL,
    verdict TEXT NOT NULL CHECK (verdict IN ('approved', 'blocked')),
    reason  TEXT NOT NULL
)
"""


def init_db(path) -> sqlite3.Connection:
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def log_decision(conn: sqlite3.Connection, p: Proposal, verdict: str, reason: str) -> None:
    conn.execute(
        "INSERT INTO decisions (ts, symbol, side, entry, stop, target, verdict, reason) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (p.created_at, p.symbol, p.side, p.entry, p.stop, p.target, verdict, reason),
    )
    conn.commit()
