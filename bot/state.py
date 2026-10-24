"""Chief's memory, kept in SQLite so a crash or restart never loses positions or repeats an order."""
import json
import sqlite3
from typing import List, Optional

SCHEMA = [
    "CREATE TABLE IF NOT EXISTS account (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    """CREATE TABLE IF NOT EXISTS positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, qty REAL NOT NULL,
        entry_fill REAL NOT NULL, stop REAL NOT NULL, target REAL NOT NULL,
        entry_time TEXT NOT NULL, entry_cost REAL NOT NULL, last_checked_ts TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, entry_time TEXT NOT NULL,
        exit_time TEXT NOT NULL, entry REAL NOT NULL, exit REAL NOT NULL, qty REAL NOT NULL,
        pnl REAL NOT NULL, return_pct REAL NOT NULL, exit_reason TEXT NOT NULL)""",
    # A signal is recorded BEFORE any order is placed. If the program dies in between, the
    # trade is skipped rather than risking a duplicate order.
    "CREATE TABLE IF NOT EXISTS signals_seen (symbol TEXT NOT NULL, created_at TEXT NOT NULL, "
    "PRIMARY KEY (symbol, created_at))",
    "CREATE TABLE IF NOT EXISTS equity_log (ts TEXT NOT NULL, equity REAL NOT NULL, "
    "cash REAL NOT NULL, open_positions INTEGER NOT NULL)",
]


class Store:
    def __init__(self, conn: sqlite3.Connection, start_equity: float):
        self.conn = conn
        for statement in SCHEMA:
            conn.execute(statement)
        conn.commit()
        if self.get("cash") is None:
            self.set("cash", start_equity)
            self.set("peak", start_equity)
            self.set("prev_equity", start_equity)
            self.set("day_start_equity", start_equity)
            self.set("day", None)
            self.set("halted", False)
            self.set("last_close", {})

    # --- key/value account fields ---
    def get(self, key: str, default=None):
        row = self.conn.execute("SELECT value FROM account WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, key: str, value) -> None:
        self.conn.execute("INSERT OR REPLACE INTO account (key, value) VALUES (?, ?)",
                          (key, json.dumps(value)))
        self.conn.commit()

    # --- positions ---
    def open_positions(self) -> List[dict]:
        cur = self.conn.execute(
            "SELECT id, symbol, qty, entry_fill, stop, target, entry_time, entry_cost, last_checked_ts "
            "FROM positions ORDER BY id")
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def add_position(self, symbol, qty, entry_fill, stop, target, entry_time, entry_cost,
                     last_checked_ts) -> None:
        self.conn.execute(
            "INSERT INTO positions (symbol, qty, entry_fill, stop, target, entry_time, entry_cost, "
            "last_checked_ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (symbol, qty, entry_fill, stop, target, entry_time, entry_cost, last_checked_ts))
        self.conn.commit()

    def set_last_checked(self, position_id: int, ts: str) -> None:
        self.conn.execute("UPDATE positions SET last_checked_ts = ? WHERE id = ?", (ts, position_id))
        self.conn.commit()

    def close_position(self, position_id: int, trade: dict) -> None:
        self.conn.execute("DELETE FROM positions WHERE id = ?", (position_id,))
        self.conn.execute(
            "INSERT INTO trades (symbol, entry_time, exit_time, entry, exit, qty, pnl, return_pct, "
            "exit_reason) VALUES (:symbol, :entry_time, :exit_time, :entry, :exit, :qty, :pnl, "
            ":return_pct, :exit_reason)", trade)
        self.conn.commit()

    def trades(self) -> List[dict]:
        cur = self.conn.execute("SELECT symbol, entry_time, exit_time, entry, exit, qty, pnl, return_pct, "
                                "exit_reason FROM trades ORDER BY id")
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    # --- idempotency ---
    def signal_seen(self, symbol: str, created_at: str) -> bool:
        return self.conn.execute("SELECT 1 FROM signals_seen WHERE symbol = ? AND created_at = ?",
                                 (symbol, created_at)).fetchone() is not None

    def mark_signal(self, symbol: str, created_at: str) -> None:
        self.conn.execute("INSERT OR IGNORE INTO signals_seen (symbol, created_at) VALUES (?, ?)",
                          (symbol, created_at))
        self.conn.commit()

    def log_equity(self, ts: str, equity: float, cash: float, open_positions: int) -> None:
        self.conn.execute("INSERT INTO equity_log (ts, equity, cash, open_positions) VALUES (?, ?, ?, ?)",
                          (ts, equity, cash, open_positions))
        self.conn.commit()
