"""Small SQLite-backed risk state store for the AI Gold Scalper server."""
import os
import sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any


class RiskStateStore:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.environ.get("RISK_STATE_DB", "data/risk_state.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    trading_day TEXT NOT NULL,
                    open_positions INTEGER NOT NULL DEFAULT 0,
                    daily_loss_usd REAL NOT NULL DEFAULT 0.0,
                    halted INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                )
                """
            )
            cur = conn.execute("SELECT COUNT(*) FROM risk_state")
            if cur.fetchone()[0] == 0:
                conn.execute(
                    "INSERT INTO risk_state (id, trading_day, open_positions, daily_loss_usd, halted, updated_at) VALUES (1, ?, 0, 0.0, 0, ?)",
                    (date.today().isoformat(), datetime.now().isoformat()),
                )
            conn.commit()
        finally:
            conn.close()

    def _rollover_if_new_day(self, conn):
        today = date.today().isoformat()
        row = conn.execute("SELECT trading_day FROM risk_state WHERE id = 1").fetchone()
        if row and row[0] != today:
            conn.execute(
                "UPDATE risk_state SET trading_day = ?, daily_loss_usd = 0.0, halted = 0, updated_at = ? WHERE id = 1",
                (today, datetime.now().isoformat()),
            )
            conn.commit()

    def load(self) -> Dict[str, Any]:
        conn = self._connect()
        try:
            self._rollover_if_new_day(conn)
            row = conn.execute(
                "SELECT trading_day, open_positions, daily_loss_usd, halted FROM risk_state WHERE id = 1"
            ).fetchone()
            return {
                "trading_day": row[0],
                "open_positions": int(row[1]),
                "daily_loss_usd": float(row[2]),
                "halted": bool(row[3]),
            }
        finally:
            conn.close()

    def save(self, state: Dict[str, Any]) -> None:
        conn = self._connect()
        try:
            self._rollover_if_new_day(conn)
            conn.execute(
                "UPDATE risk_state SET open_positions = ?, daily_loss_usd = ?, halted = ?, updated_at = ? WHERE id = 1",
                (
                    int(state.get("open_positions", 0)),
                    float(state.get("daily_loss_usd", 0.0)),
                    int(bool(state.get("halted", False))),
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
