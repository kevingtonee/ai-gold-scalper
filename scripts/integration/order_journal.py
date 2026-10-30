"""SQLite-backed order journal and reconciliation helpers."""
from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class OrderJournal:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.environ.get(
            "ORDER_JOURNAL_DB", "data/order_journal.db"
        )
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT NOT NULL UNIQUE,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    volume REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss_price REAL NOT NULL,
                    take_profit_price REAL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    retcode INTEGER,
                    broker_order_id TEXT,
                    error TEXT,
                    attempts INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _now(self):
        return datetime.now().isoformat()

    def record_intent(self, signal_id, symbol, side, volume, entry_price, stop_loss_price, take_profit_price=None, mode="paper"):
        with self._lock:
            try:
                with self._connect() as conn:
                    conn.execute(
                        """
                        INSERT INTO orders (
                            signal_id, symbol, side, volume, entry_price,
                            stop_loss_price, take_profit_price, mode, status,
                            attempts, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', 1, ?, ?)
                        """,
                        (signal_id, symbol, side, float(volume), float(entry_price),
                         float(stop_loss_price),
                         None if take_profit_price is None else float(take_profit_price),
                         mode, self._now(), self._now()),
                    )
                    conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False
            except Exception:
                return False

    def is_duplicate(self, signal_id):
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT status FROM orders WHERE signal_id = ?",
                    (signal_id,),
                ).fetchone()
        if not row:
            return False
        return row["status"] in {"pending", "submitted", "filled", "uncertain"}

    def update_status(self, signal_id, status, retcode=None, broker_order_id=None, error=None, increment_attempts=False):
        with self._lock:
            with self._connect() as conn:
                if increment_attempts:
                    conn.execute(
                        """
                        UPDATE orders
                        SET status = ?, retcode = ?, broker_order_id = ?,
                            error = ?, attempts = attempts + 1, updated_at = ?
                        WHERE signal_id = ?
                        """,
                        (status, retcode, broker_order_id, error, self._now(), signal_id),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE orders
                        SET status = ?, retcode = ?, broker_order_id = ?,
                            error = ?, updated_at = ?
                        WHERE signal_id = ?
                        """,
                        (status, retcode, broker_order_id, error, self._now(), signal_id),
                    )
                conn.commit()

    def get_order(self, signal_id):
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM orders WHERE signal_id = ?",
                    (signal_id,),
                ).fetchone()
        return dict(row) if row else None

    def list_orders(self, status=None, limit=100):
        with self._lock:
            with self._connect() as conn:
                if status:
                    rows = conn.execute(
                        "SELECT * FROM orders WHERE status = ? ORDER BY id DESC LIMIT ?",
                        (status, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM orders ORDER BY id DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
        return [dict(row) for row in rows]

    def list_uncertain(self):
        return self.list_orders(status="uncertain")
    def reconcile(self, mt5_integration, signal_id):
        order = self.get_order(signal_id)
        if not order:
            return {"status": "not_found", "signal_id": signal_id}
        if order["mode"] != "live":
            return {"status": order["status"], "signal_id": signal_id, "mode": "paper"}
        if not getattr(mt5_integration, "connected", False):
            return {"status": "not_connected", "signal_id": signal_id}
        broker_order_id = order.get("broker_order_id")
        try:
            import MetaTrader5 as mt5
        except Exception:
            mt5 = None
        if mt5 is not None and broker_order_id:
            try:
                deals = mt5.history_deals_get(ticket=int(broker_order_id))
                if deals:
                    self.update_status(signal_id, "filled")
                    return {"status": "filled", "signal_id": signal_id}
            except Exception:
                pass
        return {"status": order["status"], "signal_id": signal_id}

