"""Risk state persistence tests."""
import os
import tempfile
from pathlib import Path

from core.risk_state_store import RiskStateStore


def test_risk_state_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "risk.db")
        store = RiskStateStore(db_path=db)
        store.save({"open_positions": 3, "daily_loss_usd": 12.5, "halted": True})
        loaded = store.load()
        assert loaded["open_positions"] == 3
        assert loaded["daily_loss_usd"] == 12.5
        assert loaded["halted"] is True
