"""Tests for order journal."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest

from integration.order_journal import OrderJournal


def test_record_and_duplicate(tmp_path):
    journal = OrderJournal(db_path=str(tmp_path / "journal.db"))
    assert journal.record_intent("sig-1", "XAUUSD", "BUY", 0.1, 2000.0, 1990.0) is True
    assert journal.is_duplicate("sig-1") is True
    assert journal.record_intent("sig-1", "XAUUSD", "BUY", 0.1, 2000.0, 1990.0) is False


def test_update_and_get(tmp_path):
    journal = OrderJournal(db_path=str(tmp_path / "journal.db"))
    journal.record_intent("sig-2", "XAUUSD", "BUY", 0.1, 2000.0, 1990.0)
    journal.update_status("sig-2", "filled", retcode=10009, broker_order_id="123")
    order = journal.get_order("sig-2")
    assert order["status"] == "filled"
    assert order["retcode"] == 10009
    assert order["broker_order_id"] == "123"


def test_list_uncertain(tmp_path):
    journal = OrderJournal(db_path=str(tmp_path / "journal.db"))
    journal.record_intent("sig-3", "XAUUSD", "BUY", 0.1, 2000.0, 1990.0)
    journal.update_status("sig-3", "uncertain", error="timeout")
    uncertain = journal.list_uncertain()
    assert len(uncertain) == 1
    assert uncertain[0]["signal_id"] == "sig-3"


def test_reconcile_paper(tmp_path):
    journal = OrderJournal(db_path=str(tmp_path / "journal.db"))
    journal.record_intent("sig-4", "XAUUSD", "BUY", 0.1, 2000.0, 1990.0, mode="paper")
    result = journal.reconcile(object(), "sig-4")
    assert result["status"] == "pending"
    assert result["mode"] == "paper"
def test_reconcile_live_not_connected(tmp_path):
    journal = OrderJournal(db_path=str(tmp_path / "journal.db"))
    journal.record_intent("sig-5", "XAUUSD", "BUY", 0.1, 2000.0, 1990.0, mode="live")
    result = journal.reconcile(object(), "sig-5")
    assert result["status"] == "not_connected"

