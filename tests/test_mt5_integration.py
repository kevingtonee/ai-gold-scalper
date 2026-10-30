"""Tests for optional MT5 integration wrapper."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import types

import pytest

from integration.mt5_integration import MT5Integration, MT5_AVAILABLE


def test_package_availability_flag_is_bool():
    assert isinstance(MT5_AVAILABLE, bool)


def test_connect_without_package_returns_error(monkeypatch):
    import integration.mt5_integration as mod

    monkeypatch.setattr(mod, "MT5_AVAILABLE", False)
    monkeypatch.setattr(mod, "mt5", None)
    mt5 = MT5Integration()
    result = mt5.connect()
    assert result["connected"] is False
    assert "not installed" in result["error"]


def test_validate_order_rejects_bad_side_and_prices():
    mt5 = MT5Integration()
    result = mt5.validate_order(
        symbol="XAUUSD",
        volume=0.1,
        side="HOLD",
        entry_price=2000.0,
        stop_loss_price=1990.0,
    )
    assert result["valid"] is False
    assert any("side must be BUY or SELL" in e for e in result["errors"])


def test_validate_order_direction_rules():
    mt5 = MT5Integration()
    bad_buy = mt5.validate_order("XAUUSD", 0.1, "BUY", 2000.0, 2010.0)
    bad_sell = mt5.validate_order("XAUUSD", 0.1, "SELL", 2000.0, 1990.0)
    assert bad_buy["valid"] is False
    assert bad_sell["valid"] is False


def test_execute_trade_defaults_to_paper_mode():
    mt5 = MT5Integration(allow_live_trading=False)
    result = mt5.execute_trade(
        symbol="XAUUSD",
        side="BUY",
        volume=0.1,
        entry_price=2000.0,
        stop_loss_price=1990.0,
        signal_id="sig-1",
    )
    assert result["status"] == "paper"
    assert result["executed"] is False
    assert result["live"] is False


def test_duplicate_signal_is_rejected():
    mt5 = MT5Integration(allow_live_trading=False)
    first = mt5.execute_trade(
        "XAUUSD", "BUY", 0.1, 2000.0, 1990.0, signal_id="dup"
    )
    second = mt5.execute_trade(
        "XAUUSD", "BUY", 0.1, 2000.0, 1990.0, signal_id="dup"
    )
    assert first["status"] == "paper"
    assert second["status"] == "duplicate_signal"


def test_rejected_order_does_not_execute():
    mt5 = MT5Integration(allow_live_trading=False)
    result = mt5.execute_trade(
        "XAUUSD", "BUY", 0.1, 2000.0, 2010.0, signal_id="bad"
    )
    assert result["status"] == "rejected"
    assert result["executed"] is False


def test_live_mode_requires_connection():
    mt5 = MT5Integration(allow_live_trading=True)
    result = mt5.execute_trade(
        "XAUUSD",
        "BUY",
        0.1,
        2000.0,
        1990.0,
        allow_live=True,
        signal_id="live-1",
    )
    assert result["status"] == "not_connected"
    assert result["executed"] is False


def test_live_mode_with_mocked_mt5(monkeypatch):
    import integration.mt5_integration as mod

    fake_mt5 = types.SimpleNamespace()
    fake_mt5.ORDER_TYPE_BUY = 0
    fake_mt5.ORDER_TYPE_SELL = 1
    fake_mt5.ORDER_FILLING_FOK = 0
    fake_mt5.ORDER_FILLING_IOC = 1
    fake_mt5.ORDER_FILLING_RETURN = 2
    fake_mt5.TRADE_ACTION_DEAL = 2
    fake_mt5.ORDER_TIME_GTC = 3
    fake_mt5.ORDER_FILLING_IOC = 4

    class FakeResult:
        def _asdict(self):
            return {"retcode": 10009, "order": 123}

    def fake_order_send(request):
        return FakeResult()

    def fake_last_error():
        return "ok"

    fake_mt5.order_send = fake_order_send
    fake_mt5.last_error = fake_last_error

    monkeypatch.setattr(mod, "MT5_AVAILABLE", True)
    monkeypatch.setattr(mod, "mt5", fake_mt5)

    mt5 = MT5Integration(allow_live_trading=True)
    monkeypatch.setattr(mt5, "_connected", True)
    # Bypass symbol/tick lookups by making validation independent of MT5 data.
    monkeypatch.setattr(mt5, "get_symbol_info", lambda symbol: None)
    monkeypatch.setattr(mt5, "get_tick", lambda symbol: None)

    result = mt5.execute_trade(
        "XAUUSD",
        "BUY",
        0.1,
        2000.0,
        1990.0,
        allow_live=True,
        signal_id="live-2",
    )
    assert result["live"] is True
    assert result["executed"] is True
    assert result["status"] == "filled"
