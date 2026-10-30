"""Tests for risk-aware trading workflow."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest

from integration.trading_workflow import TradingWorkflow
from core.enhanced_ai_server_consolidated import RiskManager


class FakeMT5:
    def __init__(self, symbol_info=None, validation=None, execution=None):
        self._symbol_info = symbol_info
        self._validation = validation or {"valid": True, "errors": []}
        self._execution = execution or {"status": "paper", "executed": False, "live": False}

    def get_symbol_info(self, symbol):
        return self._symbol_info

    def validate_order(self, **kwargs):
        return self._validation

    def execute_trade(self, **kwargs):
        return self._execution


def _risk_manager(balance=10000.0):
    cfg = {
        "account_balance": balance,
        "account_currency": "USD",
        "risk_percent_per_trade": 0.01,
        "max_daily_loss_percent": 0.05,
        "max_drawdown_percent": 0.20,
        "max_positions": 5,
    }
    specs = {
        "XAUUSD": {
            "contract_size": 100.0,
            "tick_size": 0.01,
            "tick_value": 1.0,
            "volume_min": 0.01,
            "volume_max": 100.0,
            "volume_step": 0.01,
            "stop_level_min_points": 10.0,
        }
    }
    return RiskManager(cfg, specs)


def test_risk_rejected():
    mt5 = FakeMT5()
    rm = _risk_manager(balance=0.0)
    wf = TradingWorkflow(mt5, rm)
    result = wf.run("XAUUSD", "BUY", 2000.0, 1990.0)
    assert result["status"] == "risk_rejected"
    assert result["risk"] is None


def test_broker_validation_rejected():
    mt5 = FakeMT5(validation={"valid": False, "errors": ["spread exceeds configured limit"]})
    rm = _risk_manager()
    wf = TradingWorkflow(mt5, rm)
    result = wf.run("XAUUSD", "BUY", 2000.0, 1990.0)
    assert result["status"] == "broker_validation_rejected"
    assert result["errors"] == ["spread exceeds configured limit"]


def test_paper_execution_success():
    mt5 = FakeMT5(
        symbol_info={
            "volume_min": 0.01,
            "volume_max": 100.0,
            "volume_step": 0.01,
        },
        execution={"status": "paper", "executed": False, "live": False},
    )
    rm = _risk_manager()
    wf = TradingWorkflow(mt5, rm)
    result = wf.run("XAUUSD", "BUY", 2000.0, 1990.0, signal_id="wf-1")
    assert result["status"] == "paper"
    assert result["risk"] is not None
    assert result["risk"]["volume"] == 0.1
    assert result["execution"]["live"] is False
