"""
P0 Trading Safety — position sizing and risk validation tests.

No live or demo orders are placed; all execution is mocked/isolated.
"""
import pytest

from core.enhanced_ai_server_consolidated import RiskManager

SYMBOL_SPECS = {
    "contract_size": 100.0,
    "tick_size": 0.01,
    "tick_value": 1.0,
    "volume_min": 0.01,
    "volume_max": 100.0,
    "volume_step": 0.01,
    "stop_level_min_points": 10.0,
}

ACCOUNT_BALANCE = 10000.0
RISK_PCT = 0.01


def _rm(balance=ACCOUNT_BALANCE, risk_pct=RISK_PCT):
    risk_cfg = {
        'account_balance': balance,
        'account_currency': 'USD',
        'risk_percent_per_trade': risk_pct,
        'max_daily_loss_percent': 0.05,
        'max_drawdown_percent': 0.20,
        'max_positions': 5,
    }
    return RiskManager(risk_cfg, SYMBOL_SPECS)


def test_position_size_zero_balance_rejected():
    rm = _rm(balance=0.0)
    result = rm.calculate_position_size(
        entry_price=2000.0, stop_loss_price=1990.0, symbol="XAUUSD",
        symbol_specs=SYMBOL_SPECS,
    )
    assert result is None


def test_position_size_negative_balance_rejected():
    rm = _rm(balance=-100.0)
    result = rm.calculate_position_size(
        entry_price=2000.0, stop_loss_price=1990.0, symbol="XAUUSD",
        symbol_specs=SYMBOL_SPECS,
    )
    assert result is None


def test_position_size_non_finite_balance_rejected():
    rm = _rm(balance=float("nan"))
    result = rm.calculate_position_size(
        entry_price=2000.0, stop_loss_price=1990.0, symbol="XAUUSD",
        symbol_specs=SYMBOL_SPECS,
    )
    assert result is None


def test_position_size_zero_stop_distance_rejected():
    rm = _rm()
    result = rm.calculate_position_size(
        entry_price=2000.0, stop_loss_price=2000.0, symbol="XAUUSD",
        symbol_specs=SYMBOL_SPECS,
    )
    assert result is None


def test_position_size_negative_stop_distance_rejected():
    rm = _rm()
    result = rm.calculate_position_size(
        entry_price=2000.0, stop_loss_price=2010.0, symbol="XAUUSD",
        symbol_specs=SYMBOL_SPECS,
    )
    assert result is None


def test_position_size_non_finite_prices_rejected():
    rm = _rm()
    for entry, sl in [(float("nan"), 1990.0), (2000.0, float("inf")),
                      (float("inf"), 1990.0)]:
        result = rm.calculate_position_size(
            entry_price=entry, stop_loss_price=sl, symbol="XAUUSD",
            symbol_specs=SYMBOL_SPECS,
        )
        assert result is None, f"entry={entry} sl={sl}"


def test_position_size_missing_symbol_specs_rejected():
    rm = _rm()
    result = rm.calculate_position_size(
        entry_price=2000.0, stop_loss_price=1990.0, symbol="XAUUSD",
        symbol_specs=None,
    )
    assert result is None


def test_position_size_incomplete_symbol_specs_rejected():
    rm = _rm()
    incomplete = {"contract_size": 100.0}
    result = rm.calculate_position_size(
        entry_price=2000.0, stop_loss_price=1990.0, symbol="XAUUSD",
        symbol_specs=incomplete,
    )
    assert result is None


def test_position_size_broker_aware_formula():
    rm = _rm()
    result = rm.calculate_position_size(
        entry_price=2000.0, stop_loss_price=1990.0, symbol="XAUUSD",
        symbol_specs=SYMBOL_SPECS,
    )
    assert result is not None
    assert result["volume"] == 0.1
    assert result["risk_amount_usd"] == 100.0
    assert result["stop_distance_price"] == 10.0


def test_position_size_not_balance_divided_by_price():
    rm = _rm()
    result = rm.calculate_position_size(
        entry_price=2000.0, stop_loss_price=1990.0, symbol="XAUUSD",
        symbol_specs=SYMBOL_SPECS,
    )
    assert result["volume"] != pytest.approx(0.05)
    assert result["volume"] == 0.1


def test_position_size_below_min_rounded_up_rejected():
    rm = _rm(risk_pct=0.000001)
    result = rm.calculate_position_size(
        entry_price=2000.0, stop_loss_price=1990.0, symbol="XAUUSD",
        symbol_specs=SYMBOL_SPECS,
    )
    assert result is None


def test_position_size_rounded_down_to_step():
    rm = _rm()
    result = rm.calculate_position_size(
        entry_price=2000.0, stop_loss_price=1999.3, symbol="XAUUSD",
        symbol_specs=SYMBOL_SPECS,
    )
    assert result is not None
    assert result["volume"] == 1.42


def test_position_size_above_max_clamped_to_max():
    # Tight stop + large risk -> raw lots exceed volume_max -> clamp to max.
    rm = _rm(risk_pct=0.5)
    result = rm.calculate_position_size(
        entry_price=2000.0, stop_loss_price=1999.6, symbol="XAUUSD",
        symbol_specs=SYMBOL_SPECS,
    )
    assert result is not None
    assert result["volume"] == SYMBOL_SPECS["volume_max"]
    assert result["clamped"] is True


def test_position_size_stop_too_close_rejected():
    rm = _rm()
    result = rm.calculate_position_size(
        entry_price=2000.0, stop_loss_price=1999.95, symbol="XAUUSD",
        symbol_specs=SYMBOL_SPECS,
    )
    assert result is None


def test_position_size_respects_max_positions():
    rm = _rm()
    rm.open_positions = 5
    result = rm.calculate_position_size(
        entry_price=2000.0, stop_loss_price=1990.0, symbol="XAUUSD",
        symbol_specs=SYMBOL_SPECS,
    )
    assert result is None


def test_position_size_daily_loss_limit_rejected():
    rm = _rm()
    rm.daily_loss_usd = 600.0
    result = rm.calculate_position_size(
        entry_price=2000.0, stop_loss_price=1990.0, symbol="XAUUSD",
        symbol_specs=SYMBOL_SPECS,
    )
    assert result is None


def test_invalid_trade_direction_rejected():
    rm = _rm()
    result = rm.calculate_position_size(
        entry_price=2000.0, stop_loss_price=1990.0, symbol="XAUUSD",
        symbol_specs=SYMBOL_SPECS, side="INVALID",
    )
    assert result is None


def test_valid_buy_and_sell_directions():
    rm = _rm()
    # BUY: stop below entry; SELL: stop above entry
    # BUY: entry 2000, stop 1990 (below entry = valid for BUY)
    result_buy = rm.calculate_position_size(
        entry_price=2000.0, stop_loss_price=1990.0, symbol="XAUUSD",
        symbol_specs=SYMBOL_SPECS, side="BUY",
    )
    assert result_buy is not None
    assert result_buy["volume"] == 0.1
    # SELL: entry 2000, stop 2010 (above entry = valid for SELL)
    result_sell = rm.calculate_position_size(
        entry_price=2000.0, stop_loss_price=2010.0, symbol="XAUUSD",
        symbol_specs=SYMBOL_SPECS, side="SELL",
    )
    assert result_sell is not None
    assert result_sell["volume"] == 0.1


def test_default_lot_size_respects_risk_budget():
    rm = _rm(risk_pct=0.000001)
    result = rm.calculate_position_size(
        entry_price=2000.0, stop_loss_price=1990.0, symbol="XAUUSD",
        symbol_specs=SYMBOL_SPECS,
    )
    assert result is None