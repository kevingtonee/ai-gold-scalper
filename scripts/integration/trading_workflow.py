"""Risk-aware trading workflow integrating RiskManager and MT5Integration."""
from __future__ import annotations

from typing import Any, Dict, Optional

try:
    from core.enhanced_ai_server_consolidated import RiskManager
except Exception:  # pragma: no cover - import path fallback
    try:
        from enhanced_ai_server_consolidated import RiskManager
    except Exception:
        RiskManager = None  # type: ignore


class TradingWorkflowError(Exception):
    pass


class TradingWorkflow:
    """Coordinates risk sizing, MT5 validation, and execution."""

    def __init__(self, mt5_integration, risk_manager, journal=None):
        self.mt5 = mt5_integration
        self.risk_manager = risk_manager
        self.journal = journal

    def run(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: Optional[float] = None,
        allow_live: bool = False,
        signal_id: Optional[str] = None,
        spread_limit_points: Optional[float] = None,
    ) -> Dict[str, Any]:
        if RiskManager is None:
            raise TradingWorkflowError("RiskManager is unavailable")

        if self.journal is not None and signal_id and self.journal.is_duplicate(signal_id):
            return {
                "status": "duplicate_signal",
                "risk": None,
                "execution": None,
                "signal_id": signal_id,
            }

        mode = "live" if allow_live and getattr(self.mt5, "allow_live_trading", False) else "paper"

        mt5_info = self.mt5.get_symbol_info(symbol)
        required = ['contract_size', 'tick_size', 'volume_min', 'volume_max',
                    'volume_step', 'stop_level_min_points']
        symbol_specs = None
        if isinstance(mt5_info, dict):
            symbol_specs = {
                "contract_size": mt5_info.get("trade_contract_size", mt5_info.get("contract_size")),
                "tick_size": mt5_info.get("trade_tick_size", mt5_info.get("tick_size")),
                "tick_value": mt5_info.get("trade_tick_value", mt5_info.get("tick_value", 0.0)),
                "volume_min": mt5_info.get("volume_min"),
                "volume_max": mt5_info.get("volume_max"),
                "volume_step": mt5_info.get("volume_step"),
                "stop_level_min_points": mt5_info.get(
                    "trade_stops_level", mt5_info.get("stop_level_min_points", 0.0)
                ),
            }
        if isinstance(symbol_specs, dict) and all(k in symbol_specs and symbol_specs[k] is not None for k in required):
            specs_map = {symbol: symbol_specs}
        else:
            specs_map = getattr(self.risk_manager, "symbol_specs", {}) or {}

        # Use a request-scoped risk manager if the caller's manager is shared.
        request_risk_manager = self.risk_manager
        try:
            request_risk_manager.symbol_specs = specs_map or request_risk_manager.symbol_specs
        except Exception:
            pass

        risk_result = request_risk_manager.calculate_position_size(
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            symbol=symbol,
            symbol_specs=specs_map.get(symbol) if isinstance(specs_map, dict) else None,
            side=side,
        )

        if not risk_result:
            if self.journal is not None and signal_id:
                self.journal.record_intent(
                    signal_id=signal_id,
                    symbol=symbol,
                    side=side,
                    volume=0.0,
                    entry_price=entry_price,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price,
                    mode=mode,
                )
                self.journal.update_status(signal_id, "risk_rejected")
            return {
                "status": "risk_rejected",
                "risk": None,
                "execution": None,
            }

        volume = float(risk_result["volume"])
        if self.journal is not None and signal_id:
            self.journal.record_intent(
                signal_id=signal_id,
                symbol=symbol,
                side=side,
                volume=volume,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                mode=mode,
            )

        validation = self.mt5.validate_order(
            symbol=symbol,
            volume=volume,
            side=side,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            spread_limit_points=spread_limit_points,
        )
        if not validation["valid"]:
            if self.journal is not None and signal_id:
                self.journal.update_status(
                    signal_id,
                    "broker_validation_rejected",
                    error="; ".join(validation["errors"]),
                )
            return {
                "status": "broker_validation_rejected",
                "risk": risk_result,
                "execution": None,
                "errors": validation["errors"],
            }

        execution = self.mt5.execute_trade(
            symbol=symbol,
            side=side,
            volume=volume,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            allow_live=allow_live,
            signal_id=signal_id,
        )

        if self.journal is not None and signal_id:
            status = execution.get("status", "unknown")
            journal_status = status
            if status == "rejected":
                journal_status = "rejected"
            elif status == "not_connected":
                journal_status = "uncertain"
            elif status == "error":
                journal_status = "uncertain"
            self.journal.update_status(
                signal_id,
                journal_status,
                retcode=execution.get("retcode"),
                broker_order_id=str(execution.get("order")) if execution.get("order") is not None else None,
                error=execution.get("error"),
            )

        return {
            "status": execution.get("status", "unknown"),
            "risk": risk_result,
            "execution": execution,
        }
