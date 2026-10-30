"""AI Gold Scalper - MetaTrader 5 integration helpers.

Conservative by design: MT5 is optional, execution defaults to paper mode.
"""

from __future__ import annotations

import os
import threading
import time
import math
from typing import Any, Dict, Optional

try:
    import MetaTrader5 as mt5  # type: ignore
    MT5_AVAILABLE = True
except Exception:  # pragma: no cover
    mt5 = None  # type: ignore
    MT5_AVAILABLE = False


class MT5IntegrationError(Exception):
    pass


class MT5Integration:
    def __init__(self, allow_live_trading: Optional[bool] = None):
        if allow_live_trading is None:
            allow_live_trading = os.environ.get(
                "MT5_ALLOW_LIVE_TRADING", ""
            ).strip().lower() in {"1", "true", "yes", "on"}
        self.allow_live_trading = bool(allow_live_trading)
        self._connected = False
        self._lock = threading.RLock()
        self._recent_signal_ids: Dict[str, float] = {}
        self._duplicate_window_seconds = float(
            os.environ.get("MT5_DUPLICATE_SIGNAL_WINDOW_SECONDS", "60")
        )

    @staticmethod
    def is_package_available() -> bool:
        return MT5_AVAILABLE

    def connect(self, login=None, password=None, server=None, timeout=10000):
        if not MT5_AVAILABLE:
            return {"connected": False, "error": "MetaTrader5 package is not installed"}
        with self._lock:
            try:
                kwargs = {"timeout": timeout}
                if login is not None:
                    kwargs["login"] = login
                if password is not None:
                    kwargs["password"] = password
                if server is not None:
                    kwargs["server"] = server
                initialized = mt5.initialize(**kwargs)
                if not initialized:
                    self._connected = False
                    return {"connected": False, "error": str(mt5.last_error())}
                self._connected = True
                account = None
                try:
                    info = mt5.account_info()
                    account = info._asdict() if info else None
                except Exception:
                    account = None
                return {"connected": True, "account": account}
            except Exception as exc:
                self._connected = False
                return {"connected": False, "error": str(exc)}

    def disconnect(self):
        with self._lock:
            if MT5_AVAILABLE:
                try:
                    mt5.shutdown()
                except Exception:
                    pass
            self._connected = False

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    def get_symbol_info(self, symbol):
        if not MT5_AVAILABLE or not self._connected:
            return None
        try:
            info = mt5.symbol_info(symbol)
            return info._asdict() if info is not None else None
        except Exception:
            return None

    def get_tick(self, symbol):
        if not MT5_AVAILABLE or not self._connected:
            return None
        try:
            tick = mt5.symbol_info_tick(symbol)
            return tick._asdict() if tick is not None else None
        except Exception:
            return None

    def get_rates(self, symbol, timeframe, count=100):
        if not MT5_AVAILABLE or not self._connected or count <= 0:
            return None
        try:
            return mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        except Exception:
            return None
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.disconnect()
        return False

    def validate_order(self, symbol, volume, side, entry_price, stop_loss_price, spread_limit_points=None):
        errors = []
        if side not in {"BUY", "SELL"}:
            errors.append("side must be BUY or SELL")
        try:
            volume = float(volume)
            entry_price = float(entry_price)
            stop_loss_price = float(stop_loss_price)
        except (TypeError, ValueError):
            return {"valid": False, "errors": ["non-numeric order inputs"]}
        for name, value in (("volume", volume), ("entry_price", entry_price), ("stop_loss_price", stop_loss_price)):
            if not math.isfinite(value):
                errors.append(f"{name} must be finite")
        if entry_price <= 0 or stop_loss_price <= 0 or volume <= 0:
            errors.append("volume, entry_price, and stop_loss_price must be positive")
        if side == "BUY" and not (stop_loss_price < entry_price):
            errors.append("BUY stop loss must be below entry price")
        if side == "SELL" and not (stop_loss_price > entry_price):
            errors.append("SELL stop loss must be above entry price")
        info = self.get_symbol_info(symbol)
        if info is not None:
            volume_min = float(info.get("volume_min", 0.0) or 0.0)
            volume_max = float(info.get("volume_max", 0.0) or 0.0)
            volume_step = float(info.get("volume_step", 0.0) or 0.0)
            if volume_min > 0 and volume < volume_min:
                errors.append("volume below broker minimum")
            if volume_max > 0 and volume > volume_max:
                errors.append("volume above broker maximum")
            if volume_step > 0:
                ratio = volume / volume_step
                if abs(ratio - round(ratio)) > 1e-9:
                    errors.append("volume is not a valid broker step")
        if spread_limit_points is not None:
            tick = self.get_tick(symbol)
            if tick is not None:
                bid = float(tick.get("bid", 0.0) or 0.0)
                ask = float(tick.get("ask", 0.0) or 0.0)
                point = 0.0
                if info is not None:
                    point = float(info.get("point", 0.0) or 0.0)
                if point > 0:
                    spread_points = (ask - bid) / point
                    if spread_points > float(spread_limit_points):
                        errors.append("spread exceeds configured limit")
        return {"valid": not errors, "errors": errors}

    def _is_duplicate(self, signal_id):
        if not signal_id:
            return False
        now = time.time()
        with self._lock:
            expired = [k for k, ts in self._recent_signal_ids.items() if now - ts > self._duplicate_window_seconds]
            for key in expired:
                del self._recent_signal_ids[key]
            if signal_id in self._recent_signal_ids:
                return True
            self._recent_signal_ids[signal_id] = now
            return False

    def execute_trade(self, symbol, side, volume, entry_price, stop_loss_price, take_profit_price=None, allow_live=False, signal_id=None):
        validation = self.validate_order(symbol, volume, side, entry_price, stop_loss_price)
        if not validation["valid"]:
            return {"executed": False, "live": False, "status": "rejected", "errors": validation["errors"]}
        if self._is_duplicate(signal_id):
            return {"executed": False, "live": False, "status": "duplicate_signal", "signal_id": signal_id}
        live_requested = bool(allow_live and self.allow_live_trading)
        if not live_requested:
            return {
                "executed": False,
                "live": False,
                "status": "paper",
                "symbol": symbol,
                "side": side,
                "volume": float(volume),
                "entry_price": float(entry_price),
                "stop_loss_price": float(stop_loss_price),
                "take_profit_price": None if take_profit_price is None else float(take_profit_price),
                "signal_id": signal_id,
            }
        if not MT5_AVAILABLE or not self._connected:
            return {"executed": False, "live": False, "status": "not_connected", "error": "MT5 is not available or not connected"}
        try:
            order_type = mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL
            info = self.get_symbol_info(symbol)
            if info is not None:
                filling_mode = int(info.get("filling_mode", 0) or 0)
                if filling_mode & 1:
                    type_filling = mt5.ORDER_FILLING_IOC
                elif filling_mode & 2:
                    type_filling = mt5.ORDER_FILLING_RETURN
                else:
                    type_filling = mt5.ORDER_FILLING_FOK
            else:
                type_filling = mt5.ORDER_FILLING_IOC
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(volume),
                "type": order_type,
                "price": float(entry_price),
                "sl": float(stop_loss_price),
                "deviation": 20,
                "magic": 20250726,
                "comment": "ai-gold-scalper",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": type_filling,
            }
            if take_profit_price is not None:
                request["tp"] = float(take_profit_price)
            result = mt5.order_send(request)
            if result is None:
                return {"executed": False, "live": True, "status": "error", "error": str(mt5.last_error())}
            result_dict = result._asdict() if hasattr(result, "_asdict") else dict(result)
            executed = int(result_dict.get("retcode", -1)) == 10009
            return {
                "executed": executed,
                "live": True,
                "status": "filled" if executed else "rejected",
                "retcode": result_dict.get("retcode"),
                "order": result_dict.get("order"),
                "result": result_dict,
            }
        except Exception as exc:
            return {"executed": False, "live": True, "status": "error", "error": str(exc)}

