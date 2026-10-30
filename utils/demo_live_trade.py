"""Strictly guarded DEMO live-order test.

By default this runs in PAPER mode. Live mode requires ALL of:
- --live
- MT5_ALLOW_LIVE_TRADING=true
- MT5_LIVE_DEMO_CONFIRM=I_UNDERSTAND_THIS_PLACES_A_REAL_DEMO_ORDER
- an MT5 server name containing "Demo"
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from core.enhanced_ai_server_consolidated import RiskManager
from integration.mt5_integration import MT5Integration
from integration.order_journal import OrderJournal
from integration.trading_workflow import TradingWorkflow


def _live_allowed(args):
    if not args.live:
        return False, "paper mode selected"
    if os.environ.get("MT5_ALLOW_LIVE_TRADING", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return False, "MT5_ALLOW_LIVE_TRADING is not enabled"
    if os.environ.get("MT5_LIVE_DEMO_CONFIRM", "").strip() != "I_UNDERSTAND_THIS_PLACES_A_REAL_DEMO_ORDER":
        return False, "MT5_LIVE_DEMO_CONFIRM is missing or incorrect"
    return True, "live mode enabled"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Place a tiny real demo order. Requires all safety gates.")
    args = parser.parse_args()

    login = os.environ.get("MT5_LOGIN")
    password = os.environ.get("MT5_PASSWORD")
    server = os.environ.get("MT5_SERVER")
    if not login or not password or not server:
        print("Set MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER first.")
        return 1
    if "demo" not in server.lower():
        print("Refusing to run: server name must contain 'Demo'.")
        return 1

    live_allowed, reason = _live_allowed(args)
    allow_live = live_allowed
    mode = "live" if allow_live else "paper"
    print(f"Mode: {mode} ({reason})")

    journal = OrderJournal()
    mt5 = MT5Integration(allow_live_trading=allow_live)

    try:
        status = mt5.connect(login=int(login), password=password, server=server)
        if not status.get("connected"):
            print("MT5 connection failed:", status.get("error"))
            return 1

        symbol = "XAUUSD"
        info = mt5.get_symbol_info(symbol)
        tick = mt5.get_tick(symbol)
        if not info or not tick:
            print("Unable to read symbol/tick data.")
            return 1

        ask = float(tick["ask"])
        entry_price = ask
        stop_loss_price = round(ask - 1.0, 2)
        volume = 0.01  # Fixed tiny demo size.

        risk_manager = RiskManager(
            {
                "account_balance": 500.0,
                "account_currency": "USD",
                "risk_percent_per_trade": 0.002,
                "max_daily_loss_percent": 0.05,
                "max_drawdown_percent": 0.20,
                "max_positions": 5,
            },
            {symbol: info},
        )

        workflow = TradingWorkflow(mt5, risk_manager, journal=journal)
        result = workflow.run(
            symbol=symbol,
            side="BUY",
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=round(entry_price + 2.0, 2),
            allow_live=allow_live,
            signal_id="demo-live-guarded-1",
            spread_limit_points=1000,
        )

        print("Entry:", entry_price)
        print("Stop:", stop_loss_price)
        print("Volume:", result.get("risk", {}).get("volume"))
        print("Result:", result)
        print("Journal:", journal.get_order("demo-live-guarded-1"))
        return 0
    finally:
        mt5.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
