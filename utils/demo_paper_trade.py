"""End-to-end DEMO paper-mode workflow. No live orders are placed."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from core.enhanced_ai_server_consolidated import RiskManager
from integration.mt5_integration import MT5Integration
from integration.order_journal import OrderJournal
from integration.trading_workflow import TradingWorkflow


def main():
    login = os.environ.get("MT5_LOGIN")
    password = os.environ.get("MT5_PASSWORD")
    server = os.environ.get("MT5_SERVER")
    if not login or not password or not server:
        print("Set MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER first.")
        return 1

    os.environ.setdefault("MT5_ALLOW_LIVE_TRADING", "false")

    journal = OrderJournal()
    mt5 = MT5Integration(allow_live_trading=False)

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
        stop_loss_price = round(ask - 1.0, 2)  # 100-point stop for demo paper test

        risk_manager = RiskManager(
            {
                "account_balance": 500.0,
                "account_currency": "USD",
                "risk_percent_per_trade": 0.01,
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
            signal_id="demo-paper-1",
            spread_limit_points=1000,
        )

        print("Entry:", entry_price)
        print("Stop:", stop_loss_price)
        print("Result:", result)
        print("Journal:", journal.get_order("demo-paper-1"))
        return 0
    finally:
        mt5.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
