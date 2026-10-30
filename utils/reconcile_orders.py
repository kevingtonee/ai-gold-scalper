"""Manual reconciliation helper for uncertain or pending orders."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from integration.order_journal import OrderJournal
from integration.mt5_integration import MT5Integration


def _connect_mt5_if_configured():
    login = os.environ.get("MT5_LOGIN")
    password = os.environ.get("MT5_PASSWORD")
    server = os.environ.get("MT5_SERVER")
    if not login or not password or not server:
        return None
    mt5 = MT5Integration()
    status = mt5.connect(login=int(login), password=password, server=server)
    if not status.get("connected"):
        print(f"MT5 connection failed: {status.get('error')}")
        mt5.disconnect()
        return None
    return mt5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Reconcile all orders, not only uncertain ones.")
    args = parser.parse_args()

    journal = OrderJournal()
    if args.all:
        orders = journal.list_orders(limit=1000)
    else:
        orders = journal.list_uncertain()

    print(f"Found {len(orders)} order(s) to reconcile.")
    if not orders:
        return 0

    mt5 = _connect_mt5_if_configured()
    try:
        for order in orders:
            signal_id = order["signal_id"]
            result = journal.reconcile(mt5, signal_id)
            updated = journal.get_order(signal_id)
            print(
                f"{signal_id}: mode={order['mode']} "
                f"reconcile={result.get('status')} "
                f"journal_status={updated['status'] if updated else 'missing'}"
            )
    finally:
        if mt5 is not None:
            mt5.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
