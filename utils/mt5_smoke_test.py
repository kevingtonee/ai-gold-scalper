"""Manual MT5 smoke test. Credentials must come from environment variables."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from integration.mt5_integration import MT5Integration


def main():
    login = os.environ.get("MT5_LOGIN")
    password = os.environ.get("MT5_PASSWORD")
    server = os.environ.get("MT5_SERVER")

    if not login or not password or not server:
        print("Set MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER first.")
        return 1

    mt5 = MT5Integration()
    try:
        print("connect:", mt5.connect(login=int(login), password=password, server=server))
        print("symbol:", mt5.get_symbol_info("XAUUSD"))
        print("tick:", mt5.get_tick("XAUUSD"))
        return 0
    finally:
        mt5.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
