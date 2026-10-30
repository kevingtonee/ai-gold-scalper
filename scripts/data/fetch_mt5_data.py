"""Fetch historical OHLCV data from MetaTrader 5 for model training.

Usage:
    python scripts/data/fetch_mt5_data.py --symbol XAUUSD --timeframe M5 --count 100000
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass


import pandas as pd

try:
    from integration.mt5_integration import MT5Integration
except Exception:  # pragma: no cover - import path fallback
    from scripts.integration.mt5_integration import MT5Integration


TIMEFRAME_MAP = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}


def fetch_mt5_data(symbol: str, timeframe: str, count: int, output: Path) -> Path:
    if timeframe not in TIMEFRAME_MAP:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    import MetaTrader5 as mt5  # type: ignore

    tf_attr = TIMEFRAME_MAP[timeframe]
    tf_value = getattr(mt5, tf_attr)

    login = os.environ.get("MT5_LOGIN")
    password = os.environ.get("MT5_PASSWORD")
    server = os.environ.get("MT5_SERVER")

    mt5_client = MT5Integration()
    status = mt5_client.connect(
        login=int(login) if login else None,
        password=password,
        server=server,
    )
    if not status.get("connected"):
        raise RuntimeError(f"MT5 connection failed: {status.get('error')}")

    try:
        # Ensure the symbol is visible in Market Watch
        try:
            mt5.symbol_select(symbol, True)
        except Exception:
            pass
        info = mt5_client.get_symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"Symbol not found in MT5: {symbol}")

        rates_chunks = []
        remaining = count
        start_pos = 0
        chunk_size = 50000
        while remaining > 0:
            this_count = min(chunk_size, remaining)
            chunk = mt5.copy_rates_from_pos(symbol, tf_value, start_pos, this_count)
            if chunk is None or len(chunk) == 0:
                break
            rates_chunks.append(chunk)
            fetched = len(chunk)
            remaining -= fetched
            start_pos += fetched
            if fetched < this_count:
                break

        if not rates_chunks:
            err = None
            try:
                err = mt5.last_error()
            except Exception:
                err = None
            raise RuntimeError(f"No rates returned from MT5 for {symbol} {timeframe}. last_error={err}")

        import numpy as np
        rates = np.concatenate(rates_chunks) if len(rates_chunks) > 1 else rates_chunks[0]


        frame = pd.DataFrame(rates)
        # MT5 rates use 'time' (epoch seconds)
        if "time" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["time"], unit="s")
        elif "timestamp" not in frame.columns:
            raise RuntimeError("MT5 rates missing time column")

        frame = frame.rename(columns={"tick_volume": "volume"}) if "tick_volume" in frame.columns else frame
        if "volume" not in frame.columns:
            frame["volume"] = frame.get("real_volume", 0)

        frame = frame[["timestamp", "open", "high", "low", "close", "volume"]]
        frame = frame.sort_values("timestamp").reset_index(drop=True)

        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        return output
    finally:
        mt5_client.disconnect()


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch MT5 OHLCV data for training")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M5", choices=sorted(TIMEFRAME_MAP))
    parser.add_argument("--count", type=int, default=100000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output:
        output = Path(args.output)
    else:
        output = Path("data") / f"ohlc_{args.symbol}_{args.timeframe}.csv"

    path = fetch_mt5_data(args.symbol, args.timeframe, args.count, output)
    print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
