"""Backfill candles from Binance's public REST endpoint."""

import requests

from server import db


# Binance.US: same klines API shape as Binance.com, but reachable from US
# (api.binance.com returns HTTP 451 for US-based IPs).
KLINES_URL = "https://api.binance.us/api/v3/klines"


def fetch_klines(symbol="BTCUSDT", interval="1m", limit=1000):
    """Fetch recent candles, including Binance's current unfinished candle."""
    try:
        response = requests.get(
            KLINES_URL,
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Binance klines request failed for {symbol}: {exc}") from exc

    return [
        {
            "symbol": symbol,
            "open_time": int(row[0]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
            "close_time": int(row[6]),
            "quote_volume": float(row[7]),
            "trades": int(row[8]),
        }
        for row in response.json()
    ]


def backfill(symbol="BTCUSDT", interval="1m", limit=1000, db_path=db.DEFAULT_DB_PATH):
    """Fetch and upsert recent candles, returning the number of rows processed."""
    rows = fetch_klines(symbol, interval, limit)
    db.init_db(db_path)
    db.upsert_candles(symbol, rows, db_path)
    return len(rows)
