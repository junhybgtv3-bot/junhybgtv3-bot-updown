"""Backfill candles from Binance's public REST endpoint."""

import time

import requests

from server import db


# Binance.US: same klines API shape as Binance.com, but reachable from US
# (api.binance.com returns HTTP 451 for US-based IPs).
KLINES_URL = "https://api.binance.us/api/v3/klines"


def fetch_klines(symbol="BTCUSDT", interval="1m", limit=1000, *,
                 start_ms=None, end_ms=None, max_retries=5):
    """Fetch recent candles, including Binance's current unfinished candle.

    Transient failures (timeouts, resets, HTTP 429/418) are retried with
    exponential backoff so a long backfill survives a single bad request.
    """
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms is not None:
        params["startTime"] = start_ms
    if end_ms is not None:
        params["endTime"] = end_ms
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(
                KLINES_URL,
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            last_exc = exc
            if attempt == max_retries:
                raise RuntimeError(
                    f"Binance klines request failed for {symbol}: {exc}"
                ) from exc
            wait = 2 ** (attempt + 1)
            print(f"klines request failed ({exc}), retrying in {wait}s "
                  f"(attempt {attempt + 1}/{max_retries})", flush=True)
            time.sleep(wait)

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


def backfill_range(symbol="BTCUSDT", interval="1m", *, start_ms, end_ms,
                   db_path=db.DEFAULT_DB_PATH):
    """Upsert an inclusive open-time range; return rows processed, not new rows.

    Each page is committed independently, so rerunning after a failed request
    is safe. Progress is printed every 20 requests.
    """
    if start_ms < 0 or end_ms < start_ms:
        raise ValueError("Expected 0 <= start_ms <= end_ms")
    db.init_db(db_path)
    cursor, total, requests_made = start_ms, 0, 0
    while cursor <= end_ms:
        rows = fetch_klines(symbol, interval, 1000,
                            start_ms=cursor, end_ms=end_ms)
        requests_made += 1
        if not rows:
            break
        times = [row["open_time"] for row in rows]
        if (times[0] < cursor or times[-1] > end_ms
                or any(a >= b for a, b in zip(times, times[1:]))):
            raise RuntimeError("Binance returned unordered or out-of-range candles")
        db.upsert_candles(symbol, rows, db_path)
        total += len(rows)
        if requests_made % 20 == 0:
            print(f"Backfill: {requests_made} requests, {total:,} rows upserted",
                  flush=True)
        cursor = times[-1] + 1
        if cursor <= end_ms:
            time.sleep(0.3)
    return total
