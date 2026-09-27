"""Ingest completed one-minute candles from Binance.US's public stream."""

import json
import time
from contextlib import closing

import websocket

from server import db


STREAM_URL = "wss://stream.binance.us:9443/ws/{symbol}@kline_1m"
MAX_ATTEMPTS = 5


def parse_message(raw_json):
    """Pure parser: return a completed candle, or None for an open candle."""
    kline = json.loads(raw_json)["k"]
    if kline["x"] is not True:
        return None
    return {
        "open_time": int(kline["t"]),
        "open": float(kline["o"]),
        "high": float(kline["h"]),
        "low": float(kline["l"]),
        "close": float(kline["c"]),
        "volume": float(kline["v"]),
        "close_time": int(kline["T"]),
        "quote_volume": float(kline["q"]),
        "trades": int(kline["n"]),
    }


def handle_message(raw_json, symbol="BTCUSDT", db_path=db.DEFAULT_DB_PATH):
    """Store only a closed candle; return its open time, or None if unfinished."""
    candle = parse_message(raw_json)
    if candle is None:
        return None
    db.init_db(db_path)
    db.upsert_candles(symbol.upper(), [candle], db_path)
    return candle["open_time"]


def run_live(symbol="BTCUSDT", db_path=db.DEFAULT_DB_PATH, max_candles=None):
    """Store closed candles until the limit, returning the number processed.

    Retry transport failures with 1, 2, 4, 8 second delays, raising after
    five failed connections. A stored candle resets the failure budget.
    Consecutive replays after reconnect do not count toward the limit.
    """
    if max_candles is not None and (
        isinstance(max_candles, bool)
        or not isinstance(max_candles, int)
        or max_candles < 0
    ):
        raise ValueError("max_candles must be a nonnegative integer or None")
    db.init_db(db_path)
    stored = 0
    failures = 0
    latest_counted = None
    url = STREAM_URL.format(symbol=symbol.lower())
    while max_candles is None or stored < max_candles:
        try:
            with closing(websocket.create_connection(url, timeout=20)) as stream:
                while max_candles is None or stored < max_candles:
                    raw = stream.recv()
                    if not raw:
                        raise websocket.WebSocketConnectionClosedException(
                            "Binance.US closed the stream"
                        )
                    open_time = handle_message(raw, symbol, db_path)
                    if open_time is not None:
                        if latest_counted is None or open_time > latest_counted:
                            stored += 1
                            latest_counted = open_time
                            failures = 0
        except (websocket.WebSocketException, OSError) as exc:
            failures += 1
            if failures >= MAX_ATTEMPTS:
                raise RuntimeError(
                    f"Binance.US stream failed after {MAX_ATTEMPTS} attempts"
                ) from exc
            time.sleep(2 ** (failures - 1))
    return stored
