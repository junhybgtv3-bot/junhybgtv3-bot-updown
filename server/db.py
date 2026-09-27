"""SQLite storage for candles; keep database access behind these helpers."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path


DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "updown.db"


@contextmanager
def _connect(db_path):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def init_db(db_path=DEFAULT_DB_PATH):
    """Create the candle table if needed."""
    with _connect(db_path) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT,
                open_time INTEGER,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                close_time INTEGER,
                quote_volume REAL,
                trades INTEGER,
                PRIMARY KEY (symbol, open_time)
            )
        """)


def upsert_candles(symbol, rows, db_path=DEFAULT_DB_PATH):
    """Insert candles, replacing existing rows with the same symbol and time."""
    with _connect(db_path) as connection:
        connection.executemany("""
            INSERT OR REPLACE INTO candles (
                symbol, open_time, open, high, low, close, volume,
                close_time, quote_volume, trades
            ) VALUES (
                :symbol, :open_time, :open, :high, :low, :close, :volume,
                :close_time, :quote_volume, :trades
            )
        """, ({**row, "symbol": symbol} for row in rows))


def get_latest_open_time(symbol, db_path=DEFAULT_DB_PATH):
    """Return the latest stored open time in milliseconds, or None."""
    with _connect(db_path) as connection:
        return connection.execute(
            "SELECT MAX(open_time) FROM candles WHERE symbol = ?", (symbol,)
        ).fetchone()[0]


def count_candles(symbol, db_path=DEFAULT_DB_PATH):
    """Return the number of stored candles for a symbol."""
    with _connect(db_path) as connection:
        return connection.execute(
            "SELECT COUNT(*) FROM candles WHERE symbol = ?", (symbol,)
        ).fetchone()[0]
