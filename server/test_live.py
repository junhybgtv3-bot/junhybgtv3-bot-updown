"""Offline tests and an explicitly selected, 150-second live smoke test."""

import json
import signal
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from server import db, live


def payload(closed=True, open_time=1700000000000):
    return json.dumps({"k": {
        "t": open_time, "o": "60000.1", "h": "60010.2", "l": "59990.3",
        "c": "60005.4", "v": "1.25", "T": open_time + 59999,
        "q": "75000.5", "n": 12, "x": closed,
    }})


class UnitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "candles.db"
        db.init_db(self.path)

    def test_only_closed_candle_is_stored_with_all_fields(self):
        self.assertIsNone(live.handle_message(payload(False), "BTCUSDT", self.path))
        self.assertEqual(db.count_candles("BTCUSDT", self.path), 0)
        live.handle_message(payload(), "BTCUSDT", self.path)
        live.handle_message(payload(False, 1700000060000), "BTCUSDT", self.path)
        live.handle_message(payload(), "BTCUSDT", self.path)
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute("SELECT * FROM candles").fetchall()
        self.assertEqual(rows, [("BTCUSDT", 1700000000000, 60000.1,
                                 60010.2, 59990.3, 60005.4, 1.25,
                                 1700000059999, 75000.5, 12)])

    def test_requires_boolean_true(self):
        for flag in (False, "true", 1, None):
            self.assertIsNone(live.parse_message(payload(flag)))

    @patch("server.live.time.sleep")
    @patch("server.live.websocket.create_connection")
    def test_retry_exhaustion(self, connect, sleep):
        connect.side_effect = live.websocket.WebSocketConnectionClosedException()
        with self.assertRaisesRegex(RuntimeError, "5 attempts"):
            live.run_live(db_path=self.path, max_candles=1)
        self.assertEqual(connect.call_count, 5)
        self.assertEqual(sleep.call_args_list, [call(1), call(2), call(4), call(8)])

    @patch("server.live.time.sleep")
    @patch("server.live.websocket.create_connection")
    def test_disconnect_replay_and_limit(self, connect, sleep):
        first, second = MagicMock(), MagicMock()
        first.recv.side_effect = [payload(False), payload(), ""]
        second.recv.side_effect = [payload(), payload(True, 1700000060000)]
        connect.side_effect = [first, second]
        self.assertEqual(live.run_live(db_path=self.path, max_candles=2), 2)
        self.assertEqual(db.count_candles("BTCUSDT", self.path), 2)
        first.close.assert_called_once()
        second.close.assert_called_once()
        sleep.assert_called_once_with(1)
        self.assertEqual(connect.call_args.args[0],
                         "wss://stream.binance.us:9443/ws/btcusdt@kline_1m")


class LiveSmoke(unittest.TestCase):
    def test_real_stream(self):
        def deadline(signum, frame):
            raise AssertionError("No closed candle stored within 150 seconds")

        previous = signal.signal(signal.SIGALRM, deadline)
        try:
            signal.alarm(150)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "live.db"
                self.assertEqual(live.run_live("BTCUSDT", path, max_candles=1), 1)
                self.assertEqual(db.count_candles("BTCUSDT", path), 1)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
