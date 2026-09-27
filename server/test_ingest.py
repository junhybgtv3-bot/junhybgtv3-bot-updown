"""Offline pagination and idempotence checks."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests

from server import db, ingest


def raw_candle(stamp):
    return [stamp, "100", "101", "99", "100", "2", stamp + 59999, "200", 3]


class BackfillTests(unittest.TestCase):
    @patch("server.ingest.time.sleep")
    @patch("server.ingest.requests.get")
    def test_pagination_bounds_and_idempotence(self, get, sleep):
        def respond(url, params, timeout):
            self.assertEqual(url, ingest.KLINES_URL)
            self.assertEqual(params["endTime"], 120000)
            self.assertEqual(params["limit"], 1000)
            stamps = {0: [0, 60000], 60001: [120000]}[params["startTime"]]
            return Mock(json=lambda: [raw_candle(s) for s in stamps])
        get.side_effect = respond
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candles.db"
            for _ in range(2):
                self.assertEqual(ingest.backfill_range(
                    start_ms=0, end_ms=120000, db_path=path), 3)
                self.assertEqual(db.count_candles("BTCUSDT", path), 3)
            self.assertEqual([r["open_time"] for r in db.get_candles(
                "BTCUSDT", 60000, 120000, path)], [60000, 120000])
        self.assertEqual(sleep.call_count, 2)

    @patch("server.ingest.requests.get")
    def test_empty_and_nonadvancing_page(self, get):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candles.db"
            get.return_value = Mock(json=lambda: [])
            self.assertEqual(ingest.backfill_range(
                start_ms=60000, end_ms=120000, db_path=path), 0)
            get.return_value = Mock(json=lambda: [raw_candle(0)])
            with self.assertRaisesRegex(RuntimeError, "out-of-range"):
                ingest.backfill_range(start_ms=60000, end_ms=120000, db_path=path)

    @patch("server.ingest.requests.get")
    def test_http_failure_is_not_silent(self, get):
        get.return_value.raise_for_status.side_effect = requests.HTTPError("503")
        with self.assertRaisesRegex(RuntimeError, "request failed"):
            ingest.fetch_klines()


if __name__ == "__main__":
    unittest.main()
