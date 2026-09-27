"""Offline numerical and causality checks: python -m unittest server.test_features."""

import copy
import unittest

import numpy as np
import pandas as pd

from server.features import compute_features, feature_names


def candles(prices):
    return [dict(open_time=i * 60000, open=float(p), high=float(p + 2),
                 low=float(p - 2), close=float(p), volume=float(i + 1))
            for i, p in enumerate(prices)]


class FeatureTests(unittest.TestCase):
    def test_returns_and_schema(self):
        rows = candles(range(100, 900, 10))
        result = compute_features(rows)
        self.assertEqual(len(result), len(rows))
        self.assertEqual(list(result.columns), feature_names())
        self.assertTrue(np.isnan(result.return_1m.iloc[0]))
        for n in (1, 5, 15, 60):
            self.assertTrue(result[f"return_{n}m"].iloc[:n].isna().all())
            for i in range(n, len(rows)):
                self.assertEqual(result[f"return_{n}m"].iloc[i],
                                 rows[i]["close"] / rows[i - n]["close"] - 1)
        for n in (7, 14):
            self.assertTrue(result[f"rsi_{n}"].iloc[:n].isna().all())
            self.assertTrue(result[f"rsi_{n}"].dropna().eq(100).all())

    def test_flat_prices(self):
        result = compute_features(candles([100] * 80))
        for col in ("ema9_ratio", "ema21_ratio"):
            np.testing.assert_allclose(result[col].dropna(), 1)
        for col in ("ema9_21_diff", "macd_line", "macd_signal", "macd_hist",
                    "vol_std_5m", "vol_std_15m"):
            np.testing.assert_allclose(result[col].dropna(), 0)
        for col in ("rsi_7", "rsi_14"):
            np.testing.assert_allclose(result[col].dropna(), 50)
        np.testing.assert_allclose(result.atr_14.dropna(), 4)

    def test_wilder_seed_and_recurrence(self):
        prices = [100, 102, 101, 104, 102, 106, 103, 108, 106]
        result = compute_features(candles(prices))
        self.assertAlmostEqual(result.rsi_7.iloc[7], 100 * 14 / 20)
        self.assertAlmostEqual(result.rsi_7.iloc[8], 100 * 12 / (12 + 50 / 7))
        rising = compute_features(candles(range(100, 400, 10)))
        self.assertAlmostEqual(rising.atr_14.iloc[13], (4 + 13 * 12) / 14)
        self.assertAlmostEqual(rising.atr_14.iloc[14], ((4 + 13 * 12) / 14 * 13 + 12) / 14)
        falling = compute_features(candles(range(400, 100, -10)))
        self.assertTrue(falling.rsi_7.dropna().eq(0).all())

    def test_volume_and_log_volatility(self):
        rows = candles(range(100, 900, 10))
        result = compute_features(rows)
        self.assertEqual(result.volume_change_5m.iloc[5], 6 / 3 - 1)
        self.assertAlmostEqual(result.volume_zscore_20m.iloc[19],
                               (20 - 10.5) / np.std(np.arange(1, 21), ddof=1))
        self.assertAlmostEqual(result.vol_std_5m.iloc[5],
                               np.std(np.diff(np.log(np.arange(100, 151, 10))), ddof=1))
        for row in rows:
            row["volume"] = 0
        result = compute_features(rows)
        self.assertTrue(result[["volume_change_5m", "volume_zscore_20m"]].isna().all().all())

    def test_no_lookahead_and_no_mutation(self):
        rows = candles(200 + np.sin(np.arange(90)) * 10)
        original = copy.deepcopy(rows)
        baseline = compute_features(rows)
        self.assertEqual(rows, original)
        rows[-1]["close"] += 20
        changed = compute_features(rows)
        pd.testing.assert_frame_equal(baseline.iloc[:-1], changed.iloc[:-1], check_exact=True)
        for size in (1, 7, 14, 26, 34, 60):
            pd.testing.assert_frame_equal(baseline.iloc[:size], compute_features(original[:size]),
                                          check_exact=True)
        for col in ("rsi_7", "rsi_14"):
            self.assertTrue(baseline[col].dropna().between(0, 100).all())

    def test_warmups_and_empty(self):
        result = compute_features(candles(range(100, 900, 10)))
        starts = {"ema9_ratio": 8, "ema21_ratio": 20, "ema9_21_diff": 20,
                  "macd_line": 25, "macd_signal": 33, "macd_hist": 33,
                  "atr_14": 13, "vol_std_5m": 5, "vol_std_15m": 15,
                  "volume_change_5m": 5, "volume_zscore_20m": 19}
        for name, start in starts.items():
            self.assertEqual(result[name].first_valid_index(), start, name)
        empty = compute_features([])
        self.assertEqual(len(empty), 0)
        self.assertEqual(list(empty.columns), feature_names())

    def test_invalid_input(self):
        for rows in ([{}], candles([0]), candles([np.nan]), candles([100, 110])[::-1]):
            with self.assertRaises(ValueError):
                compute_features(rows)


if __name__ == "__main__":
    unittest.main()
