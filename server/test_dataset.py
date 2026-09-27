"""Offline dataset checks: python -m unittest server.test_dataset."""

import copy
import unittest

import pandas as pd

from server.dataset import build_dataset, time_split
from server.features import compute_features, feature_names


def candles(prices):
    return [dict(open_time=i * 60000, open=float(p), high=float(p + 1),
                 low=float(p - 1), close=float(p), volume=float(i + 1))
            for i, p in enumerate(prices)]


class DatasetTests(unittest.TestCase):
    def test_sharp_jump_exact_horizon(self):
        rows = candles([100] * 100 + [101] * 30)
        original = copy.deepcopy(rows)
        X, y, timestamps = build_dataset(rows)
        self.assertEqual(list(X.index), list(range(85, 100)))
        self.assertEqual(y.tolist(), [1] * 15)
        self.assertEqual(timestamps.tolist(), [i * 60000 for i in range(85, 100)])
        self.assertEqual(list(X.columns), feature_names())
        pd.testing.assert_frame_equal(X, compute_features(rows).loc[85:99])
        self.assertEqual(rows, original)

    def test_down_jump(self):
        X, y, _ = build_dataset(candles([100] * 100 + [99] * 30))
        self.assertEqual(list(X.index), list(range(85, 100)))
        self.assertEqual(y.tolist(), [0] * 15)

    def test_flat_and_empty(self):
        for rows in ([], candles([100] * 150), candles([100, 101])):
            X, y, timestamps = build_dataset(rows)
            self.assertEqual((len(X), len(y), len(timestamps)), (0, 0, 0))
            self.assertEqual(list(X.columns), feature_names())
            self.assertEqual(y.dtype, "int64")

    def test_warmup_tail_and_custom_horizon(self):
        rows = candles(range(100, 300))
        for horizon in (1, 15, 30, 250):
            X, y, timestamps = build_dataset(rows, horizon_min=horizon)
            self.assertEqual(list(X.index), list(range(60, max(60, 200 - horizon))))
            self.assertFalse(X.isna().any().any())
            self.assertTrue(y.eq(1).all())
            self.assertTrue(X.index.equals(y.index))
            self.assertTrue(X.index.equals(timestamps.index))

    def test_label_no_lookahead(self):
        rows = candles(range(100, 250))
        i = 80
        X, y, _ = build_dataset(rows)
        rows[i + 16]["close"] = 50
        changed_X, changed_y, _ = build_dataset(rows)
        self.assertEqual(y.loc[i], changed_y.loc[i])
        pd.testing.assert_series_equal(X.loc[i], changed_X.loc[i])
        # The exact endpoint does affect this label.
        rows[i + 15]["close"] = 50
        self.assertEqual(build_dataset(rows)[1].loc[i], 0)
        # Without the endpoint, the row cannot receive a label.
        self.assertNotIn(i, build_dataset(rows[:i + 15])[1].index)

    def test_threshold_is_strict(self):
        # Binary-exact returns avoid decimal floating-point boundary ambiguity.
        for after in (112.5, 87.5):
            rows = candles([100] * 100 + [after] * 30)
            self.assertTrue(build_dataset(rows, threshold=0.125)[0].empty)
            self.assertEqual(len(build_dataset(rows, threshold=0.124)[0]), 15)

    def test_chronological_split(self):
        X, y, timestamps = build_dataset(candles(range(100, 1178)))
        parts = time_split(X, y, timestamps)
        self.assertEqual([len(a) for a, _ in parts], [702, 150, 151])
        for a, b in parts:
            self.assertTrue(a.index.equals(b.index))
        train, val, test = [timestamps.loc[a.index] for a, _ in parts]
        self.assertLess(train.max(), val.min())
        self.assertLess(val.max(), test.min())
        pd.testing.assert_frame_equal(pd.concat([a for a, _ in parts]), X)
        pd.testing.assert_series_equal(pd.concat([b for _, b in parts]), y)

    def test_invalid_parameters_and_split(self):
        for horizon in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                build_dataset([], horizon_min=horizon)
        for threshold in (-0.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                build_dataset([], threshold=threshold)
        X, y, timestamps = build_dataset(candles(range(100, 300)))
        for train, val in ((-0.1, 0.15), (0.9, 0.2), (float("nan"), 0.1)):
            with self.assertRaises(ValueError):
                time_split(X, y, timestamps, train, val)
        for a, b, t in ((X, y.iloc[:-1], timestamps),
                        (X, y.reset_index(drop=True), timestamps),
                        (X.iloc[::-1], y.iloc[::-1], timestamps.iloc[::-1])):
            with self.assertRaises(ValueError):
                time_split(a, b, t)
        empty = build_dataset([])
        self.assertTrue(all(a.empty and b.empty for a, b in time_split(*empty)))


if __name__ == "__main__":
    unittest.main()
