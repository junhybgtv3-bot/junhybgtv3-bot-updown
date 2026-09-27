"""Backfill 120 days and train chronological baselines: python -m server.train."""

from datetime import datetime, timezone
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from server import db
from server.dataset import build_dataset, time_split
from server.ingest import backfill_range


def print_calibration(y, probabilities):
    """Print all ten bins, including empty bins and probability exactly 1."""
    bins = np.minimum((np.asarray(probabilities) * 10).astype(int), 9)
    labels = np.asarray(y)
    print("bin          count   mean_pred   actual_positive")
    for i in range(10):
        mask = bins == i
        count = int(mask.sum())
        predicted = f"{np.mean(probabilities[mask]):.6f}" if count else "n/a"
        actual = f"{labels[mask].mean():.6f}" if count else "n/a"
        bracket = "]" if i == 9 else ")"
        print(f"[{i / 10:.1f}, {(i + 1) / 10:.1f}{bracket} {count:7d} "
              f"{predicted:>11} {actual:>17}")


def prepare_data(*, _timings=None):
    """Return the existing splits; optional timings preserve main's reporting."""
    pipeline_start = time.perf_counter()
    # Exclude the currently open candle; exactly 120 days of completed minutes.
    end_exclusive = int(time.time() * 1000) // 60000 * 60000
    start_ms = end_exclusive - 120 * 24 * 60 * 60000
    end_ms = end_exclusive - 1
    for name, stamp in (("Start (inclusive)", start_ms),
                        ("End (exclusive)", end_exclusive)):
        print(f"{name}: {datetime.fromtimestamp(stamp / 1000, timezone.utc).isoformat()}",
              flush=True)
    rows = backfill_range(start_ms=start_ms, end_ms=end_ms,
                          db_path=db.DEFAULT_DB_PATH)
    print(f"Backfill rows upserted: {rows:,}; "
          f"time: {time.perf_counter() - pipeline_start:.2f}s", flush=True)
    candles = db.get_candles("BTCUSDT", start_ms, end_ms)
    print(f"Loaded candles: {len(candles):,}; "
          f"total BTCUSDT rows in DB: {db.count_candles('BTCUSDT'):,}", flush=True)
    if (not candles
            or any(b["open_time"] <= a["open_time"]
                   for a, b in zip(candles, candles[1:]))
            or any(c["close_time"] >= end_exclusive for c in candles)):
        raise RuntimeError(
            "Candles are empty, unordered, or include the still-open minute")
    # The exchange itself can have real holes (Binance.US returned no
    # candles for ~9h on 2026-08-31). Split into contiguous segments and
    # build the dataset per segment, so neither features nor labels ever
    # cross a gap; rows too close to a boundary are dropped by
    # build_dataset's NaN/unavailable-label filtering.
    segments, current = [], [candles[0]]
    for candle in candles[1:]:
        if candle["open_time"] - current[-1]["open_time"] == 60000:
            current.append(candle)
        else:
            segments.append(current)
            current = [candle]
    segments.append(current)
    for i, seg in enumerate(segments):
        first = datetime.fromtimestamp(seg[0]["open_time"] / 1000,
                                       timezone.utc).isoformat()
        last = datetime.fromtimestamp(seg[-1]["open_time"] / 1000,
                                      timezone.utc).isoformat()
        print(f"Segment {i}: {len(seg):,} candles, {first} -> {last}",
              flush=True)
    parts = [build_dataset(seg, horizon_min=15, threshold=0.0005)
             for seg in segments]
    X = pd.concat([part[0] for part in parts], ignore_index=True)
    y = pd.concat([part[1] for part in parts], ignore_index=True)
    timestamps = pd.concat([part[2] for part in parts], ignore_index=True)
    if len(X) == 0:
        raise RuntimeError("No usable rows after gap segmentation")
    if _timings is not None:
        _timings["training_start"] = time.perf_counter()
    splits = time_split(X, y, timestamps, purge=15)
    print(f"Dataset rows: {len(X):,}; features: {X.shape[1]}", flush=True)
    for name, (features, labels) in zip(("train", "validation", "test"), splits):
        print(f"{name}: rows={len(features):,}, mean(y)={labels.mean():.6f}", flush=True)
        if features.empty:
            raise RuntimeError(f"Empty {name} split")
    return splits


def main():
    pipeline_start = time.perf_counter()
    timings = {}
    splits = prepare_data(_timings=timings)
    training_start = timings["training_start"]
    (X_train, y_train), _, (X_test, y_test) = splits
    if y_train.nunique() != 2:
        raise RuntimeError("Training requires both classes")
    models = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "lgbm": LGBMClassifier(n_estimators=300, max_depth=5,
                               learning_rate=0.03, random_state=42, n_jobs=2,
                               verbosity=-1),
    }
    output = Path(__file__).resolve().parent.parent / "models"
    output.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        fit_start = time.perf_counter()
        model.fit(X_train, y_train)
        print(f"{name} fit time: {time.perf_counter() - fit_start:.2f}s", flush=True)
        probabilities = model.predict_proba(X_test)[:, 1]
        print(f"{name}: test accuracy={accuracy_score(y_test, probabilities >= 0.5):.6f}, "
              f"Brier={brier_score_loss(y_test, probabilities):.6f}")
        print(f"{name} test calibration:")
        print_calibration(y_test, probabilities)
        path = output / f"{name}.pkl"
        with path.open("wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved: {path} ({path.stat().st_size:,} bytes)", flush=True)
    print(f"Training time (including dataset/evaluation/save): "
          f"{time.perf_counter() - training_start:.2f}s")
    print(f"Total pipeline time: {time.perf_counter() - pipeline_start:.2f}s")


if __name__ == "__main__":
    main()
