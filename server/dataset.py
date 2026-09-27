"""Pure dataset construction from ascending one-minute candle dictionaries.

No network or database access is performed. Input follows compute_features;
horizon_min counts candles, so callers must supply contiguous one-minute data
for a horizon measured in minutes. Outputs retain original candle row indices.
"""

from numbers import Integral

import numpy as np
import pandas as pd

from server.features import compute_features, feature_names


def build_dataset(candles: list[dict], horizon_min: int = 15,
                  threshold: float = 0.0005) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return features, integer binary labels, and open_time Series.

    Label i compares close[i + horizon_min] with close[i]. Strictly positive
    threshold crossings are UP (1), negative crossings DOWN (0); equality,
    FLAT returns, and unavailable future closes are -1 and are dropped.
    Rows with any NaN feature are also dropped. Input is never mutated.
    """
    if isinstance(horizon_min, bool) or not isinstance(horizon_min, Integral) or horizon_min <= 0:
        raise ValueError("horizon_min must be a positive integer")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    features = compute_features(candles)
    close = pd.Series([row["close"] for row in candles], dtype=float)
    returns = close.shift(-int(horizon_min)) / close - 1
    labels = pd.Series(-1, index=features.index, dtype="int64", name="label")
    labels.loc[returns > threshold] = 1
    labels.loc[returns < -threshold] = 0
    timestamps = pd.Series([row["open_time"] for row in candles],
                           index=features.index, name="open_time",
                           dtype=None if candles else "int64")
    keep = labels.ne(-1) & features.notna().all(axis=1)
    return (features.loc[keep, feature_names()].copy(),
            labels.loc[keep].copy(), timestamps.loc[keep].copy())


def time_split(X: pd.DataFrame, y: pd.Series, timestamps: pd.Series,
               train_frac: float = 0.7, val_frac: float = 0.15,
               purge: int = 0):
    """Split aligned rows chronologically, without shuffling or sorting.

    Train and validation sizes are floor(n * fraction); test gets the rest.
    Fractions must be nonnegative and sum to at most one. Empty partitions
    are allowed. Timestamps must be nonmissing and strictly increasing.
    purge drops that many trailing rows from the train and validation
    partitions, so labels computed from a forward horizon never leak across
    a partition boundary (pass purge=horizon_min when labels look ahead).
    """
    if (not np.isfinite([train_frac, val_frac]).all()
            or train_frac < 0 or val_frac < 0 or train_frac + val_frac > 1):
        raise ValueError("Split fractions must be finite, nonnegative, and sum to <= 1")
    if isinstance(purge, bool) or not isinstance(purge, Integral) or purge < 0:
        raise ValueError("purge must be a nonnegative integer")
    if not (len(X) == len(y) == len(timestamps)):
        raise ValueError("X, y, and timestamps must have equal lengths")
    if not X.index.equals(y.index) or not X.index.equals(timestamps.index):
        raise ValueError("X, y, and timestamps must have aligned indices")
    if (timestamps.isna().any() or not timestamps.is_monotonic_increasing
            or timestamps.duplicated().any()):
        raise ValueError("timestamps must be strictly increasing and nonmissing")
    train_end = int(len(X) * train_frac)
    val_end = train_end + int(len(X) * val_frac)

    def _purged(start: int, end: int) -> tuple[int, int]:
        return (start, max(start, end - purge))

    (ts, te), (vs, ve), (es, ee) = (_purged(0, train_end),
                                    _purged(train_end, val_end),
                                    (val_end, len(X)))
    return tuple((X.iloc[start:end].copy(), y.iloc[start:end].copy())
                 for start, end in ((ts, te), (vs, ve), (es, ee)))
