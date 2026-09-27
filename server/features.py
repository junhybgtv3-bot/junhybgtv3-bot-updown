"""Causal features for ascending, contiguous one-minute candle dictionaries.

Input is a list of dictionaries containing open_time/open/high/low/close/volume;
extra fields are ignored. Prices must be positive, volume nonnegative, and all
numeric values finite. Missing minutes are not filled: horizons count candles.
Output has a fresh RangeIndex, only feature columns, and never drops rows.
Insufficient history and undefined divisions produce NaN.

EMAs use adjust=False, seeded by the first close, and are hidden until span
observations exist. MACD signal needs nine available MACD values. Wilder RSI
uses n price changes and ATR uses n true ranges (first TR is high-low), seeded
by their arithmetic mean, then updated with alpha=1/n. Flat RSI is 50; gains
without losses give 100. Rolling standard deviations use ddof=1. Volume z-score
uses the current inclusive 20-candle window; volume change uses five strictly
previous candles. Zero volume baselines or zero variance give NaN.
"""

import numpy as np
import pandas as pd


_FEATURES = (
    "return_1m", "return_5m", "return_15m", "return_60m",
    "ema9_ratio", "ema21_ratio", "ema9_21_diff", "rsi_14", "rsi_7",
    "macd_line", "macd_signal", "macd_hist", "vol_std_5m", "vol_std_15m",
    "atr_14", "volume_change_5m", "volume_zscore_20m",
)


def feature_names() -> list[str]:
    """Return the feature columns in their stable output order."""
    return list(_FEATURES)


def _wilder(values: pd.Series, period: int) -> pd.Series:
    """Smooth a series with at most a leading NaN using an SMA seed."""
    result = pd.Series(np.nan, index=values.index, dtype=float)
    start = values.first_valid_index()
    if start is None or len(values) < start + period:
        return result
    seed = start + period - 1
    result.iloc[seed] = values.iloc[start:seed + 1].mean()
    for i in range(seed + 1, len(values)):
        result.iloc[i] = (result.iloc[i - 1] * (period - 1) + values.iloc[i]) / period
    return result


def _rsi(close: pd.Series, period: int) -> pd.Series:
    change = close.diff()
    gain = _wilder(change.clip(lower=0), period)
    loss = _wilder(-change.clip(upper=0), period)
    total = gain + loss
    return (100 * gain / total.replace(0, np.nan)).mask(total.eq(0), 50.0)


def compute_features(candles: list[dict]) -> pd.DataFrame:
    """Compute features without mutating input or accessing external state.

    Raises ValueError for invalid numbers, missing required fields, or times
    that are not strictly ascending. An empty list yields an empty frame with
    the complete feature schema. See module docstring for warm-up conventions.
    """
    if not candles:
        return pd.DataFrame(columns=feature_names(), dtype=float)
    data = pd.DataFrame(candles)
    required = ["open_time", "open", "high", "low", "close", "volume"]
    if not set(required).issubset(data.columns):
        raise ValueError("Missing required candle fields")
    data = data[required].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("Candle values must be finite")
    if not data.open_time.diff().iloc[1:].gt(0).all():
        raise ValueError("open_time must be strictly ascending")
    if (data[["open", "high", "low", "close"]] <= 0).any().any() or (data.volume < 0).any():
        raise ValueError("Prices must be positive and volume nonnegative")
    close, volume = data.close, data.volume
    out = pd.DataFrame(index=data.index)
    for n in (1, 5, 15, 60):
        out[f"return_{n}m"] = close / close.shift(n) - 1
    ema9 = close.ewm(span=9, adjust=False, min_periods=9).mean()
    ema21 = close.ewm(span=21, adjust=False, min_periods=21).mean()
    out["ema9_ratio"] = close / ema9
    out["ema21_ratio"] = close / ema21
    out["ema9_21_diff"] = ema9 - ema21
    for n in (14, 7):
        out[f"rsi_{n}"] = _rsi(close, n)
    out["macd_line"] = (
        close.ewm(span=12, adjust=False, min_periods=12).mean()
        - close.ewm(span=26, adjust=False, min_periods=26).mean()
    )
    out["macd_signal"] = out.macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd_hist"] = out.macd_line - out.macd_signal
    log_return = np.log(close).diff()
    for n in (5, 15):
        out[f"vol_std_{n}m"] = log_return.rolling(n).std(ddof=1)
    true_range = pd.concat([
        data.high - data.low,
        (data.high - close.shift()).abs(),
        (data.low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    out["atr_14"] = _wilder(true_range, 14)
    out["volume_change_5m"] = volume / volume.shift().rolling(5).mean().replace(0, np.nan) - 1
    out["volume_zscore_20m"] = (
        (volume - volume.rolling(20).mean())
        / volume.rolling(20).std(ddof=1).replace(0, np.nan)
    )
    return out[feature_names()]
