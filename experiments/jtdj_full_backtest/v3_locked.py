from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import pandas as pd

# JTDJ V3 locked challenger
#
# Discovery:
#   - Yahoo 2020-2022 was used as the primary discovery window.
# Validation:
#   - Yahoo 2023-2026 holdout
#   - Kaggle/Stooq US stocks 2000-2017 independent historical cross-check
#
# IMPORTANT: These thresholds are now frozen. Do not tune them on future OOS data.

RS20_MIN = 0.03
EVENT_AGE_MIN = 10
EVENT_AGE_MAX = 20


def locked_filter(df: pd.DataFrame) -> pd.Series:
    return (
        (df["relative_strength_20"] > RS20_MIN)
        & (df["event_age"] >= EVENT_AGE_MIN)
        & (df["event_age"] <= EVENT_AGE_MAX)
    )


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"trades": 0}
    r = df["net_return"].astype(float)
    gross_profit = r[r > 0].sum()
    gross_loss = -r[r <= 0].sum()
    return {
        "trades": int(len(df)),
        "symbols": int(df["symbol"].nunique()),
        "win_rate": float((r > 0).mean()),
        "avg_return": float(r.mean()),
        "median_return": float(r.median()),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else None,
        "best_trade": float(r.max()),
        "worst_trade": float(r.min()),
    }


def apply_locked_filter(input_csv: str, output_csv: str | None = None) -> dict:
    df = pd.read_csv(input_csv)
    out = df.loc[locked_filter(df)].copy()
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
    return summarize(out)
