from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

def max_drawdown(eq: pd.Series) -> float:
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())

def perf(eq: pd.DataFrame) -> dict:
    if eq.empty:
        return {}
    s = eq.set_index("date")["equity"].astype(float)
    r = s.pct_change().fillna(0.0)
    days = (s.index[-1] - s.index[0]).days
    years = days / 365.25 if days > 0 else np.nan
    cagr = float((s.iloc[-1] / s.iloc[0]) ** (1.0 / years) - 1.0) if years and years > 0 else np.nan
    sd = float(r.std(ddof=1))
    sharpe = float(r.mean() / sd * math.sqrt(252.0)) if sd > 0 else np.nan
    return {
        "start_equity": float(s.iloc[0]),
        "end_equity": float(s.iloc[-1]),
        "total_return": float(s.iloc[-1] / s.iloc[0] - 1.0),
        "cagr": cagr,
        "sharpe": sharpe,
        "mdd": max_drawdown(s),
        "daily_vol": sd,
        "trading_days": int(len(s)),
    }

def load_locked_trades(path: str) -> pd.DataFrame:
    x = pd.read_csv(path)
    for c in ["signal_date","entry_date","exit_date"]:
        x[c] = pd.to_datetime(x[c])
    # Locked V3 thresholds. Never tune in this portfolio script.
    x = x[
        (x["relative_strength_20"] > 0.03)
        & (x["event_age"] >= 10)
        & (x["event_age"] <= 20)
    ].copy()
    x = x.sort_values(["entry_date","relative_strength_20","rr_at_entry"],
                      ascending=[True,False,False]).reset_index(drop=True)
    return x

def load_prices(root: Path, symbols: set[str], source: str) -> dict[str,pd.DataFrame]:
    out = {}
    if source == "yahoo":
        for sym in sorted(symbols):
            p = root / f"{sym.lower()}.parquet"
            if not p.exists():
                p = root / f"{sym.upper()}.parquet"
            if not p.exists():
                continue
            try:
                df = pd.read_parquet(p)
            except Exception:
                continue
            df.columns = [str(c).lower() for c in df.columns]
            if not {"date","open","close"}.issubset(df.columns):
                continue
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            for c in ["open","close"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            if "adj_close" in df.columns:
                adj = pd.to_numeric(df["adj_close"], errors="coerce")
                raw = pd.to_numeric(df["close"], errors="coerce")
                factor = (adj/raw).replace([np.inf,-np.inf],np.nan)
                factor = factor.where((factor>0)&factor.notna(),1.0)
                df["open"] = df["open"] * factor
                df["close"] = np.where(adj.notna()&(adj>0),adj,df["close"])
            df = df.dropna(subset=["date","open","close"]).drop_duplicates("date").sort_values("date")
            out[sym] = df[["date","open","close"]].set_index("date")
    else:
        lookup = {}
        for p in root.rglob("*.txt"):
            if "stocks" in [q.lower() for q in p.parts]:
                sym = p.name.lower().replace(".us.txt","").upper()
                if sym in symbols and sym not in lookup:
                    lookup[sym]=p
        for sym,p in lookup.items():
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            df.columns=[str(c).lower() for c in df.columns]
            if not {"date","open","close"}.issubset(df.columns):
                continue
            df["date"]=pd.to_datetime(df["date"],errors="coerce")
            for c in ["open","close"]:
                df[c]=pd.to_numeric(df[c],errors="coerce")
            df=df.dropna(subset=["date","open","close"]).drop_duplicates("date").sort_values("date")
            out[sym]=df[["date","open","close"]].set_index("date")
    return out

def simulate(trades: pd.DataFrame, prices: dict[str,pd.DataFrame], max_positions: int,
             initial_cash: float=100_000.0) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    start = trades["entry_date"].min()
    end = trades["exit_date"].max()
    all_dates = sorted(set().union(*[
        set(df.loc[(df.index>=start)&(df.index<=end)].index) for df in prices.values()
    ]))
    if not all_dates:
        return pd.DataFrame(),pd.DataFrame(),{}
    entries = {d:g.copy() for d,g in trades.groupby("entry_date")}
    exit_lookup = trades.set_index(["symbol","entry_date"])["exit_date"].to_dict()
    trade_lookup = trades.set_index(["symbol","entry_date"]).to_dict("index")

    cash = float(initial_cash)
    pos = {}
    ledger = []
    curve = []
    skipped = 0
    accepted = 0

    for d in all_dates:
        # Exit first at the known V2 execution price; frees slots/cash for same-day entries.
        for key,p in list(pos.items()):
            if p["exit_date"] == d:
                cash += p["shares"] * p["exit_price"]
                ledger.append({
                    "symbol":p["symbol"],"entry_date":p["entry_date"],"exit_date":d,
                    "capital":p["capital"],"pnl":p["shares"]*p["exit_price"]-p["capital"],
                    "return_on_allocated":p["exit_price"]/p["entry_price"]-1.0,
                })
                del pos[key]

        # Mark current equity at today's close before sizing new entries.
        marked = cash
        for p in pos.values():
            px = prices.get(p["symbol"])
            if px is not None and d in px.index:
                marked += p["shares"] * float(px.loc[d,"close"])
            else:
                marked += p["shares"] * p["last_close"]
        equity_pre = marked

        # Fill slots with same-day signals ranked by locked RS then RR.
        if d in entries:
            g = entries[d]
            for _,t in g.iterrows():
                if len(pos) >= max_positions:
                    skipped += 1
                    continue
                sym = str(t["symbol"])
                if any(p["symbol"] == sym for p in pos.values()):
                    skipped += 1
                    continue
                px = prices.get(sym)
                if px is None or d not in px.index:
                    skipped += 1
                    continue
                entry_price = float(t["entry"])
                exit_price = float(t["exit"])
                if entry_price <= 0 or exit_price <= 0:
                    skipped += 1
                    continue
                target_capital = equity_pre / max_positions
                capital = min(target_capital, cash)
                if capital <= 1e-9:
                    skipped += 1
                    continue
                shares = capital / entry_price
                cash -= capital
                key = f"{sym}|{d.date()}|{accepted}"
                pos[key] = {
                    "symbol":sym,"entry_date":d,"exit_date":pd.Timestamp(t["exit_date"]),
                    "entry_price":entry_price,"exit_price":exit_price,"shares":shares,
                    "capital":capital,"last_close":entry_price,
                }
                accepted += 1

        # End-of-day MTM.
        equity = cash
        for p in pos.values():
            px = prices.get(p["symbol"])
            if px is not None and d in px.index:
                p["last_close"] = float(px.loc[d,"close"])
            equity += p["shares"] * p["last_close"]
        curve.append({"date":d,"equity":equity,"cash":cash,"positions":len(pos)})

    curve = pd.DataFrame(curve)
    ledger = pd.DataFrame(ledger)
    stats = perf(curve)
    stats.update({
        "max_positions":max_positions,
        "accepted_trades":accepted,
        "skipped_signals":skipped,
        "avg_positions":float(curve["positions"].mean()) if not curve.empty else np.nan,
        "max_positions_used":int(curve["positions"].max()) if not curve.empty else 0,
        "cash_fraction_avg":float((curve["cash"]/curve["equity"]).mean()) if not curve.empty else np.nan,
    })
    return curve, ledger, stats

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--trades",required=True)
    ap.add_argument("--price-root",required=True)
    ap.add_argument("--source",choices=["yahoo","kaggle"],required=True)
    ap.add_argument("--output-dir",required=True)
    args=ap.parse_args()
    trades=load_locked_trades(args.trades)
    prices=load_prices(Path(args.price_root),set(trades["symbol"].astype(str)),args.source)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    result={"source":args.source,"locked_trades":len(trades),"priced_symbols":len(prices),"variants":{}}
    for mp in [5,10,20]:
        curve,ledger,stats=simulate(trades,prices,mp)
        curve.to_csv(out/f"equity_max{mp}.csv",index=False)
        ledger.to_csv(out/f"ledger_max{mp}.csv",index=False)
        result["variants"][f"max_{mp}"]=stats
    (out/"portfolio_summary.json").write_text(json.dumps(result,indent=2))
    print("RESULT_JSON="+json.dumps(result,separators=(",",":")),flush=True)

if __name__=="__main__":
    main()
