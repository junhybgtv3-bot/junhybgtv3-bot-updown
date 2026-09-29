from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

DATASET_ID = "AmirTrader/YahooFinance"

def clip01(x):
    return np.clip(x, 0.0, 1.0)

def safe_pf(returns: np.ndarray) -> float:
    wins = returns[returns > 0].sum()
    losses = -returns[returns <= 0].sum()
    return float(wins / losses) if losses > 0 else float("inf")

def t_stat(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return float("nan")
    sd = returns.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(returns.mean() / (sd / math.sqrt(len(returns))))

@dataclass
class Trade:
    symbol: str
    signal_date: str
    entry_date: str
    exit_date: str
    entry: float
    exit: float
    net_return: float
    gross_return: float
    hold_days: int
    reason: str
    setup_score: float
    accumulation_score: float
    rr_at_signal: float
    support_distance_pct: float
    red_signal: bool
    dollar_volume_20: float

def load_adjusted(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    if df.empty:
        return None
    df.columns = [str(c).lower() for c in df.columns]
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return None
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
    if len(df) < 260:
        return None

    if "adj_close" in df.columns:
        adj = pd.to_numeric(df["adj_close"], errors="coerce")
        factor = (adj / df["close"]).replace([np.inf, -np.inf], np.nan)
        factor = factor.where((factor > 0) & factor.notna(), 1.0)
        for c in ["open", "high", "low", "close"]:
            if c == "close":
                df[c] = np.where(adj.notna() & (adj > 0), adj, df[c])
            else:
                df[c] = df[c] * factor
    return df.reset_index(drop=True)

def indicators(df: pd.DataFrame) -> dict[str, np.ndarray]:
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    n = len(df)

    prev = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h-l, np.abs(h-prev), np.abs(l-prev)]), axis=0)
    tr[0] = h[0] - l[0]
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    atrp = atr / c

    s_c, s_h, s_l, s_v = map(pd.Series, [c, h, l, v])
    ma = {w: s_c.rolling(w, min_periods=w).mean().to_numpy() for w in [5,20,60,120,224]}
    low = {w: s_l.rolling(w, min_periods=w).min().to_numpy() for w in [10,20,60,120]}
    high = {w: s_h.rolling(w, min_periods=w).max().to_numpy() for w in [20,60,120]}
    vol_med20 = s_v.rolling(20, min_periods=20).median().to_numpy()
    rv = v / np.where(vol_med20 > 0, vol_med20, np.nan)
    body = c / o - 1.0
    clv = np.divide(c-l, h-l, out=np.full(n, 0.5), where=(h-l)!=0)
    clv = clip01(clv)
    range_atr = (h-l) / atr
    dollar_volume20 = pd.Series(c*v).rolling(20, min_periods=20).median().to_numpy()

    low120 = low[120]
    high120 = s_h.rolling(120, min_periods=120).max().to_numpy()
    price_pos120 = np.divide(c-low120, high120-low120,
                             out=np.full(n, 0.5), where=(high120-low120)>0)
    price_pos120 = clip01(price_pos120)
    move_norm = body / np.maximum(atrp, 0.01)

    acc = 100.0 * (
        0.30 * clip01((move_norm - 0.5) / 2.5) +
        0.25 * clip01((rv - 1.2) / 3.0) +
        0.15 * clip01((clv - 0.45) / 0.45) +
        0.10 * clip01((range_atr - 0.8) / 2.2) +
        0.20 * clip01((0.55 - price_pos120) / 0.45)
    )
    acc = np.where((body >= 0.10) & (rv >= 3.0), np.minimum(100.0, acc + 12.0), acc)
    acc[:119] = np.nan
    acc_rollmax = pd.Series(acc).rolling(120, min_periods=1).max().to_numpy()

    acc_trigger_low = pd.Series(np.where(acc >= 45.0, l, np.nan)).ffill(limit=119).to_numpy()
    recent_atrp = pd.Series(atrp).rolling(10, min_periods=10).median().to_numpy()
    prior_atrp = pd.Series(atrp).shift(20).rolling(20, min_periods=20).median().to_numpy()
    compression = clip01((prior_atrp - recent_atrp) / np.maximum(prior_atrp, 1e-6) + 0.5)

    s20 = (c / np.roll(c, 19) - 1.0) / 19.0
    s60 = (c / np.roll(c, 59) - 1.0) / 59.0
    s20[:19] = np.nan
    s60[:59] = np.nan
    stabilized = clip01(1.0 - np.abs(s20) / 0.01)
    improving = clip01((s20 - s60 + 0.002) / 0.008)
    base = 100.0 * (0.45*compression + 0.30*stabilized + 0.25*improving)

    ma_score = np.full(n, 0.25)
    ma_score[(ma[5] > ma[20])] = 0.65
    ma_score[(ma[5] > ma[20]) & (ma[20] > ma[60])] = 1.0
    pressure = clip01(1.0 - (high[20]-c) / np.maximum(2.0*atr, 1e-9))
    trend = 100.0 * (0.50*ma_score + 0.30*improving + 0.20*pressure)

    return dict(o=o,h=h,l=l,c=c,v=v,atr=atr,atrp=atrp,body=body,ma=ma,low=low,high=high,
                acc=acc,acc_rollmax=acc_rollmax,acc_low=acc_trigger_low,base=base,trend=trend,
                dollar_volume20=dollar_volume20)

def support_cluster(candidates: list[float], atr: float) -> tuple[float, float]:
    vals = [x for x in candidates if np.isfinite(x) and x > 0]
    if not vals:
        return float("nan"), 0.0
    tol = 0.8 * atr
    best = [vals[0]]
    for x in vals:
        cluster = [y for y in vals if abs(y-x) <= tol]
        if len(cluster) > len(best):
            best = cluster
    return float(np.median(best)), min(1.0, len(best)/3.0)

def backtest_symbol(symbol: str, df: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp,
                    setup_threshold: float, red_mode: str, cost_side: float,
                    max_hold: int, min_price: float, min_dollar_volume: float) -> list[Trade]:
    d = indicators(df)
    dates = df["date"].to_numpy()
    start_i = int(np.searchsorted(dates, np.datetime64(start), side="left"))
    end_i = int(np.searchsorted(dates, np.datetime64(end), side="right")) - 1
    start_i = max(start_i, 223)
    end_i = min(end_i, len(df)-2)
    if start_i > end_i:
        return []

    trades = []
    i = start_i
    while i <= end_i:
        price = d["c"][i]
        atr = d["atr"][i]
        acc = d["acc_rollmax"][i]
        base = d["base"][i]
        trend = d["trend"][i]
        dv20 = d["dollar_volume20"][i]
        red = bool(d["c"][i] < d["o"][i])
        if (not np.isfinite(atr) or atr <= 0 or not np.isfinite(acc) or acc < 45 or
            not np.isfinite(base) or not np.isfinite(trend) or price < min_price or
            not np.isfinite(dv20) or dv20 < min_dollar_volume or
            (red_mode == "red" and not red) or (red_mode == "green" and red)):
            i += 1
            continue

        support, conf = support_cluster([
            d["acc_low"][i], d["low"][20][i], d["low"][60][i], d["low"][10][i]
        ], atr)
        if not np.isfinite(support) or support <= 0 or price <= support:
            i += 1
            continue

        res_candidates = [d["high"][20][i], d["high"][60][i],
                          d["ma"][60][i], d["ma"][120][i], d["ma"][224][i]]
        above = [x for x in res_candidates if np.isfinite(x) and x > price*1.01]
        resistance = min(above) if above else d["high"][120][i]
        if not np.isfinite(resistance) or resistance <= price:
            i += 1
            continue

        rr = (resistance-price) / max(price-support, 1e-9)
        support_distance_pct = (price-support)/support
        support_distance_atr = (price-support)/atr
        chase = support_distance_pct > 0.08 and support_distance_atr > 2.0
        prox = 100.0*clip01(1.0-max(support_distance_pct/0.08, support_distance_atr/2.0))
        setup = 100.0*(0.30*acc/100.0 + 0.20*base/100.0 + 0.15*conf +
                       0.15*clip01(rr/3.0) + 0.10*prox/100.0 + 0.10*trend/100.0)
        if chase or rr < 1.5 or setup < setup_threshold:
            i += 1
            continue

        ei = i+1
        entry = d["o"][ei]
        stop = support - 0.15*atr
        target = resistance
        if not np.isfinite(entry) or entry <= stop or entry >= target:
            i += 1
            continue

        xj = None
        exit_price = None
        reason = None
        last = min(ei+max_hold-1, len(df)-1, end_i+1)
        for j in range(ei, last+1):
            op, hi, lo, cl = d["o"][j], d["h"][j], d["l"][j], d["c"][j]
            if op <= stop:
                exit_price, xj, reason = op, j, "gap_stop"
                break
            if op >= target:
                exit_price, xj, reason = op, j, "gap_target"
                break
            if lo <= stop:
                exit_price, xj, reason = stop, j, "stop"
                break
            if hi >= target:
                exit_price, xj, reason = target, j, "target"
                break
            if j == last:
                exit_price, xj, reason = cl, j, "time"
                break
        if xj is None:
            i += 1
            continue

        gross = exit_price/entry - 1.0
        net = (exit_price*(1.0-cost_side))/(entry*(1.0+cost_side)) - 1.0
        trades.append(Trade(
            symbol=symbol,
            signal_date=str(pd.Timestamp(dates[i]).date()),
            entry_date=str(pd.Timestamp(dates[ei]).date()),
            exit_date=str(pd.Timestamp(dates[xj]).date()),
            entry=float(entry), exit=float(exit_price), net_return=float(net), gross_return=float(gross),
            hold_days=int(xj-ei+1), reason=str(reason), setup_score=float(setup),
            accumulation_score=float(acc), rr_at_signal=float(rr),
            support_distance_pct=float(support_distance_pct), red_signal=red,
            dollar_volume_20=float(dv20)
        ))
        i = xj + 1
    return trades

def summarize(trades: list[Trade]) -> dict:
    if not trades:
        return {"trades":0}
    r = np.array([t.net_return for t in trades], dtype=float)
    wins = r > 0
    return {
        "trades": len(trades),
        "symbols_with_trades": len({t.symbol for t in trades}),
        "win_rate": float(wins.mean()),
        "avg_return": float(r.mean()),
        "median_return": float(np.median(r)),
        "profit_factor": safe_pf(r),
        "t_stat": t_stat(r),
        "avg_hold_days": float(np.mean([t.hold_days for t in trades])),
        "best_trade": float(r.max()),
        "worst_trade": float(r.min()),
        "target_rate": float(np.mean(["target" in t.reason for t in trades])),
        "stop_rate": float(np.mean(["stop" in t.reason for t in trades])),
    }

def year_summary(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    x = pd.DataFrame([asdict(t) for t in trades])
    x["year"] = pd.to_datetime(x["signal_date"]).dt.year
    rows=[]
    for y,g in x.groupby("year"):
        rr=g["net_return"].to_numpy(float)
        rows.append({"year":int(y),"trades":len(g),"win_rate":float((rr>0).mean()),
                     "avg_return":float(rr.mean()),"profit_factor":safe_pf(rr),"t_stat":t_stat(rr)})
    return pd.DataFrame(rows)

def bootstrap_ci(returns: np.ndarray, seed=7, n=3000) -> dict:
    if len(returns) < 10:
        return {}
    rng=np.random.default_rng(seed)
    means=np.empty(n)
    for i in range(n):
        means[i]=rng.choice(returns,size=len(returns),replace=True).mean()
    return {"mean_return_ci95_low":float(np.quantile(means,.025)),
            "mean_return_ci95_high":float(np.quantile(means,.975))}

def download_data(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    root = snapshot_download(repo_id=DATASET_ID, repo_type="dataset", local_dir=str(data_dir),
                             allow_patterns=["data/daily/*.parquet"], max_workers=8)
    return Path(root)/"data"/"daily"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/tmp/yahoo_finance")
    ap.add_argument("--output-dir", default="results/jtdj_full")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2026-09-25")
    ap.add_argument("--setup", type=float, default=70.0)
    ap.add_argument("--cost-side", type=float, default=0.001)
    ap.add_argument("--max-hold", type=int, default=60)
    ap.add_argument("--min-price", type=float, default=2.0)
    ap.add_argument("--min-dollar-volume", type=float, default=1_000_000.0)
    args=ap.parse_args()

    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    daily=download_data(Path(args.data_dir))
    files=sorted(daily.glob("*.parquet"))
    print(f"DATA_FILES={len(files)}")

    modes={"base":"any","red_only":"red","green_only":"green"}
    all_trades={k:[] for k in modes}
    valid_symbols=0
    rows_total=0
    errors=[]
    start=pd.Timestamp(args.start)
    end=pd.Timestamp(args.end)

    for idx,path in enumerate(files,1):
        symbol=path.stem.upper()
        df=load_adjusted(path)
        if df is None:
            continue
        valid_symbols+=1
        rows_total+=len(df)
        for name,mode in modes.items():
            try:
                all_trades[name].extend(backtest_symbol(
                    symbol,df,start=start,end=end,setup_threshold=args.setup,red_mode=mode,
                    cost_side=args.cost_side,max_hold=args.max_hold,min_price=args.min_price,
                    min_dollar_volume=args.min_dollar_volume
                ))
            except Exception as e:
                errors.append({"symbol":symbol,"mode":name,"error":repr(e)})
        if idx % 500 == 0:
            print(f"PROGRESS {idx}/{len(files)} valid={valid_symbols} base_trades={len(all_trades['base'])} red={len(all_trades['red_only'])}")

    summary={
        "dataset":DATASET_ID,
        "requested_period":{"start":args.start,"end":args.end},
        "data_files":len(files),"valid_symbols":valid_symbols,"rows_total":rows_total,
        "parameters":{"setup_threshold":args.setup,"cost_side":args.cost_side,"max_hold":args.max_hold,
                      "min_price":args.min_price,"min_dollar_volume":args.min_dollar_volume,
                      "entry":"next trading day open","stop":"support - 0.15 ATR",
                      "target":"signal-day resistance","same_day_stop_target":"stop first"},
        "strategies":{}
    }
    for name,tr in all_trades.items():
        s=summarize(tr)
        if tr:
            s.update(bootstrap_ci(np.array([x.net_return for x in tr],float)))
        summary["strategies"][name]=s
        pd.DataFrame([asdict(x) for x in tr]).to_csv(out/f"trades_{name}.csv",index=False)
        year_summary(tr).to_csv(out/f"year_{name}.csv",index=False)

    for name,tr in all_trades.items():
        for label,lo,hi in [("2010_2017","2010-01-01","2017-12-31"),("2018_2026","2018-01-01","2026-09-25")]:
            sub=[x for x in tr if lo <= x.signal_date <= hi]
            summary["strategies"][name][label]=summarize(sub)

    with open(out/"summary.json","w") as f:
        json.dump(summary,f,indent=2)
    with open(out/"errors.json","w") as f:
        json.dump(errors[:1000],f,indent=2)
    print("RESULT_JSON="+json.dumps(summary,separators=(",",":")))

if __name__=="__main__":
    main()
