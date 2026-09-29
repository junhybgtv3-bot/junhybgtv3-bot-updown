from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from run import Trade, backtest_symbol, summarize, year_summary, bootstrap_ci

def load_stooq_txt(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or len(df) < 260:
        return None
    df.columns = [str(c).lower() for c in df.columns]
    required = {"date","open","high","low","close","volume"}
    if not required.issubset(df.columns):
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date","open","high","low","close","volume"])
    df = df[(df["open"]>0)&(df["high"]>0)&(df["low"]>0)&(df["close"]>0)]
    if len(df) < 260:
        return None
    return df.sort_values("date").drop_duplicates("date",keep="last").reset_index(drop=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output-dir", default="results/jtdj_kaggle_full")
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--end", default="2017-11-10")
    ap.add_argument("--setup", type=float, default=70.0)
    ap.add_argument("--cost-side", type=float, default=0.001)
    ap.add_argument("--max-hold", type=int, default=60)
    ap.add_argument("--min-price", type=float, default=2.0)
    ap.add_argument("--min-dollar-volume", type=float, default=1_000_000.0)
    args=ap.parse_args()

    root=Path(args.root)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    files=[p for p in root.rglob("*.txt") if "stocks" in [x.lower() for x in p.parts]]
    files=sorted(files)
    print(f"STOCK_FILES={len(files)}", flush=True)

    modes={"base":"any","red_only":"red","green_only":"green"}
    all_trades={k:[] for k in modes}
    valid=0; rows=0; errors=[]
    start=pd.Timestamp(args.start); end=pd.Timestamp(args.end)

    for idx,path in enumerate(files,1):
        symbol=path.name.lower().replace(".us.txt","").upper()
        df=load_stooq_txt(path)
        if df is None:
            continue
        valid+=1; rows+=len(df)
        for name,mode in modes.items():
            try:
                all_trades[name].extend(backtest_symbol(
                    symbol,df,start=start,end=end,setup_threshold=args.setup,red_mode=mode,
                    cost_side=args.cost_side,max_hold=args.max_hold,min_price=args.min_price,
                    min_dollar_volume=args.min_dollar_volume
                ))
            except Exception as e:
                errors.append({"symbol":symbol,"mode":name,"error":repr(e)})
        if idx % 250 == 0:
            print(f"PROGRESS={idx}/{len(files)} valid={valid} base={len(all_trades['base'])} red={len(all_trades['red_only'])} green={len(all_trades['green_only'])}", flush=True)

    summary={
        "dataset":"Kaggle borismarjanovic/price-volume-data-for-all-us-stocks-etfs",
        "requested_period":{"start":args.start,"end":args.end},
        "stock_files":len(files),"valid_symbols":valid,"rows_total":rows,
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

        for label,lo,hi in [
            ("2000_2008","2000-01-01","2008-12-31"),
            ("2009_2013","2009-01-01","2013-12-31"),
            ("2014_2017","2014-01-01","2017-11-10"),
        ]:
            sub=[x for x in tr if lo <= x.signal_date <= hi]
            summary["strategies"][name][label]=summarize(sub)

    with open(out/"summary.json","w") as f:
        json.dump(summary,f,indent=2)
    with open(out/"errors.json","w") as f:
        json.dump(errors[:1000],f,indent=2)
    print("RESULT_JSON="+json.dumps(summary,separators=(",",":")), flush=True)

if __name__=="__main__":
    main()
