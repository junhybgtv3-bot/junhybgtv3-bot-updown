from __future__ import annotations
import argparse, json, subprocess, zipfile
from pathlib import Path
import pandas as pd
from huggingface_hub import snapshot_download
from portfolio_v3 import load_locked_trades, load_prices, simulate

DATASET_ID="AmirTrader/YahooFinance"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--artifact-zip",required=True)
    ap.add_argument("--work-dir",default="/tmp/jtdj_quick")
    ap.add_argument("--output-dir",default="results/jtdj_v3_portfolio_quick")
    args=ap.parse_args()
    work=Path(args.work_dir); work.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.artifact_zip) as z:
        z.extractall(work/"artifact")
    trades_path=work/"artifact"/"trades_core.csv"
    trades=load_locked_trades(str(trades_path))
    syms=sorted(set(trades["symbol"].astype(str)))
    print(f"LOCKED_TRADES={len(trades)} SYMBOLS={len(syms)}",flush=True)

    patterns=[]
    for s in syms:
        patterns += [f"data/daily/{s.lower()}.parquet",f"data/daily/{s.upper()}.parquet"]
    root=Path(snapshot_download(repo_id=DATASET_ID,repo_type="dataset",local_dir=str(work/"yahoo"),
                                allow_patterns=patterns,max_workers=8))
    daily=root/"data"/"daily"
    prices=load_prices(daily,set(syms),"yahoo")
    print(f"PRICED_SYMBOLS={len(prices)}",flush=True)

    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    result={"locked_trades":len(trades),"symbols":len(syms),"priced_symbols":len(prices),"variants":{}}
    for mp in [1,2,3,5,10,20]:
        curve,ledger,stats=simulate(trades,prices,mp,initial_cash=100000.0,cost_side=.001)
        curve.to_csv(out/f"equity_max{mp}.csv",index=False)
        ledger.to_csv(out/f"ledger_max{mp}.csv",index=False)
        result["variants"][f"max_{mp}"]=stats
    (out/"portfolio_summary.json").write_text(json.dumps(result,indent=2))
    print("RESULT_JSON="+json.dumps(result,separators=(",",":")),flush=True)

if __name__=="__main__":
    main()
