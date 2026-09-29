from __future__ import annotations
import argparse, json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from run import backtest_symbol, summarize, year_summary, bootstrap_ci
from run_kaggle import load_stooq_txt

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    ap.add_argument("--output-dir",default="results/jtdj_kaggle_red")
    ap.add_argument("--start",default="2000-01-01")
    ap.add_argument("--end",default="2017-11-10")
    args=ap.parse_args()
    root=Path(args.root); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    files=sorted([p for p in root.rglob("*.txt") if "stocks" in [x.lower() for x in p.parts]])
    print(f"STOCK_FILES={len(files)}",flush=True)
    trades=[]; valid=0; rows=0
    start=pd.Timestamp(args.start); end=pd.Timestamp(args.end)
    for idx,p in enumerate(files,1):
        df=load_stooq_txt(p)
        if df is None: continue
        valid+=1; rows+=len(df)
        symbol=p.name.lower().replace(".us.txt","").upper()
        trades.extend(backtest_symbol(symbol,df,start=start,end=end,setup_threshold=70.0,
                                      red_mode="red",cost_side=.001,max_hold=60,
                                      min_price=2.0,min_dollar_volume=1_000_000.0))
        if idx%250==0:
            print(f"PROGRESS={idx}/{len(files)} valid={valid} trades={len(trades)}",flush=True)
    s=summarize(trades)
    if trades:
        s.update(bootstrap_ci(np.array([x.net_return for x in trades],float)))
    summary={"dataset":"Kaggle full US stocks","stock_files":len(files),"valid_symbols":valid,
             "rows_total":rows,"period":{"start":args.start,"end":args.end},
             "strategy":"JTDJ setup>=70 + red signal + RR>=1.5 + no-chase; next-open entry; 10bp/side",
             "results":s}
    for label,lo,hi in [("2000_2008","2000-01-01","2008-12-31"),
                        ("2009_2013","2009-01-01","2013-12-31"),
                        ("2014_2017","2014-01-01","2017-11-10")]:
        sub=[x for x in trades if lo<=x.signal_date<=hi]
        summary[label]=summarize(sub)
    pd.DataFrame([asdict(x) for x in trades]).to_csv(out/"trades.csv",index=False)
    year_summary(trades).to_csv(out/"year.csv",index=False)
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    print("RESULT_JSON="+json.dumps(summary,separators=(",",":")),flush=True)

if __name__=="__main__":
    main()
