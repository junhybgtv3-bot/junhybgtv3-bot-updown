from __future__ import annotations
import argparse, json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from v2_engine import backtest_v2, market_context, summarize, bootstrap_ci

DATASET_ID="AmirTrader/YahooFinance"

def load_parquet(path: Path):
    try:
        df=pd.read_parquet(path)
    except Exception:
        return None
    if df.empty:
        return None
    df.columns=[str(c).lower() for c in df.columns]
    if not {"date","open","high","low","close","volume"}.issubset(df.columns):
        return None
    if "adj_close" in df.columns:
        df["adj_close"]=pd.to_numeric(df["adj_close"],errors="coerce")
        for c in ["open","high","low","close","volume"]:
            df[c]=pd.to_numeric(df[c],errors="coerce")
        factor=(df["adj_close"]/df["close"]).replace([np.inf,-np.inf],np.nan)
        factor=factor.where((factor>0)&factor.notna(),1.0)
        df["open"]=df["open"]*factor
        df["high"]=df["high"]*factor
        df["low"]=df["low"]*factor
        df["close"]=np.where(df["adj_close"].notna()&(df["adj_close"]>0),df["adj_close"],df["close"])
    return df

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",default="/tmp/yahoo_v2")
    ap.add_argument("--output-dir",default="results/jtdj_v2_yahoo")
    ap.add_argument("--start",default="2020-01-01")
    ap.add_argument("--end",default="2026-09-25")
    args=ap.parse_args()
    root=Path(snapshot_download(repo_id=DATASET_ID,repo_type="dataset",local_dir=args.data_dir,
                                allow_patterns=["data/daily/*.parquet"],max_workers=8))
    daily=root/"data"/"daily"
    files=sorted(daily.glob("*.parquet"))
    spy_candidates=[p for p in files if p.stem.upper()=="SPY"]
    if spy_candidates:
        spy_path=spy_candidates[0]
        print(f"SPY_FILE={spy_path.name}",flush=True)
        spy=load_parquet(spy_path)
    else:
        print(f"SPY_FILE_EXTERNAL={args.spy_csv}",flush=True)
        spy=pd.read_csv(args.spy_csv)
        spy.columns=[str(x).lower() for x in spy.columns]
    if spy is None or spy.empty:
        raise RuntimeError("Failed to load SPY benchmark")
    spy_ctx=market_context(spy)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    variants={"core":(False,False),"rs_regime":(True,True)}
    result={k:[] for k in variants}
    valid=0
    for idx,p in enumerate(files,1):
        df=load_parquet(p)
        if df is None or len(df)<260:
            continue
        valid+=1
        sym=p.stem.upper()
        for name,(req_rs,req_reg) in variants.items():
            result[name].extend(backtest_v2(sym,df,spy_ctx,args.start,args.end,require_rs=req_rs,require_spy_regime=req_reg))
        if idx%500==0:
            print(f"PROGRESS {idx}/{len(files)} valid={valid} core={len(result['core'])} rs={len(result['rs_regime'])}",flush=True)
    summary={"dataset":DATASET_ID,"files":len(files),"valid_symbols":valid,
             "period":{"start":args.start,"end":args.end},"strategies":{}}
    for name,tr in result.items():
        s=summarize(tr); s.update(bootstrap_ci(tr))
        for label,lo,hi in [("2020_2022","2020-01-01","2022-12-31"),
                            ("2023_2024","2023-01-01","2024-12-31"),
                            ("2025_2026","2025-01-01","2026-09-25")]:
            s[label]=summarize([x for x in tr if lo<=x.signal_date<=hi])
        summary["strategies"][name]=s
        pd.DataFrame([asdict(x) for x in tr]).to_csv(out/f"trades_{name}.csv",index=False)
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    print("RESULT_JSON="+json.dumps(summary,separators=(",",":")),flush=True)

if __name__=="__main__":
    main()
