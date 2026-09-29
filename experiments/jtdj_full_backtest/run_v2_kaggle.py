from __future__ import annotations
import argparse, json
from dataclasses import asdict
from pathlib import Path
import pandas as pd
from v2_engine import backtest_v2, market_context, summarize, bootstrap_ci

def load_txt(path: Path):
    try:
        df=pd.read_csv(path)
    except Exception:
        return None
    if df.empty or len(df)<260:
        return None
    df.columns=[str(c).lower() for c in df.columns]
    if not {"date","open","high","low","close","volume"}.issubset(df.columns):
        return None
    return df

def find_spy(root: Path):
    cands=[p for p in root.rglob("spy.us.txt")]
    if not cands:
        raise RuntimeError("SPY file not found")
    return cands[0]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    ap.add_argument("--output-dir",default="results/jtdj_v2_kaggle")
    ap.add_argument("--start",default="2000-01-01")
    ap.add_argument("--end",default="2017-11-10")
    args=ap.parse_args()
    root=Path(args.root)
    spy=load_txt(find_spy(root))
    spy_ctx=market_context(spy)
    files=sorted([p for p in root.rglob("*.txt") if "stocks" in [x.lower() for x in p.parts]])
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    variants={"core":(False,False),"rs_regime":(True,True)}
    result={k:[] for k in variants}; valid=0
    seen=set()
    for idx,p in enumerate(files,1):
        sym=p.name.lower().replace(".us.txt","").upper()
        # Archive sometimes contains duplicate Stocks directories; only one file per ticker.
        if sym in seen:
            continue
        seen.add(sym)
        df=load_txt(p)
        if df is None:
            continue
        valid+=1
        for name,(req_rs,req_reg) in variants.items():
            result[name].extend(backtest_v2(sym,df,spy_ctx,args.start,args.end,require_rs=req_rs,require_spy_regime=req_reg))
        if valid%250==0:
            print(f"PROGRESS valid={valid} unique={len(seen)} core={len(result['core'])} rs={len(result['rs_regime'])}",flush=True)
    summary={"dataset":"Kaggle price-volume-data-for-all-us-stocks-etfs","raw_stock_files":len(files),
             "unique_symbols_seen":len(seen),"valid_symbols":valid,
             "period":{"start":args.start,"end":args.end},"strategies":{}}
    for name,tr in result.items():
        s=summarize(tr); s.update(bootstrap_ci(tr))
        for label,lo,hi in [("2000_2008","2000-01-01","2008-12-31"),
                            ("2009_2013","2009-01-01","2013-12-31"),
                            ("2014_2017","2014-01-01","2017-11-10")]:
            s[label]=summarize([x for x in tr if lo<=x.signal_date<=hi])
        summary["strategies"][name]=s
        pd.DataFrame([asdict(x) for x in tr]).to_csv(out/f"trades_{name}.csv",index=False)
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    print("RESULT_JSON="+json.dumps(summary,separators=(",",":")),flush=True)

if __name__=="__main__":
    main()
