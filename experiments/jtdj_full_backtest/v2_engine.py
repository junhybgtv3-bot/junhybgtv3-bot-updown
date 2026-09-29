from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def safe_pf(r):
    r=np.asarray(r,dtype=float)
    gp=r[r>0].sum()
    gl=-r[r<=0].sum()
    return float(gp/gl) if gl>0 else float("inf")


def t_stat(r):
    r=np.asarray(r,dtype=float)
    if len(r)<2:
        return float("nan")
    sd=r.std(ddof=1)
    return float(r.mean()/(sd/math.sqrt(len(r)))) if sd>0 else float("nan")


@dataclass
class TradeV2:
    symbol: str
    signal_date: str
    event_date: str
    entry_date: str
    exit_date: str
    entry: float
    exit: float
    net_return: float
    gross_return: float
    hold_days: int
    reason: str
    event_score: float
    event_rvol: float
    event_age: int
    down_volume_share_10: float
    support_distance_pct: float
    rr_at_entry: float
    relative_strength_20: float
    spy_regime: bool
    signal_green: bool


def load_market_frame(df: pd.DataFrame) -> pd.DataFrame:
    x=df.copy()
    x.columns=[str(c).lower() for c in x.columns]
    x["date"]=pd.to_datetime(x["date"],errors="coerce")
    for c in ["open","high","low","close","volume"]:
        x[c]=pd.to_numeric(x[c],errors="coerce")
    x=x.dropna(subset=["date","open","high","low","close","volume"])
    x=x[(x["open"]>0)&(x["high"]>0)&(x["low"]>0)&(x["close"]>0)]
    return x.sort_values("date").drop_duplicates("date",keep="last").reset_index(drop=True)


def compute_arrays(df: pd.DataFrame):
    o=df["open"].to_numpy(float); h=df["high"].to_numpy(float)
    l=df["low"].to_numpy(float); c=df["close"].to_numpy(float)
    v=df["volume"].to_numpy(float); n=len(df)
    prev=np.r_[np.nan,c[:-1]]
    tr=np.nanmax(np.vstack([h-l,np.abs(h-prev),np.abs(l-prev)]),axis=0)
    tr[0]=h[0]-l[0]
    atr=pd.Series(tr).rolling(14,min_periods=14).mean().to_numpy()
    atrp=atr/c
    sc=pd.Series(c); sh=pd.Series(h); sl=pd.Series(l); sv=pd.Series(v)
    ma20=sc.rolling(20,min_periods=20).mean().to_numpy()
    ma60=sc.rolling(60,min_periods=60).mean().to_numpy()
    ma200=sc.rolling(200,min_periods=200).mean().to_numpy()
    low10=sl.rolling(10,min_periods=10).min().to_numpy()
    low20=sl.rolling(20,min_periods=20).min().to_numpy()
    low60=sl.rolling(60,min_periods=60).min().to_numpy()
    high20_prev=sh.shift(1).rolling(20,min_periods=20).max().to_numpy()
    high60_prev=sh.shift(1).rolling(60,min_periods=60).max().to_numpy()
    high120_prev=sh.shift(1).rolling(120,min_periods=120).max().to_numpy()
    vol_med20=sv.rolling(20,min_periods=20).median().to_numpy()
    rv=v/np.where(vol_med20>0,vol_med20,np.nan)
    body=c/o-1.0
    clv=np.divide(c-l,h-l,out=np.full(n,.5),where=(h-l)!=0)
    clv=clip01(clv)
    range_atr=(h-l)/atr
    low120=sl.rolling(120,min_periods=120).min().to_numpy()
    high120=sh.rolling(120,min_periods=120).max().to_numpy()
    pos=np.divide(c-low120,high120-low120,out=np.full(n,.5),where=(high120-low120)>0)
    pos=clip01(pos)
    move_norm=body/np.maximum(atrp,.01)
    acc=100.0*(.30*clip01((move_norm-.5)/2.5)+.25*clip01((rv-1.2)/3.0)+
                 .15*clip01((clv-.45)/.45)+.10*clip01((range_atr-.8)/2.2)+
                 .20*clip01((.55-pos)/.45))
    acc=np.where((body>=.10)&(rv>=3.0),np.minimum(100.0,acc+12.0),acc)
    acc[:119]=np.nan

    eligible=(acc>=60.0)&(acc<80.0)&(body>=0.02)&(rv>=1.5)
    raw_idx=np.where(eligible,np.arange(n),np.nan)
    # most recent event at least 5 bars old and at most 40 bars old
    event_idx=pd.Series(raw_idx).shift(5).ffill(limit=35).to_numpy()

    down=np.where(c<prev,v,0.0)
    down10=pd.Series(down).rolling(10,min_periods=10).sum().to_numpy()
    vol10=sv.rolling(10,min_periods=10).sum().to_numpy()
    down_share=np.divide(down10,vol10,out=np.full(n,np.nan),where=vol10>0)
    ret20=sc.pct_change(20).to_numpy()
    dollar_volume20=pd.Series(c*v).rolling(20,min_periods=20).median().to_numpy()
    ma20_slope=np.r_[np.full(5,np.nan),ma20[5:]/ma20[:-5]-1.0]

    return dict(o=o,h=h,l=l,c=c,v=v,atr=atr,rv=rv,acc=acc,event_idx=event_idx,
                down_share=down_share,ret20=ret20,ma20=ma20,ma60=ma60,ma200=ma200,
                low10=low10,low20=low20,low60=low60,
                high20_prev=high20_prev,high60_prev=high60_prev,high120_prev=high120_prev,
                dollar_volume20=dollar_volume20,ma20_slope=ma20_slope)


def market_context(spy: pd.DataFrame):
    spy=load_market_frame(spy)
    c=spy["close"].to_numpy(float)
    ret20=pd.Series(c).pct_change(20).to_numpy()
    ma200=pd.Series(c).rolling(200,min_periods=200).mean().to_numpy()
    regime=c>ma200
    return pd.DataFrame({"date":spy["date"],"spy_ret20":ret20,"spy_regime":regime})


def support_cluster(vals,atr):
    vals=[float(x) for x in vals if np.isfinite(x) and x>0]
    if not vals:
        return np.nan,0
    tol=.8*atr
    best=[]
    for x in vals:
        cl=[y for y in vals if abs(y-x)<=tol]
        if len(cl)>len(best):
            best=cl
    return float(np.median(best)),len(best)


def backtest_v2(symbol,df,spy_ctx,start,end,require_rs=True,require_spy_regime=True,
                cost_side=.001,max_hold=40,min_price=2.0,min_dollar_volume=1_000_000.0):
    df=load_market_frame(df)
    if len(df)<260:
        return []
    m=df[["date"]].merge(spy_ctx,on="date",how="left")
    A=compute_arrays(df)
    spy_ret=m["spy_ret20"].to_numpy(float)
    spy_reg=m["spy_regime"].fillna(False).to_numpy(bool)
    rs20=A["ret20"]-spy_ret
    dates=df["date"].to_numpy()
    si=max(223,int(np.searchsorted(dates,np.datetime64(pd.Timestamp(start)),side="left")))
    ei=min(len(df)-2,int(np.searchsorted(dates,np.datetime64(pd.Timestamp(end)),side="right"))-1)
    out=[]
    i=si
    while i<=ei:
        jv=A["event_idx"][i]
        if not np.isfinite(jv):
            i+=1; continue
        j=int(jv)
        age=i-j
        if age<5 or age>40:
            i+=1; continue
        price=A["c"][i]; atr=A["atr"][i]
        if (not np.isfinite(atr) or atr<=0 or price<min_price or
            not np.isfinite(A["dollar_volume20"][i]) or A["dollar_volume20"][i]<min_dollar_volume):
            i+=1; continue
        if require_spy_regime and not spy_reg[i]:
            i+=1; continue
        if require_rs and (not np.isfinite(rs20[i]) or rs20[i]<=0):
            i+=1; continue

        # Follow-through / absorption: event must hold, not collapse after spike.
        lows=A["l"][j+1:i+1]
        if len(lows)==0:
            i+=1; continue
        if np.nanmin(lows) < A["l"][j]-0.25*A["atr"][j]:
            i+=1; continue
        if A["c"][i] < 0.97*A["c"][j]:
            i+=1; continue
        if not np.isfinite(A["down_share"][i]) or A["down_share"][i]>.55:
            i+=1; continue
        # Cooling after event; avoid a second blow-off spike.
        if np.isfinite(A["rv"][i]) and A["rv"][i]>1.8:
            i+=1; continue
        # Transition/base confirmation.
        if not np.isfinite(A["ma20_slope"][i]) or A["ma20_slope"][i] < -0.01:
            i+=1; continue
        if A["c"][i] < 0.97*A["ma20"][i]:
            i+=1; continue

        support,conf=support_cluster([A["l"][j],A["low10"][i],A["low20"][i],A["low60"][i]],atr)
        if conf<2 or not np.isfinite(support) or price<=support:
            i+=1; continue
        dist=(price-support)/support
        if dist<.01 or dist>.06:
            i+=1; continue

        candidates=[A["high20_prev"][i],A["high60_prev"][i],A["high120_prev"][i]]
        above=[x for x in candidates if np.isfinite(x) and x>price*1.03]
        if not above:
            i+=1; continue
        resistance=min(above)

        entry_i=i+1
        entry=A["o"][entry_i]
        stop=support-.50*atr
        if entry<=stop:
            i+=1; continue
        risk=entry-stop
        rr=(resistance-entry)/risk
        if rr<2.0:
            i+=1; continue
        target=resistance

        last=min(entry_i+max_hold-1,len(df)-1,ei+1)
        xj=None; xp=None; reason=None
        for k in range(entry_i,last+1):
            op,hi,lo,cl=A["o"][k],A["h"][k],A["l"][k],A["c"][k]
            if op<=stop:
                xp,xj,reason=op,k,"gap_stop"; break
            if op>=target:
                xp,xj,reason=op,k,"gap_target"; break
            if lo<=stop:
                xp,xj,reason=stop,k,"stop"; break
            if hi>=target:
                xp,xj,reason=target,k,"target"; break
            if k==last:
                xp,xj,reason=cl,k,"time"; break
        if xj is None:
            i+=1; continue
        gross=xp/entry-1
        net=(xp*(1-cost_side))/(entry*(1+cost_side))-1
        out.append(TradeV2(
            symbol=symbol,signal_date=str(pd.Timestamp(dates[i]).date()),
            event_date=str(pd.Timestamp(dates[j]).date()),
            entry_date=str(pd.Timestamp(dates[entry_i]).date()),
            exit_date=str(pd.Timestamp(dates[xj]).date()),
            entry=float(entry),exit=float(xp),net_return=float(net),gross_return=float(gross),
            hold_days=int(xj-entry_i+1),reason=reason,event_score=float(A["acc"][j]),
            event_rvol=float(A["rv"][j]),event_age=int(age),
            down_volume_share_10=float(A["down_share"][i]),support_distance_pct=float(dist),
            rr_at_entry=float(rr),relative_strength_20=float(rs20[i]) if np.isfinite(rs20[i]) else np.nan,
            spy_regime=bool(spy_reg[i]),signal_green=bool(A["c"][i]>A["o"][i])
        ))
        i=xj+1
    return out


def summarize(trades):
    if not trades:
        return {"trades":0}
    r=np.array([x.net_return for x in trades],float)
    return {
        "trades":len(trades),
        "symbols_with_trades":len({x.symbol for x in trades}),
        "win_rate":float((r>0).mean()),
        "avg_return":float(r.mean()),
        "median_return":float(np.median(r)),
        "profit_factor":safe_pf(r),
        "t_stat":t_stat(r),
        "avg_hold_days":float(np.mean([x.hold_days for x in trades])),
        "best_trade":float(r.max()),"worst_trade":float(r.min()),
        "target_rate":float(np.mean(["target" in x.reason for x in trades])),
        "stop_rate":float(np.mean(["stop" in x.reason for x in trades])),
    }


def bootstrap_ci(trades,seed=17,n=2500):
    if len(trades)<20:
        return {}
    r=np.array([x.net_return for x in trades],float)
    rng=np.random.default_rng(seed)
    means=np.array([rng.choice(r,size=len(r),replace=True).mean() for _ in range(n)])
    return {"mean_return_ci95_low":float(np.quantile(means,.025)),
            "mean_return_ci95_high":float(np.quantile(means,.975))}
