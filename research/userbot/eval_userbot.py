import sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/claude/lab'); sys.path.insert(0,'/home/claude/bots')
import numpy as np, pandas as pd
import userbot_v2 as U
d=pd.read_parquet('m5.parquet'); h=pd.read_parquet('hist.parquet')
X=pd.read_parquet('/home/claude/lab/agents/cache_features.parquet',columns=['decision_time']); t=X.decision_time
def load(s):
    p=np.full(len(X),np.nan); w=pd.read_parquet(f'/home/claude/lab/agents/v3ens_{s}_base.parquet'); p[w.i.values]=w.prob.values; return p
pb,pr=load('bull'),load('bear')
rq=lambda p,q: pd.Series(p,index=t).rolling('90D',min_periods=2000).quantile(q).shift(1).values
A=pd.DataFrame({'dt':t,'pb':pb,'pr':pr,'b50':rq(pb,.5),'b65':rq(pb,.65),'r50':rq(pr,.5),'r65':rq(pr,.65)}).dropna()
bar=h.bar.values; slot=d.slot.values[bar]; atr=d.atr.values[bar]; cost=d.cost.values[bar]
h['close_time']=h.time+pd.Timedelta(minutes=5)
h['risk_usd']=3*atr*1.0                                   # 0.01 lot = 1 oz -> $ per R
ok=h.R.notna()&h.score.notna()&(slot>=6)&(U.SLOTS-1-slot>=12)&(cost/(3*atr)<=U.MAX_COST_R)&(h.time>='2013-01-01')
h=h[ok.values].copy()
g=pd.merge_asof(h.sort_values('close_time'),A,left_on='close_time',right_on='dt',direction='backward',tolerance=pd.Timedelta(minutes=30))
g=g.dropna(subset=['pb']).sort_values(['bar','setup','dir']).reset_index(drop=True)
L=g.dir==1
own=np.where(L,g.pb,g.pr); opp=np.where(L,g.pr,g.pb)
o50=np.where(L,g.b50,g.r50); o65=np.where(L,g.b65,g.r65); p50=np.where(L,g.r50,g.b50)
std=own>=o50; strict=own>=o65; dual=strict&(opp<=p50)
def stats(name,mask,score_col='score'):
    x=g[mask].copy()
    if score_col is None: x['score']=0.0
    tk=U._one_at_a_time(x)
    yr=tk.groupby(tk.time.dt.year).R.sum(); s=U.r_stats(tk.R.values)
    usd=(tk.R*tk.risk_usd).sum(); 
    print(f"{name:46s} trades {s['n']:5d} ({s['n']/14:.0f}/yr) | avg {s['avg']:+.3f}R | total {s['total']:+7.1f}R | PF {s['pf']:.2f} | win {s['win']*100:.0f}% | ${usd:+7.0f} @0.01 | yrs+ {(yr>0).sum()}/14 | t {s['t']:+.1f}")
    return tk
print("=== The user's M5 bot replicated, 2013-2026 (14 years), one trade at a time, costs 0.20 ===")
stats('A. every setup, no model',np.ones(len(g),bool),None)
base=stats("B. bot's own model (score > 0)  = LIVE RULE",(g.score>0).values)
stats('C. agents only (strict), no bot model',strict,None)
stats('D. bot model + agents standard',(g.score>0).values&std)
best=stats('E. bot model + agents strict',(g.score>0).values&strict)
stats('F. bot model + agents dual-strict',(g.score>0).values&dual)
for nm,tk in [('B live rule',base),('E +agents strict',best)]:
    print(f"\n{nm} — R by year:", tk.groupby(tk.time.dt.year).R.sum().round(1).to_dict())
    print(f"{nm} — buys {int((tk.dir==1).sum())} {tk.R[tk.dir==1].sum():+.1f}R | sells {int((tk.dir==-1).sum())} {tk.R[tk.dir==-1].sum():+.1f}R")
g.to_parquet('userbot_merged.parquet')
