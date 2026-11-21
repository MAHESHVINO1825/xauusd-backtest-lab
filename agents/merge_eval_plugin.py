import warnings, sys; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from core import data as D
from agents import merge_backtest as M
tag=sys.argv[1]
X=pd.read_parquet('agents/cache_features.parquet'); L=pd.read_parquet('agents/cache_labels.parquet')
t=X.decision_time
def load(side):
    p=np.full(len(X),np.nan); w=pd.read_parquet(f'agents/{tag}_{side}_base.parquet'); p[w.i.values]=w.prob.values; return p
pb,pr=load('bull'),load('bear')
def roll_q(p,q):
    s=pd.Series(p,index=t).rolling('90D',min_periods=2000).quantile(q).shift(1)   # prior 90 days only
    return s.values
th={('bull',.5):roll_q(pb,.5),('bull',.65):roll_q(pb,.65),('bear',.5):roll_q(pr,.5),('bear',.65):roll_q(pr,.65)}
years=t.dt.year.values; arr=D.m1_arrays(0.20); Y0=2013; rng=np.random.default_rng(1)
modes={'standard':lambda: (pb>=th['bull',.5], pr>=th['bear',.5]),
       'strict':lambda: (pb>=th['bull',.65], pr>=th['bear',.65]),
       'dual-strict':lambda: ((pb>=th['bull',.65])&(pr<=th['bear',.5]), (pr>=th['bear',.65])&(pb<=th['bull',.5]))}
res=[]
for name,(ls,ss) in M.bot_signals(X).items():
    ok=np.isfinite(pb)&np.isfinite(pr)&(years>=Y0); ls=ls&ok; ss=ss&ok
    base=M.summarise(M.run(ls,ss,L,arr),Y0)
    line=f"{name:26s} ALONE R-PF {base['pf_r']:.2f} {base['R']:+5.0f}R ${base['usd']:+6.0f} yrs+{base['yrs']}/14"
    for mn,f in modes.items():
        al,as_=f(); al=ls&al; as_=ss&as_
        s=M.summarise(M.run(al,as_,L,arr),Y0)
        ctrl=[M.summarise(M.run(ls&(rng.random(len(X))<al.sum()/max(ls.sum(),1)), ss&(rng.random(len(X))<as_.sum()/max(ss.sum(),1)),L,arr),Y0)['R'] for _ in range(40)]
        beat=np.mean(np.array(ctrl)<s['R'])*100
        line+=f" | {mn}: n{s['n']} PF {s['pf']:.2f} R-PF {s['pf_r']:.2f} {s['R']:+5.0f}R ${s['usd']:+5.0f} DD${s['dd']:.0f} yrs+{s['yrs']} beat{beat:.0f}%"
        res.append(dict(bot=name,mode=mn,**{k:v for k,v in s.items()},beat=beat,base_R=base['R'],base_pf_r=base['pf_r'],base_usd=base['usd'],base_yrs=base['yrs']))
    print(line, flush=True)
pd.DataFrame(res).to_csv(f'agents/merge_results_{tag}.csv',index=False)
