"""The user's engine (14 setups + LightGBM score + 3ATR stop / 2R / lock / trail) on M5, M15, M30, with and without agents."""
import sys, time, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/claude/lab'); sys.path.insert(0,'/home/claude/bots')
import numpy as np, pandas as pd
import userbot_v2 as U
from core import data as D
tf=int(sys.argv[1]); k=tf//5
t0=time.time()
m1=D.load_m1(); m1=m1[m1.time>='2010-01-01']
b=m1.set_index('time').resample(f'{tf}min',label='left',closed='left').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna(subset=['open']).reset_index()
b=b.rename(columns={'volume':'tick_volume'}); b['spread']=0.0
U.WARMUP=max(600,3000//k)
d=U.tm_prepare(U.add_indicators(b)); d['cost']=float(sys.argv[2]) if len(sys.argv)>2 else 0.20
cands=U.detect_setups(d)
bar=cands.bar.to_numpy().astype(np.int64)
max_bars=144//k
horizon=np.minimum(max_bars,(U.SLOTS-1-d.slot.to_numpy()[bar])//k).astype(np.int64)
R,out,held,mae,mfe=U._outcomes(bar,cands.dir.to_numpy().astype(np.int64),horizon,d.open.to_numpy(),d.high.to_numpy(),d.low.to_numpy(),d.close.to_numpy(),d.atr.to_numpy(),d.cost.to_numpy(),*[float(x) for x in U.PLAN])
h=cands.copy(); h['time']=d.time.to_numpy()[bar]; h['R']=R; h['held']=held; h['horizon']=horizon
X=U.build_features(d,cands)
score=np.full(len(h),np.nan)
for Y in range(2013,2027):
    s=pd.Timestamp(f'{Y}-01-01'); m,n=U.train_model(X,h,s); sel=((h.time>=s)&(h.time<pd.Timestamp(f'{Y+1}-01-01'))).values
    score[sel]=m.predict(X[sel])
h['score']=score
slot=d.slot.values[bar]; atr=d.atr.values[bar]
h['risk_usd']=3*atr; h['close_time']=h.time+pd.Timedelta(minutes=tf)
ok=h.R.notna()&h.score.notna()&(slot>=6)&((U.SLOTS-1-slot)>=12)&(d.cost.values[bar]/(3*atr)<=U.MAX_COST_R)&(h.time>='2013-01-01')
h=h[ok.values]
X0=pd.read_parquet('/home/claude/lab/agents/cache_features.parquet',columns=['decision_time']); t=X0.decision_time
def load(s):
    p=np.full(len(X0),np.nan); w=pd.read_parquet(f'/home/claude/lab/agents/v3ens_{s}_base.parquet'); p[w.i.values]=w.prob.values; return p
pb,pr=load('bull'),load('bear'); rq=lambda p,q: pd.Series(p,index=t).rolling('90D',min_periods=2000).quantile(q).shift(1).values
A=pd.DataFrame({'dt':t,'pb':pb,'pr':pr,'b50':rq(pb,.5),'b65':rq(pb,.65),'r50':rq(pr,.5),'r65':rq(pr,.65)}).dropna()
g=pd.merge_asof(h.sort_values('close_time'),A,left_on='close_time',right_on='dt',direction='backward',tolerance=pd.Timedelta(minutes=45)).dropna(subset=['pb']).sort_values(['bar','setup','dir']).reset_index(drop=True)
L=g.dir==1; own=np.where(L,g.pb,g.pr); opp=np.where(L,g.pr,g.pb)
strict=own>=np.where(L,g.b65,g.r65); dual=strict&(opp<=np.where(L,g.r50,g.b50))
g.to_parquet(f'merged_M{tf}_{d.cost.iat[0]:.2f}.parquet')
def st(name,mask,use_score=True):
    x=g[mask].copy()
    if not use_score: x['score']=0.0
    tk=U._one_at_a_time(x); s=U.r_stats(tk.R.values); yr=tk.groupby(tk.time.dt.year).R.sum()
    print(f"M{tf:<2} {name:34s} trades {s['n']:5d} ({s['n']/14:4.0f}/yr) avg {s['avg']:+.3f}R total {s['total']:+7.1f}R PF {s['pf']:.2f} win {s['win']*100:.0f}% ${(tk.R*tk.risk_usd).sum():+6.0f} yrs+ {(yr>0).sum()}/14 t {s['t']:+.1f}",flush=True)
    return tk
st('every setup',np.ones(len(g),bool),False)
st('bot model (live rule)',(g.score>0).values)
st('bot model + agents strict',(g.score>0).values&strict)
tk=st('bot model + agents dual-strict',(g.score>0).values&dual)
st('bot model > 0.05 + agents dual-strict',(g.score>0.05).values&dual)
print(f"M{tf} dual-strict R by year:", tk.groupby(tk.time.dt.year).R.sum().round(1).to_dict(), f"[{time.time()-t0:.0f}s]")
