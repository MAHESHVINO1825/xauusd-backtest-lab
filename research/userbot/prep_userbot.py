"""Replicate the user's M5 bot on 2010-2026 data using ITS OWN functions."""
import sys, time, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/claude/lab'); sys.path.insert(0,'/home/claude/bots')
import numpy as np, pandas as pd
import userbot_v2 as U
from core import data as D
t0=time.time()
m1=D.load_m1(); m1=m1[m1.time>='2010-01-01']
m5=m1.set_index('time').resample('5min',label='left',closed='left').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna(subset=['open']).reset_index()
m5=m5.rename(columns={'volume':'tick_volume'}); m5['spread']=0.0
d=U.tm_prepare(U.add_indicators(m5)); d['cost']=np.maximum(d.spread.fillna(0)*0.001,0.15)+0.05   # = the bot's own cost rule -> 0.20
print('M5 frame', len(d), round(time.time()-t0)); 
cands=U.detect_setups(d); hist=U.real_outcomes(d,cands); X=U.build_features(d,cands)
print('candidates', len(cands), round(time.time()-t0))
d.to_parquet('/home/claude/bots/m5.parquet'); hist.to_parquet('/home/claude/bots/hist.parquet'); X.to_parquet('/home/claude/bots/X.parquet')
# yearly walk-forward of the bot's own LightGBM model (it retrains monthly live; yearly here for time)
score=np.full(len(hist),np.nan)
for Y in range(2013,2027):
    start=pd.Timestamp(f'{Y}-01-01'); model,n=U.train_model(X,hist,start)
    sel=(hist.time>=start)&(hist.time<pd.Timestamp(f'{Y+1}-01-01'))
    score[sel.values]=model.predict(X[sel.values]); print(Y,'trained on',n,round(time.time()-t0),flush=True)
hist['score']=score; hist.to_parquet('/home/claude/bots/hist.parquet'); print('done', round(time.time()-t0))
