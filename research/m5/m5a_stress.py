import numpy as np, pandas as pd
exec(open("m5grades.py").read().split('print(f"M5 candidates')[0])
base = d.copy()
pf = lambda r: r[r > 0].sum() / -r[r < 0].sum()
def sim_take(x, max_open=4):
    x = x.sort_values(["time"]); op = []; take = []; dc = {}
    for r in x.itertuples():
        op = [q for q in op if q[0] > r.time]; dk = r.time.normalize()
        if dc.get(dk, 0) >= 15 or len(op) >= max_open: continue
        if any(q[2] != r.dir for q in op) or any(q[3] == r.setup for q in op) or any(q[4] == r.time for q in op): continue
        op.append((r.close_time, r.g, r.dir, r.setup, r.time)); take.append(r.Index); dc[dk] = dc.get(dk, 0) + 1
    return take
A_ = base[base.g == "A"]
take = sim_take(A_)
t = base.loc[take].copy()
print(f"M5 grade-A engine: {len(t)} trades, {len(t)/days:.2f}/day")
for extra, lab in ((0.0, "cost 0.20 (as tested)"), (0.10, "cost 0.30"), (0.20, "cost 0.40")):
    R = t.R - extra / t.risk_usd
    a, b = R[t.yr <= 2019], R[t.yr >= 2020]
    yrs = R.groupby(t.yr).sum()
    print(f"  {lab:22s} PF {pf(R):.2f} total {R.sum():+5.0f}R | 2013-19 {a.mean():+.3f} 2020-26 {b.mean():+.3f} (t {b.mean()/b.std()*np.sqrt(len(b)):+.1f}) yrs+ {(yrs>0).sum()}/{len(yrs)}")
R3 = t.R - 0.10 / t.risk_usd
print("\nYear by year at cost 0.30 (R, trades, avg risk $):")
print(pd.DataFrame(dict(R=R3.groupby(t.yr).sum().round(1), n=t.groupby("yr").size(), risk=t.groupby("yr").risk_usd.mean().round(1))).T.to_string())
print("\nBy setup (cost 0.30):"); print(pd.DataFrame(dict(n=t.groupby("setup").size(), avgR=R3.groupby(t.setup).mean().round(3), totR=R3.groupby(t.setup).sum().round(1))).sort_values("totR").to_string())
print("\nBy UTC hour block (cost 0.30):"); hb = pd.cut(t.hr, [-1, 6, 12, 16, 23], labels=["00-06 Asia", "07-12 London", "13-16 NY", "17-23 late"])
print(pd.DataFrame(dict(n=t.groupby(hb).size(), avgR=R3.groupby(hb).mean().round(3))).to_string())
# overlap with the M30 grade-A engine (same day)
m30 = pd.read_parquet("/home/claude/bots/merged_M30.parquet"); m30["time"] = pd.to_datetime(m30.time)
bb = m30.dir > 0
a30 = m30[(m30.score > 0) & np.where(bb, (m30.pb >= m30.b65) & (m30.pr <= m30.r50), (m30.pr >= m30.r65) & (m30.pb <= m30.b50))]
d5 = R3.groupby(t.time.dt.normalize()).sum(); d30 = a30.R.groupby(a30.time.dt.normalize()).sum()
j = pd.concat([d5, d30], axis=1, keys=["m5", "m30"]).fillna(0)
print(f"\nDaily R correlation M5-A vs M30-A: {j.corr().iloc[0,1]:.2f}")
rec = t[t.time >= "2025-11-01"]; Rr = R3[t.time >= "2025-11-01"]
print(f"Last 12 months: {len(rec)} trades ({len(rec)/250:.1f}/day) avg {Rr.mean():+.3f}R PF {pf(Rr):.2f} total {Rr.sum():+.1f}R, avg risk ${rec.risk_usd.mean():.1f}/trade at 0.01 lot")
