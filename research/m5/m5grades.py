"""User's 14 M5 setups + Agent Bull/Bear grades (out-of-sample agent probabilities), several trades at once."""
import numpy as np, pandas as pd
d = pd.read_parquet("/home/claude/bots/userbot_merged.parquet")
d["time"] = pd.to_datetime(d.time); d["close_time"] = d.time + pd.Timedelta(minutes=5) + pd.to_timedelta(d.held * 5, unit="min")
d["yr"] = d.time.dt.year; d["hr"] = d.time.dt.hour
b = d.dir > 0
d["g"] = np.where(d.score <= 0, "D", np.where(b, np.where((d.pb >= d.b65) & (d.pr <= d.r50), "A", np.where(d.pb >= d.b65, "B", np.where(d.pb >= d.b50, "C", "D"))),
                                         np.where((d.pr >= d.r65) & (d.pb <= d.b50), "A", np.where(d.pr >= d.r65, "B", np.where(d.pr >= d.r50, "C", "D")))))
days = d.time.dt.normalize().nunique()
pf = lambda r: r[r > 0].sum() / -r[r < 0].sum()
print(f"M5 candidates {len(d)} over {days} days ({d.time.min():%Y-%m} to {d.time.max():%Y-%m})")
for g in "ABCD":
    x = d[d.g == g]; a, b_ = x[x.yr <= 2019], x[x.yr >= 2020]
    print(f"  grade {g}: {len(x)/days:5.2f}/day  avg {x.R.mean():+.3f}R PF {pf(x.R):.2f} | 2013-19 {a.R.mean():+.3f}  2020-26 {b_.R.mean():+.3f}")
# setup whitelist chosen on 2013-2019 grade A/B only
ab = d[d.g.isin(["A", "B"])]
is_ = ab[ab.yr <= 2019].groupby("setup").R.agg(["mean", "count"])
white = is_[(is_["mean"] > 0.02) & (is_["count"] >= 50)].index.tolist()
print("\nsetups positive 2013-19 (grade A/B):", white)
def sim(x, max_open=4, caps=dict(A=4, B=2, C=1), per_bar=1, day_cap=15):
    x = x.sort_values(["time", "g"]); op = []; take = []; dc = {}
    for r in x.itertuples():
        op = [q for q in op if q[0] > r.time]; dk = r.time.normalize()
        if dc.get(dk, 0) >= day_cap or len(op) >= max_open or sum(q[1] == r.g for q in op) >= caps.get(r.g, 0): continue
        if any(q[2] != r.dir for q in op) or any(q[3] == r.setup for q in op) or sum(q[4] == r.time for q in op) >= per_bar: continue
        op.append((r.close_time, r.g, r.dir, r.setup, r.time)); take.append(r.Index); dc[dk] = dc.get(dk, 0) + 1
    t = d.loc[take]
    eq = t.sort_values("close_time").R.cumsum(); dd = (eq.cummax() - eq).max()
    yrs = t.groupby("yr").R.sum(); a, b_ = t[t.yr <= 2019].R, t[t.yr >= 2020].R
    usd = (t.R * t.risk_usd).sum()
    return f"{len(t)/days:4.2f}/day PF {pf(t.R):.2f} total {t.R.sum():+6.0f}R (${usd:+,.0f}) maxDD {dd:4.0f}R yrs+ {(yrs>0).sum()}/{len(yrs)} | 2013-19 {a.mean():+.3f} 2020-26 {b_.mean():+.3f} (t {b_.mean()/b_.std()*np.sqrt(len(b_)):+.1f}) avg risk ${t.risk_usd.mean():.1f}"
print("\nMulti-trade engine on M5 (max 4 open, one per setup, no hedge, 1 new per bar):")
for name, m in [("A only", d.g == "A"), ("A+B", d.g.isin(["A", "B"])), ("A+B+C", d.g.isin(["A", "B", "C"])),
                ("A only, IS-positive setups", (d.g == "A") & d.setup.isin(white)), ("A+B, IS-positive setups", d.g.isin(["A", "B"]) & d.setup.isin(white)),
                ("A+B, IS setups, London+NY 07-17 UTC", d.g.isin(["A", "B"]) & d.setup.isin(white) & d.hr.between(7, 16))]:
    print(f"  {name:38s} {sim(d[m])}")
