import numpy as np, pandas as pd
exec(open("m5a_costr.py").read().split("for cost in")[0])
pf = lambda r: r[r > 0].sum() / -r[r < 0].sum()
R = t.R - 0.10 / t.risk_usd; cr = 0.30 / t.risk_usd; m = cr <= 0.08; x = R[m]; tt = t[m]
yv = x.groupby(tt.yr).sum(); a, b = x[tt.yr <= 2019], x[tt.yr >= 2020]
eq = x.cumsum(); dd = (eq.cummax() - eq).max()
print(f"M5 grade A + cost <= 8% of risk (cost 0.30): {len(x)} trades {len(x)/days:.2f}/day, avg {x.mean():+.3f}R PF {pf(x):.2f} total {x.sum():+.0f}R maxDD {dd:.0f}R yrs+ {(yv>0).sum()}/{len(yv)}")
print(f"  2013-19 {a.mean():+.3f}R (n {len(a)}, t {a.mean()/a.std()*np.sqrt(len(a)):+.1f}) | 2020-26 {b.mean():+.3f}R (n {len(b)}, t {b.mean()/b.std()*np.sqrt(len(b)):+.1f})")
print("  by year:", yv.round(1).to_dict())
rec = tt.time >= "2025-11-01"
print(f"  last 12 months: {rec.sum()} trades ({rec.sum()/250:.1f}/day) avg {x[rec].mean():+.3f}R PF {pf(x[rec]):.2f} ${(x[rec]*tt.risk_usd[rec]).sum():+,.0f} at 0.01 lot, avg risk ${tt.risk_usd[rec].mean():.1f}")
# random control: same trades count, random grade (all candidates with score>0, same cost rule)
rng = np.random.default_rng(7); pool = d[(d.score > 0) & (0.30 / d.risk_usd <= 0.08)]
Rp = pool.R - 0.10 / pool.risk_usd
sims = [Rp.sample(len(x), random_state=int(s)).mean() for s in rng.integers(0, 1e9, 300)]
print(f"  random control (score>0, same cost rule, no agent grade): avg {np.mean(sims):+.3f}R, 95th pct {np.percentile(sims,95):+.3f}R -> grade A {'beats' if x.mean() > np.percentile(sims,95) else 'does NOT beat'} it")
