import numpy as np, pandas as pd
exec(open("m5a_stress.py").read().split("R3 = t.R")[0].replace('print(f"M5 grade-A', '_=(f"').replace('    print(f"  {lab', '    _=(f"  {lab'))
pf = lambda r: r[r > 0].sum() / -r[r < 0].sum()
for cost in (0.30, 0.40):
    R = t.R - (cost - 0.20) / t.risk_usd
    cr = cost / t.risk_usd          # cost as a share of the stop distance
    print(f"\nTotal cost {cost:.2f}: grade-A M5 trades bucketed by cost/risk")
    for lo, hi in ((0, .02), (.02, .03), (.03, .05), (.05, .08), (.08, 1)):
        m = (cr >= lo) & (cr < hi)
        if m.sum() < 30: continue
        x = R[m]; yrs = sorted(t.yr[m].unique())
        print(f"  cost {lo:.0%}-{hi:.0%} of risk: {m.sum():5d} trades  avg {x.mean():+.3f}R  PF {pf(x):.2f}  t {x.mean()/x.std()*np.sqrt(len(x)):+.1f}  years {yrs[0]}-{yrs[-1]} ({len(yrs)})")
    for cap in (0.03, 0.04, 0.05):
        m = cr <= cap; x = R[m]; yv = x.groupby(t.yr[m]).sum()
        a, b = x[t.yr[m] <= 2019], x[t.yr[m] >= 2020]
        print(f"  RULE cost <= {cap:.0%} of risk: {m.sum()} trades ({m.sum()/days:.2f}/day overall), PF {pf(x):.2f}, total {x.sum():+.0f}R, yrs+ {(yv>0).sum()}/{len(yv)}, "
              f"2013-19 {a.mean() if len(a) else float('nan'):+.3f} (n {len(a)}) 2020-26 {b.mean():+.3f} (n {len(b)}, t {b.mean()/b.std()*np.sqrt(len(b)):+.1f})")
