import numpy as np, pandas as pd
from orb import t, o, h, l, c, SESS, starts, run
X = pd.read_parquet("/home/claude/lab/agents/cache_features.parquet", columns=["decision_time"]); dt = X.decision_time
def load(s):
    p = np.full(len(X), np.nan); w = pd.read_parquet(f"/home/claude/lab/agents/v3ens_{s}_base.parquet"); p[w.i.values] = w.prob.values; return p
pb, pr = load("bull"), load("bear")
rq = lambda p, q: pd.Series(p, index=dt).rolling("90D", min_periods=2000).quantile(q).shift(1).values
A = pd.DataFrame(dict(dt=dt, pb=pb, pr=pr, b50=rq(pb, .5), b65=rq(pb, .65), r50=rq(pr, .5), r65=rq(pr, .65))).dropna().sort_values("dt")
A["bullA"] = (A.pb >= A.b65) & (A.pr <= A.r50); A["bearA"] = (A.pr >= A.r65) & (A.pb <= A.b50)
A["bullB"] = A.pb >= A.b65; A["bearB"] = A.pr >= A.r65
res = []
for name, (tz, hh, mm) in SESS.items():
    st = starts(tz, hh, mm); y = pd.to_datetime(st).year.values
    for N in (15, 30, 60):
        g = pd.merge_asof(pd.DataFrame(dict(k=np.arange(len(st)), at=pd.to_datetime(st) + pd.Timedelta(minutes=N))), A, left_on="at", right_on="dt", direction="backward", tolerance=pd.Timedelta(minutes=30))
        for T in (1.5, 2.0, 3.0):
            for H in (120, 240):
                R, usd, side, rng = run(t, o, h, l, c, st, N, T, H, 0, 0.30, 0.0, 1e9)
                for flt in ("A", "B"):
                    agree = ((side == 1) & g[f"bull{flt}"].fillna(False).values) | ((side == -1) & g[f"bear{flt}"].fillna(False).values)
                    ok = (~np.isnan(R)) & np.asarray(agree, dtype=bool); a, b = ok & (y <= 2019), ok & (y >= 2020)
                    if a.sum() < 30 or b.sum() < 30: continue
                    res.append(dict(session=name, N=N, T=T, H=H, agents=flt, is_R=R[a].mean(), is_t=R[a].mean() / R[a].std() * np.sqrt(a.sum()), n_is=a.sum(),
                                    os_R=R[b].mean(), os_t=R[b].mean() / R[b].std() * np.sqrt(b.sum()), n_os=b.sum(), os_usd=usd[b].sum()))
F = pd.DataFrame(res); pd.set_option("display.width", 220)
print(len(F), "variants. Top 12 by 2013-19 t:"); print(F.sort_values("is_t", ascending=False).head(12).round(3).to_string(index=False))
print("share positive OOS:", (F.os_R > 0).mean().round(2), " both:", ((F.os_R > 0) & (F.is_R > 0)).mean().round(2))
