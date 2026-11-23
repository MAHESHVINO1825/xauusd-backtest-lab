import numpy as np, pandas as pd
from numba import njit
d = pd.read_parquet("m1a.parquet"); d = d[d.time >= "2024-12-01"].reset_index(drop=True)
t = d.time.values; o, h, l, c, a = (d[k].to_numpy(np.float64) for k in ("open", "high", "low", "close", "atr"))
yr = d.time.dt.year.to_numpy(); hr = d.time.dt.hour.to_numpy()
# minutes since previous bar (skip trades across gaps / weekend)
gap = np.r_[999, np.diff(t).astype("timedelta64[m]").astype(np.int64)]
ema = lambda x, n: pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()
e20, e50 = ema(c, 20), ema(c, 50)
# agents (out-of-sample walk-forward probabilities, 15-min grid) -> state per M1 bar
X = pd.read_parquet("/home/claude/lab/agents/cache_features.parquet", columns=["decision_time"]); dt = X.decision_time
def load(s):
    p = np.full(len(X), np.nan); w = pd.read_parquet(f"/home/claude/lab/agents/v3ens_{s}_base.parquet"); p[w.i.values] = w.prob.values; return p
pb, pr = load("bull"), load("bear")
rq = lambda p, q: pd.Series(p, index=dt).rolling("90D", min_periods=2000).quantile(q).shift(1).values
A = pd.DataFrame(dict(dt=dt, pb=pb, pr=pr, b50=rq(pb, .5), b65=rq(pb, .65), r50=rq(pr, .5), r65=rq(pr, .65))).dropna().sort_values("dt")
A["st"] = np.where((A.pb >= A.b65) & (A.pr <= A.r50), 1, np.where((A.pr >= A.r65) & (A.pb <= A.b50), -1, 0))
g = pd.merge_asof(pd.DataFrame(dict(ct=d.time + pd.Timedelta(minutes=1))), A[["dt", "st"]], left_on="ct", right_on="dt", direction="backward", tolerance=pd.Timedelta(minutes=20))
agent = g.st.fillna(0).to_numpy()

@njit(cache=True)
def sim(sig, o, h, l, c, a, gap, S, T, H, cost):
    """sig[i] = +1/-1 entry at close of bar i. One trade at a time. Returns entry idx, R, usd."""
    n = len(c); out_i = []; out_R = []; out_u = []; i = 0
    while i < n - 1:
        s = sig[i]
        if s == 0 or a[i] <= 0:
            i += 1; continue
        risk = S * a[i]; ent = c[i]; stop = ent - s * risk; tgt = ent + s * T * risk; ex = np.nan; j = i + 1
        while j < n and j <= i + H:
            if gap[j] > 5: ex = c[j - 1]; break
            if s == 1:
                if l[j] <= stop: ex = min(o[j], stop); break
                if h[j] >= tgt: ex = tgt; break
            else:
                if h[j] >= stop: ex = max(o[j], stop); break
                if l[j] <= tgt: ex = tgt; break
            j += 1
        if np.isnan(ex): ex = c[min(j, n - 1) if j <= i + H else j - 1]
        u = s * (ex - ent) - cost
        out_i.append(i); out_R.append(u / risk); out_u.append(u); i = j + 1
    return np.array(out_i), np.array(out_R), np.array(out_u)

rng_ = h - l; body = c - o
pos = np.where(rng_ > 0, (c - l) / np.where(rng_ > 0, rng_, 1), 0.5)
prev = lambda x: np.r_[x[0], x[:-1]]
hiN = {N: pd.Series(h).rolling(N).max().shift(1).to_numpy() for N in (15, 30, 60)}
loN = {N: pd.Series(l).rolling(N).min().shift(1).to_numpy() for N in (15, 30, 60)}
active = (gap[:] <= 5) & (yr >= 2025)
fam = {}
for k in (2.0, 3.0):
    up = (rng_ > k * prev(a)) & (pos > 0.8) & (body > 0); dn = (rng_ > k * prev(a)) & (pos < 0.2) & (body < 0)
    fam[f"burst{k:g}-follow"] = np.where(up, 1, np.where(dn, -1, 0))
    fam[f"burst{k:g}-fade"] = np.where(up, -1, np.where(dn, 1, 0))
    fam[f"burst{k:g}-follow+agentA"] = np.where(up & (agent == 1), 1, np.where(dn & (agent == -1), -1, 0))
    fam[f"burst{k:g}-fade+agentA"] = np.where(up & (agent == -1), -1, np.where(dn & (agent == 1), 1, 0))
cross_up = (c > e20) & (prev(c) <= prev(e20)); cross_dn = (c < e20) & (prev(c) >= prev(e20))
fam["agentA pullback-reclaim EMA20"] = np.where(cross_up & (agent == 1) & (e20 > e50), 1, np.where(cross_dn & (agent == -1) & (e20 < e50), -1, 0))
for N in (15, 30, 60):
    fam[f"agentA break {N}-bar high/low"] = np.where((c > hiN[N]) & (agent == 1), 1, np.where((c < loN[N]) & (agent == -1), -1, 0))
    fam[f"break {N}-bar (no agent)"] = np.where(c > hiN[N], 1, np.where(c < loN[N], -1, 0))
res = []
for name, sig in fam.items():
    sig = np.where(active, sig, 0).astype(np.int64)
    for S in (2.0, 3.0, 5.0):
        for T in (1.0, 1.5, 2.0, 3.0):
            for H in (15, 30, 60, 120):
                ii, R, u = sim(sig, o, h, l, c, a, gap, S, T, H, 0.30)
                if len(R) < 60: continue
                y = yr[ii]; A_, B_ = R[y == 2025], R[y == 2026]
                if len(A_) < 30 or len(B_) < 30: continue
                res.append(dict(family=name, S=S, T=T, H=H, n25=len(A_), R25=A_.mean(), t25=A_.mean() / A_.std() * np.sqrt(len(A_)),
                                n26=len(B_), R26=B_.mean(), t26=B_.mean() / B_.std() * np.sqrt(len(B_)), usd26=u[y == 2026].sum(),
                                pf26=u[y == 2026][u[y == 2026] > 0].sum() / -u[y == 2026][u[y == 2026] < 0].sum(), perday26=len(B_) / 225))
F = pd.DataFrame(res); F.to_csv("m1_grid.csv", index=False)
pd.set_option("display.width", 230)
print(len(F), "variants. Top 15 chosen ONLY on 2025, then unseen 2026 (Jan-Nov). Cost 0.30, $ at 0.01 lot:")
print(F.sort_values("t25", ascending=False).head(15).round(3).to_string(index=False))
print("\nshare positive 2026:", (F.R26 > 0).mean().round(2), "| positive both years:", ((F.R26 > 0) & (F.R25 > 0)).mean().round(2))
print("\nBest variant per family on 2025, its 2026 result:")
best = F.loc[F.groupby("family").t25.idxmax()].sort_values("t25", ascending=False)
print(best[["family", "S", "T", "H", "R25", "t25", "R26", "t26", "pf26", "perday26", "usd26"]].round(3).to_string(index=False))
