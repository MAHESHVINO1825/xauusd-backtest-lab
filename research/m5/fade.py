"""Fade the opening-range breakout: when an M5 candle closes beyond the range, trade back INTO the range.
Stop = S x range beyond the entry, target = T x risk (or the range middle). One trade per session per day. Cost 0.30."""
import numpy as np, pandas as pd
from numba import njit
from orb import t, o, h, l, c, SESS, starts

@njit(cache=True)
def run(t, o, h, l, c, st, N, S, T, H, tmid, cost):
    n = len(st); R = np.full(n, np.nan); usd = np.full(n, np.nan); i = 0
    for k in range(n):
        s = st[k]
        while i < len(t) and t[i] < s: i += 1
        if i >= len(t): break
        if t[i] - s > 120 * 60_000_000_000: continue
        j = i; hi = -1e18; lo = 1e18
        while j < len(t) and t[j] < s + N * 60_000_000_000:
            hi = max(hi, h[j]); lo = min(lo, l[j]); j += 1
        if j >= len(t) or hi <= lo: continue
        r = hi - lo; end = s + (N + H) * 60_000_000_000; e = j; d_ = 0; ent = 0.0
        while e < len(t) and t[e] < end:
            if (t[e] // 60_000_000_000 + 1) % 5 == 0:
                if c[e] > hi: d_ = -1; ent = c[e]; break
                if c[e] < lo: d_ = 1; ent = c[e]; break
            e += 1
        if d_ == 0: continue
        risk = S * r
        stop = ent - d_ * risk
        tgt = (hi + lo) / 2 if tmid else ent + d_ * T * risk
        if d_ * (tgt - ent) <= 0: continue
        ex = np.nan; m = e + 1
        while m < len(t) and t[m] < end:
            if d_ == 1:
                if l[m] <= stop: ex = min(o[m], stop); break
                if h[m] >= tgt: ex = tgt; break
            else:
                if h[m] >= stop: ex = max(o[m], stop); break
                if l[m] <= tgt: ex = tgt; break
            m += 1
        if np.isnan(ex): ex = c[min(m, len(t) - 1) - 1]
        usd[k] = d_ * (ex - ent) - cost; R[k] = usd[k] / risk
    return R, usd

res = []
for name, (tz, hh, mm) in SESS.items():
    st = starts(tz, hh, mm); y = pd.to_datetime(st).year.values
    for N in (15, 30, 60):
        for S in (0.5, 1.0):
            for T, tm in ((1.0, 0), (1.5, 0), (2.0, 0), (0, 1)):
                for H in (60, 120, 240):
                    R, usd = run(t, o, h, l, c, st, N, S, T, H, tm, 0.30)
                    ok = ~np.isnan(R); a, b = ok & (y <= 2019), ok & (y >= 2020)
                    f = lambda m: (R[m].mean(), R[m].mean() / R[m].std() * np.sqrt(m.sum()), usd[m].sum(), usd[m][usd[m] > 0].sum() / -usd[m][usd[m] < 0].sum())
                    ia, it, iu, ipf = f(a); oa, ot, ou, opf = f(b)
                    yrs = pd.Series(usd[ok]).groupby(y[ok]).sum()
                    res.append(dict(session=name, N=N, stopx=S, tgt="mid" if tm else T, H=H, is_R=ia, is_t=it, is_pf=ipf, os_R=oa, os_t=ot, os_pf=opf, os_usd=ou, n_os=b.sum(), yrs=f"{(yrs>0).sum()}/{len(yrs)}"))
F = pd.DataFrame(res); F.to_csv("fade_grid.csv", index=False)
pd.set_option("display.width", 220)
print("FADE grid:", len(F), "variants. Top 15 by 2013-2019 t, then unseen 2020-2026:")
print(F.sort_values("is_t", ascending=False).head(15).round(3).to_string(index=False))
print("\nShare positive 2020-26:", (F.os_R > 0).mean().round(2), "| positive both periods:", ((F.os_R > 0) & (F.is_R > 0)).mean().round(2))
print(F.groupby("session")[["is_R", "os_R"]].median().round(3))
