"""Opening-range breakout on gold. Range = first N minutes after a session open (local time, DST-correct).
Entry: first M5 close beyond the range (after it is complete). Stop: other side of range (or range mid).
Target: T x risk. Exit at target/stop or after H minutes. One trade per session per day. Cost 0.30 per trade."""
import numpy as np, pandas as pd
from numba import njit
d = pd.read_parquet("m1.parquet")
t = d.time.values.astype("datetime64[ns]").astype(np.int64)
o, h, l, c = (d[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
yr = d.time.dt.year.to_numpy()
SESS = {"Tokyo 09:00": ("Asia/Tokyo", 9, 0), "London 08:00": ("Europe/London", 8, 0), "COMEX 08:20 NY": ("America/New_York", 8, 20), "NYSE 09:30 NY": ("America/New_York", 9, 30)}

def starts(tz, hh, mm):
    days = pd.date_range("2013-01-01", "2026-11-14", freq="D")
    loc = pd.DatetimeIndex([pd.Timestamp(x.date()).replace(hour=hh, minute=mm) for x in days]).tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    loc = loc[~loc.isna()]
    u = loc.tz_convert("UTC").tz_localize(None)
    u = u[u.dayofweek < 5]
    return u.values.astype("datetime64[ns]").astype(np.int64)

@njit(cache=True)
def run(t, o, h, l, c, st, N, T, H, stopmode, cost, minr, maxr):
    n = len(st); R = np.full(n, np.nan); side = np.zeros(n); usd = np.full(n, np.nan); rng = np.full(n, np.nan)
    i = 0
    for k in range(n):
        s = st[k]
        while i < len(t) and t[i] < s: i += 1
        if i >= len(t): break
        if t[i] - s > 120 * 60_000_000_000: continue          # market closed / holiday gap
        j = i; hi = -1e18; lo = 1e18
        while j < len(t) and t[j] < s + N * 60_000_000_000:
            hi = max(hi, h[j]); lo = min(lo, l[j]); j += 1
        if j >= len(t) or hi <= lo: continue
        r = hi - lo; rng[k] = r
        if r < minr or r > maxr: continue
        end = s + (N + H) * 60_000_000_000
        e = j; d_ = 0; ent = 0.0
        while e < len(t) and t[e] < end:                      # entry on an M5 close beyond the range
            if (t[e] // 60_000_000_000 + 1) % 5 == 0:
                if c[e] > hi: d_ = 1; ent = c[e]; break
                if c[e] < lo: d_ = -1; ent = c[e]; break
            e += 1
        if d_ == 0: continue
        stop = (lo if d_ == 1 else hi) if stopmode == 0 else (hi + lo) / 2
        risk = d_ * (ent - stop)
        if risk <= 0.05: continue
        tgt = ent + d_ * T * risk; ex = np.nan; m = e + 1
        while m < len(t) and t[m] < end:
            if d_ == 1:
                if l[m] <= stop: ex = min(o[m], stop); break
                if h[m] >= tgt: ex = max(o[m], tgt) if o[m] > tgt else tgt; break
            else:
                if h[m] >= stop: ex = max(o[m], stop); break
                if l[m] <= tgt: ex = min(o[m], tgt) if o[m] < tgt else tgt; break
            m += 1
        if np.isnan(ex):
            ex = c[min(m, len(t) - 1) - 1]
        side[k] = d_; usd[k] = d_ * (ex - ent) - cost; R[k] = usd[k] / risk
    return R, usd, side, rng

if __name__ == "__main__":
    res = []
    yr_of = lambda st: pd.to_datetime(st).year.values
    for name, (tz, hh, mm) in SESS.items():
        st = starts(tz, hh, mm); y = yr_of(st)
        for N in (15, 30, 60):
            for T in (1.0, 1.5, 2.0, 3.0):
                for H in (120, 240):
                    for sm in (0, 1):
                        R, usd, side, rng = run(t, o, h, l, c, st, N, T, H, sm, 0.30, 0.0, 1e9)
                        ok = ~np.isnan(R)
                        a, b = ok & (y <= 2019), ok & (y >= 2020)
                        f = lambda m: (R[m].mean(), R[m].mean() / R[m].std() * np.sqrt(m.sum()), usd[m].sum(), usd[m][usd[m] > 0].sum() / -usd[m][usd[m] < 0].sum(), m.sum())
                        ia, it, iu, ipf, ini = f(a); oa, ot, ou, opf, oni = f(b)
                        yrs = pd.Series(usd[ok]).groupby(y[ok]).sum()
                        res.append(dict(session=name, N=N, T=T, H=H, stop="range" if sm == 0 else "mid", is_R=ia, is_t=it, os_R=oa, os_t=ot, os_pf=opf, os_usd=ou, n_os=oni, yrs=f"{(yrs>0).sum()}/{len(yrs)}"))
    R_ = pd.DataFrame(res)
    R_.to_csv("orb_grid.csv", index=False)
    pd.set_option("display.width", 220)
    print("ORB grid:", len(R_), "variants. Top 15 chosen ONLY by 2013-2019 t, then their unseen 2020-2026 result:")
    print(R_.sort_values("is_t", ascending=False).head(15).round(3).to_string(index=False))
    print("\nShare of variants positive 2020-26:", (R_.os_R > 0).mean().round(2), "| positive in BOTH periods:", ((R_.os_R > 0) & (R_.is_R > 0)).mean().round(2))
    print("\nBest session overall (median OOS R across its variants):"); print(R_.groupby("session")[["is_R", "os_R"]].median().round(3))
