"""The five pre-registered hypotheses. Grids are fixed here BEFORE results are seen."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core import data as D
from core import indicators as I
from core.engine import simulate

from . import harness as H

# ------------------------------------------------------------------ helpers


def tod_minutes(b):
    return (b["hour"].values * 60 + b["minute"].values).astype(int)


def hm(s):
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def in_window(tod, start, end):
    s, e = hm(start), hm(end)
    return (tod >= s) & (tod < e) if s < e else (tod >= s) | (tod < e)


def daily_trend(lookback) -> pd.Series:
    """+1/-1 time-series momentum signal known at each day's close (22:00 UTC)."""
    c = H.daily()["close"]
    if lookback == "blend":
        s = sum(np.sign(c / c.shift(L) - 1) for L in (21, 63, 252))
        return np.sign(s).fillna(0)
    return np.sign(c / c.shift(int(lookback)) - 1).fillna(0)


def trend_on_bars(lookback, b=None) -> np.ndarray:
    """Daily signal mapped to M15 bars without look-ahead: bars of day k use the signal from the
    close of day k-1; the 22:00 rollover hour keeps the older signal (no trading in the spread spike)."""
    b = H.bars15() if b is None else b
    s = daily_trend(lookback)
    days = pd.Index(s.index)
    k = days.get_indexer(b["day"].values)
    prev1 = np.where(k >= 1, s.values[np.clip(k - 1, 0, None)], 0)
    prev2 = np.where(k >= 2, s.values[np.clip(k - 2, 0, None)], 0)
    return np.where(b["hour"].values == 22, prev2, prev1).astype(np.int8)


# ------------------------------------------------------------------ S1 time-series momentum
S1_GRID = [dict(lookback=L, mode=m) for L in (21, 63, 126, 252, "blend") for m in ("long_short", "long_only")]


def s1_positions(lookback, mode, b=None):
    p = trend_on_bars(lookback, b)
    if mode == "long_only":
        p = np.where(p > 0, 1, 0)
    return p.astype(np.int8)


# ------------------------------------------------------------------ S2 / S3 Asian drift
S2_GRID = [dict(entry=e, exit=x) for e in ("23:15", "00:00", "01:00") for x in ("05:00", "06:00", "07:00")]


def s2_positions(entry, exit, b=None):
    b = H.bars15() if b is None else b
    w = in_window(tod_minutes(b), entry, exit)
    return w.astype(np.int8)


def s3_positions(entry, exit, b=None):
    return (s2_positions(entry, exit, b) * (trend_on_bars("blend", b) > 0)).astype(np.int8)


# ------------------------------------------------------------------ S5 US-session short when trend down
S5_GRID = [dict(entry=e, exit=x) for e in ("13:30", "14:30", "15:30") for x in ("17:00", "19:00", "20:30")]


def s5_positions(entry, exit, b=None):
    b = H.bars15() if b is None else b
    w = in_window(tod_minutes(b), entry, exit)
    return -(w & (trend_on_bars("blend", b) < 0)).astype(np.int8)


# ------------------------------------------------------------------ S4 volatility-expansion breakout
S4_GRID = [dict(tf=tf, n=n, sl=sl, rr=rr) for tf in ("H1", "H4") for n in (10, 20) for (sl, rr) in ((1.5, 2.0), (2.0, 3.0))]
S4_HOLD_MIN = 5 * 24 * 60

_m1 = {}


def m1(spread_mult=1.0):
    key = spread_mult
    if key not in _m1:
        _m1[key] = D.m1_arrays(H.SPREAD * spread_mult)
    return _m1[key]


def s4_signals(tf, n):
    b = D.bars(tf)
    h, l, c = b["high"].values, b["low"].values, b["close"].values
    a = I.atr(h, l, c, 14)
    pr = pd.Series(a).rolling(250, min_periods=100).rank(pct=True).values * 100
    was_low = pd.Series(pr < 30).rolling(10, min_periods=1).max().values.astype(bool)
    expand = (pr >= 50) & was_low
    hh = pd.Series(h).rolling(n).max().shift(1).values
    ll = pd.Series(l).rolling(n).min().shift(1).values
    ls = I.edge(expand & (c > hh))
    ss = I.edge(expand & (c < ll)) & ~ls
    return b, a, ls, ss


def s4_run(tf, n, sl, rr, spread_mult=1.0, delay=0, sigs=None):
    b, a, ls, ss = sigs if sigs is not None else s4_signals(tf, n)
    if delay:
        ls = np.r_[np.zeros(delay, bool), ls[:-delay]]
        ss = np.r_[np.zeros(delay, bool), ss[:-delay]]
    mm = m1(spread_mult)
    ent = D.entry_index(b)
    res = simulate(ls, ss, ent, a, mm["open"], mm["high"], mm["low"], mm["close"], mm["spread"], mm["minutes"],
                   sl, rr, S4_HOLD_MIN, 10.0, H.SLIP, 0.0, H.SWAP_PER_LOT_NIGHT, H.LOT, 1e9)
    return s4_trades(res, mm)


def s4_trades(res, mm):
    ent, ext, d, lot, ep, xp, pnl = res[:7]
    t = pd.Series(mm["time"][ent])
    tr = pd.DataFrame({"entry_time": t.values, "exit_time": mm["time"][ext], "side": d, "pnl": pnl,
                       "ret": pnl / (ep * H.CONTRACT * H.LOT) * 1e4, "reason": res[8]})
    tr["era"] = H.era_of(tr["entry_time"])
    return tr


# ------------------------------------------------------------------ S6 reopen drift (registered 2026-11-21 after the
# UTC fix, from the 2020-26 hourly profile; the 2015-2019 era had NOT been viewed on corrected data at that point)
S6_GRID = [dict(entry=e, exit=x) for e in ("22:15", "22:30", "23:00") for x in ("00:00", "01:00", "02:00")]


def s6_positions(entry, exit, b=None):
    return s2_positions(entry, exit, b)


# ------------------------------------------------------------------ S7 reopen drift anchored to NY 18:00 reopen
S7_GRID = [dict(entry=e, exit=x) for e in (15, 30) for x in (60, 120)]


def s7_positions(entry, exit, b=None):
    b = H.bars15() if b is None else b
    s = b["since_reopen"].values
    return ((s >= entry) & (s < exit)).astype(np.int8)
