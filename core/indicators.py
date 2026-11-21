"""Indicator library. Each indicator returns a (bull, bear) pair of boolean
arrays describing its state at every bar CLOSE (no look-ahead).

Directional indicators vote long or short. Filters (direction=None) are true for
both sides and only restrict *when* trades can happen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

# name, group, directional?, plain-English bull rule (bear is the mirror)
INDICATORS = [
    ("EMA cross", "Trend", True, "EMA 9 above EMA 21"),
    ("SMA cross", "Trend", True, "SMA 20 above SMA 50"),
    ("Supertrend", "Trend", True, "Supertrend (10, 3) in up mode"),
    ("ADX", "Trend", True, "ADX 14 > 20 and +DI above -DI"),
    ("Ichimoku", "Trend", True, "Close above the Kumo cloud"),
    ("Parabolic SAR", "Trend", True, "Close above SAR (0.02, 0.2)"),
    ("HTF EMA bias", "Trend", True, "Last closed H4 bar above H4 EMA 50"),
    ("RSI", "Momentum", True, "RSI 14 below 40 (buy the pullback)"),
    ("Stoch RSI", "Momentum", True, "Stoch RSI %K crossed above %D under 20 (last 3 bars)"),
    ("MACD", "Momentum", True, "MACD (12, 26, 9) line above signal"),
    ("CCI", "Momentum", True, "CCI 20 crossed back above -100 (last 3 bars)"),
    ("Williams %R", "Momentum", True, "%R 14 crossed back above -80 (last 3 bars)"),
    ("Bollinger", "Volatility", True, "Tagged lower band (20, 2) in last 3 bars and closed back inside"),
    ("Keltner", "Volatility", True, "Close above upper Keltner (20, 2 ATR) — breakout"),
    ("Donchian breakout", "Volatility", True, "Close above previous 20-bar high"),
    ("ATR expansion", "Volatility", False, "Filter: ATR 14 > 1.1 x its 50-bar average"),
    ("VWAP", "Structure & time", True, "Close above daily VWAP"),
    ("Pivot points", "Structure & time", True, "Close above the daily pivot P"),
    ("Prev-day high/low", "Structure & time", True, "Close above previous day's high"),
    ("Session filter", "Structure & time", False, "Filter: London + New York hours, 07:00–17:00 UTC"),
    ("Fractal swing", "Structure & time", True, "Close above the last confirmed fractal high"),
]
NAMES = [i[0] for i in INDICATORS]
GROUPS = {}
for _n, _g, _d, _r in INDICATORS:
    GROUPS.setdefault(_g, []).append(_n)
DIRECTIONAL = {n: d for n, _, d, _ in INDICATORS}
RULES = {n: r for n, _, _, r in INDICATORS}


def _L(n, s):
    return max(2, int(round(n * s)))


def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values


def sma(x, n):
    return pd.Series(x).rolling(n).mean().values


def atr(h, l, c, n=14):
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().values


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.clip(d, 0, None)).ewm(alpha=1 / n, adjust=False).mean()
    dn = pd.Series(np.clip(-d, 0, None)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50).values


def recent(mask, n=3):
    return pd.Series(mask.astype(np.int8)).rolling(n, min_periods=1).max().values.astype(bool)


@njit(cache=True)
def _supertrend(h, l, c, a, mult):
    n = len(c)
    d = np.ones(n, dtype=np.int8)
    up = np.zeros(n)
    dn = np.zeros(n)
    for i in range(n):
        m = (h[i] + l[i]) / 2
        bu = m + mult * a[i]
        bl = m - mult * a[i]
        if i == 0 or np.isnan(a[i]):
            up[i] = bu
            dn[i] = bl
            continue
        up[i] = bu if (bu < up[i - 1] or c[i - 1] > up[i - 1]) else up[i - 1]
        dn[i] = bl if (bl > dn[i - 1] or c[i - 1] < dn[i - 1]) else dn[i - 1]
        if d[i - 1] == 1:
            d[i] = -1 if c[i] < dn[i] else 1
        else:
            d[i] = 1 if c[i] > up[i] else -1
    return d


@njit(cache=True)
def _psar(h, l, step, mx):
    n = len(h)
    sar = np.zeros(n)
    bull = True
    af = step
    ep = h[0]
    sar[0] = l[0]
    for i in range(1, n):
        s = sar[i - 1] + af * (ep - sar[i - 1])
        if bull:
            s = min(s, l[i - 1], l[i - 2] if i > 1 else l[i - 1])
            if l[i] < s:
                bull = False
                s = ep
                ep = l[i]
                af = step
            elif h[i] > ep:
                ep = h[i]
                af = min(af + step, mx)
        else:
            s = max(s, h[i - 1], h[i - 2] if i > 1 else h[i - 1])
            if h[i] > s:
                bull = True
                s = ep
                ep = h[i]
                af = step
            elif l[i] < ep:
                ep = l[i]
                af = min(af + step, mx)
        sar[i] = s
    return sar


def _day_key(times):
    # trading day starts at the 22:00 UTC rollover
    return (pd.to_datetime(times) + pd.Timedelta(hours=2)).normalize().values


def compute_states(df: pd.DataFrame, names, scale: float = 1.0, h4_close=None) -> dict:
    """Return {name: (bull, bear)} boolean arrays for the given TF bars."""
    o, h, l, c = (df[k].values.astype(np.float64) for k in ("open", "high", "low", "close"))
    v = df["volume"].values.astype(np.float64)
    n = len(c)
    a14 = atr(h, l, c, _L(14, scale))
    out = {}
    for name in names:
        if name == "EMA cross":
            f, s = ema(c, _L(9, scale)), ema(c, _L(21, scale))
            out[name] = (f > s, f < s)
        elif name == "SMA cross":
            f, s = sma(c, _L(20, scale)), sma(c, _L(50, scale))
            out[name] = (f > s, f < s)
        elif name == "Supertrend":
            d = _supertrend(h, l, c, atr(h, l, c, _L(10, scale)), 3.0 * scale)
            out[name] = (d == 1, d == -1)
        elif name == "ADX":
            p = _L(14, scale)
            upm = np.r_[0, np.diff(h)]
            dnm = np.r_[0, -np.diff(l)]
            pdm = np.where((upm > dnm) & (upm > 0), upm, 0.0)
            ndm = np.where((dnm > upm) & (dnm > 0), dnm, 0.0)
            a = atr(h, l, c, p)
            pdi = 100 * pd.Series(pdm).ewm(alpha=1 / p, adjust=False).mean().values / a
            ndi = 100 * pd.Series(ndm).ewm(alpha=1 / p, adjust=False).mean().values / a
            dx = 100 * np.abs(pdi - ndi) / np.where(pdi + ndi == 0, np.nan, pdi + ndi)
            adx = pd.Series(dx).ewm(alpha=1 / p, adjust=False).mean().values
            out[name] = ((adx > 20) & (pdi > ndi), (adx > 20) & (ndi > pdi))
        elif name == "Ichimoku":
            t9, k26, s52 = _L(9, scale), _L(26, scale), _L(52, scale)
            hs, ls = pd.Series(h), pd.Series(l)
            ten = (hs.rolling(t9).max() + ls.rolling(t9).min()) / 2
            kij = (hs.rolling(k26).max() + ls.rolling(k26).min()) / 2
            sa = ((ten + kij) / 2).shift(k26).values
            sb = ((hs.rolling(s52).max() + ls.rolling(s52).min()) / 2).shift(k26).values
            top, bot = np.fmax(sa, sb), np.fmin(sa, sb)
            out[name] = (c > top, c < bot)
        elif name == "Parabolic SAR":
            sar = _psar(h, l, 0.02 * scale, 0.2)
            out[name] = (c > sar, c < sar)
        elif name == "HTF EMA bias":
            if h4_close is None:
                out[name] = (np.ones(n, bool), np.ones(n, bool))
            else:
                ends, closes = h4_close
                e = ema(closes, _L(50, scale))
                idx = np.searchsorted(ends, df["end"].values, side="right") - 1
                ok = idx >= 0
                cl = np.where(ok, closes[np.clip(idx, 0, None)], np.nan)
                em = np.where(ok, e[np.clip(idx, 0, None)], np.nan)
                out[name] = (cl > em, cl < em)
        elif name == "RSI":
            r = rsi(c, _L(14, scale))
            out[name] = (r < 40, r > 60)
        elif name == "Stoch RSI":
            p = _L(14, scale)
            r = pd.Series(rsi(c, p))
            st = (r - r.rolling(p).min()) / (r.rolling(p).max() - r.rolling(p).min()).replace(0, np.nan) * 100
            k = st.rolling(3).mean()
            d = k.rolling(3).mean()
            kv, dv = k.values, d.values
            kp, dp = np.r_[np.nan, kv[:-1]], np.r_[np.nan, dv[:-1]]
            xu = (kv > dv) & (kp <= dp) & (kv < 20)
            xd = (kv < dv) & (kp >= dp) & (kv > 80)
            out[name] = (recent(xu), recent(xd))
        elif name == "MACD":
            m = ema(c, _L(12, scale)) - ema(c, _L(26, scale))
            sg = ema(m, _L(9, scale))
            out[name] = (m > sg, m < sg)
        elif name == "CCI":
            p = _L(20, scale)
            tp = pd.Series((h + l + c) / 3)
            md = tp.rolling(p).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
            cci = ((tp - tp.rolling(p).mean()) / (0.015 * md)).values
            pv = np.r_[np.nan, cci[:-1]]
            out[name] = (recent((cci > -100) & (pv <= -100)), recent((cci < 100) & (pv >= 100)))
        elif name == "Williams %R":
            p = _L(14, scale)
            hh, ll = pd.Series(h).rolling(p).max().values, pd.Series(l).rolling(p).min().values
            wr = -100 * (hh - c) / np.where(hh - ll == 0, np.nan, hh - ll)
            pv = np.r_[np.nan, wr[:-1]]
            out[name] = (recent((wr > -80) & (pv <= -80)), recent((wr < -20) & (pv >= -20)))
        elif name == "Bollinger":
            p = _L(20, scale)
            mid = sma(c, p)
            sd = pd.Series(c).rolling(p).std().values
            lo, hi = mid - 2 * sd, mid + 2 * sd
            out[name] = (recent(l < lo) & (c > lo), recent(h > hi) & (c < hi))
        elif name == "Keltner":
            p = _L(20, scale)
            mid = ema(c, p)
            a = atr(h, l, c, p)
            out[name] = (c > mid + 2 * a, c < mid - 2 * a)
        elif name == "Donchian breakout":
            p = _L(20, scale)
            hh = pd.Series(h).rolling(p).max().shift(1).values
            ll = pd.Series(l).rolling(p).min().shift(1).values
            out[name] = (c > hh, c < ll)
        elif name == "ATR expansion":
            x = a14 > 1.1 * sma(a14, _L(50, scale))
            out[name] = (x, x)
        elif name == "VWAP":
            day = _day_key(df["time"].values)
            tp = (h + l + c) / 3
            vv = np.where(v <= 0, 1.0, v)
            g = pd.Series(day)
            vw = (pd.Series(tp * vv).groupby(g).cumsum() / pd.Series(vv).groupby(g).cumsum()).values
            out[name] = (c > vw, c < vw)
        elif name in ("Pivot points", "Prev-day high/low"):
            day = _day_key(df["time"].values)
            dd = pd.DataFrame({"d": day, "h": h, "l": l, "c": c}).groupby("d").agg(h=("h", "max"), l=("l", "min"), c=("c", "last"))
            prev = dd.shift(1)
            ph = pd.Series(day).map(prev["h"]).values
            pl = pd.Series(day).map(prev["l"]).values
            pc = pd.Series(day).map(prev["c"]).values
            if name == "Pivot points":
                p = (ph + pl + pc) / 3
                out[name] = (c > p, c < p)
            else:
                out[name] = (c > ph, c < pl)
        elif name == "Session filter":
            hr = pd.to_datetime(df["time"]).dt.hour.values
            x = (hr >= 7) & (hr < 17)
            out[name] = (x, x)
        elif name == "Fractal swing":
            fh = (h > np.r_[h[1:], np.inf]) & (h > np.r_[h[2:], np.inf, np.inf]) & (h > np.r_[-np.inf, h[:-1]]) & (h > np.r_[-np.inf, -np.inf, h[:-2]])
            fl = (l < np.r_[l[1:], -np.inf]) & (l < np.r_[l[2:], -np.inf, -np.inf]) & (l < np.r_[np.inf, l[:-1]]) & (l < np.r_[np.inf, np.inf, l[:-2]])
            # a fractal at bar i is only confirmed at bar i+2
            lvl_h = pd.Series(np.where(fh, h, np.nan)).shift(2).ffill().values
            lvl_l = pd.Series(np.where(fl, l, np.nan)).shift(2).ffill().values
            out[name] = (c > lvl_h, c < lvl_l)
        else:
            raise KeyError(name)
        b, s = out[name]
        out[name] = (np.nan_to_num(b, nan=0).astype(bool), np.nan_to_num(s, nan=0).astype(bool))
    return out


def edge(mask: np.ndarray) -> np.ndarray:
    """True only on the bar where the combined condition switches on."""
    prev = np.r_[False, mask[:-1]]
    return mask & ~prev
