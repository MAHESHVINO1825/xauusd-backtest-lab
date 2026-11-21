"""Extra feature group (prefix ext_) — new information beyond the v1 library. Same no-look-ahead rules."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core import indicators as I

from . import features as F


def _eff_ratio(c, n):
    ch = np.abs(c - np.r_[np.full(n, np.nan), c[:-n]])
    vol = pd.Series(np.abs(np.diff(c, prepend=c[0]))).rolling(n).sum().values
    return ch / np.where(vol == 0, np.nan, vol)


def _streak(c):
    s = np.sign(np.diff(c, prepend=c[0]))
    out = np.zeros(len(c))
    for i in range(1, len(c)):
        out[i] = out[i - 1] + s[i] if s[i] == np.sign(out[i - 1]) or out[i - 1] == 0 else s[i]
    return out


def _tf_block(b: pd.DataFrame, p: str) -> pd.DataFrame:
    h, l, c, v = b["high"].values, b["low"].values, b["close"].values, b["volume"].values.astype(float)
    a = I.atr(h, l, c, 14)
    f = pd.DataFrame(index=b.index)
    f[f"ext_{p}_er10"] = _eff_ratio(c, 10)
    f[f"ext_{p}_er30"] = _eff_ratio(c, 30)
    f[f"ext_{p}_z20"] = (c - pd.Series(c).rolling(20).mean().values) / pd.Series(c).rolling(20).std().values
    f[f"ext_{p}_streak"] = _streak(c)
    vz = pd.Series(np.log1p(v))
    f[f"ext_{p}_vol_z"] = ((vz - vz.rolling(50).mean()) / vz.rolling(50).std()).values
    f[f"ext_{p}_upvol_share"] = pd.Series(np.where(c > b["open"].values, v, 0)).rolling(10).sum().values / \
        pd.Series(v).rolling(10).sum().replace(0, np.nan).values
    return f


def build_ext(X: pd.DataFrame, m1: pd.DataFrame) -> pd.DataFrame:
    m1 = m1[["time", "open", "high", "low", "close", "volume"]].sort_values("time").reset_index(drop=True)
    out = X[["decision_time", "close"]].copy()
    for p, mins in (("M15", 15), ("H1", 60)):
        b = F.resample(m1, mins)
        f = _tf_block(b, p)
        f["_end"] = b["end"].values
        out = pd.merge_asof(out.sort_values("decision_time"), f.sort_values("_end"), left_on="decision_time",
                            right_on="_end", direction="backward").drop(columns="_end")
    # previous completed day's candle + efficiency
    d = F.daily_bars(m1)
    o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
    rng = np.maximum(h - l, 1e-9)
    a = I.atr(h, l, c, 14)
    df = pd.DataFrame({"_end": d["end"].values,
                       "ext_D1_body": (c - o) / rng, "ext_D1_closepos": (c - l) / rng, "ext_D1_range_atr": rng / a,
                       "ext_D1_inside": ((h <= np.r_[np.nan, h[:-1]]) & (l >= np.r_[np.nan, l[:-1]])).astype(int),
                       "ext_D1_er20": _eff_ratio(c, 20), "ext_D1_streak": _streak(c),
                       "_dclose": c, "_datr": a})
    out = pd.merge_asof(out.sort_values("decision_time"), df.sort_values("_end"), left_on="decision_time",
                        right_on="_end", direction="backward").drop(columns="_end")
    # intraday: session VWAP (from 22:00 UTC rollover), reopen gap, Asian / London ranges so far
    mm = m1.copy()
    mm["day"] = (mm["time"] + pd.Timedelta(hours=2)).dt.normalize()
    tp = (mm["high"] + mm["low"] + mm["close"]) / 3
    vv = mm["volume"].clip(lower=1)
    g = mm.groupby("day")
    mm["vwap"] = (tp * vv).groupby(mm["day"]).cumsum() / vv.groupby(mm["day"]).cumsum()
    mm["day_open"] = g["open"].transform("first")
    hr = mm["time"].dt.hour
    asia = (hr >= 22) | (hr < 7)
    lon = (hr >= 7) & (hr < 13)
    for nm, msk in (("asia", asia), ("london", lon)):
        hh = mm["high"].where(msk).groupby(mm["day"]).cummax()
        ll = mm["low"].where(msk).groupby(mm["day"]).cummin()
        mm[f"{nm}_rng"] = (hh - ll).groupby(mm["day"]).ffill()
    mm["_end"] = mm["time"] + pd.Timedelta(minutes=1)
    cols = ["_end", "vwap", "day_open", "asia_rng", "london_rng"]
    out = pd.merge_asof(out.sort_values("decision_time"), mm[cols].sort_values("_end"), left_on="decision_time",
                        right_on="_end", direction="backward").drop(columns="_end")
    c, a = out["close"].values, out["_datr"].values
    out["ext_dist_vwap"] = (c - out["vwap"].values) / a
    out["ext_reopen_gap"] = (out["day_open"].values - out["_dclose"].values) / a
    out["ext_asia_rng_atr"] = out["asia_rng"].values / a
    out["ext_london_rng_atr"] = out["london_rng"].values / a
    # calendar / events
    t = out["decision_time"]
    out["ext_dom"] = t.dt.day
    out["ext_days_to_month_end"] = (t + pd.offsets.MonthEnd(0)).dt.day - t.dt.day
    nfp = (t.dt.weekday == 4) & (t.dt.day <= 7)
    ny = t.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    nfp_min = (ny.dt.hour * 60 + ny.dt.minute) - (8 * 60 + 30)       # minutes since 08:30 NY release
    out["ext_nfp_day"] = nfp.astype(int)
    out["ext_nfp_minutes"] = np.where(nfp, nfp_min, np.nan)
    keep = ["decision_time"] + [c for c in out.columns if c.startswith("ext_")]
    return out[keep].replace([np.inf, -np.inf], np.nan).sort_values("decision_time").reset_index(drop=True)
