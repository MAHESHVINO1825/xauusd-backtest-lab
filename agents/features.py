"""Feature library shared by Agent Bull and Agent Bear.

The SAME function builds features for backtests and for live trading: give it M1 (or M15) OHLCV
bars in UTC and it returns one row per M15 decision time (= M15 bar close). Every value in a row
uses only bars that had CLOSED by that time (merge_asof on bar END times), so there is no look-ahead.

Feature groups (prefix -> meaning)
  cdl_   candlestick patterns on M15 / H1 / H4
  str_   chart structure: swings, HH/HL/LH/LL, double top/bottom, breakouts, compression
  lvl_   key levels: previous day/week high-low-close, today's open, round numbers, Asian range
  ind_   indicators: RSI, Stochastic, MACD, ADX/DI, Bollinger, Keltner, EMA distances & slopes, Supertrend
  vol_   volatility: ATR percentiles, realised-vol ratio, range expansion, tick-volume
  mom_   momentum: returns over several horizons
  tim_   time: hour, weekday, US daylight-saving, minutes since the 18:00 New York reopen
  reg_   regime: daily trend, distance from 200-day average, 60-day return
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core import indicators as I

TF_MIN = {"M15": 15, "H1": 60, "H4": 240}


# ------------------------------------------------------------------ helpers
def resample(m: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """OHLCV bars labelled by START time; adds END time (when the bar is complete)."""
    x = m.set_index("time").resample(f"{minutes}min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["open"])
    x = x.reset_index()
    x["end"] = x["time"] + pd.Timedelta(minutes=minutes)
    return x


def daily_bars(m: pd.DataFrame) -> pd.DataFrame:
    """Trading days on the 22:00 UTC rollover; END = the following 22:00 UTC."""
    d = m.assign(day=(m["time"] + pd.Timedelta(hours=2)).dt.normalize())
    x = d.groupby("day").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                             close=("close", "last"), volume=("volume", "sum"), n=("close", "size"))
    x = x[x["n"] >= 60].reset_index().rename(columns={"day": "time"})
    x["end"] = x["time"] + pd.Timedelta(hours=22)   # day D runs D-1 22:00 -> D 22:00
    return x


def safe_div(a, b):
    b = np.where(np.abs(b) < 1e-12, np.nan, b)
    return a / b


# ------------------------------------------------------------------ candlestick patterns
def candle_features(b: pd.DataFrame, p: str) -> pd.DataFrame:
    o, h, l, c = (b[k].values for k in ("open", "high", "low", "close"))
    rng = np.maximum(h - l, 1e-9)
    body = c - o
    ab = np.abs(body)
    up = h - np.maximum(o, c)
    lo = np.minimum(o, c) - l
    o1, h1, l1, c1 = (np.r_[np.nan, x[:-1]] for x in (o, h, l, c))
    o2, c2 = (np.r_[np.nan, np.nan, x[:-2]] for x in (o, c))
    ab1 = np.abs(c1 - o1)
    a = I.atr(h, l, c, 14)
    f = pd.DataFrame(index=b.index)
    f[f"cdl_{p}_body"] = body / rng
    f[f"cdl_{p}_upwick"] = up / rng
    f[f"cdl_{p}_lowick"] = lo / rng
    f[f"cdl_{p}_closepos"] = (c - l) / rng
    f[f"cdl_{p}_size_atr"] = rng / a
    f[f"cdl_{p}_bull_engulf"] = ((c > o) & (c1 < o1) & (c >= o1) & (o <= c1) & (ab > ab1)).astype(int)
    f[f"cdl_{p}_bear_engulf"] = ((c < o) & (c1 > o1) & (c <= o1) & (o >= c1) & (ab > ab1)).astype(int)
    f[f"cdl_{p}_hammer"] = ((lo >= 2 * ab) & (up <= 0.25 * rng) & (rng > 0.5 * a)).astype(int)
    f[f"cdl_{p}_shooting_star"] = ((up >= 2 * ab) & (lo <= 0.25 * rng) & (rng > 0.5 * a)).astype(int)
    f[f"cdl_{p}_doji"] = (ab <= 0.1 * rng).astype(int)
    f[f"cdl_{p}_inside"] = ((h <= h1) & (l >= l1)).astype(int)
    f[f"cdl_{p}_outside"] = ((h > h1) & (l < l1)).astype(int)
    f[f"cdl_{p}_marubozu"] = (ab >= 0.85 * rng).astype(int) * np.sign(body)
    mid2 = (o2 + c2) / 2
    small1 = ab1 <= 0.3 * np.r_[np.nan, rng[:-1]]
    f[f"cdl_{p}_morning_star"] = ((c2 < o2) & small1 & (c > o) & (c > mid2)).astype(int)
    f[f"cdl_{p}_evening_star"] = ((c2 > o2) & small1 & (c < o) & (c < mid2)).astype(int)
    up3 = (c > o) & (c1 > o1) & (c2 > o2) & (c > c1) & (c1 > c2)
    dn3 = (c < o) & (c1 < o1) & (c2 < o2) & (c < c1) & (c1 < c2)
    f[f"cdl_{p}_three_soldiers"] = up3.astype(int)
    f[f"cdl_{p}_three_crows"] = dn3.astype(int)
    bull = f[[f"cdl_{p}_bull_engulf", f"cdl_{p}_hammer", f"cdl_{p}_morning_star", f"cdl_{p}_three_soldiers"]].sum(axis=1)
    bear = f[[f"cdl_{p}_bear_engulf", f"cdl_{p}_shooting_star", f"cdl_{p}_evening_star", f"cdl_{p}_three_crows"]].sum(axis=1)
    f[f"cdl_{p}_bull_count3"] = bull.rolling(3, min_periods=1).sum()
    f[f"cdl_{p}_bear_count3"] = bear.rolling(3, min_periods=1).sum()
    return f


# ------------------------------------------------------------------ chart structure
def structure_features(b: pd.DataFrame, p: str) -> pd.DataFrame:
    h, l, c = b["high"].values, b["low"].values, b["close"].values
    a = I.atr(h, l, c, 14)
    n = len(c)
    # fractal swing (2 bars each side) confirmed 2 bars later -> shift(2)
    fh = np.zeros(n, bool)
    fl = np.zeros(n, bool)
    fh[2:-2] = (h[2:-2] > h[1:-3]) & (h[2:-2] > h[:-4]) & (h[2:-2] > h[3:-1]) & (h[2:-2] > h[4:])
    fl[2:-2] = (l[2:-2] < l[1:-3]) & (l[2:-2] < l[:-4]) & (l[2:-2] < l[3:-1]) & (l[2:-2] < l[4:])
    sh = pd.Series(np.where(fh, h, np.nan)).shift(2)
    sl = pd.Series(np.where(fl, l, np.nan)).shift(2)
    sh1 = sh.ffill().values
    sl1 = sl.ffill().values
    # previous distinct swing values
    sh2 = sh.dropna().shift(1).reindex(range(n)).ffill().values
    sl2 = sl.dropna().shift(1).reindex(range(n)).ffill().values
    f = pd.DataFrame(index=b.index)
    f[f"str_{p}_hh"] = (sh1 > sh2).astype(int)
    f[f"str_{p}_hl"] = (sl1 > sl2).astype(int)
    f[f"str_{p}_lh"] = (sh1 < sh2).astype(int)
    f[f"str_{p}_ll"] = (sl1 < sl2).astype(int)
    f[f"str_{p}_trend_score"] = f[f"str_{p}_hh"] + f[f"str_{p}_hl"] - f[f"str_{p}_lh"] - f[f"str_{p}_ll"]
    f[f"str_{p}_dist_swing_high"] = (sh1 - c) / a
    f[f"str_{p}_dist_swing_low"] = (c - sl1) / a
    f[f"str_{p}_double_top"] = ((np.abs(sh1 - sh2) < 0.3 * a) & (c < sh1)).astype(int)
    f[f"str_{p}_double_bottom"] = ((np.abs(sl1 - sl2) < 0.3 * a) & (c > sl1)).astype(int)
    f[f"str_{p}_contracting"] = ((sh1 < sh2) & (sl1 > sl2)).astype(int)   # triangle-like
    for k in (20, 50):
        hh = pd.Series(h).rolling(k).max().shift(1).values
        ll = pd.Series(l).rolling(k).min().shift(1).values
        f[f"str_{p}_breakout{k}"] = (c > hh).astype(int) - (c < ll).astype(int)
        f[f"str_{p}_pos_in_range{k}"] = safe_div(c - ll, hh - ll)
        f[f"str_{p}_range{k}_atr"] = (hh - ll) / a
    return f


# ------------------------------------------------------------------ indicators
def indicator_features(b: pd.DataFrame, p: str) -> pd.DataFrame:
    h, l, c = b["high"].values, b["low"].values, b["close"].values
    a = I.atr(h, l, c, 14)
    f = pd.DataFrame(index=b.index)
    f[f"ind_{p}_rsi"] = I.rsi(c, 14)
    r = pd.Series(f[f"ind_{p}_rsi"].values)
    f[f"ind_{p}_stochrsi"] = ((r - r.rolling(14).min()) / (r.rolling(14).max() - r.rolling(14).min()).replace(0, np.nan)).values
    m = I.ema(c, 12) - I.ema(c, 26)
    f[f"ind_{p}_macd_hist"] = (m - I.ema(m, 9)) / a
    st = I.compute_states(b.assign(end=b["end"]), ["ADX", "Supertrend"])
    f[f"ind_{p}_adx_bull"] = st["ADX"][0].astype(int)
    f[f"ind_{p}_adx_bear"] = st["ADX"][1].astype(int)
    f[f"ind_{p}_supertrend"] = st["Supertrend"][0].astype(int) - st["Supertrend"][1].astype(int)
    mid = I.sma(c, 20)
    sd = pd.Series(c).rolling(20).std().values
    f[f"ind_{p}_bb_pctb"] = safe_div(c - (mid - 2 * sd), 4 * sd)
    f[f"ind_{p}_bb_width_atr"] = 4 * sd / a
    for n in (20, 50, 200):
        e = I.ema(c, n)
        f[f"ind_{p}_dist_ema{n}"] = (c - e) / a
        f[f"ind_{p}_slope_ema{n}"] = (e - np.r_[np.full(5, np.nan), e[:-5]]) / a
    return f


def vol_mom_features(b: pd.DataFrame, p: str, horizons=(1, 4, 16)) -> pd.DataFrame:
    h, l, c, v = b["high"].values, b["low"].values, b["close"].values, b["volume"].values.astype(float)
    a = I.atr(h, l, c, 14)
    f = pd.DataFrame(index=b.index)
    f[f"vol_{p}_atr_pctile"] = pd.Series(a / c).rolling(500, min_periods=100).rank(pct=True).values
    f[f"vol_{p}_range_exp"] = (h - l) / a
    f[f"vol_{p}_relvol"] = v / pd.Series(v).rolling(20, min_periods=5).mean().values
    lc = np.log(c)
    for k in horizons:
        f[f"mom_{p}_r{k}"] = (lc - np.r_[np.full(k, np.nan), lc[:-k]]) / (a / c)
    return f


# ------------------------------------------------------------------ daily context, levels, regime
def daily_features(d: pd.DataFrame) -> pd.DataFrame:
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    a = I.atr(h, l, c, 14)
    lc = pd.Series(np.log(c))
    f = pd.DataFrame(index=d.index)
    f["lvl_d_prev_high"] = h
    f["lvl_d_prev_low"] = l
    f["lvl_d_prev_close"] = c
    f["lvl_w_high"] = pd.Series(h).rolling(5).max().values
    f["lvl_w_low"] = pd.Series(l).rolling(5).min().values
    f["d_atr"] = a
    s200 = I.sma(c, 200)
    f["reg_dist_ma200"] = (c - s200) / a
    f["reg_dist_ma50"] = (c - I.sma(c, 50)) / a
    f["reg_r20"] = lc.diff(20).values
    f["reg_r60"] = lc.diff(60).values
    f["reg_r5"] = lc.diff(5).values
    f["reg_r1"] = lc.diff(1).values
    f["reg_trend_blend"] = np.sign(np.sign(lc.diff(21)) + np.sign(lc.diff(63)) + np.sign(lc.diff(252))).values
    rv = lc.diff()
    f["reg_rv_ratio"] = (rv.rolling(5).std() / rv.rolling(60).std()).values
    f["reg_atr_pctile"] = pd.Series(a / c).rolling(250, min_periods=60).rank(pct=True).values
    return f


# ------------------------------------------------------------------ main
def build(m1: pd.DataFrame) -> pd.DataFrame:
    """m1: DataFrame time (UTC, naive), open, high, low, close, volume — M1 or M15 bars."""
    m1 = m1[["time", "open", "high", "low", "close", "volume"]].sort_values("time").reset_index(drop=True)
    tf = {k: resample(m1, v) for k, v in TF_MIN.items()}
    base = tf["M15"][["time", "end", "open", "high", "low", "close"]].copy()
    out = base.rename(columns={"end": "decision_time"})
    for p, b in tf.items():
        parts = [candle_features(b, p), vol_mom_features(b, p)]
        if p in ("H1", "H4"):
            parts += [structure_features(b, p), indicator_features(b, p)]
        if p == "M15":
            parts += [indicator_features(b, p)[[f"ind_{p}_rsi", f"ind_{p}_bb_pctb", f"ind_{p}_dist_ema20", f"ind_{p}_macd_hist"]]]
        feat = pd.concat(parts, axis=1)
        feat["_end"] = b["end"].values
        if p == "M15":
            out = pd.concat([out, feat.drop(columns="_end").reset_index(drop=True)], axis=1)
        else:
            out = pd.merge_asof(out.sort_values("decision_time"), feat.sort_values("_end"),
                                left_on="decision_time", right_on="_end", direction="backward").drop(columns="_end")
    # daily context (completed days only)
    d = daily_bars(m1)
    df = daily_features(d)
    df["_end"] = d["end"].values
    out = pd.merge_asof(out.sort_values("decision_time"), df.sort_values("_end"), left_on="decision_time",
                        right_on="_end", direction="backward").drop(columns="_end")
    c, a = out["close"].values, out["d_atr"].values
    for k in ("lvl_d_prev_high", "lvl_d_prev_low", "lvl_d_prev_close", "lvl_w_high", "lvl_w_low"):
        out[k] = (c - out[k].values) / a
    for step in (10, 50):
        rn = np.round(c / step) * step
        out[f"lvl_round{step}"] = (c - rn) / a
    # today's open and Asian range so far (from the 22:00 UTC rollover)
    day = (out["decision_time"] - pd.Timedelta(minutes=1) + pd.Timedelta(hours=2)).dt.normalize()
    g = out.groupby(day)
    out["lvl_today_open"] = (c - g["open"].transform("first").values) / a
    out["lvl_today_high"] = (g["high"].cummax().values - c) / a
    out["lvl_today_low"] = (c - g["low"].cummin().values) / a
    # time
    t = out["decision_time"]
    out["tim_hour"] = t.dt.hour + t.dt.minute / 60
    out["tim_weekday"] = t.dt.weekday
    ny = t.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    out["tim_us_dst"] = (ny.dt.tz_localize(None) - t + pd.Timedelta(hours=5)).dt.total_seconds().gt(0).astype(int)
    out["tim_since_reopen"] = ((ny.dt.hour * 60 + ny.dt.minute).values - 18 * 60) % 1440
    out = out.drop(columns=["d_atr"]).replace([np.inf, -np.inf], np.nan)
    return out.sort_values("decision_time").reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.split("_")[0] in ("cdl", "str", "lvl", "ind", "vol", "mom", "tim", "reg")]
