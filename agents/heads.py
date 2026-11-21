"""Strategy-specific heads.

S6 head (Agent Bull): compact logistic model on 16 context features measured AT the entry moment
                      (23:00 UTC, price = the open of that minute). Approves ~50% of S6 longs.
S5 head (Agent Bear): gradient-boosting model on the full 162-feature library at the 15:30 UTC decision.
                      Approves ~50% of S5 shorts.
Both were validated with yearly walk-forward (see MODEL_CARD.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core import indicators as I

from . import features as F

S6_FEATS = ["r_1h", "r_4h", "r_24h", "r_session", "r_d1", "r_d5", "r_d20", "atr_rel", "atr_pctile",
            "dist_ma50", "dist_ma200", "vol_ratio", "trend", "weekday", "month", "us_dst"]


def _asof_open(times, q, opens):
    i = np.searchsorted(times, q, side="right") - 1
    return np.where(i >= 0, opens[np.clip(i, 0, None)], np.nan)


def context_features(m1: pd.DataFrame, entry_times, session_start_utc="22:00") -> pd.DataFrame:
    """16 context features for trades entered at `entry_times` (UTC), using bars up to and including
    the open price of the entry minute. Works on history and live (pass the latest M1 bars)."""
    t = pd.to_datetime(pd.Series(entry_times)).values
    T, O = m1["time"].values, m1["open"].values
    p0 = _asof_open(T, t, O)
    hr = np.timedelta64(1, "h")
    f = pd.DataFrame({"entry_time": t})
    f["r_1h"] = np.log(p0 / _asof_open(T, t - hr, O))
    f["r_4h"] = np.log(p0 / _asof_open(T, t - 4 * hr, O))
    f["r_24h"] = np.log(p0 / _asof_open(T, t - 24 * hr, O))
    sh, sm = map(int, session_start_utc.split(":"))
    s0 = pd.to_datetime(t).normalize() + pd.Timedelta(hours=sh, minutes=sm)
    s0 = np.where(s0.values > t, (s0 - pd.Timedelta(days=1)).values, s0.values)
    f["r_session"] = np.log(p0 / _asof_open(T, s0, O))
    d = F.daily_bars(m1)
    c = d["close"].values
    a = I.atr(d["high"].values, d["low"].values, c, 14)
    tday = (pd.to_datetime(t) + pd.Timedelta(hours=2)).normalize().values
    k = np.searchsorted(d["time"].values, tday, side="left") - 1
    ok = k >= 0
    kk = np.clip(k, 0, None)
    cs = pd.Series(c)
    lc = np.log(cs)
    pct = pd.Series(a / c).rolling(250, min_periods=60).rank(pct=True).values
    rv5, rv60 = lc.diff().rolling(5).std().values, lc.diff().rolling(60).std().values
    trend = np.sign(np.sign(lc.diff(21)) + np.sign(lc.diff(63)) + np.sign(lc.diff(252))).fillna(0).values
    take = lambda arr: np.where(ok, arr[kk], np.nan)  # noqa: E731
    f["r_d1"] = take(lc.diff(1).values)
    f["r_d5"] = take(lc.diff(5).values)
    f["r_d20"] = take(lc.diff(20).values)
    f["atr_rel"] = take(a / c)
    f["atr_pctile"] = take(pct)
    f["dist_ma50"] = take((c - cs.rolling(50).mean().values) / a)
    f["dist_ma200"] = take((c - cs.rolling(200).mean().values) / a)
    f["vol_ratio"] = take(rv5 / rv60)
    f["trend"] = take(trend)
    tt = pd.to_datetime(t)
    f["weekday"] = tt.weekday
    f["month"] = tt.month
    ny = tt.tz_localize("UTC").tz_convert("America/New_York")
    f["us_dst"] = np.array([x.dst().total_seconds() > 0 for x in ny]).astype(int)
    return f.replace([np.inf, -np.inf], np.nan)


def s6_model():
    return make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))


def s5_model():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=200, min_samples_leaf=60,
                                          l2_regularization=2.0, max_features=0.3, random_state=1)
