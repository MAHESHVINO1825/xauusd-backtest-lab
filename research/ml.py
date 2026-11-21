"""Diagnostics + walk-forward ML meta-labelling for S5 and S6.

Every feature uses only data available BEFORE the trade's entry bar opens.
Walk-forward: for each year Y from 2017, train on all trades before Y, predict trades in Y.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core import indicators as I

from . import harness as H
from . import strategies as S

STRATS = {
    "S5": dict(fn=S.s5_positions, cfg=dict(entry="15:30", exit="20:30"), session_start="07:00"),
    "S6": dict(fn=S.s6_positions, cfg=dict(entry="23:00", exit="00:00"), session_start="22:00"),
    "S7": dict(fn=S.s7_positions, cfg=dict(entry=15, exit=60), session_start="22:00"),
}


def _asof_open(times: np.ndarray, t_query: np.ndarray, opens: np.ndarray) -> np.ndarray:
    """Open price of the last M15 bar starting at or before t_query."""
    i = np.searchsorted(times, t_query, side="right") - 1
    return np.where(i >= 0, opens[np.clip(i, 0, None)], np.nan)


def features(name: str) -> pd.DataFrame:
    st = STRATS[name]
    b = H.bars15()
    tr = H.run_positions(st["fn"](**st["cfg"]))
    t = pd.to_datetime(tr["entry_time"]).values
    T, O = b["time"].values, b["open"].values
    p0 = _asof_open(T, t, O)
    hr = np.timedelta64(1, "h")
    f = pd.DataFrame(index=tr.index)
    f["r_1h"] = np.log(p0 / _asof_open(T, t - hr, O))
    f["r_4h"] = np.log(p0 / _asof_open(T, t - 4 * hr, O))
    f["r_24h"] = np.log(p0 / _asof_open(T, t - 24 * hr, O))
    # session-so-far move
    sh, sm = map(int, st["session_start"].split(":"))
    sess0 = pd.to_datetime(t).normalize() + pd.Timedelta(hours=sh, minutes=sm)
    sess0 = np.where(sess0.values > t, (sess0 - pd.Timedelta(days=1)).values, sess0.values)
    f["r_session"] = np.log(p0 / _asof_open(T, sess0, O))
    # daily context: only days completed before the trade's own trading day
    d = H.daily()
    c = d["close"].values
    a = I.atr(d["high"].values, d["low"].values, c, 14)
    tday = (pd.to_datetime(t) + pd.Timedelta(hours=2)).normalize().values
    k = np.searchsorted(d.index.values, tday, side="left") - 1
    ok = k >= 0
    kk = np.clip(k, 0, None)
    cs = pd.Series(c)
    atr_pct = pd.Series(a / c).rolling(250, min_periods=60).rank(pct=True).values
    sma50, sma200 = cs.rolling(50).mean().values, cs.rolling(200).mean().values
    rv5 = pd.Series(np.log(c)).diff().rolling(5).std().values
    rv60 = pd.Series(np.log(c)).diff().rolling(60).std().values
    f["r_d1"] = np.where(ok, np.log(c[kk] / c[np.clip(kk - 1, 0, None)]), np.nan)
    f["r_d5"] = np.where(ok, np.log(c[kk] / c[np.clip(kk - 5, 0, None)]), np.nan)
    f["r_d20"] = np.where(ok, np.log(c[kk] / c[np.clip(kk - 20, 0, None)]), np.nan)
    f["atr_rel"] = np.where(ok, a[kk] / c[kk], np.nan)
    f["atr_pctile"] = np.where(ok, atr_pct[kk], np.nan)
    f["dist_ma50"] = np.where(ok, (c[kk] - sma50[kk]) / a[kk], np.nan)
    f["dist_ma200"] = np.where(ok, (c[kk] - sma200[kk]) / a[kk], np.nan)
    f["vol_ratio"] = np.where(ok, rv5[kk] / rv60[kk], np.nan)
    f["trend"] = S.daily_trend("blend").reindex(d.index).values[kk] * ok
    f["weekday"] = pd.to_datetime(t).weekday
    f["month"] = pd.to_datetime(t).month
    ny = pd.to_datetime(t).tz_localize("UTC").tz_convert("America/New_York")
    f["us_dst"] = np.array([x.dst().total_seconds() > 0 for x in ny]).astype(int)
    out = pd.concat([tr.reset_index(drop=True), f.reset_index(drop=True)], axis=1)
    out["year"] = pd.to_datetime(out["entry_time"]).dt.year
    out["win"] = (out["ret"] > 0).astype(int)
    return out.replace([np.inf, -np.inf], np.nan)


FEATS = ["r_1h", "r_4h", "r_24h", "r_session", "r_d1", "r_d5", "r_d20", "atr_rel", "atr_pctile",
         "dist_ma50", "dist_ma200", "vol_ratio", "trend", "weekday", "month", "us_dst"]


def walk_forward(df: pd.DataFrame, model="gbm", start_year=2017, keep=0.5, seed=0):
    """Expanding-window yearly walk-forward. Takes the top `keep` share of trades by predicted
    probability (threshold learned from the TRAIN predictions, so no peeking at the test year)."""
    rows = []
    for y in range(start_year, int(df["year"].max()) + 1):
        trn, tst = df[df["year"] < y].dropna(subset=FEATS), df[df["year"] == y]
        tst = tst.dropna(subset=FEATS)
        if len(tst) == 0:
            continue
        if model == "gbm":
            m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                               min_samples_leaf=40, l2_regularization=1.0, random_state=seed)
        else:
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500))
        m.fit(trn[FEATS], trn["win"])
        thr = np.quantile(m.predict_proba(trn[FEATS])[:, 1], 1 - keep)
        pr = m.predict_proba(tst[FEATS])[:, 1]
        r = tst.assign(prob=pr, take=pr >= thr)
        rows.append(r)
    return pd.concat(rows)


def summary(wf: pd.DataFrame, label: str, rng=np.random.default_rng(1)):
    base = H.stats(wf["ret"].values)
    filt = H.stats(wf.loc[wf["take"], "ret"].values)
    # random-filter control: keep the same share at random, 500 times
    frac = wf["take"].mean()
    rnd = [H.stats(wf["ret"].values[rng.random(len(wf)) < frac])["pf"] for _ in range(500)]
    pct = np.mean(np.array(rnd) < filt["pf"]) * 100
    yb = wf.groupby("year")["ret"].sum()
    yf = wf[wf["take"]].groupby("year")["ret"].sum()
    print(f"  {label:22s} all PF {base['pf']:.2f} (n{base['n']}, t {base['t']:+.2f}) → filtered PF {filt['pf']:.2f} "
          f"(n{filt['n']}, t {filt['t']:+.2f}) · beats {pct:.0f}% of random filters")
    print("     yearly bp  base:", {int(k): int(v) for k, v in yb.items()})
    print("     yearly bp  filt:", {int(k): int(v) for k, v in yf.reindex(yb.index, fill_value=0).items()})
    return dict(base=base, filt=filt, beats=pct)
