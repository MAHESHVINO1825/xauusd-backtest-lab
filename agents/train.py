"""Label opportunities and train Agent Bull / Agent Bear with yearly walk-forward.

Label (triple barrier, walked on the real M1 path, costs included):
  Bull: buy at ask on the next minute; WIN if price reaches +1.5 x ATR(H1) before -1.0 x ATR(H1) within 8 hours.
  Bear: mirror image, sell at bid, exit at ask.
"""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from numba import njit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from core import data as D
from core import indicators as I

from . import features as F

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
TP_ATR, SL_ATR, HORIZON_MIN = 1.5, 1.0, 480
SPREAD, SLIP = 0.20, 0.05


@njit(cache=True)
def triple_barrier(entry, atr, o, h, l, c, sp, mins, side, tp_k, sl_k, horizon):
    n = len(entry)
    win = np.zeros(n, np.int8)
    r = np.full(n, np.nan)
    for i in range(n):
        e = entry[i]
        a = atr[i]
        if e < 0 or e >= len(o) - 1 or not (a > 0):
            continue
        if side == 1:
            ep = o[e] + sp[e] + SLIP
            tp, sl = ep + tp_k * a, ep - sl_k * a
        else:
            ep = o[e] - SLIP
            tp, sl = ep - tp_k * a, ep + sl_k * a
        t0 = mins[e]
        j = e
        res = np.nan
        while j < len(o):
            if side == 1:
                if l[j] <= sl:
                    res = (sl - ep) / a; break
                if h[j] >= tp:
                    res = (tp - ep) / a; win[i] = 1; break
            else:
                if h[j] + sp[j] >= sl:
                    res = (ep - sl) / a; break
                if l[j] + sp[j] <= tp:
                    res = (ep - tp) / a; win[i] = 1; break
            if mins[j] - t0 >= horizon:
                res = ((c[j] - ep) if side == 1 else (ep - c[j] - sp[j])) / a
                break
            j += 1
        r[i] = res
    return win, r


def h1_atr_at(X: pd.DataFrame, m1: pd.DataFrame) -> np.ndarray:
    b = F.resample(m1, 60)
    a = pd.DataFrame({"_end": b["end"], "atr": I.atr(b["high"].values, b["low"].values, b["close"].values, 14)})
    return pd.merge_asof(X[["decision_time"]], a, left_on="decision_time", right_on="_end", direction="backward")["atr"].values


def make_labels(X: pd.DataFrame) -> pd.DataFrame:
    m1 = D.load_m1()
    arr = D.m1_arrays(SPREAD)
    entry = np.searchsorted(arr["time"], X["decision_time"].values, side="left").astype(np.int64)
    atr = h1_atr_at(X, m1)
    out = pd.DataFrame({"decision_time": X["decision_time"], "atr_h1": atr, "entry_m1": entry})
    for name, side in (("bull", 1), ("bear", -1)):
        w, r = triple_barrier(entry, atr, arr["open"], arr["high"], arr["low"], arr["close"], arr["spread"],
                              arr["minutes"], side, TP_ATR, SL_ATR, HORIZON_MIN)
        out[f"y_{name}"] = w
        out[f"r_{name}"] = r
    return out


def model():
    return HistGradientBoostingClassifier(max_leaf_nodes=31, learning_rate=0.05, max_iter=300, min_samples_leaf=300,
                                          l2_regularization=1.0, max_features=0.5, random_state=7)


def walk_forward(X, L, side, cols, start=2017, embargo_days=2):
    rows, models = [], {}
    yr = X["decision_time"].dt.year.values
    ok = np.isfinite(L[f"r_{side}"].values)
    for y in range(start, int(yr.max()) + 1):
        cut = pd.Timestamp(f"{y}-01-01") - pd.Timedelta(days=embargo_days)
        tr = ok & (X["decision_time"].values < cut.to_datetime64())
        te = ok & (yr == y)
        m = model().fit(X.loc[tr, cols], L.loc[tr, f"y_{side}"])
        p_tr = m.predict_proba(X.loc[tr, cols])[:, 1]
        p = m.predict_proba(X.loc[te, cols])[:, 1]
        thr = {q: float(np.quantile(p_tr, q)) for q in (0.9, 0.95, 0.99)}
        rows.append(pd.DataFrame({"decision_time": X.loc[te, "decision_time"].values, "prob": p,
                                  "y": L.loc[te, f"y_{side}"].values, "r": L.loc[te, f"r_{side}"].values,
                                  "thr90": thr[0.9], "thr95": thr[0.95], "thr99": thr[0.99], "year": y}))
        models[y] = m
    return pd.concat(rows, ignore_index=True), models


def fit_final(X, L, side, cols):
    ok = np.isfinite(L[f"r_{side}"].values)
    m = model().fit(X.loc[ok, cols], L.loc[ok, f"y_{side}"])
    p = m.predict_proba(X.loc[ok, cols])[:, 1]
    thr = {str(q): float(np.quantile(p, q)) for q in (0.9, 0.95, 0.99)}
    return m, thr
