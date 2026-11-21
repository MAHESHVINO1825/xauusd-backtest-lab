"""Out-of-sample evaluation of a walk-forward agent run."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from core import data as D
from core.engine import simulate

from . import train as T

_arr = {}


def arrays():
    if "a" not in _arr:
        _arr["a"] = D.m1_arrays(T.SPREAD)
    return _arr["a"]


def pf(x):
    x = np.asarray(x, float)
    gp, gl = x[x > 0].sum(), -x[x < 0].sum()
    return gp / gl if gl > 0 else np.nan


def trade(wf: pd.DataFrame, L: pd.DataFrame, side: str, mask: np.ndarray):
    """Trade the chosen decision points, one position at a time, real M1 path, 0.01 lot."""
    a = arrays()
    lab = L.set_index("decision_time").loc[wf["decision_time"]]
    ent = lab["entry_m1"].values
    atr = lab["atr_h1"].values
    sig = mask.astype(bool)
    z = np.zeros(len(sig), bool)
    ls, ss = (sig, z) if side == "bull" else (z, sig)
    res = simulate(ls, ss, ent, atr, a["open"], a["high"], a["low"], a["close"], a["spread"], a["minutes"],
                   T.SL_ATR, T.TP_ATR / T.SL_ATR, T.HORIZON_MIN, 10.0, T.SLIP, 0.0, 30.0, 0.01, 1e9)
    t = pd.Series(a["time"][res[0]])
    return pd.DataFrame({"time": t, "year": t.dt.year, "pnl": res[6], "r": res[7]})


def report(wf: pd.DataFrame, L: pd.DataFrame, side: str, q="thr99", n_ctrl=100, seed=3):
    rng = np.random.default_rng(seed)
    print(f"\n######## Agent {side.upper()} — out-of-sample, each year scored by a model trained only on earlier years")
    rows = []
    for y, g in wf.groupby("year"):
        auc = roc_auc_score(g["y"], g["prob"])
        top = g[g["prob"] >= g[q]]
        rows.append((y, auc, g["y"].mean(), top["y"].mean() if len(top) else np.nan, len(top)))
    t = pd.DataFrame(rows, columns=["year", "AUC", "base_win", "top_win", "n_top"])
    print(t.round(3).to_string(index=False))
    sel = (wf["prob"] >= wf[q]).values
    tr = trade(wf, L, side, sel)
    # control: same number of signals at random decision points with the same hour-of-day mix
    hrs = wf["decision_time"].dt.hour.values
    want = pd.Series(hrs[sel]).value_counts()
    ctrl_pf, ctrl_net = [], []
    for _ in range(n_ctrl):
        m = np.zeros(len(wf), bool)
        for h, k in want.items():
            idx = np.where(hrs == h)[0]
            m[rng.choice(idx, min(k, len(idx)), replace=False)] = True
        c = trade(wf, L, side, m)
        ctrl_pf.append(pf(c["pnl"]))
        ctrl_net.append(c["r"].sum())
    beats = np.mean(np.array(ctrl_net) < tr["r"].sum()) * 100
    eq = 1000 + tr["pnl"].cumsum()
    print(f"  TRADED (top {100 - int(q[3:])}% signals, one at a time): {len(tr)} trades, win {100 * (tr['pnl'] > 0).mean():.0f}%, "
          f"PF {pf(tr['pnl']):.2f}, {tr['r'].sum():+.0f} R, ${tr['pnl'].sum():+.0f} at 0.01 lot, max DD ${(eq.cummax() - eq).max():.0f}")
    print(f"  random hour-matched control: median PF {np.nanmedian(ctrl_pf):.2f} · agent beats {beats:.0f}% of controls")
    print("  $ by year:", tr.groupby("year")["pnl"].sum().round(0).astype(int).to_dict())
    print("  PF by year:", tr.groupby("year")["pnl"].apply(pf).round(2).to_dict())
    return t, tr, beats
