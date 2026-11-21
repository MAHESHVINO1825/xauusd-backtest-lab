"""Backtest: do Agent Bull / Agent Bear improve OTHER bots when merged as an approval layer?

Each bot produces its own long/short signals. Merged version = a long is taken only if Agent Bull
approves it, a short only if Agent Bear approves it. Agent probabilities come from yearly walk-forward
models (each year scored by a model trained only on earlier years). The approval threshold for year Y
is the median agent probability at that bot's own signals in years before Y — no look-ahead.

Same exit for every bot: stop 1.5 x ATR(H1), target 2R, max hold 24h, one trade at a time, 0.01 lot,
spread 0.20 x hour profile, slippage 0.05/side, swap. Real M1 price path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core import data as D
from core.engine import simulate

SL_ATR, RR, HOLD = 1.5, 2.0, 24 * 60


def bot_signals(X: pd.DataFrame) -> dict:
    """Five common bots, built from the feature table (all values known at the decision time)."""
    def turn(s, val):
        s = np.asarray(s)
        prev = np.r_[np.nan, s[:-1]]
        return (s == val) & (prev != val)
    e = np.sign(X["ind_H1_dist_ema50"].values - X["ind_H1_dist_ema20"].values)    # = sign(EMA20 - EMA50)
    st = X["ind_H1_supertrend"].values
    mh = np.sign(X["ind_H1_macd_hist"].values)
    rsi = X["ind_M15_rsi"].values
    rp = np.r_[np.nan, rsi[:-1]]
    dc = X["str_H4_breakout20"].values
    return {
        "EMA 20/50 cross (H1)": (turn(e, 1), turn(e, -1)),
        "Supertrend flip (H1)": (turn(st, 1), turn(st, -1)),
        "MACD cross (H1)": (turn(mh, 1), turn(mh, -1)),
        "RSI 30/70 reversal (M15)": ((rsi > 30) & (rp <= 30), (rsi < 70) & (rp >= 70)),
        "Donchian-20 breakout (H4)": (turn(dc, 1), turn(dc, -1)),
    }


def run(ls, ss, L, arr):
    res = simulate(ls, ss, L["entry_m1"].values, L["atr_h1"].values, arr["open"], arr["high"], arr["low"], arr["close"],
                   arr["spread"], arr["minutes"], SL_ATR, RR, HOLD, 10.0, 0.05, 0.0, 30.0, 0.01, 1e9)
    t = pd.Series(arr["time"][res[0]])
    return pd.DataFrame({"time": t, "year": t.dt.year, "side": res[2], "pnl": res[6], "r": res[7]})


def pf(x):
    x = np.asarray(x, float)
    gp, gl = x[x > 0].sum(), -x[x < 0].sum()
    return gp / gl if gl > 0 else np.nan


def approve(sig: np.ndarray, prob: np.ndarray, years: np.ndarray, start: int) -> np.ndarray:
    """Approve a signal if the agent's probability beats the median at this bot's signals in earlier years."""
    out = np.zeros_like(sig)
    for y in np.unique(years[sig]):
        if y < start:
            continue
        prev = sig & (years < y) & np.isfinite(prob)
        cur = sig & (years == y)
        if prev.sum() < 30:
            continue
        thr = np.nanmedian(prob[prev])
        out |= cur & (prob >= thr)
    return out


def summarise(tr: pd.DataFrame, y0: int) -> dict:
    t = tr[tr["year"] >= y0]
    yr = t.groupby("year")["r"].sum()
    eq = 1000 + t["pnl"].cumsum()
    return dict(n=len(t), pf=pf(t["pnl"]), pf_r=pf(t["r"]), R=t["r"].sum(), usd=t["pnl"].sum(),
                dd=(eq.cummax() - eq).max() if len(t) else 0, yrs=int((yr > 0).sum()), nyrs=len(yr))
