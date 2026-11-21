"""Research harness for hypothesis-driven XAUUSD strategies.

Eras (fixed before any test was run):
    sealed      2015-01 .. 2019-12   opened ONCE at the very end
    train       2020-01 .. 2022-12   parameter choice
    test        2023-01 .. 2024-12   ranking
    validation  2025-01 .. 2026-11   check (already partly seen in earlier runs)

Costs: spread 0.20 x hour-of-day multiplier (rollover hours dearer), slippage 0.05 per side,
swap 30 USD per lot per night on any position held through 22:00 UTC (x3 Wednesday), 0.01 lot.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import data as D  # noqa: E402

LOT = 0.01
CONTRACT = 100.0
SPREAD = 0.20
SLIP = 0.05
SWAP_PER_LOT_NIGHT = 30.0
CAPITAL = 1000.0

ERAS = [("sealed", "2015-01-01", "2020-01-01"), ("train", "2020-01-01", "2023-01-01"),
        ("test", "2023-01-01", "2025-01-01"), ("validation", "2025-01-01", "2027-01-01")]
OPEN_ERAS = ["train", "test", "validation"]

TRIALS_FILE = os.path.join(os.path.dirname(__file__), "trial_log.jsonl")


def log_trial(rec: dict):
    with open(TRIALS_FILE, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")


def era_of(t: pd.Series) -> np.ndarray:
    out = np.empty(len(t), dtype=object)
    for name, a, b in ERAS:
        out[(t >= a) & (t < b)] = name
    return out


_cache = {}


def bars15() -> pd.DataFrame:
    """M15 bars 2015-2026 with hour-scaled spread; times are bar starts (UTC)."""
    if "m15" not in _cache:
        b = D.bars("M15").copy()
        b["hour"] = b["time"].dt.hour
        b["minute"] = b["time"].dt.minute
        b["spread"] = SPREAD * D.spread_shape()[b["hour"].values]
        # minutes since the daily 18:00 New York reopen, and the measured reopen spread widening
        ny = b["time"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
        b["since_reopen"] = ((ny.dt.hour * 60 + ny.dt.minute).values - 18 * 60) % 1440
        rp = os.path.join(D.DATA_DIR, "spread_after_reopen.csv")
        if os.path.exists(rp):
            tab = pd.read_csv(rp, index_col=0)["mult"]
            m = b["since_reopen"] < 240
            b.loc[m, "spread"] = SPREAD * tab.reindex((b.loc[m, "since_reopen"] // 15) * 15).values
        b["day"] = (b["time"] + pd.Timedelta(hours=2)).dt.normalize()   # trading day starts 22:00 UTC
        b["era"] = era_of(b["time"])
        _cache["m15"] = b.reset_index(drop=True)
    return _cache["m15"]


def daily() -> pd.DataFrame:
    """Daily bars on the 22:00 UTC rollover."""
    if "d1" not in _cache:
        b = bars15()
        d = b.groupby("day").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                                 close=("close", "last"), n=("close", "size"))
        d = d[d["n"] >= 20]
        _cache["d1"] = d
    return _cache["d1"]


# ------------------------------------------------------------------ position backtest
def run_positions(pos: np.ndarray, spread_mult=1.0, delay=0, b: pd.DataFrame | None = None) -> pd.DataFrame:
    """pos[t] = target position (-1/0/+1) held from the OPEN of bar t to the open of bar t+1.
    Returns one row per trade (contiguous run of same non-zero position)."""
    b = bars15() if b is None else b
    pos = np.asarray(pos, dtype=np.int8)
    if delay:
        pos = np.r_[np.zeros(delay, np.int8), pos[:-delay]]
    o = b["open"].values
    nxt = np.r_[o[1:], o[-1]]
    pnl_bar = pos * (nxt - o) * CONTRACT * LOT
    # costs on position changes: half spread + slippage per unit of change
    prev = np.r_[0, pos[:-1]]
    dpos = np.abs(pos.astype(np.int16) - prev)
    cost = dpos * (b["spread"].values * spread_mult / 2 + SLIP) * CONTRACT * LOT
    # swap: holding a position across the 22:00 UTC bar boundary
    roll = (b["hour"].values == 22) & (b["minute"].values == 0)
    wd = b["time"].dt.weekday.values
    nights = np.where(roll, np.where(wd == 2, 3, 1), 0)
    swap = (prev != 0) & (pos != 0) & (nights > 0)
    swap_cost = swap * nights * SWAP_PER_LOT_NIGHT * LOT
    net = pnl_bar - cost - swap_cost
    # trades = runs
    change = np.r_[True, pos[1:] != pos[:-1]]
    run_id = np.cumsum(change)
    ret = net / (o * CONTRACT * LOT)   # fraction of notional: scale-free across price levels
    df = pd.DataFrame({"run": run_id, "pos": pos, "net": net, "ret": ret, "t": b["time"].values})
    df = df[df["pos"] != 0]
    if df.empty:
        return pd.DataFrame(columns=["entry_time", "exit_time", "side", "pnl", "ret", "bars", "era"])
    g = df.groupby("run")
    tr = pd.DataFrame({"entry_time": g["t"].first(), "exit_time": g["t"].last(), "side": g["pos"].first(),
                       "pnl": g["net"].sum(), "ret": g["ret"].sum() * 1e4, "bars": g["net"].size()}).reset_index(drop=True)
    tr["era"] = era_of(tr["entry_time"])
    return tr


# ------------------------------------------------------------------ metrics
def stats(p: np.ndarray) -> dict:
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return dict(n=0, pf=0.0, net=0.0, win=0.0, t=0.0, dd=0.0)
    gp, gl = p[p > 0].sum(), -p[p < 0].sum()
    eq = CAPITAL + np.cumsum(p)
    peak = np.maximum.accumulate(np.r_[CAPITAL, eq])[1:]
    sd = p.std(ddof=1) if n > 1 else 0
    return dict(n=n, pf=float(gp / gl) if gl > 0 else 99.0, net=float(p.sum()), win=float((p > 0).mean() * 100),
                t=float(p.mean() / sd * np.sqrt(n)) if sd > 0 else 0.0, dd=float(((peak - eq) / peak).max() * 100))


def by_era(tr: pd.DataFrame, include_sealed=False, col="ret") -> dict:
    eras = [e for e, _, _ in ERAS if include_sealed or e != "sealed"]
    out = {}
    for e in eras:
        x = tr[tr["era"] == e]
        s = stats(x[col].values)
        s["usd"] = float(x["pnl"].sum())
        s["dd_usd_pct"] = stats(x["pnl"].values)["dd"]
        out[e] = s
    return out


def yearly(tr: pd.DataFrame, eras=("train", "test", "validation")) -> pd.Series:
    t = tr[tr["era"].isin(eras)]
    return t.groupby(pd.to_datetime(t["entry_time"]).dt.year)["ret"].sum()


def fmt(m: dict) -> str:
    return " | ".join(f"{e[:5]} PF {v['pf']:.2f} n{v['n']} {v['net']:+.0f}bp ${v['usd']:+.0f} t {v['t']:+.1f}" for e, v in m.items())


@dataclass
class Verdict:
    name: str
    params: dict
    eras: dict
    checks: dict

    @property
    def passed(self):
        return all(self.checks.values())
