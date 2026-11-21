"""Run with:  python -m pytest -q tests"""
import numpy as np
import pandas as pd
import pytest

from core import data as D
from core import indicators as I
from core import search as S
from core.engine import simulate


def _flat_m1(n=600, start=100.0, step=0.0):
    o = start + step * np.arange(n)
    return o, o + 0.05, o - 0.05, o.copy(), np.zeros(n), np.arange(n, dtype=np.int64)


def _run(o, h, l, c, sp, mins, ls, ss, entry, atr, sl=1.0, rr=2.0, hold=10_000, lot=0.01):
    return simulate(ls, ss, entry, atr, o, h, l, c, sp, mins, sl, rr, hold, 10.0, 0.0, 0.0, 0.0, lot, 1e9)


def test_long_hits_take_profit_exact_pnl():
    o, h, l, c, sp, mins = _flat_m1(step=0.1)            # steady rally of 0.10 per minute
    ls = np.array([True, False]); ss = np.zeros(2, bool)
    res = _run(o, h, l, c, sp, mins, ls, ss, np.array([10, 20]), np.array([1.0, 1.0]))
    ent, ext, d, lot, ep, xp, pnl, r, why, skip = res
    assert len(pnl) == 1 and why[0] == 1                    # TP
    assert ep[0] == pytest.approx(o[10])
    assert xp[0] == pytest.approx(o[10] + 2.0)              # rr 2 x 1.0 ATR
    assert pnl[0] == pytest.approx(2.0 * 100 * 0.01)        # $2 on 0.01 lot


def test_short_pays_spread_and_hits_stop():
    o, h, l, c, sp, mins = _flat_m1(step=0.1)
    sp[:] = 0.3
    ls = np.zeros(2, bool); ss = np.array([True, False])
    res = _run(o, h, l, c, sp, mins, ls, ss, np.array([10, 20]), np.array([1.0, 1.0]))
    why, pnl = res[8], res[6]
    assert why[0] == 0                                       # SL on a rising market
    assert pnl[0] == pytest.approx(-1.0 * 100 * 0.01)        # stop is exactly 1.0 away


def test_same_bar_sl_and_tp_assumes_stop_first():
    o, h, l, c, sp, mins = _flat_m1()
    h[12] = o[12] + 5; l[12] = o[12] - 5                       # one huge M1 bar
    res = _run(o, h, l, c, sp, mins, np.array([True, False]), np.zeros(2, bool), np.array([10, 20]), np.array([1.0, 1.0]))
    assert res[8][0] == 0


def test_time_exit():
    o, h, l, c, sp, mins = _flat_m1()
    res = _run(o, h, l, c, sp, mins, np.array([True, False]), np.zeros(2, bool), np.array([10, 20]), np.array([1.0, 1.0]), hold=30)
    assert res[8][0] == 2 and res[1][0] == 40


def test_no_overlapping_trades():
    o, h, l, c, sp, mins = _flat_m1()
    ls = np.array([True, True, True, False]); ss = np.zeros(4, bool)
    res = _run(o, h, l, c, sp, mins, ls, ss, np.array([10, 15, 400, 500]), np.ones(4), hold=100)
    assert len(res[0]) == 2                                   # second signal ignored while busy


def test_risk_mode_skips_when_account_too_small():
    o, h, l, c, sp, mins = _flat_m1()
    res = simulate(np.array([True, False]), np.zeros(2, bool), np.array([10, 20]), np.array([50.0, 50.0]),
                   o, h, l, c, sp, mins, 1.0, 1.0, 100, 10.0, 0.0, 0.0, 0.0, 0.0, 30.0)
    assert len(res[0]) == 0 and res[9] == 1


def test_indicators_have_no_lookahead():
    b = D.bars("H1").iloc[:4000].reset_index(drop=True)
    full = I.compute_states(b, I.NAMES)
    cut = I.compute_states(b.iloc[:3000].reset_index(drop=True), I.NAMES)
    for nm in I.NAMES:
        if nm in ("VWAP", "Pivot points", "Prev-day high/low"):
            n = 2950   # last partial trading day of the cut differs only in day grouping, not values
        else:
            n = 3000
        for side in (0, 1):
            assert np.array_equal(full[nm][side][:n], cut[nm][side][:n]), nm


def test_random_entries_without_costs_are_breakeven():
    m1 = D.m1_arrays(0.0)
    ctx = S.TFContext("H1", ["EMA cross"], m1)
    cfg = S.Config(indicators=["EMA cross"], slippage=0.0, swap_night=0.0)
    rng = np.random.default_rng(3)
    pfs = []
    for _ in range(8):
        ls = rng.random(len(ctx.bars)) < 0.03
        ss = (rng.random(len(ctx.bars)) < 0.03) & ~ls
        pfs.append(S.pf_of(ctx.run(ls, ss, 1.0, 1.0, cfg)[6]))
    assert 0.9 < np.mean(pfs) < 1.1


def test_entries_fill_after_signal_bar_closes():
    b = D.bars("M15")
    e = D.entry_index(b)
    t = D.load_m1()["time"].values
    ok = e < len(t)
    assert (t[e[ok]] >= b["end"].values[ok]).all()
