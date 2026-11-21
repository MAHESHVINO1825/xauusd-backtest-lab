"""Run the five pre-registered hypotheses through the research gates.

    python -m research.run_top5            # open eras only (train / test / validation)
    python -m research.run_top5 --unseal   # ALSO report the sealed 2015-2019 era for the chosen configs
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

from . import harness as H
from . import strategies as S

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(20261121)
OPEN = H.OPEN_ERAS


def open_pf(tr, eras=("test", "validation")):
    p = tr.loc[tr["era"].isin(eras), "ret"].values
    return H.stats(p)["pf"]


def open_net(tr):
    return tr.loc[tr["era"].isin(OPEN), "ret"].sum()


def checks_for(tr, grid_pfs, stress_tr, delay_tr, ctrl_nets):
    m = H.by_era(tr)
    allopen = H.stats(tr.loc[tr["era"].isin(OPEN), "ret"].values)
    allopen["usd"] = float(tr.loc[tr["era"].isin(OPEN), "pnl"].sum())
    allopen["dd_usd"] = H.stats(tr.loc[tr["era"].isin(OPEN), "pnl"].values)["dd"]
    yrs = H.yearly(tr)
    return {
        "train PF ≥ 1.1": m["train"]["pf"] >= 1.1,
        "test PF ≥ 1.1": m["test"]["pf"] >= 1.1,
        "validation PF ≥ 1.1": m["validation"]["pf"] >= 1.1,
        "t-stat 2020–26 ≥ 2": allopen["t"] >= 2.0,
        "spread ×1.5 still PF > 1 (test+val)": open_pf(stress_tr) > 1.0,
        "one-bar delay still PF > 1 (test+val)": open_pf(delay_tr) > 1.0,
        "grid median PF > 1.05 (test+val)": float(np.median(grid_pfs)) > 1.05,
        "≥ 60% of years 2020–26 profitable": (yrs > 0).mean() >= 0.6 if len(yrs) else False,
        "beats 95% of random controls": (np.mean(np.array(ctrl_nets) < open_net(tr)) >= 0.95) if len(ctrl_nets) else False,
        "max DD (2020–26) ≤ 25% of $1000": allopen["dd_usd"] <= 25,
    }, m, allopen, yrs


def summarise(name, grid, trades_by_cfg, chosen_i, checks, m, allopen, yrs, ctrl_nets, extra=None):
    cfg = grid[chosen_i]
    rec = dict(name=name, chosen=cfg, eras=m, all_open=allopen, years=yrs.round(0).to_dict(),
               checks={k: bool(v) for k, v in checks.items()}, passed=int(sum(checks.values())),
               ctrl_p95=float(np.percentile(ctrl_nets, 95)) if len(ctrl_nets) else None,
               grid=[dict(cfg=g, **{e: H.stats(t.loc[t["era"] == e, "ret"].values) for e in OPEN})
                     for g, t in zip(grid, trades_by_cfg)])
    if extra:
        rec.update(extra)
    print(f"\n=== {name}  chosen {cfg}  gates {rec['passed']}/{len(checks)}")
    print("   ", H.fmt(m))
    rank = float(np.mean(np.array(ctrl_nets) < allopen["net"]) * 100) if len(ctrl_nets) else 0
    rec["ctrl_rank"] = rank
    print(f"    2020–26: PF {allopen['pf']:.2f} n{allopen['n']} {allopen['net']:+.0f}bp ${allopen['usd']:+.0f} t {allopen['t']:+.2f} $DD {allopen['dd_usd']:.1f}% · beats {rank:.0f}% of random controls")
    print("    years:", {int(k): int(v) for k, v in yrs.items()})
    for k, v in checks.items():
        print(f"    {'PASS' if v else 'fail'}  {k}")
    return rec


def choose(trades_by_cfg):
    sc = []
    for t in trades_by_cfg:
        s = H.stats(t.loc[t["era"] == "train", "ret"].values)
        sc.append(s["pf"] if s["n"] >= 30 else 0)
    return int(np.argmax(sc))


# ------------------------------------------------------------------ position families
def run_position_family(name, grid, posfn, control):
    b = H.bars15()
    trades = []
    for cfg in grid:
        tr = H.run_positions(posfn(**cfg))
        trades.append(tr)
        H.log_trial(dict(family=name, cfg=cfg, eras=H.by_era(tr)))
    i = choose(trades)
    cfg = grid[i]
    pos = posfn(**cfg)
    tr = trades[i]
    stress = H.run_positions(pos, spread_mult=1.5)
    delay = H.run_positions(pos, delay=1)
    grid_pfs = [open_pf(t) for t in trades]
    ctrl = control(cfg, pos)
    chk, m, allopen, yrs = checks_for(tr, grid_pfs, stress, delay, ctrl)
    rec = summarise(name, grid, trades, i, chk, m, allopen, yrs, ctrl)
    tr.to_parquet(os.path.join(OUT, f"trades_{name.split()[0]}.parquet"), index=False)
    return rec, pos


def ctrl_random_window(gate=None, side=1, n=200):
    """Same-length window at a random time of day, same trend gate."""
    b = H.bars15()
    tod = S.tod_minutes(b)

    def f(cfg, pos):
        s, e = S.hm(cfg["entry"]), S.hm(cfg["exit"])
        length = (e - s) % 1440
        g = gate() if gate else np.ones(len(b), bool)
        nets = []
        for _ in range(n):
            st = int(RNG.integers(0, 96)) * 15
            w = ((tod - st) % 1440) < length
            p = (w & g).astype(np.int8) * side
            nets.append(open_net(H.run_positions(p)))
        return nets
    return f


def ctrl_random_trend(n=200):
    """Random daily long/short series with the same switch rate and long share."""
    b = H.bars15()
    days = pd.Index(H.daily().index)
    k = days.get_indexer(b["day"].values)

    def f(cfg, pos):
        sig = S.daily_trend(cfg["lookback"]).values
        if cfg["mode"] == "long_only":
            sig = np.where(sig > 0, 1, 0)
        sw = np.mean(sig[1:] != sig[:-1])
        vals = np.unique(sig[sig != 0]) if cfg["mode"] == "long_only" else np.array([-1, 1])
        p_long = np.mean(sig > 0)
        nets = []
        for _ in range(n):
            r = np.empty(len(sig))
            cur = 1 if RNG.random() < p_long else (0 if cfg["mode"] == "long_only" else -1)
            for j in range(len(sig)):
                if RNG.random() < sw:
                    cur = 1 if RNG.random() < p_long else (0 if cfg["mode"] == "long_only" else -1)
                r[j] = cur
            pb = np.where(k >= 1, r[np.clip(k - 1, 0, None)], 0).astype(np.int8)
            nets.append(open_net(H.run_positions(pb)))
        return nets
    return f


# ------------------------------------------------------------------ S4 (event engine)
def run_s4():
    name = "S4 Vol-expansion breakout"
    trades, sigcache = [], {}
    for cfg in S.S4_GRID:
        key = (cfg["tf"], cfg["n"])
        if key not in sigcache:
            sigcache[key] = S.s4_signals(*key)
        tr = S.s4_run(**cfg, sigs=sigcache[key])
        trades.append(tr)
        H.log_trial(dict(family=name, cfg=cfg, eras=H.by_era(tr)))
    i = choose(trades)
    cfg = S.S4_GRID[i]
    sig = sigcache[(cfg["tf"], cfg["n"])]
    tr = trades[i]
    stress = S.s4_run(**cfg, spread_mult=1.5, sigs=sig)
    delay = S.s4_run(**cfg, delay=1, sigs=sig)
    grid_pfs = [open_pf(t) for t in trades]
    # random-anchor control: same number of entries at random bars, same exits, same side mix
    b, a, ls, ss = sig
    n_l, n_s = int(ls.sum()), int(ss.sum())
    nets = []
    for _ in range(200):
        pick = RNG.choice(len(b) - 1, n_l + n_s, replace=False)
        rl = np.zeros(len(b), bool); rs = np.zeros(len(b), bool)
        rl[pick[:n_l]] = True; rs[pick[n_l:]] = True
        nets.append(open_net(S.s4_run(**cfg, sigs=(b, a, rl, rs))))
    chk, m, allopen, yrs = checks_for(tr, grid_pfs, stress, delay, nets)
    rec = summarise(name, S.S4_GRID, trades, i, chk, m, allopen, yrs, nets)
    tr.to_parquet(os.path.join(OUT, "trades_S4.parquet"), index=False)
    return rec


def main(unseal=False):
    t0 = time.time()
    recs, positions = [], {}
    r, p = run_position_family("S1 Time-series momentum (D1)", S.S1_GRID, S.s1_positions, ctrl_random_trend())
    recs.append(r); positions["S1"] = p
    r, p = run_position_family("S2 Asian-session long drift", S.S2_GRID, S.s2_positions, ctrl_random_window())
    recs.append(r); positions["S2"] = p
    r, p = run_position_family("S3 Asian drift + D1 uptrend", S.S2_GRID, S.s3_positions,
                               ctrl_random_window(gate=lambda: S.trend_on_bars("blend") > 0))
    recs.append(r); positions["S3"] = p
    recs.append(run_s4())
    r, p = run_position_family("S5 US-session short in D1 downtrend", S.S5_GRID, S.s5_positions,
                               ctrl_random_window(gate=lambda: S.trend_on_bars("blend") < 0, side=-1))
    recs.append(r); positions["S5"] = p
    if unseal:
        print("\n=== SEALED ERA 2015–2019 (opened once) ===")
        for rec in recs:
            key = rec["name"].split()[0]
            tr = pd.read_parquet(os.path.join(OUT, f"trades_{key}.parquet"))
            s = H.stats(tr.loc[tr["era"] == "sealed", "ret"].values)
            rec["sealed"] = s
            print(f"  {rec['name']:45s} PF {s['pf']:.2f} n{s['n']} net {s['net']:+.0f} t {s['t']:+.2f} DD {s['dd']:.1f}%")
    with open(os.path.join(OUT, "top5_results.json"), "w") as f:
        json.dump(recs, f, default=float, indent=1)
    n_trials = sum(1 for _ in open(H.TRIALS_FILE))
    print(f"\nDone in {time.time() - t0:.0f}s · trials logged so far: {n_trials}")


if __name__ == "__main__":
    main(unseal="--unseal" in sys.argv)
