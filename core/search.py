"""Combination search + validation gauntlet."""
from __future__ import annotations

import itertools
import json
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import data as D
from . import indicators as I
from .engine import REASONS, simulate

RESULTS_DIR = os.path.join(D.ROOT, "results")

EXIT_PRESETS = {
    "Scalping": [(1.0, 1.0), (1.0, 1.5), (1.5, 1.0), (1.5, 2.0)],
    "Swing": [(2.0, 1.5), (2.0, 2.5), (3.0, 1.5), (3.0, 2.5)],
}
HOLD_MIN = {"Scalping": 240, "Swing": 7200}

GATE_NAMES = [
    "Enough trades", "PF test ≥ 1.3", "Cost stress", "Rolling windows",
    "Param stability", "Yearly consistency", "Beats random", "Monte Carlo DD", "Holdout",
]
GATE_HELP = [
    "≥ 100 trades in train and ≥ 30 in test",
    "Profit factor on the 2023–24 test period ≥ 1.3",
    "Spread ×1.5 and double slippage: test PF still ≥ 1.1",
    "≥ 65% of 6-month windows (2015–24) profitable",
    "Indicator lengths ±20% and stop ±25%: median test PF ≥ 1.1, none below 0.9",
    "≥ 70% of calendar years (2015–24) profitable",
    "Net profit beats 95% of 200 random-entry runs with the same exits",
    "95th-percentile drawdown of 1,000 reshuffles ≤ 35% of capital",
    "Holdout (2025–2026): PF ≥ 1.1 with ≥ 10 trades",
]


@dataclass
class Config:
    indicators: list
    max_k: int = 2
    timeframes: list = field(default_factory=lambda: ["M15"])
    style: str = "Scalping"
    direction: str = "Both"
    capital: float = 1000.0
    lot_mode: str = "Fixed lot"      # "Fixed lot" or "Risk %"
    fixed_lot: float = 0.01
    risk_pct: float = 1.0
    max_forced_risk_pct: float = 3.0
    base_spread: float = 0.20
    slippage: float = 0.05
    commission: float = 0.0
    swap_night: float = 5.0
    finalists: int = 30

    @property
    def risk_usd(self):
        return self.capital * self.risk_pct / 100.0

    @property
    def lot_arg(self):
        return self.fixed_lot if self.lot_mode == "Fixed lot" else 0.0


# ---------------------------------------------------------------- metrics
def metrics(pnl: np.ndarray, r: np.ndarray, capital: float) -> dict:
    n = len(pnl)
    if n == 0:
        return dict(trades=0, win=0.0, pf=0.0, net=0.0, ret=0.0, dd=0.0, exp_r=0.0, avg_win=0.0, avg_loss=0.0)
    gp = pnl[pnl > 0].sum()
    gl = -pnl[pnl < 0].sum()
    eq = capital + np.cumsum(pnl)
    peak = np.maximum.accumulate(np.r_[capital, eq])[1:]
    dd = float(((peak - eq) / peak).max()) if n else 0.0
    return dict(
        trades=n, win=float((pnl > 0).mean() * 100), pf=float(gp / gl) if gl > 0 else (99.0 if gp > 0 else 0.0),
        net=float(pnl.sum()), ret=float(pnl.sum() / capital * 100), dd=dd * 100,
        exp_r=float(r.mean()), avg_win=float(pnl[pnl > 0].mean()) if gp > 0 else 0.0,
        avg_loss=float(pnl[pnl < 0].mean()) if gl > 0 else 0.0,
    )


def pf_of(p):
    gp, gl = p[p > 0].sum(), -p[p < 0].sum()
    return gp / gl if gl > 0 else (99.0 if gp > 0 else 0.0)


# ---------------------------------------------------------------- context
class TFContext:
    """Everything needed to backtest on one timeframe, computed once."""

    def __init__(self, tf: str, names, m1: dict, scale: float = 1.0):
        self.tf = tf
        self.bars = D.bars(tf)
        self.entry = D.entry_index(self.bars)
        b = self.bars
        self.atr = I.atr(b["high"].values, b["low"].values, b["close"].values, 14)
        h4 = D.bars("H4")
        self.h4 = (h4["end"].values, h4["close"].values.astype(np.float64))
        self.states = I.compute_states(b, names, scale=scale, h4_close=self.h4)
        self.m1 = m1
        self.period = D.period_of(m1["time"])  # per M1 bar

    def signals(self, combo, direction):
        bull = np.ones(len(self.bars), bool)
        bear = np.ones(len(self.bars), bool)
        for nm in combo:
            b, s = self.states[nm]
            bull &= b
            bear &= s
        ls = I.edge(bull) if direction in ("Long", "Both") else np.zeros_like(bull)
        ss = I.edge(bear) if direction in ("Short", "Both") else np.zeros_like(bear)
        return ls, ss

    def run(self, ls, ss, sl_atr, rr, cfg: Config, spread_mult=1.0, slip_mult=1.0):
        m = self.m1
        spread = m["spread"] * spread_mult if spread_mult != 1.0 else m["spread"]
        return simulate(ls, ss, self.entry, self.atr, m["open"], m["high"], m["low"], m["close"],
                        spread, m["minutes"], sl_atr, rr, HOLD_MIN[cfg.style], cfg.risk_usd,
                        cfg.slippage * slip_mult, cfg.commission, cfg.swap_night, cfg.lot_arg,
                        cfg.capital * cfg.max_forced_risk_pct / 100.0)


def combos_for(cfg: Config):
    out = []
    for k in range(1, cfg.max_k + 1):
        for c in itertools.combinations(cfg.indicators, k):
            if any(I.DIRECTIONAL[n] for n in c):
                out.append(c)
    return out


def count_runs(cfg: Config) -> int:
    return len(combos_for(cfg)) * len(cfg.timeframes) * len(EXIT_PRESETS[cfg.style])


# ---------------------------------------------------------------- gauntlet
def rolling_windows(times, pnl, mask):
    t = pd.to_datetime(times[mask])
    p = pnl[mask]
    if len(p) == 0:
        return 0.0, []
    key = t.year.values * 2 + (t.month.values > 6)
    res = []
    for k in np.unique(key):
        pp = p[key == k]
        if len(pp) >= 5:
            res.append((f"{k // 2}H{k % 2 + 1}", float(pf_of(pp)), float(pp.sum())))
    if not res:
        return 0.0, res
    return float(np.mean([x[2] > 0 for x in res]) * 100), res


def monte_carlo(pnl, capital, n=1000, seed=1):
    rng = np.random.default_rng(seed)
    if len(pnl) < 5:
        return 100.0, None
    dds = np.empty(n)
    paths = np.empty((n, len(pnl)))
    for i in range(n):
        p = rng.choice(pnl, len(pnl), replace=True)
        eq = capital + np.cumsum(p)
        peak = np.maximum.accumulate(np.r_[capital, eq])[1:]
        dds[i] = ((peak - eq) / peak).max()
        paths[i] = eq
    q = np.percentile(paths, [5, 50, 95], axis=0)
    return float(np.percentile(dds, 95) * 100), q


def random_benchmark(ctx: TFContext, n_trades, long_frac, sl_atr, rr, cfg, n=200, seed=7):
    rng = np.random.default_rng(seed)
    ent_period = ctx.period[np.clip(ctx.entry, 0, len(ctx.period) - 1)]
    pool = np.where(ent_period <= 1)[0]
    if n_trades < 5 or len(pool) < n_trades:
        return np.array([])
    nets = np.empty(n)
    for i in range(n):
        pick = rng.choice(pool, n_trades, replace=False)
        ls = np.zeros(len(ctx.bars), bool)
        ss = np.zeros(len(ctx.bars), bool)
        isl = rng.random(n_trades) < long_frac
        ls[pick[isl]] = True
        ss[pick[~isl]] = True
        res = ctx.run(ls, ss, sl_atr, rr, cfg)
        nets[i] = res[6].sum()
    return nets


def trades_frame(ctx: TFContext, res) -> pd.DataFrame:
    ent, ext, d, lot, ep, xp, pnl, r, why, _skip = res
    t = ctx.m1["time"]
    return pd.DataFrame({
        "entry_time": t[ent], "exit_time": t[ext], "side": np.where(d == 1, "BUY", "SELL"),
        "lots": lot, "entry": ep.round(2), "exit": xp.round(2), "pnl": pnl.round(2), "r": r.round(3),
        "reason": [REASONS[int(w)] for w in why], "minutes": ctx.m1["minutes"][ext] - ctx.m1["minutes"][ent],
        "period": np.array(D.PERIODS)[ctx.period[ent]],
    })


def gauntlet(ctx_by_scale, combo, sl_atr, rr, cfg: Config):
    ctx = ctx_by_scale[1.0]
    ls, ss = ctx.signals(combo, cfg.direction)
    res = ctx.run(ls, ss, sl_atr, rr, cfg)
    tr = trades_frame(ctx, res)
    pnl, r = tr["pnl"].values, tr["r"].values
    per = ctx.period[res[0]]
    m = {p: metrics(pnl[per == i], r[per == i], cfg.capital) for i, p in enumerate(D.PERIODS)}
    is_mask = per <= 1
    g = []
    g.append(m["train"]["trades"] >= 100 and m["test"]["trades"] >= 30)
    g.append(m["test"]["pf"] >= 1.3)
    st = ctx.run(ls, ss, sl_atr, rr, cfg, spread_mult=1.5, slip_mult=2.0)
    st_pf = pf_of(st[6][ctx.period[st[0]] == 1])
    g.append(st_pf >= 1.1)
    roll_pct, roll = rolling_windows(ctx.m1["time"][res[0]], pnl, is_mask)
    g.append(roll_pct >= 65)
    neigh = []
    for sc in (0.8, 1.2):
        c2 = ctx_by_scale[sc]
        a, b = c2.signals(combo, cfg.direction)
        rr_ = c2.run(a, b, sl_atr, rr, cfg)
        neigh.append(pf_of(rr_[6][c2.period[rr_[0]] == 1]))
    for f in (0.75, 1.25):
        rr_ = ctx.run(ls, ss, sl_atr * f, rr, cfg)
        neigh.append(pf_of(rr_[6][ctx.period[rr_[0]] == 1]))
    g.append(float(np.median(neigh)) >= 1.1 and min(neigh) >= 0.9)
    yrs = pd.to_datetime(tr["entry_time"][is_mask]).dt.year
    ys = pd.Series(pnl[is_mask]).groupby(yrs.values).agg(["sum", "size"])
    ys = ys[ys["size"] >= 10]
    year_pct = float((ys["sum"] > 0).mean() * 100) if len(ys) else 0.0
    g.append(year_pct >= 70)
    nis = int(is_mask.sum())
    long_frac = float((tr["side"][is_mask] == "BUY").mean()) if nis else 0.5
    rnd = random_benchmark(ctx, nis, long_frac, sl_atr, rr, cfg)
    is_net = float(pnl[is_mask].sum())
    rnd_pct = float((rnd < is_net).mean() * 100) if len(rnd) else 0.0
    g.append(rnd_pct >= 95)
    mc_dd, mc_q = monte_carlo(pnl[is_mask], cfg.capital)
    g.append(mc_dd <= 35)
    g.append(m["holdout"]["pf"] >= 1.1 and m["holdout"]["trades"] >= 10)
    detail = dict(
        skipped=int(res[9]), stress_pf=st_pf, neighbours=neigh, roll_pct=roll_pct, roll=roll, year_pct=year_pct,
        years=ys.reset_index().values.tolist(), rnd_pct=rnd_pct,
        rnd_p95=float(np.percentile(rnd, 95)) if len(rnd) else None, is_net=is_net, mc_dd95=mc_dd,
    )
    return [bool(x) for x in g], m, detail, tr, mc_q


# ---------------------------------------------------------------- main run
def run_search(cfg: Config, progress=None, log=None):
    t0 = time.time()
    say = log or (lambda s: None)
    m1 = D.m1_arrays(cfg.base_spread)
    combos = combos_for(cfg)
    presets = EXIT_PRESETS[cfg.style]
    total = len(combos) * len(cfg.timeframes) * len(presets)
    rows, done = [], 0
    ctxs = {}
    for tf in cfg.timeframes:
        say(f"Preparing {tf} indicators…")
        ctx = TFContext(tf, cfg.indicators, m1)
        ctxs[tf] = ctx
        for combo in combos:
            ls, ss = ctx.signals(combo, cfg.direction)
            for sl_atr, rr in presets:
                res = ctx.run(ls, ss, sl_atr, rr, cfg)
                pnl, per = res[6], ctx.period[res[0]]
                row = {"combo": " + ".join(combo), "tf": tf, "sl_atr": sl_atr, "rr": rr, "skipped": int(res[9])}
                for i, p in enumerate(D.PERIODS):
                    pp = pnl[per == i]
                    row[f"{p}_trades"] = len(pp)
                    row[f"{p}_pf"] = pf_of(pp)
                    row[f"{p}_net"] = float(pp.sum())
                rows.append(row)
                done += 1
                if progress and done % 25 == 0:
                    progress(done / total * 0.8, f"Screening {done:,}/{total:,} · {tf}")
    df = pd.DataFrame(rows)
    say(f"Screened {len(df):,} backtests in {time.time() - t0:.0f}s")
    df["screen"] = (df["train_trades"] >= 60) & (df["train_pf"] >= 1.1) & (df["test_trades"] >= 20) & (df["test_pf"] >= 1.0)
    fin = df[df["screen"]].sort_values("test_pf", ascending=False).head(cfg.finalists)
    say(f"{int(df['screen'].sum())} passed the screen; validating top {len(fin)}")

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(RESULTS_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)
    finals = []
    scaled = {}
    for n_i, (_, row) in enumerate(fin.iterrows()):
        tf = row["tf"]
        if tf not in scaled:
            scaled[tf] = {1.0: ctxs[tf], 0.8: TFContext(tf, cfg.indicators, m1, 0.8), 1.2: TFContext(tf, cfg.indicators, m1, 1.2)}
        combo = tuple(row["combo"].split(" + "))
        gates, m, det, tr, mc_q = gauntlet(scaled[tf], combo, row["sl_atr"], row["rr"], cfg)
        sid = f"s{n_i:03d}"
        tr.to_parquet(os.path.join(out_dir, f"trades_{sid}.parquet"), index=False)
        if mc_q is not None:
            np.save(os.path.join(out_dir, f"mc_{sid}.npy"), mc_q)
        finals.append(dict(id=sid, combo=row["combo"], tf=tf, sl_atr=row["sl_atr"], rr=row["rr"],
                           gates=gates, passed=int(sum(gates)), metrics=m, detail=det))
        if progress:
            progress(0.8 + 0.2 * (n_i + 1) / max(1, len(fin)), f"Validating {n_i + 1}/{len(fin)} · {row['combo']} {tf}")
    df.to_parquet(os.path.join(out_dir, "screen.parquet"), index=False)
    meta = dict(run_id=run_id, config=asdict(cfg), total=total, screened=int(df["screen"].sum()),
                seconds=round(time.time() - t0, 1), data_first=str(m1["time"][0]), data_last=str(m1["time"][-1]))
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(dict(meta=meta, finalists=finals), f, default=float, indent=1)
    say(f"Done in {time.time() - t0:.0f}s · {sum(1 for x in finals if x['passed'] == 9)} passed all 9 gates")
    return out_dir


def list_runs():
    if not os.path.isdir(RESULTS_DIR):
        return []
    return sorted([d for d in os.listdir(RESULTS_DIR) if os.path.exists(os.path.join(RESULTS_DIR, d, "run.json"))], reverse=True)


def load_run(run_id):
    p = os.path.join(RESULTS_DIR, run_id)
    with open(os.path.join(p, "run.json")) as f:
        js = json.load(f)
    js["screen"] = pd.read_parquet(os.path.join(p, "screen.parquet"))
    js["dir"] = p
    return js
