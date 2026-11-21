# XAUUSD Backtest Lab

Research lab for gold (XAUUSD): an indicator-combination backtester with a dashboard, a hypothesis-driven
research harness, and two trained AI agents — **Agent Bull** and **Agent Bear**.

## Run the dashboard
Double-click **`run_backtest_lab.bat`** (first launch installs packages, 2–4 min) → http://localhost:8501

## What's inside
| Folder | What it is |
|---|---|
| `app.py`, `core/` | Dashboard + combination search: 21 indicators, M5–H4, scalping/swing, $500–2000, 9 validation gates |
| `research/` | Hypothesis tests (S1–S7), walk-forward ML filters, pre-registration log, **FINDINGS.md** |
| `agents/` | **Agent Bull / Agent Bear**: 162-feature library, training, saved models, S5/S6 heads — see `agents/MODEL_CARD.md` |
| `data/` | XAUUSD M1 2007-01 → 2026-11, **UTC** (20 years) |
| `tests/` | 12 tests: engine maths, no look-ahead, agents, live-window consistency |

## Data (UTC)
* 2007–2024: XAU/USD 1-minute archive (GitHub `parshant65-lang`). **Its timestamps are broker server time
  (New York + 7h), not UTC** — converted here and verified against Dukascopy ticks (offset 0, corr 0.97–0.99).
* 2025–2026: Dukascopy tick data (GitHub `Touhid-HZ/xauusd-tick-data-*`) → M1 bid bars + real spread.
* `spread_hour_shape.csv`, `spread_after_reopen.csv`: measured spread profiles (rollover / 18:00 NY reopen widening).

## Research eras
Sealed 2015–19 (opened once per hypothesis) · Train 2020–22 · Test 2023–24 · Validation 2025–26.
Dashboard split: train 2015–22 · test 2023–24 · holdout 2025–26.

## Current candidates (details: research/FINDINGS.md, agents/MODEL_CARD.md)
| Strategy | Rule | With agent head (walk-forward, out-of-sample) |
|---|---|---|
| **S6** | Long 23:00 → 00:00 UTC (Asian reopen) | Bull head: PF 1.16 → **1.36**, beats 99% of random filters |
| **S5** | Short 15:30 → 20:30 UTC on daily-downtrend days | Bear head: PF 1.20 → **1.45**, beats 95% of random filters |
Both are small, regime-dependent edges (strongest 2023–26). **Forward-test on demo before live money.**

**Agent Bull v2** also trades on its own: top-5% signals, stop 1×ATR(H1), target 2×ATR, max 16h → PF 1.28 out-of-sample.
**Agent Bear** sniper mode (top-5% + daily downtrend) → PF 1.36 but only ~10 trades/year.

## Merge the agents into ANY bot (v3)
```python
from agents.plugin import GoldAgents
ga = GoldAgents(mode="strict")            # "dual-strict" for reversal / scalping bots
verdict = ga.check("BUY", m1_utc)         # or GoldAgents.m1_from_mt5(mt5, "XAUUSD")
if verdict["approve"]: place_order()
```
Backtest 2013–2026 on 5 different bots: the agents improved all of them; Donchian breakout went from −22 R to
**+89 R (R-PF 1.40, 12/14 years)**, MACD −172 R → +48 R, EMA cross −81 R → +36 R, RSI −577 R → +29 R.
Full tables: `agents/MERGE_RESULTS.md`. Example: `examples/merge_into_any_bot.py`.

## Use the agents from a bot (lower level)
```python
from agents.agent import BullAgent, BearAgent
bull, bear = BullAgent(), BearAgent()
bull.score(m1_utc)                 # general long-opportunity probability + active bullish patterns
bull.approve_s6(m1_utc, now_utc)   # at 23:00 UTC
bear.approve_s5(m1_utc)            # at 15:30 UTC
```
Retrain yearly: `python -m agents.build_models`. Tests: `python -m pytest -q tests`.
