# Agent Bull & Agent Bear — model card (v3 ensemble, 2026-11-21)

## v3 (current)
* Data: XAUUSD M1 **2007-01 → 2026-11 (20 years, UTC)** — 2007–2014 added (2008 and 2013 crashes).
* Each agent = average of two gradient-boosting models (fast + full), 188 features, recency weighting (2-year half-life).
* Bull target: +2R before −1R within 16h. Bear target: +1.5R before −1R within 8h.
* **Merge-ready**: `agents/plugin.py` (`GoldAgents.check("BUY"|"SELL", m1)`), self-calibrating thresholds,
  modes standard / strict / dual-strict, MT5 helper with automatic broker-time → UTC conversion.
* Results: see **agents/MERGE_RESULTS.md** (5 bots, 2013–2026) — agents improved every bot; 4 of 5 became profitable.
* S5/S6 heads were retrained on the 20-year data with the same design (validation numbers below are from the 2015–2026 study).

---

## (previous) v2 summary

## v2 summary
| Role | Version | Out-of-sample result (walk-forward 2017–26, costs included) |
|---|---|---|
| **Bull — standalone long signals** | **v2**: 188 features, small GBM, recency half-life 2y, target +2R before −1R within 16h | top-5% signals: 479 trades, **PF 1.28**, +46 R, +$615 @0.01, max DD $187; beats 100% of hour-matched random entries (median PF 0.96); PF>1 in 7/10 years |
| Bull — S6 filter | v1 head | PF 1.16 → **1.36** |
| **Bear — standalone "sniper"** | v1 general + only when the daily trend is down | 99 trades, **PF 1.36** (R-PF 1.28), +15 R, 7/10 years positive — few trades, treat as low-confidence |
| Bear — S5 filter | v1 head | PF 1.20 → **1.45** |

### What the improvement study found (one change at a time, fast walk-forward 2019–26)
* Candlestick patterns: removing all 42 changed AUC by ±0.0005 — **no measurable value**; kept only for explanations.
* +26 extra features (efficiency ratio, session VWAP, Asian/London range, reopen gap, streaks, NFP timing, month-turn): small gain.
* Smaller / more regularised models beat bigger ones (overfitting is the main enemy).
* Recency weighting (2-year half-life) helped both agents' ranking.
* **The target was the biggest lever for Bull:** +2R within 16h raised top-5% R/trade from +0.06 to +0.20 (7/8 years).
  Short targets (1R in 4h, 2R in 8h) score higher AUC but lose after costs.
* Bear: longer targets and the v2 changes did NOT help standalone shorting (gold's long uptrend); v1 + downtrend gate is best.
* Heads: recency weighting, extra features and stacking the Bull v2 score did not improve the S5/S6 heads — v1 heads kept.
* Not yet tested (needs data the cloud workspace cannot download): US dollar (EURUSD / DXY), US yields, S&P 500, real news calendar.

---


Built 2026-11-21 on XAUUSD M1 data 2015-01-02 → 2026-11-15 (UTC, corrected — see research/FINDINGS.md).

## What each agent is
| | Agent Bull | Agent Bear |
|---|---|---|
| General scorer | P(long reaches +1.5×ATR(H1) before −1.0×ATR within 8h) | P(short reaches +1.5×ATR(H1) before −1.0×ATR within 8h) |
| Strategy head | **S6 head** — approves/rejects the 23:00→00:00 UTC long | **S5 head** — approves/rejects the 15:30→20:30 UTC short (D1 downtrend days) |
| Model | Gradient boosting, 162 features | Gradient boosting, 162 features |
| Head model | Logistic regression, 16 context features at the entry minute | Gradient boosting, 162 features |

**Features (162):** candlestick patterns on M15/H1/H4 (engulfing, hammer, shooting star, doji, inside/outside bar,
marubozu, morning/evening star, three soldiers/crows, wick/body geometry), chart structure on H1/H4 (fractal swings,
HH/HL/LH/LL, double top/bottom, triangle-like contraction, 20/50-bar breakouts, position in range), key levels
(previous day/week high-low-close, today's open/high/low, $10/$50 round numbers), indicators (RSI, Stoch-RSI, MACD,
ADX/DI, Supertrend, Bollinger %b/width, EMA 20/50/200 distance & slope), volatility (ATR percentile, range expansion,
tick volume), momentum returns, time (hour, weekday, US DST, minutes since the 18:00 NY reopen) and regime
(daily trend, distance from 200-day MA, 5/20/60-day returns, realised-vol ratio). No look-ahead (tested).

## Out-of-sample results (yearly walk-forward: each year predicted by a model trained only on earlier years)
Costs: spread 0.20 × hour profile (reopen widening measured from Dukascopy), slippage 0.05/side, swap $30/lot/night, 0.01 lot.

### General scorers (2017–2026)
| | AUC per year | Base win rate | Top-5% win rate | Traded top 5% (one at a time) | vs random entries at same hours |
|---|---|---|---|---|---|
| Bull | 0.56 – 0.62 (all 10 years > 0.5) | 27% | 43% | 506 trades, PF 1.01 | beats 100% (random PF 0.90) |
| Bear | 0.56 – 0.65 (all 10 years > 0.5) | 27% | 42% | 469 trades, PF 1.08 | beats 100% (random PF 0.95) |
**Reading:** real, stable predictive skill — but not enough on its own to beat costs. Use as a filter / confirmation.

### Strategy heads (the useful part)
| | No filter | With head | Random-filter control |
|---|---|---|---|
| **S6 + Bull head** (2017–26) | PF 1.16, 2,542 trades | **PF 1.36**, 1,291 trades, +$850, max DD $94 | beats 99% |
| **S5 + Bear head** (2018–26) | PF 1.20, 720 trades | **PF 1.45**, 336 trades, +$327 | beats 95% |
| S6 + Bull *general* scorer | PF 1.15 | PF 1.19 | 71% — not useful |
| S5 + Bear *general* scorer | PF 1.20 | PF 1.09 | 22% — harmful |

## Limits — read before trading
* Edges are small and **regime-dependent**: most S6 profit came in 2025–26 (gold bull market); 2017–23 ≈ flat.
* S6 and S5 themselves failed the sealed 2015–19 era test (PF 0.88 / 0.81). The heads improve them; they do not make them proven.
* Several filter variants were tried for S5 before the GBM head passed (multiple-testing risk). Forward-test before real money.
* Retrain yearly: `python -m agents.build_models`.
* Live data must be converted to **UTC** before calling the agents.
