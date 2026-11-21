# XAUUSD research log

## 2026-11-21 — Round 1: literature top-5 + reopen drift

**Data correction (critical).** The GitHub M1 source (2015 – mid-2025) is stamped in broker server time
(New York + 7h → UTC+2 winter / UTC+3 summer), not UTC as its README claims. It was converted to UTC, and
2025–2026 was replaced with Dukascopy tick-derived M1 bars (true UTC, with real bid/ask spread).
Verified offset 0 against Dukascopy ticks in 2024 and 2026. **Every result produced before this fix is void,
including the 2026-11-20 dashboard runs** (Session filter, VWAP, pivots and prev-day levels were all on shifted hours).

**Costs:** spread 0.20 × hour-of-day profile, slippage 0.05/side, swap $30/lot/night (×3 Wed), 0.01 lot.
**Eras:** sealed 2015–19 · train 2020–22 · test 2023–24 · validation 2025–26. Statistics in % of notional (bp).

| Strategy | Best config (chosen on train) | 2020–26 gates | Sealed 2015–19 | Verdict |
|---|---|---|---|---|
| S1 Time-series momentum D1 | blend 21/63/252, long-short | 5/10 | not opened | fail |
| S2 Asian drift long | 01:00→07:00 | 3/10 | not opened | fail |
| S3 Asian drift + uptrend | 01:00→07:00 | 3/10 | not opened | fail |
| S4 Vol-expansion breakout | H4, N20, SL 2 ATR, 3R | 4/10 | not opened | fail |
| S5 US-session short, D1 downtrend | 15:30→20:30 | **9/10** (PF 1.25, beats 97% random) | **PF 0.81** | fail — regime-specific |
| S6 Reopen drift long | 23:00→00:00 | **9/10** (PF 1.27, t 2.69, DD 9.9%) | **PF 0.88** (still beats 90% random) | fail — recent-regime only |

**Hourly profile (UTC, corrected, 2020–26):** strongest hours 21–23 UTC (23:00 t = 4.3) and 01:00, 06:00.
In 2015–19 the reopen hour was still relatively strong (top 10% of windows) but too small to beat costs.

**Takeaways**
1. Edges seen in 2023–26 (Asian-reopen long, US-afternoon short in downtrends) did not exist net of costs in 2015–19.
   Either they are new (Asian / central-bank demand regime) or they are luck. Only forward data can separate the two.
2. Time-of-day effects in gold are real in gross terms but small (≈1–4 bp/hour); 0.20 spread ≈ 0.5–1.5 bp per trade eats most of it.
3. Total trials logged this round: 54 configurations across 6 families.

## 2026-11-21 — Round 2: S5 / S6 deep-dive + walk-forward ML

**Cost model update:** measured Dukascopy spread after the 18:00 NY reopen (×1.43 first 15 min, ×1.23 next 15, ~×1.15 after) now applied.

**S6 diagnostics (long 23:00→00:00 UTC):** works in bull regimes — top quartile distance above 200-day MA PF 1.73 (t 4.1),
after a strong week PF 1.60 (t 3.7); US winter PF 1.48 vs summer 0.97. Fails in flat markets (most of 2015–19).
**S7** (same idea anchored to NY reopen +15 min) was worse (PF 1.12) — rejected.

**S5 diagnostics (short 15:30→20:30 in D1 downtrend):** best after the London/NY morning rallied (top quartile PF 1.78, t 2.9);
too few trades (~90/yr) for ML.

**Walk-forward ML (train on years < Y, predict Y, 2017–2026; keep top 50% by predicted win probability):**
| | No filter | Logistic filter |
|---|---|---|
| S6 PF / t | 1.16 / 2.04 | **1.32 / 2.65** (beats 94% of random filters) |
| S6 trades/yr, $ at 0.01 lot | 257/yr, +$838 (DD $119) | 128/yr, +$638 (DD $147), avg +$0.50/trade |
| S5 | PF 1.10 | no improvement (≈ random filter) |
Model weights: + distance above 200-day MA, − last-hour / session return (buy dips), − US summer, + 20-day return.

**Verdict:** S6 + ML filter = best candidate so far, but a SMALL, regime-dependent edge (strong 2025–26, ~flat 2017–23).
Next step is a demo forward test, not more fitting.

## 2026-11-21 — Round 3: Agent Bull / Agent Bear
General scorers (162 features, triple-barrier labels, walk-forward 2017–26): AUC 0.56–0.65 every year; top-5% signals win
42–43% vs 27% base; standalone PF 1.01 (bull) / 1.08 (bear) — beat 100% of hour-matched random entries but not costs.
Used as generic filters on S5/S6 they did NOT help. Strategy-specific heads did:
S6 + Bull head (16 context features at entry): PF 1.16 → 1.36 (beats 99% random filters), +$850 2017–26, max DD $94.
S5 + Bear head (GBM, 162 features): PF 1.20 → 1.45 (beats 95% random filters), 7/9 years positive.
Details: agents/MODEL_CARD.md.

## 2026-11-21 — Round 4: agent improvement study → v2
Bull v2 (188 features, small GBM, recency 2y, target 2R/16h): standalone PF 1.01 → **1.28** (479 trades 2017–26, beats 100% random).
Bear: no v2 change helped standalone; best = v1 + daily-downtrend gate, PF 1.36 on 99 trades. Heads unchanged (v1 best).
Candlestick patterns add no measurable value. Biggest lever = the prediction target (longer, 2:1).
Next data to add (needs laptop / MT5 export): EURUSD or DXY, US 10y yield, S&P 500, economic calendar.

## 2026-11-21 — Round 5: 20-year data, v3 ensemble agents, merge-ready plugin
Data extended to 2007–2026 (2007–14 broker time converted to UTC week by week, anchored on the 17:00 NY daily break).
Agents v3 = ensemble (fast + full GBM), 188 features, recency 2y; walk-forward 2012–2026.
Merge backtest (5 bots, 2013–26, self-calibrating plugin policy): agents improved every bot; best modes →
Donchian +89R (R-PF 1.40, 12/14 yrs), MACD +48R, EMA +36R, RSI +29R, Supertrend −5R (≈ breakeven). Details agents/MERGE_RESULTS.md.
Standalone: Bull top-2% +82R (11/15 yrs); Bear sniper +137R (R-PF 1.19, 10/15 yrs, ~87 trades/yr).

## 2026-11-21 — Round 6: Mahe's running M5 bot + low-timeframe search
Replicated the user's M5 bot with its own code on 2010–2026: live rule ≈ breakeven over 14 years (−61R, PF 0.98, 6/14 yrs;
strong only 2024–26). + agents dual-strict → +89R, PF 1.08, 10/14 yrs. Same engine on M30 + agents dual-strict →
**+170R, PF 1.22, 11/14 yrs, t 4.0**, survives cost 0.30 (PF 1.20). Simple indicator bots on M5/M15 lose even with agents.
Details: research/userbot/README.md.
