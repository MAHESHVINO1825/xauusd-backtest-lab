# Pre-registration log

## 2026-11-21 — Top-5 hypotheses (from literature review)
S1 TSMOM D1, S2 Asian drift, S3 Asian drift + D1 uptrend, S4 vol-expansion breakout, S5 US-session short in D1 downtrend.
Grids: see strategies.py (S1_GRID, S2_GRID, S4_GRID, S5_GRID). Selection by TRAIN (2020-22) PF.
Gates: see run_top5.checks_for.

## 2026-11-21 — DATA CORRECTION
Source M1 data 2015 – mid-2025 was stamped in broker server time (New York + 7h = UTC+2/UTC+3), not UTC.
Converted to UTC (server − 7h as America/New_York → UTC); 2025–2026 replaced with Dukascopy tick-derived M1 (true UTC).
Verified against Dukascopy ticks (2024 w10/w30/w45, 2026 w10/w30): best alignment offset 0, corr 0.97–0.99.
All results before this correction (incl. the 2026-11-20 dashboard runs) are void.

## 2026-11-21 — S6 reopen drift (registered BEFORE running)
Hypothesis: long from shortly after the 22:00 UTC reopen into the first Asian hours.
Grid: entry ∈ {22:15, 22:30, 23:00} × exit ∈ {00:00, 01:00, 02:00}. Same gates and selection rule.
Basis: 2020–26 hourly profile on corrected data (hours 22–23 UTC strongest). This is data-informed, so the
2015–2019 sealed era (not yet viewed on corrected data) is the decisive out-of-sample test.

## 2026-11-21 — Unsealing 2015–2019 for survivors (9/10 gates): S5 {15:30→20:30} and S6 {23:00→00:00}. Chosen configs fixed before unsealing.

## 2026-11-21 — S7 Reopen drift anchored to the NY 18:00 reopen (registered BEFORE running)
Motivation: S6 diagnostics — edge strong in US winter (when 23:00 UTC = reopen hour), absent in summer.
Grid: entry = reopen + {15, 30} min; exit = reopen + {60, 120} min; long only. Costs: Dukascopy 2025–26
measured spread multiplier by minute-since-reopen (15-min buckets) × 0.20 base. Same gates.
Then: walk-forward logistic filter (keep 50%), identical settings to the S6 test — no re-tuning.
