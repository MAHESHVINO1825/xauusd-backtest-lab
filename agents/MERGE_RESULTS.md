# Merging Agent Bull & Agent Bear into other bots — backtest results

**Setup.** Real XAUUSD M1 data (UTC). Agents = v3 ensemble, trained on 2007–2026 history with yearly walk-forward:
every year from 2012 is scored by models trained only on earlier years. Results below are 2013–2026 (14 years).
Plugin policy exactly as in `agents/plugin.py`: each agent's score is compared with its own scores over the
previous 90 days (self-calibrating, no look-ahead).

Same exit for every bot: stop 1.5 × ATR(H1), target 2R, max 24 h, one trade at a time, 0.01 lot,
spread 0.20 × hour profile, slippage 0.05/side, swap $30/lot/night.
"R" = result in risk units (fair across 2013's $1,200 gold and 2026's $4,000 gold). "beats" = share of
random filters (same number of trades kept at random) that the agent filter beats.

| Bot | Bot alone | + agents, best mode | Mode |
|---|---|---|---|
| Donchian-20 breakout (H4) | −22 R, PF 0.97 (R), 5/14 yrs | **+89 R, R-PF 1.40, PF$ 1.43, 12/14 yrs, DD $185** (beats 100%) | strict |
| MACD cross (H1) | −172 R, R-PF 0.94, 5/14 yrs | **+48 R, R-PF 1.10, PF$ 1.34, 9/14 yrs, DD $182** (beats 100%) | dual-strict |
| EMA 20/50 cross (H1) | −81 R, R-PF 0.91, 3/14 yrs | **+36 R, R-PF 1.12, PF$ 1.34, 11/14 yrs, DD $114** (beats 100%) | strict |
| RSI 30/70 reversal (M15) | −577 R, R-PF 0.84, 0/14 yrs | **+29 R, R-PF 1.03, PF$ 1.12, 6/14 yrs** (beats 100%) | dual-strict |
| Supertrend flip (H1) | −173 R, R-PF 0.86, 3/14 yrs | −5 R, R-PF 0.97, PF$ 1.13, 7/14 yrs (beats 70%) | dual-strict |

All modes, all bots (n trades / R-PF / R / years positive):

| Bot | standard | strict | dual-strict |
|---|---|---|---|
| EMA 20/50 cross | 699 / 1.11 / +46 / 10 | 473 / 1.12 / +36 / 11 | 215 / 1.10 / +13 / 8 |
| Supertrend flip | 869 / 0.97 / −18 / 6 | 584 / 0.92 / −31 / 4 | 234 / 0.97 / −5 / 7 |
| MACD cross | 2484 / 1.05 / +77 / 8 | 1787 / 1.07 / +73 / 8 | 818 / 1.10 / +48 / 9 |
| RSI 30/70 reversal | 3620 / 0.89 / −258 / 3 | 2705 / 0.94 / −102 / 5 | 1455 / 1.03 / +29 / 6 |
| Donchian-20 breakout | 595 / 1.27 / +90 / 10 | 407 / 1.40 / +89 / 12 | 225 / 1.69 / +76 / 12 |

**Reading.**
* The agents improved **every** bot in every mode versus the bot alone; in 14 of 15 bot/mode cases they beat
  ≥ 88% of random filters, so it is skill, not just "trading less".
* Trend-following / breakout bots: use **strict**. Reversal / oscillator bots: use **dual-strict**.
* The agents cannot rescue a bot with no edge at all (Supertrend flip stays ≈ breakeven); they turn weak bots
  with some information into profitable ones.
* Choosing the best mode per bot from these tables is itself a selection — forward-test before going live.

## Agents trading on their own (2012–2026, 15 years, same self-calibrating thresholds)
| Agent | Trades | PF ($) | R-PF | Result | Years positive |
|---|---|---|---|---|---|
| Bull, top 2% signals (stop 1 ATR, target 2R, max 16h) | 1,149 | 1.22 | 1.11 | +82 R, +$765 @0.01, DD $200 | 11/15 |
| Bull, top 5% | 2,307 | 1.17 | 1.05 | +79 R, +$1,216 | 9/15 |
| Bear sniper: top 5% + daily downtrend (target 1.5R, max 8h) | 1,304 | 1.08 | 1.19 | +137 R, +$202 | 10/15 |
| Bear, top 5% (no trend gate) | 3,073 | 1.04 | 1.08 | +135 R, +$330 | 10/15 |

Training on 2007–2014 (2008 crash, 2013 crash) made Bear far more useful: sniper trades went from ~10 to ~87 a year.
