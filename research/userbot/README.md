# Analysis of Mahe's running M5 bot (xauusd_agent.py / xauusd_agent_v2.py)

The bot file itself is NOT stored here (it contains account passwords and Telegram tokens).
To re-run: copy `xauusd_agent_v2.py` to `/home/claude/bots/userbot_v2.py` (or edit the import), then
`python prep_userbot.py`, `python eval_userbot.py`, `python multitf.py 30 [cost]`.

The scripts import the bot's OWN functions (14 setup detectors, features, LightGBM score, 3×ATR stop / 2R /
lock at +0.6R → +0.15R / trail from +1R) and replay them on 2010–2026 UTC data with a yearly walk-forward of
the bot's model. Agents = v3 ensemble walk-forward scores, self-calibrating 90-day thresholds (plugin policy).

## Results 2013–2026 (one trade at a time, cost 0.20)
| Version | Trades/yr | Total | PF | Years + | t |
|---|---|---|---|---|---|
| M5 live rule (score > 0) | 708 | −61 R | 0.98 | 6/14 | −0.7 |
| M5 + agents dual-strict | 208 | +89 R | 1.08 | 10/14 | 1.7 |
| M15 + agents dual-strict | 245 | +126 R | 1.11 | 11/14 | 2.3 |
| **M30 + agents dual-strict** | **183** | **+170 R** | **1.22** | **11/14** | **4.0** |
| M30 + agents strict | 310 | +182 R | 1.13 | 11/14 | 3.2 |
| M30 dual-strict @ cost 0.30 | 146 | +119 R | 1.20 | 10/14 | 3.1 |
| M30 dual-strict @ cost 0.40 | 104 | +54 R | 1.12 | 9/14 | 1.7 |

Live demo (16–20 Nov 2026): account 115820371 +$110 (6 trades), account 5058919823 +$57 (27 trades);
nearly all profit from sells during a ~$340 fall in gold — consistent with the bot's strong 2024–26 backtest years.

## M30 + Agents upgrade (2026-11-21)
`patch_m30_agents.py <src_bot.py> <dst.py> <magic_base>` turns the M5 bot into a multi-timeframe bot (default M30)
with Agent Bull/Bear approval (dual-strict). Separate MAGIC (base+30) and DB per timeframe.

Offline checks on the patched bot:
- `--mode once` and Time Machine OK on M30.
- Bot's own walk-forward (no agents), 36 unseen months 2023-12..2026-11: 1,749 trades, PF 1.04, +22R.
- AgentGate replay 2026-10-05..09: 53 candidate signals, 4 approved, 49 vetoed (agents are selective by design).
- Yearly backtest 2013-26, M30 + agents dual-strict: +170R, PF 1.22, 11/14 years positive (cost 0.30: PF 1.20).

## M5 scalping verdict
16 variants (agents + M5 confirmation, tight/wide exits): PF 0.77-1.12, t <= 1.6. More trades = worse.
No separate M5 scalping engine is recommended. More daily trades -> run several validated engines in parallel.
