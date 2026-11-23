# M5 research (2026-11-24) - pick on 2013-2019, verify on unseen 2020-2026, cost 0.30 unless noted
| Idea | Variants | Positive on unseen 2020-26 | Verdict |
|---|---|---|---|
| Time-of-day drift (NY hour, 1-4h hold) | 87 | 23%, best IS pick lost OOS | no edge |
| Opening-range breakout (Tokyo/London/COMEX/NYSE) | 192 | 3% | no edge (consistently negative) |
| Fade the opening-range breakout | 288 | 0% | no edge |
| ORB only in agent direction (grade A/B) | 144 | 24%, 0% in both periods | no edge |
| Earlier: agents + M5 EMA confirmation (scalp.py) | 16 | - | PF 0.77-1.12, no edge |
| **User's 14 M5 setups, agent grade A only, multi-trade** | 1 | yes | **modest edge** |

M5 grade-A engine (max 4 open, 1 per setup, no hedge, 1 new per bar) + rule cost <= 8% of stop distance, cost 0.30:
3,190 trades (0.82/day; 2.7/day in the last 12 months), avg +0.045R, PF 1.13, +143R, max DD 47R, 11/14 years positive.
2013-19 +0.053R (t 1.6) | 2020-26 +0.042R (t 2.2). Beats a random-pick control (95th pct +0.012R).
Last 12 months: +0.048R, PF 1.14, about +$782 at 0.01 lot (avg risk $21/trade). Daily correlation with the M30 grade-A engine 0.20.
Cost sensitivity: without the cost rule PF 1.11 / 1.05 / 0.99 at cost 0.20 / 0.30 / 0.40. Grade B and C fail on unseen years.
Caveat: the 8% cost rule was chosen after looking at cost buckets (mild selection); the edge is thin.
