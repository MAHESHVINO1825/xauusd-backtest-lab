"""GoldAgents — plug Agent Bull and Agent Bear into ANY XAUUSD bot as an approval layer.

Minimal use (any bot, any strategy):

    from agents.plugin import GoldAgents
    ga = GoldAgents(mode="strict")                 # "standard" | "strict" | "dual-strict"
    ...
    # your bot found a BUY or SELL signal:
    verdict = ga.check("BUY", m1_utc)              # m1_utc: last ~14 months of M1 bars in UTC
    if verdict["approve"]:
        place_order(...)
    print(verdict["reason"])                       # human-readable, ready for Telegram

With MetaTrader 5 directly:

    import MetaTrader5 as mt5
    m1 = GoldAgents.m1_from_mt5(mt5, "XAUUSD")     # fetches bars and converts broker time -> UTC
    ga.check("SELL", m1)

Modes (validated out-of-sample 2013–2026 on five different bots, see agents/MERGE_RESULTS.md):
  standard     approve if the agent for that direction is above its typical (median) level
  strict       approve only the agent's top ~35% — best for trend-following / breakout bots
  dual-strict  strict, AND the opposite agent must be below its median — best for reversal / scalping bots

Self-calibrating thresholds: each agent's current score is compared with that agent's own scores at
every 15-minute point of the previous 90 days (computed from the same M1 window you pass in), so the
plugin needs no history of your bot and adapts as markets change. Exactly this policy was backtested.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .agent import BearAgent, BullAgent
from . import features as F

MODES = ("standard", "strict", "dual-strict")


class GoldAgents:
    def __init__(self, mode: str = "strict", models_dir: str | None = None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        kw = {"models_dir": models_dir} if models_dir else {}
        self.bull, self.bear = BullAgent(**kw), BearAgent(**kw)
        self.mode = mode
        self._cal = None

    # ---------------------------------------------------------------- data helpers
    @staticmethod
    def m1_from_mt5(mt5, symbol: str = "XAUUSD", days: int = 430) -> pd.DataFrame:
        """Fetch M1 bars from a running MT5 terminal and convert broker server time to UTC.
        The broker offset is measured from the latest tick (server time vs. real UTC)."""
        n = days * 1440
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, min(n, 99999 * 10))
        df = pd.DataFrame(rates)
        tick = mt5.symbol_info_tick(symbol)
        server_now = pd.to_datetime(tick.time, unit="s")
        utc_now = pd.Timestamp.utcnow().tz_localize(None)
        offset_h = round((server_now - utc_now).total_seconds() / 3600)
        df["time"] = pd.to_datetime(df["time"], unit="s") - pd.Timedelta(hours=offset_h)
        df = df.rename(columns={"tick_volume": "volume"})
        return df[["time", "open", "high", "low", "close", "volume"]]

    # ---------------------------------------------------------------- scoring
    def scores(self, m1: pd.DataFrame) -> dict:
        Xb = self.bull.features(m1)
        b = self.bull.score(m1, Xb)
        r = self.bear.score(m1, Xb)
        self._cal = self._calibration(Xb)
        return {"bull": b, "bear": r}

    CAL_DAYS = 90

    def _calibration(self, Xb: pd.DataFrame) -> dict:
        """Agent probabilities at every decision point of the last 90 days before the current one."""
        t = Xb["decision_time"]
        win = (t >= t.iloc[-1] - pd.Timedelta(days=self.CAL_DAYS)) & (t < t.iloc[-1])
        return {"bull": self.bull.general.predict_proba(Xb.loc[win, self.bull.cols])[:, 1],
                "bear": self.bear.general.predict_proba(Xb.loc[win, self.bear.cols])[:, 1]}

    def _level(self, side: str, q: float) -> float:
        return float(np.quantile(self._cal[side], q))

    def check(self, direction: str, m1: pd.DataFrame, scores: dict | None = None) -> dict:
        """Approve or veto a BUY / SELL signal from any bot."""
        d = direction.upper()
        if d not in ("BUY", "SELL"):
            raise ValueError("direction must be BUY or SELL")
        s = scores or self.scores(m1)
        own, opp = ("bull", "bear") if d == "BUY" else ("bear", "bull")
        p_own, p_opp = s[own]["prob"], s[opp]["prob"]
        need = 0.65 if self.mode in ("strict", "dual-strict") else 0.50
        thr_own = self._level(own, need)
        ok = p_own >= thr_own
        why = [f"{own.capitalize()} {p_own:.2f} {'≥' if ok else '<'} {thr_own:.2f} ({int(need * 100)}th pct of last 90 days)"]
        if self.mode == "dual-strict":
            thr_opp = self._level(opp, 0.50)
            ok_opp = p_opp <= thr_opp
            ok = ok and ok_opp
            why.append(f"{opp.capitalize()} {p_opp:.2f} {'≤' if ok_opp else '>'} {thr_opp:.2f} (median)")
        pats = s[own]["patterns"]
        return {"direction": d, "approve": bool(ok), "mode": self.mode,
                "confidence": round(p_own, 4), "opposite": round(p_opp, 4),
                "patterns": pats, "decision_time": s[own]["decision_time"],
                "reason": ("APPROVED: " if ok else "VETO: ") + "; ".join(why) + (f"; patterns: {', '.join(pats)}" if pats else "")}
