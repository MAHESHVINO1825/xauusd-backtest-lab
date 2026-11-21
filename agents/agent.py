"""Agent Bull and Agent Bear — use them from any bot.

    from agents.agent import BullAgent, BearAgent
    bull, bear = BullAgent(), BearAgent()

    # m1 = DataFrame with columns time (UTC, naive), open, high, low, close, volume — at least ~300 days of M1
    #      (from MT5: mt5.copy_rates_from_pos(...); convert broker server time to UTC first!)
    bull.score(m1)              # {'prob': 0.41, 'top10': True, 'top5': False, 'patterns': [...], ...}
    bull.approve_s6(m1, now)    # {'approve': True, 'prob': 0.53, 'threshold': 0.50}  at 23:00 UTC
    bear.approve_s5(m1, now)    # same for the 15:30 UTC S5 short

IMPORTANT: features assume UTC timestamps. MT5 server time is usually UTC+2/UTC+3 — convert before calling.
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd

from . import features as F
from . import features_ext as FE
from . import heads as HD

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

BULL_PATTERNS = ["cdl_M15_bull_engulf", "cdl_H1_bull_engulf", "cdl_H1_hammer", "cdl_H1_morning_star", "cdl_H4_bull_engulf",
                 "cdl_H4_hammer", "str_H1_double_bottom", "str_H4_double_bottom", "str_H1_hl", "str_H4_hl"]
BEAR_PATTERNS = ["cdl_M15_bear_engulf", "cdl_H1_bear_engulf", "cdl_H1_shooting_star", "cdl_H1_evening_star", "cdl_H4_bear_engulf",
                 "cdl_H4_shooting_star", "str_H1_double_top", "str_H4_double_top", "str_H1_lh", "str_H4_lh"]


class _Agent:
    side = ""

    def __init__(self, models_dir: str = MODELS):
        with open(os.path.join(models_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.general = joblib.load(os.path.join(models_dir, f"{self.side}_general.joblib"))
        self.cols = self.meta["features_v3"]
        self.thr = self.meta[f"{self.side}_general_thresholds"]

    def features(self, m1: pd.DataFrame) -> pd.DataFrame:
        X = F.build(m1)
        E = FE.build_ext(X, m1)
        return pd.concat([X, E.drop(columns="decision_time")], axis=1)

    def score(self, m1: pd.DataFrame, X: pd.DataFrame | None = None) -> dict:
        """Probability that a trade opened NOW in this agent's direction hits its target before its stop.
        Bull v2: +2.0 x ATR(H1) before -1.0 x ATR within 16h.  Bear v1: +1.5 x ATR(H1) before -1.0 x ATR within 8h.
        Trade rule validated out-of-sample: act on 'top5' signals, stop = 1 x ATR(H1), target and max hold as above."""
        X = self.features(m1) if X is None else X
        row = X.iloc[[-1]]
        p = float(self.general.predict_proba(row[self.cols])[:, 1][0])
        pats = BULL_PATTERNS if self.side == "bull" else BEAR_PATTERNS
        active = [c for c in pats if c in row and row[c].iloc[0] == 1]
        out = {"agent": self.side, "decision_time": str(row["decision_time"].iloc[0]), "prob": round(p, 4),
               "top10": p >= self.thr["0.9"], "top5": p >= self.thr["0.95"], "top1": p >= self.thr["0.99"],
               "patterns": active, "target": self.meta[f"{self.side}_label"]}
        if self.side == "bear":
            # sniper mode: only act in a daily downtrend (validated PF 1.36, ~10 trades / year)
            out["daily_downtrend"] = bool(row["reg_trend_blend"].iloc[0] < 0)
            out["sniper_signal"] = bool(out["top5"] and out["daily_downtrend"])
        return out


class BullAgent(_Agent):
    side = "bull"

    def __init__(self, models_dir: str = MODELS):
        super().__init__(models_dir)
        self.s6 = joblib.load(os.path.join(models_dir, "bull_s6_head.joblib"))

    def approve_s6(self, m1: pd.DataFrame, entry_time) -> dict:
        """Call at the S6 entry (23:00 UTC) with M1 bars up to and including that minute's open."""
        cf = HD.context_features(m1, [pd.Timestamp(entry_time)])
        p = float(self.s6.predict_proba(cf[HD.S6_FEATS])[:, 1][0])
        thr = self.meta["s6_threshold"]
        return {"strategy": "S6", "approve": p >= thr, "prob": round(p, 4), "threshold": round(thr, 4)}


class BearAgent(_Agent):
    side = "bear"

    def __init__(self, models_dir: str = MODELS):
        super().__init__(models_dir)
        self.s5 = joblib.load(os.path.join(models_dir, "bear_s5_head.joblib"))

    def approve_s5(self, m1: pd.DataFrame, X: pd.DataFrame | None = None) -> dict:
        """Call at the S5 decision (15:30 UTC, after the 15:15 M15 bar closed), only on D1-downtrend days."""
        X = self.features(m1) if X is None else X
        row = X.iloc[[-1]]
        p = float(self.s5.predict_proba(row[self.meta["features_general"]])[:, 1][0])
        thr = self.meta["s5_threshold"]
        return {"strategy": "S5", "approve": p >= thr, "prob": round(p, 4), "threshold": round(thr, 4),
                "decision_time": str(row["decision_time"].iloc[0])}
