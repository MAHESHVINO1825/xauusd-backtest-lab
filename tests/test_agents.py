"""Agent tests:  python -m pytest -q tests/test_agents.py"""
import numpy as np
import pandas as pd
import pytest

from agents import features as F
from agents import features_ext as FE
from agents.agent import BearAgent, BullAgent
from core import data as D


@pytest.fixture(scope="module")
def m1():
    return D.load_m1()


def test_features_have_no_lookahead(m1):
    part = m1[(m1.time >= "2023-01-01") & (m1.time < "2024-03-01")]
    full = F.build(part)
    cut = F.build(part[part.time < "2024-02-01"])
    j = full.merge(cut, on="decision_time", suffixes=("", "_c"))
    j = j[j.decision_time < "2024-01-28"]
    for c in F.feature_columns(full):
        assert np.allclose(j[c].fillna(-9).values, j[c + "_c"].fillna(-9).values, atol=1e-9), c


def test_ext_features_have_no_lookahead(m1):
    part = m1[(m1.time >= "2023-01-01") & (m1.time < "2024-03-01")]
    X = F.build(part)
    a = FE.build_ext(X, part)
    cut = part[part.time < "2024-02-01"]
    b = FE.build_ext(X[X.decision_time < "2024-02-01"], cut)
    j = a.merge(b, on="decision_time", suffixes=("", "_c"))
    j = j[j.decision_time < "2024-01-28"]
    for c in a.columns[1:]:
        assert np.allclose(j[c].fillna(-9).values, j[c + "_c"].fillna(-9).values, atol=1e-9), c


def test_agents_load_and_score(m1):
    recent = m1[m1.time >= m1.time.iloc[-1] - pd.Timedelta(days=420)]
    bull, bear = BullAgent(), BearAgent()
    X = bear.features(recent)
    for a in (bull, bear):
        s = a.score(recent)
        assert 0 <= s["prob"] <= 1 and isinstance(s["patterns"], list)
    assert "sniper_signal" in bear.score(recent, X)
    r = bear.approve_s5(recent, F.build(recent))
    assert r["approve"] in (True, False)
    t = recent.time.iloc[-1].normalize() - pd.Timedelta(hours=1)   # 23:00 UTC of the previous day
    r = bull.approve_s6(recent[recent.time <= t], t)
    assert 0 <= r["prob"] <= 1


def test_live_window_matches_full_history(m1):
    """A live bot only has ~14 months of bars; its scores must match the full-history scores closely."""
    bull = BullAgent()
    end = pd.Timestamp("2026-06-01")
    full = bull.features(m1[m1.time < end])
    live = bull.features(m1[(m1.time < end) & (m1.time >= end - pd.Timedelta(days=420))])
    a = bull.general.predict_proba(full.iloc[-500:][bull.cols])[:, 1]
    b = bull.general.predict_proba(live.iloc[-500:][bull.cols])[:, 1]
    assert np.median(np.abs(a - b)) < 0.01


def test_plugin_check_returns_verdicts(m1):
    from agents.plugin import GoldAgents
    recent = m1[m1.time >= m1.time.iloc[-1] - pd.Timedelta(days=420)]
    for mode in ("standard", "strict", "dual-strict"):
        ga = GoldAgents(mode=mode)
        s = ga.scores(recent)
        for d in ("BUY", "SELL"):
            v = ga.check(d, recent, s)
            assert v["approve"] in (True, False) and v["direction"] == d
            assert v["reason"].startswith(("APPROVED", "VETO"))
    # strict must never approve something standard vetoes
    std, st = GoldAgents("standard"), GoldAgents("strict")
    s = std.scores(recent); st._cal = std._cal
    if st.check("BUY", recent, s)["approve"]:
        assert std.check("BUY", recent, s)["approve"]
