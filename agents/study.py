"""Fast experiment harness for improving the agents (comparisons only; finalists get the full test)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

YEARS = list(range(2019, 2027))


def gbm(**kw):
    p = dict(max_leaf_nodes=31, learning_rate=0.05, max_iter=300, min_samples_leaf=300, l2_regularization=1.0,
             max_features=0.5, random_state=7)
    p.update(kw)
    return HistGradientBoostingClassifier(**p)


def fast_wf(X, y, r, cols, years=YEARS, stride=2, weight_halflife_years=None, model=None, embargo_days=2):
    """Yearly expanding walk-forward. Training rows sub-sampled every `stride` decision points
    (neighbouring 15-min points carry nearly the same information). Returns OOS frame."""
    t = X["decision_time"]
    yr = t.dt.year.values
    ok = np.isfinite(r) & np.isfinite(y.astype(float))
    idx = np.arange(len(X))
    rows = []
    for Y in years:
        cut = (pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=embargo_days)).to_datetime64()
        tr = ok & (t.values < cut) & (idx % stride == 0)
        te = ok & (yr == Y)
        w = None
        if weight_halflife_years:
            age = (cut - t.values[tr]) / np.timedelta64(365, "D")
            w = 0.5 ** (age / weight_halflife_years)
        m = (model() if model else gbm())
        m.fit(X.loc[tr, cols], y[tr], sample_weight=w)
        p_tr = m.predict_proba(X.loc[tr, cols])[:, 1]
        p = m.predict_proba(X.loc[te, cols])[:, 1]
        rows.append(pd.DataFrame({"i": idx[te], "year": Y, "prob": p, "y": y[te], "r": r[te],
                                  "thr95": np.quantile(p_tr, 0.95), "thr90": np.quantile(p_tr, 0.90)}))
    return pd.concat(rows, ignore_index=True)


def score(wf: pd.DataFrame) -> dict:
    aucs = [roc_auc_score(g["y"], g["prob"]) for _, g in wf.groupby("year")]
    # top 5% BY TEST-YEAR RANK (removes threshold drift so experiments compare ranking quality)
    q = wf.groupby("year")["prob"].transform(lambda s: s.rank(pct=True))
    top = wf[q >= 0.95]
    yr_r = top.groupby("year")["r"].mean()
    return dict(auc=float(np.mean(aucs)), auc_min=float(np.min(aucs)), base_win=float(wf["y"].mean()),
                top5_win=float(top["y"].mean()), top5_R=float(top["r"].mean()),
                years_pos=int((yr_r > 0).sum()), n_years=int(len(yr_r)))


def line(name, s, base=None):
    d = ""
    if base:
        d = f"  ΔAUC {s['auc'] - base['auc']:+.4f}  ΔR {s['top5_R'] - base['top5_R']:+.3f}"
    return (f"{name:38s} AUC {s['auc']:.4f} (min {s['auc_min']:.3f}) | top5% win {s['top5_win']:.3f} "
            f"R/trade {s['top5_R']:+.3f} | yrs+ {s['years_pos']}/{s['n_years']}{d}")


def timed(f, *a, **k):
    t = time.time()
    out = f(*a, **k)
    return out, time.time() - t
