"""Train the final Agent Bull / Agent Bear models on ALL available history and save them.

    python -m agents.build_models          # ~6 minutes; re-run once a year (or after adding new data)

Saves to agents/models/:
    bull_general.joblib, bear_general.joblib   general opportunity scorers (162 features, triple-barrier labels)
    bull_s6_head.joblib                         approves / rejects S6 longs (16 context features)
    bear_s5_head.joblib                         approves / rejects S5 shorts (162 features)
    meta.json                                   thresholds, feature lists, training window, data hash
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from core import data as D  # noqa: E402
from research import harness as H  # noqa: E402
from research import strategies as S  # noqa: E402

from . import features as F  # noqa: E402
from . import features_ext as FE  # noqa: E402
from . import study as St  # noqa: E402
from . import heads as HD  # noqa: E402
from . import train as T  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    m1 = D.load_m1()
    X = F.build(m1)
    cols = F.feature_columns(X)
    L = T.make_labels(X)
    meta = {"trained_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
            "data_first": str(m1["time"].iloc[0]), "data_last": str(m1["time"].iloc[-1]),
            "data_rows": int(len(m1)),
            "data_hash": hashlib.md5(pd.util.hash_pandas_object(m1["close"].iloc[::997]).values.tobytes()).hexdigest(),
            "features_general": cols, "features_s6": HD.S6_FEATS,
            "label": {"tp_atr_h1": T.TP_ATR, "sl_atr_h1": T.SL_ATR, "horizon_min": T.HORIZON_MIN,
                      "spread": T.SPREAD, "slippage": T.SLIP}}
    # Bear general (v1): 162 features, 1.5R / 8h target
    m, thr = T.fit_final(X, L, "bear", cols)
    joblib.dump(m, os.path.join(OUT, "bear_general.joblib"), compress=3)
    meta["bear_general_thresholds"] = thr
    print("bear general trained", round(time.time() - t0), "s", flush=True)
    # Bull general (v2): 188 features (+ext), 2R / 16h target, small GBM, recency half-life 2 years
    E = FE.build_ext(X, m1)
    XE = pd.concat([X, E.drop(columns="decision_time")], axis=1)
    cols_v2 = cols + [c for c in E.columns if c.startswith("ext_")]
    arr = D.m1_arrays(T.SPREAD)
    w, r = T.triple_barrier(L["entry_m1"].values, L["atr_h1"].values, arr["open"], arr["high"], arr["low"], arr["close"],
                            arr["spread"], arr["minutes"], 1, 2.0, 1.0, 960)
    ok = np.isfinite(r)
    age = (XE["decision_time"].max() - XE["decision_time"]).dt.days.values / 365
    mv2 = St.gbm(max_leaf_nodes=15, min_samples_leaf=600, learning_rate=0.05, max_iter=300)
    mv2.fit(XE.loc[ok, cols_v2], w[ok], sample_weight=0.5 ** (age[ok] / 2))
    p = mv2.predict_proba(XE.loc[ok, cols_v2])[:, 1]
    joblib.dump(mv2, os.path.join(OUT, "bull_general.joblib"), compress=3)
    meta["bull_general_thresholds"] = {str(q): float(np.quantile(p, q)) for q in (0.9, 0.95, 0.99)}
    meta["features_bull_v2"] = cols_v2
    meta["bull_label"] = {"tp_atr_h1": 2.0, "sl_atr_h1": 1.0, "horizon_min": 960, "version": "v2"}
    meta["bear_label"] = {"tp_atr_h1": T.TP_ATR, "sl_atr_h1": T.SL_ATR, "horizon_min": T.HORIZON_MIN, "version": "v1"}
    print("bull v2 general trained", round(time.time() - t0), "s", flush=True)
    # S6 head (bull)
    tr = H.run_positions(S.s6_positions("23:00", "00:00"))
    tr["entry_time"] = pd.to_datetime(tr["entry_time"])
    cf = HD.context_features(m1, tr["entry_time"].values)
    y = (tr["ret"].values > 0).astype(int)
    s6 = HD.s6_model().fit(cf[HD.S6_FEATS], y)
    p = s6.predict_proba(cf[HD.S6_FEATS])[:, 1]
    joblib.dump(s6, os.path.join(OUT, "bull_s6_head.joblib"), compress=3)
    meta["s6_threshold"] = float(np.median(p))
    # S5 head (bear)
    tr = H.run_positions(S.s5_positions("15:30", "20:30"))
    tr["entry_time"] = pd.to_datetime(tr["entry_time"])
    mm = pd.merge_asof(tr.sort_values("entry_time"), X[["decision_time"] + cols].sort_values("decision_time"),
                       left_on="entry_time", right_on="decision_time", direction="backward",
                       tolerance=pd.Timedelta(hours=4)).dropna(subset=["decision_time"])
    s5 = HD.s5_model().fit(mm[cols], (mm["ret"].values > 0).astype(int))
    p = s5.predict_proba(mm[cols])[:, 1]
    joblib.dump(s5, os.path.join(OUT, "bear_s5_head.joblib"), compress=3)
    meta["s5_threshold"] = float(np.median(p))
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("saved to", OUT, "in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
