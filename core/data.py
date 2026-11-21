"""Data loading: M1 gold bars -> any timeframe, plus the spread model."""
from __future__ import annotations

import glob
import os
from functools import lru_cache

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

TIMEFRAMES = {"M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240}

# Fixed research split. Rank on TEST only; HOLDOUT is the final one-shot check.
TRAIN_END = pd.Timestamp("2023-01-01")
TEST_END = pd.Timestamp("2025-01-01")
PERIODS = ("train", "test", "holdout")


@lru_cache(maxsize=1)
def load_m1() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(DATA_DIR, "XAUUSD_M1_*.parquet")))
    if not files:
        raise FileNotFoundError(f"No XAUUSD_M1_*.parquet files in {DATA_DIR}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["time"] = pd.to_datetime(df["time"]).astype("datetime64[ns]")
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    df = df[(df["high"] >= df["low"]) & (df["close"] > 0)]
    return df.reset_index(drop=True)


@lru_cache(maxsize=1)
def spread_shape() -> np.ndarray:
    """Hour-of-day spread multiplier (median = 1.0), measured from Dukascopy 2026 ticks."""
    path = os.path.join(DATA_DIR, "spread_hour_shape.csv")
    if os.path.exists(path):
        s = pd.read_csv(path, index_col=0).iloc[:, 0]
        arr = np.ones(24)
        for h, v in s.items():
            arr[int(h)] = float(v)
        return arr
    return np.ones(24)


def m1_arrays(base_spread: float):
    """Arrays the engine needs at M1 resolution."""
    m1 = load_m1()
    t = m1["time"].values
    hours = m1["time"].dt.hour.values
    spread = base_spread * spread_shape()[hours]
    minutes = (t.astype("datetime64[m]").astype(np.int64)).astype(np.int64)
    return {
        "time": t,
        "minutes": minutes,
        "open": m1["open"].values.astype(np.float64),
        "high": m1["high"].values.astype(np.float64),
        "low": m1["low"].values.astype(np.float64),
        "close": m1["close"].values.astype(np.float64),
        "spread": spread.astype(np.float64),
    }


@lru_cache(maxsize=8)
def bars(tf: str) -> pd.DataFrame:
    """OHLCV bars for a timeframe, labelled by bar START time (UTC)."""
    m1 = load_m1().set_index("time")
    if tf == "M1":
        out = m1.copy()
    else:
        rule = f"{TIMEFRAMES[tf]}min"
        out = m1.resample(rule, label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna(subset=["open"])
    out = out.reset_index()
    out["end"] = out["time"] + pd.Timedelta(minutes=TIMEFRAMES.get(tf, 1))
    return out


def entry_index(tf_bars: pd.DataFrame) -> np.ndarray:
    """M1 index where a trade signalled at a TF bar's close is filled (next M1 open)."""
    t = load_m1()["time"].values
    return np.searchsorted(t, tf_bars["end"].values, side="left").astype(np.int64)


def period_of(times) -> np.ndarray:
    t = pd.to_datetime(pd.Series(times))
    out = np.where(t < TRAIN_END, 0, np.where(t < TEST_END, 1, 2))
    return out.astype(np.int64)


def data_health() -> dict:
    m1 = load_m1()
    t = m1["time"]
    gaps = t.diff().dt.total_seconds().div(60)
    # weekday gaps longer than 30 min that are not the weekend close
    wk = t.dt.weekday
    big = gaps[(gaps > 30) & ~((wk == 6) | (wk == 0) & (gaps > 1000))]
    by_year = m1.groupby(t.dt.year).agg(bars=("close", "size"), low=("low", "min"), high=("high", "max"))
    return {
        "first": t.iloc[0], "last": t.iloc[-1], "bars": len(m1),
        "by_year": by_year, "gaps": big.sort_values(ascending=False).head(15),
        "gap_times": t[big.sort_values(ascending=False).head(15).index],
    }
