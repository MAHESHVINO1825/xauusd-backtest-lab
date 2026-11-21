"""Continuous training pipeline — run weekly.

    python -m agents.update_pipeline            # fetch new data, retrain, test, write report
    python -m agents.update_pipeline --no-fetch # retrain on the data already present

Steps
 1. Fetch new XAUUSD tick weeks (Dukascopy, true UTC) from GitHub `Touhid-HZ/xauusd-tick-data-<year>`
    for everything after the last bar in data/, convert to M1 bid bars (+ real spread), append.
 2. Retrain all agent models on the full history (agents.build_models).
 3. Run the test suite. If any test fails, the old models are restored and the run is marked FAILED.
 4. Score the most recent 8 weeks with the new models (live monitor) and write research/out/weekly_report.md.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "agents", "models")
REPORT = os.path.join(ROOT, "research", "out", "weekly_report.md")
RAW = "https://raw.githubusercontent.com/Touhid-HZ/xauusd-tick-data-{y}/main/data/tick/XAUUSD_{y}_w{w:02d}.csv.gz"


def last_bar() -> pd.Timestamp:
    files = sorted(f for f in os.listdir(DATA) if f.startswith("XAUUSD_M1_") and f.endswith(".parquet"))
    return pd.read_parquet(os.path.join(DATA, files[-1]), columns=["time"])["time"].max()


def week_of(ts: pd.Timestamp) -> int:
    # Touhid weeks: w01 = Jan 1-7, then every 7 days
    return (ts.dayofyear - 1) // 7 + 1


def ticks_to_m1(raw: bytes) -> pd.DataFrame:
    t = pd.read_csv(io.BytesIO(raw), compression="gzip", usecols=["timestamp", "ask", "bid"])
    t["timestamp"] = pd.to_datetime(t["timestamp"])
    t = t.set_index("timestamp")
    o = t["bid"].resample("1min").ohlc()
    o["volume"] = t["bid"].resample("1min").count()
    o["spread"] = (t["ask"] - t["bid"]).resample("1min").median()
    o = o.dropna(subset=["open"])
    o.index.name = "time"
    return o.reset_index()


def fetch_new() -> int:
    lb = last_bar()
    now = pd.Timestamp.utcnow().tz_localize(None)
    added = 0
    for y in range(lb.year, now.year + 1):
        w0 = week_of(lb) if y == lb.year else 1
        w1 = week_of(now) if y == now.year else 53
        new = []
        for w in range(w0, w1 + 1):
            try:
                raw = urllib.request.urlopen(RAW.format(y=y, w=w), timeout=120).read()
            except Exception as e:  # noqa: BLE001  (week not published yet / repo for a new year missing)
                print(f"  {y} w{w:02d}: not available ({e.__class__.__name__})")
                continue
            m = ticks_to_m1(raw)
            new.append(m[m["time"] > lb])
            print(f"  {y} w{w:02d}: {len(new[-1])} new bars")
        if not new:
            continue
        path = os.path.join(DATA, f"XAUUSD_M1_{y}.parquet")
        old = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()
        df = pd.concat([old] + new).drop_duplicates("time").sort_values("time")
        df = df[df["time"].dt.year == y]
        df.to_parquet(path, index=False)
        added += sum(len(x) for x in new)
    return added


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, env={**os.environ, "PYTHONPATH": ROOT}).returncode


def monitor() -> dict:
    """Score the last 8 weeks with the freshly trained agents (in-sample for the models — a health check,
    not a performance claim)."""
    sys.path.insert(0, ROOT)
    from core import data as D  # noqa: E402
    from agents.agent import BearAgent, BullAgent  # noqa: E402
    from agents import features as F  # noqa: E402
    m1 = D.load_m1()
    recent = m1[m1["time"] >= m1["time"].max() - pd.Timedelta(days=420)]
    bull, bear = BullAgent(), BearAgent()
    Xb = bull.features(recent)
    Xr = Xb
    last = Xb["decision_time"] >= Xb["decision_time"].max() - pd.Timedelta(weeks=8)
    pb = bull.general.predict_proba(Xb.loc[last, bull.cols])[:, 1]
    pr = bear.general.predict_proba(Xb.loc[last, bear.cols])[:, 1]
    return {"bull_top5_signals_8w": int((pb >= bull.thr["0.95"]).sum()), "bear_top5_signals_8w": int((pr >= bear.thr["0.95"]).sum()),
            "bull_now": bull.score(recent, Xb), "bear_now": bear.score(recent, Xr)}


def main():
    t0 = time.time()
    fetch = "--no-fetch" not in sys.argv
    status = "OK"
    added = fetch_new() if fetch else 0
    backup = MODELS + "_backup"
    if os.path.exists(backup):
        shutil.rmtree(backup)
    shutil.copytree(MODELS, backup)
    if run([sys.executable, "-m", "agents.build_models"]) != 0:
        status = "FAILED (training)"
    elif run([sys.executable, "-m", "pytest", "-q", "-p", "no:warnings", "tests"]) != 0:
        status = "FAILED (tests)"
    if status != "OK":
        shutil.rmtree(MODELS)
        shutil.copytree(backup, MODELS)
    shutil.rmtree(backup, ignore_errors=True)
    with open(os.path.join(MODELS, "meta.json")) as f:
        meta = json.load(f)
    mon = monitor() if status == "OK" else {}
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        f.write(f"# Weekly agent update — {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}\n\n")
        f.write(f"* Status: **{status}**\n* New M1 bars added: {added:,}\n* Data: {meta['data_first']} → {meta['data_last']}\n")
        f.write(f"* Models trained: {meta['trained_at']}\n* Runtime: {time.time() - t0:.0f}s\n\n")
        if mon:
            f.write("## Live monitor (last 8 weeks)\n")
            f.write(f"* Bull top-5% signals: {mon['bull_top5_signals_8w']}\n* Bear top-5% signals: {mon['bear_top5_signals_8w']}\n")
            f.write(f"* Bull now: `{json.dumps(mon['bull_now'], default=str)}`\n* Bear now: `{json.dumps(mon['bear_now'], default=str)}`\n")
    print(open(REPORT).read())
    return 0 if status == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
