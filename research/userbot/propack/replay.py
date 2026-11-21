import os, sys, time, json, warnings, importlib
warnings.filterwarnings("ignore")
os.environ.setdefault("XAU_DB", os.environ.get("RPDB","/tmp/rp/bot.db")); os.environ.update(XAU_LAB="/home/claude/lab", XAU_DB=os.environ.get("RPDB","/tmp/rp/bot.db"), XAU_HUB="/tmp/rp/hub")
os.makedirs("/tmp/rp", exist_ok=True)
for f in (os.environ.get("RPDB","/tmp/rp/bot.db"),):
    if os.path.exists(f): os.remove(f)
sys.path.insert(0, "/home/claude/botsv3"); sys.path.insert(0, "/home/claude/botsv3/test"); sys.path.insert(0, "/home/claude/lab")
import pandas as pd, numpy as np
modname = sys.argv[1]; start, end = sys.argv[2], sys.argv[3]
sys.argv = ["x"]
B = importlib.import_module(modname)
from fake_mt5 import FakeMT5
from core import data as D
m1all = D.load_m1(); m1all = m1all[(m1all.time >= "2025-06-01")].reset_index(drop=True)
m30 = pd.read_parquet("/home/claude/botsv3/m30_test.parquet")
clock = {"t": pd.Timestamp(start)}
B.utc_now = lambda: clock["t"].to_pydatetime()
mt5 = FakeMT5(B.ACCOUNT_ID)
class Feed:
    def __init__(s): s.mt5, s.symbol, s.at = mt5, B.SYMBOL, None
    def bars(s, n): return m30[m30.time <= s.at].tail(n).reset_index(drop=True)
    def balance(s): return mt5.account_info().balance
    def equity(s): return mt5.account_info().equity
    def m1_utc(s, days=4): return m1all[(m1all.time < clock["t"]) & (m1all.time >= clock["t"] - pd.Timedelta(days=days))]
feed = Feed()
log = []
send = lambda t, html=False: log.append(t)
mem = B.Memory()
trader = B.MultiTrader(feed, mem, send)
agents = B.AgentGate(lambda: m1all[(m1all.time < clock["t"]) & (m1all.time >= clock["t"] - pd.Timedelta(days=430))])
sc = B.ProScanner(feed, B.CFG, send, mem, trader, agents)
bars = m30[(m30.time >= start) & (m30.time < end)].time.tolist()
feed.at = bars[0] - pd.Timedelta(minutes=30); clock["t"] = bars[0]
sc.scan_once(quiet=True)
mins = m1all.set_index("time")
t0 = time.time(); maxopen = 0
for bt in bars[1:]:
    close_t = bt + pd.Timedelta(minutes=30)
    for ts, r in mins.loc[clock["t"]:close_t - pd.Timedelta(minutes=1)].iterrows():
        mt5.set_bar(r.open, r.high, r.low, r.close)
        clock["t"] = ts + pd.Timedelta(minutes=1)
        trader.manage()
        maxopen = max(maxopen, len(mt5.pos))
    clock["t"] = close_t; feed.at = bt
    sc.scan_once()
print(f"replayed {len(bars)} bars in {time.time()-t0:.0f}s; max open positions at once: {maxopen}")
sh = pd.DataFrame(mem.q("SELECT * FROM shadow")); tr = pd.DataFrame(mem.q("SELECT * FROM trades"))
print("shadow signals:", len(sh), "| finished:", sh.R.notna().sum(), "| per trading day:", round(len(sh)/max(1,sh.sday.nunique()),1))
print(sh.groupby(["strategy","grade"]).size().to_string())
print("real trades:", len(tr)); 
if len(tr): print(tr[["opened","strategy","setup","grade","side","entry","exit","R","pnl","reason"]].to_string())
print("decisions:"); print(sh.decision.str.slice(0,55).value_counts().head(15).to_string())
text, real, s2 = B.weekly_report(mem); print("\n" + text)
B.hub_write(f"{B.BOT_ID}.json", B.pro_snapshot(mem, sc, trader, agents))
print("\nTelegram messages:", len(log)); print("--- sample ---"); print("\n---\n".join(log[-4:])[:2500])
print("\n--- dashboard commands ---")
B.run_hub_commands([{"cmd": "grade", "arg": "C"}, {"cmd": "pause"}, {"cmd": "auto_off"}, {"cmd": "auto_on"}, {"cmd": "close_all"}, {"cmd": "resume"}], sc, trader, send, mem)
print("\n".join(log[-6:])); print("grades now:", trader.real_grades, "paused:", sc.paused)
