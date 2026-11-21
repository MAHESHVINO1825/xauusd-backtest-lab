"""Apply the M30 + agents upgrade to a bot file. Credentials are left untouched."""
import sys, re
src, dst, magic_base = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(src, encoding="utf-8").read()
orig = s
def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n >= 1, f"pattern not found: {old[:70]!r}"
    s = s.replace(old, new) if count == 0 else s.replace(old, new, count)

# ---------------------------------------------------------------- header
rep('"""\nXAUUSD M5 AGENT', '''"""
XAUUSD AGENT v3 - TIMEFRAME SELECTABLE (M5 / M15 / M30) + AGENT BULL / AGENT BEAR APPROVAL

WHAT IS NEW IN v3 (2026-11-21)
  1. TIMEFRAME setting: "M30" (default, best backtest), "M15" or "M5". Hold time, history, warm-up, flat-by
     rule and Time Machine all scale with it. Backtest 2013-2026 of this engine + agents (dual-strict):
       M5  : 208 trades/yr, +89 R,  PF 1.08, 10/14 years profitable
       M15 : 245 trades/yr, +126 R, PF 1.11, 11/14 years
       M30 : 183 trades/yr, +170 R, PF 1.22, 11/14 years, t 4.0 (still PF 1.20 with spread 0.30)
  2. Agent Bull / Agent Bear approval (from the xauusd-backtest-lab repo): every order needs the agents' OK.
     BUY needs Bull in its top 35% of the last 90 days AND Bear below its median (dual-strict); SELL mirror.
  3. Separate MAGIC number and database per timeframe, so an M5 copy and an M30 copy can run side by side.
  Needs:  pip install scikit-learn joblib   and LAB_PATH pointing at your xauusd-backtest-lab folder.

------------------------------------------------------------------------------------------------
XAUUSD M5 AGENT''')

# ---------------------------------------------------------------- settings
rep('SYMBOL        = os.getenv("XAU_SYMBOL", "XAUUSD")   # exact name in Market Watch',
    '''SYMBOL        = os.getenv("XAU_SYMBOL", "XAUUSD")   # exact name in Market Watch
TIMEFRAME     = os.getenv("XAU_TF", "M30")         # "M30" (recommended) | "M15" | "M5"
TF_MIN        = {"M5": 5, "M15": 15, "M30": 30}[TIMEFRAME]
TF_K          = TF_MIN // 5                         # how many M5 candles fit in one candle of TIMEFRAME

# ---- Agent Bull / Agent Bear approval (xauusd-backtest-lab)
USE_AGENTS      = True                              # every order must be approved by the agents
AGENTS_REQUIRED = True                              # if the agents cannot run: no orders (alerts only)
AGENT_MODE      = "dual-strict"                     # "dual-strict" (backtested best) | "strict" | "standard"
LAB_PATH        = os.getenv("XAU_LAB", r"C:\\Users\\mahes\\Documents\\GitHub\\xauusd-backtest-lab")
AGENT_M1_DAYS   = 430                               # M1 history the agents need (about 14 months)''')
rep('HISTORY_BARS  = 150_000    # M5 bars pulled from MT5 (about 2 years); the model trains on these',
    'HISTORY_BARS  = {"M5": 150_000, "M15": 100_000, "M30": 60_000}[TIMEFRAME]   # candles pulled from MT5 for training')
rep('MIN_BARS      = 20_000     # refuse to run on less history than this',
    'MIN_BARS      = {"M5": 20_000, "M15": 8_000, "M30": 5_000}[TIMEFRAME]      # refuse to run on less history than this')
rep('MAX_BARS   = 144     # flat within 12 h or before the daily break',
    'MAX_BARS   = 144 // TF_K   # flat within 12 h or before the daily break (in candles of TIMEFRAME)')
m = re.search(r'MAGIC           = (\d+)', s)
rep(m.group(0), f'MAGIC           = {magic_base} + (0 if TIMEFRAME == "M5" else TF_MIN)   # different per timeframe')
rep('FRESH_BARS = 6               # /scan shows setups from the last 30 minutes (live trading uses the last candle only)',
    'FRESH_BARS = max(1, 6 // TF_K)   # /scan shows setups from the last 30 minutes (live trading uses the last candle only)')
rep('DB_PATH   = os.getenv("XAU_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "xauusd_agent.db"))',
    'DB_PATH   = os.getenv("XAU_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)),\n                    "xauusd_agent.db" if TIMEFRAME == "M5" else f"xauusd_agent_{TIMEFRAME}.db"))')
rep('WARMUP = 3000', 'WARMUP = max(600, 3000 // TF_K)')
rep('MIN_FULL = 250              # bars a session needs to count as "full"', 'MIN_FULL = 250 // TF_K       # candles a session needs to count as "full"')

# ---------------------------------------------------------------- outcomes / time machine / scan scaling
rep('horizon = np.minimum(max_bars, SLOTS - 1 - d.slot.to_numpy()[bar]).astype(np.int64)',
    'horizon = np.minimum(max_bars, (SLOTS - 1 - d.slot.to_numpy()[bar]) // TF_K).astype(np.int64)')
rep('''    slot0 = int(d.slot.iat[i])
    H = int(min(max_bars, SLOTS - 1 - slot0))
    if H < 6:''', '''    slot0 = int(d.slot.iat[i])
    H = int(min(max_bars, (SLOTS - 1 - slot0) // TF_K))
    window = window * TF_K
    if H * TF_K < 6:''')
rep('        s = slot0 + j\n', '        s = slot0 + j * TF_K\n')
rep('cont = nxt_ok[src] & (np.abs(slots[np.minimum(src + 1, len(slots) - 1)] - s) <= window + block)',
    'cont = nxt_ok[src] & (np.abs(slots[np.minimum(src + 1, len(slots) - 1)] - s) <= window + block * TF_K)')
rep('flat = d.time.iat[i] + pd.Timedelta(minutes=5 * (1 + min(MAX_BARS, SLOTS - 1 - int(d.slot.iat[i]))))',
    'flat = d.time.iat[i] + pd.Timedelta(minutes=TF_MIN * (1 + min(MAX_BARS, (SLOTS - 1 - int(d.slot.iat[i])) // TF_K)))')
rep('bar_close = pd.Timestamp(res["time"]) + pd.Timedelta(minutes=5)', 'bar_close = pd.Timestamp(res["time"]) + pd.Timedelta(minutes=TF_MIN)')
rep('head = f"{dot} <b>{side} XAUUSD</b> · M5 · "', 'head = f"{dot} <b>{side} XAUUSD</b> · {TIMEFRAME} · "')
rep('L = [f"XAUUSD M5 scan at {res[\'time\']} UTC"]', 'L = [f"XAUUSD {TIMEFRAME} scan at {res[\'time\']} UTC"]')
rep('flat within {MAX_BARS // 12} h', 'flat within {MAX_BARS * TF_MIN // 60} h')
rep('"ema_pullback": f"M5 {\'up\' if up else \'down\'}trend', '"ema_pullback": f"{TIMEFRAME} {\'up\' if up else \'down\'}trend')
rep('L.append(("=" if m5 == "mixed" else "+" if m5 == word else "-", f"M5 trend {m5}"))',
    'L.append(("=" if m5 == "mixed" else "+" if m5 == word else "-", f"{TIMEFRAME} trend {m5}"))')

# ---------------------------------------------------------------- MT5 timeframe
rep('r = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5, pos, min(50_000, n - pos + 1))',
    'r = mt5.copy_rates_from_pos(self.symbol, getattr(mt5, f"TIMEFRAME_{TIMEFRAME}"), pos, min(50_000, n - pos + 1))')
rep('raise RuntimeError(f"MT5 returned only {len(df):,} M5 bars;', 'raise RuntimeError(f"MT5 returned only {len(df):,} {TIMEFRAME} bars;')
rep(f'the {{self.symbol}} M5 chart back so it downloads more history.', f'the {{self.symbol}} {{TIMEFRAME}} chart back so it downloads more history.')
rep('r = self.mt5.copy_rates_from_pos(self.symbol, self.mt5.TIMEFRAME_M5, 1, 1)',
    'r = self.mt5.copy_rates_from_pos(self.symbol, getattr(self.mt5, f"TIMEFRAME_{TIMEFRAME}"), 1, 1)')

# ---------------------------------------------------------------- M1 feed for the agents + agent gate
rep('''    def equity(self):
        info = self.mt5.account_info()
        return float(info.equity) if info else 0.0
''', '''    def equity(self):
        info = self.mt5.account_info()
        return float(info.equity) if info else 0.0

    def m1_utc(self, days=AGENT_M1_DAYS):
        """M1 candles in UTC for Agent Bull / Agent Bear (columns time, open, high, low, close, volume)."""
        mt5, frames, pos, n = self.mt5, [], 1, int(days * 1440)
        while pos <= n:
            r = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1, pos, min(50_000, n - pos + 1))
            if r is None or len(r) == 0:
                break
            frames.append(pd.DataFrame(r))
            pos += len(r)
            if len(r) < 50_000:
                break
        if not frames:
            raise RuntimeError(f"no M1 bars for the agents: {mt5.last_error()}")
        df = pd.concat(frames).drop_duplicates("time").sort_values("time")
        df["time"] = self.to_utc(pd.to_datetime(df["time"], unit="s"))
        df = df.dropna(subset=["time"]).rename(columns={"tick_volume": "volume"})
        return df[["time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


import warnings as _w
_w.filterwarnings("ignore", module=r"agents(\..*)?$")

class AgentGate:
    """Agent Bull / Agent Bear approval layer (xauusd-backtest-lab, agents/plugin.py).
    One scoring per closed candle; BUY and SELL verdicts come from the same scores."""

    def __init__(self, get_m1, mode=AGENT_MODE, lab=LAB_PATH):
        self.get_m1, self.mode, self.ga, self.error, self.cache = get_m1, mode, None, None, {}
        try:
            if lab not in sys.path:
                sys.path.insert(0, lab)
            from agents.plugin import GoldAgents
            self.ga = GoldAgents(mode=mode)
        except Exception as e:                      # missing folder / packages: report, never crash the bot
            self.error = f"{type(e).__name__}: {e}"

    @property
    def ready(self):
        return self.ga is not None

    def verdicts(self, bar_time):
        """{'BUY': verdict, 'SELL': verdict} for the candle that closed at bar_time, or None + reason."""
        if not self.ready:
            return None, f"agents not loaded ({self.error})"
        key = str(bar_time)
        if key in self.cache:
            return self.cache[key], None
        try:
            m1 = self.get_m1()
            sc = self.ga.scores(m1)
            out = {d: self.ga.check(d, m1, sc) for d in ("BUY", "SELL")}
        except Exception as e:
            return None, f"agent scoring failed ({type(e).__name__}: {e})"
        self.cache = {key: out}
        return out, None
''')

# Telegram card: show the agents' verdict
rep('''    h = s["history_1y"]''', '''    ag = (res.get("agents") or {}).get(side)
    if ag:
        L.append(("🤖 Agents: <b>APPROVED</b> · " if ag["approve"] else "🤖 Agents: <b>VETO</b> · ") + e(ag["reason"]))
    elif res.get("agents_error"):
        L.append("🤖 Agents: unavailable · " + e(res["agents_error"]))
    h = s["history_1y"]''')

# Scanner: hold the gate, ask it before any order
rep('''    def __init__(self, feed, cfg, send, mem=None, trader=None):
        self.feed, self.cfg, self.send, self.mem, self.trader = feed, cfg, send, mem, trader''',
    '''    def __init__(self, feed, cfg, send, mem=None, trader=None, agents=None):
        self.feed, self.cfg, self.send, self.mem, self.trader, self.agents = feed, cfg, send, mem, trader, agents''')
rep('''        notes, traded = {}, False
        for s in res["setups"]:                              # best score first''', '''        notes, traded = {}, False
        if USE_AGENTS and self.agents is not None and any(
                s["score"] > min_score and s["risk_check"] == "PASS" for s in res["setups"]):
            res["agents"], res["agents_error"] = self.agents.verdicts(bar_time)
        for s in res["setups"]:                              # best score first''')
rep('''            elif self.paused:
                decision = "SKIP: paused"''', '''            elif USE_AGENTS and self.agents is not None and res.get("agents") is None and AGENTS_REQUIRED:
                decision = "SKIP: " + (res.get("agents_error") or "agents unavailable")
            elif USE_AGENTS and res.get("agents") and not res["agents"][s["side"]]["approve"]:
                decision = "SKIP: agents veto - " + res["agents"][s["side"]]["reason"]
            elif self.paused:
                decision = "SKIP: paused"''')
# /scan should also show the agents
rep('''        if force:
''', '''        if force:
            if USE_AGENTS and self.agents is not None and res["setups"]:
                res["agents"], res["agents_error"] = self.agents.verdicts(bar_time)
''')
# skip alerts for agent vetoes too (keep Telegram quiet like other SKIPs) — already handled by startswith("SKIP")

# run_live: build the gate, report status, /agents command
rep('''    sc = Scanner(feed, cfg, send, mem, trader)''', '''    agents = AgentGate(lambda: feed.m1_utc()) if USE_AGENTS else None
    sc = Scanner(feed, cfg, send, mem, trader, agents)''')
rep('''+ f"\\n{dash}\\n/help for commands" + note)''', '''+ f"\\nTimeframe: {TIMEFRAME} · magic {MAGIC} · memory {os.path.basename(DB_PATH)}"
         + ("" if not USE_AGENTS else f"\\nAgents ({AGENT_MODE}): " + ("loaded" if agents.ready else f"NOT LOADED - {agents.error}"
            + (" - no orders until fixed" if AGENTS_REQUIRED else "")))
         + f"\\n{dash}\\n/help for commands" + note)''')
rep('''    @bot.message_handler(commands=["status"])''', '''    @bot.message_handler(commands=["agents"])
    @only_me
    def _agents(m):
        if not USE_AGENTS or agents is None:
            return send("Agents are switched off (USE_AGENTS = False).")
        with MT5_LOCK:
            v, err = agents.verdicts(f"manual {time.time()}")
        send(err or "\\n".join(f"{d}: {v[d]['reason']}" for d in ("BUY", "SELL")))

    @bot.message_handler(commands=["status"])''')
rep('"/validate - walk-forward test (takes a few minutes)\\n', '"/validate - walk-forward test (takes a few minutes)\\n/agents - Agent Bull / Bear verdict now\\n')
rep('send(f"Model trained on {primed[\'trained_on\']:,} past setups ({sc.history}). Waiting for the next M5 candle.")',
    'send(f"Model trained on {primed[\'trained_on\']:,} past setups ({sc.history}). Waiting for the next {TIMEFRAME} candle.")')
open(dst, "w", encoding="utf-8").write(s)
print("patched", dst, "| changed chars:", len(s) - len(orig))
