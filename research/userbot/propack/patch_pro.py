"""Add the PRO PACK to a bot file:  python patch_pro.py <src> <dst> <BOT1|BOT2>"""
import re
import sys

src, dst, which = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(src, encoding="utf-8").read()
pack = open("propack.py", encoding="utf-8").read()

PROFILES = {
    "BOT1": dict(id="BOT1-INTRADAY", title="Bot 1 · Intraday Momentum", grades='("A", "B", "C")',
                 strategies='("macd_h1", "ema_h1")'),
    "BOT2": dict(id="BOT2-SWING", title="Bot 2 · Swing + Agent Sniper", grades='("A",)',
                 strategies='("donchian_h4", "bull_top2", "bear_sniper")'),
}
P = PROFILES[which]


def rep(old, new, count=1):
    global s
    if old not in s:
        raise SystemExit(f"anchor not found: {old[:70]!r}")
    s = s.replace(old, new, count)


settings = f'''
# ---- PRO PACK (2026-11-21): several strategies, grades, several trades at once, shadow trades, shared dashboard
BOT_ID        = "{P['id']}"              # name on Telegram and on the shared dashboard
BOT_TITLE     = "{P['title']}"
STRATEGIES    = {P['strategies']}   # lab strategies this bot trades next to the {{TIMEFRAME}} setup engine
ENGINE_REAL_GRADES = {P['grades']}          # engine grades that place REAL orders (others = shadow only). Dashboard buttons change it
#   grade A = agents dual-strict (best), B = strict, C = standard, D = agents disagree / model score <= 0
MAX_OPEN      = 4                # open trades at the same time (all strategies of this bot)
GRADE_SLOTS   = {{"A": 4, "B": 2, "C": 1}}   # per grade: B and C can never fill the room an A trade needs
MAX_NEW_PER_CANDLE = 2           # new real trades per closed candle
DAILY_STOP_USD = 140.0           # no new trades once today's closed trades lost this much
HUB_DIR       = os.getenv("XAU_HUB", os.path.join(os.path.expanduser("~"), "xau_bots_hub"))   # shared by both bots
DASH_PIN      = "2580"           # dashboard password (phone / browser). Change it.
DASH_PUBLIC   = True             # True: phones on the same Wi-Fi can open http://<laptop-ip>:{{DASH_PORT}}
TUNNEL        = True             # True + cloudflared installed: a public https link for the phone (sent on Telegram)
WEEKLY_REPORT_UTC = (5, 1)       # weekly report on Saturday from 01:00 UTC (after the Friday close)
'''
LINE = "AGENT_M1_DAYS   = 430                               # M1 history the agents need (about 14 months)\n"
rep(LINE, LINE + settings.replace("{TIMEFRAME}", "TIMEFRAME"))
s = re.sub(r"MAX_TRADES_DAY  = \d+ ", "MAX_TRADES_DAY  = 15 ", s, count=1)

# pack goes after the Scanner class (it subclasses Trader and Scanner)
rep("\ndef run_validation(", "\n" + pack + "\n\ndef run_validation(")

# ---- run_live wiring
rep("    trader = Trader(feed, mem, send)\n", "    trader = MultiTrader(feed, mem, send)\n")
rep("    sc = Scanner(feed, cfg, send, mem, trader, agents)\n", "    sc = ProScanner(feed, cfg, send, mem, trader, agents)\n"
    "    hub = Hub(send)\n\n    def send_doc(data, name):\n        try:\n            bio = io.BytesIO(data)\n            bio.name = name\n"
    "            bot.send_document(chat, bio)\n        except Exception as e:\n            print('telegram document failed:', e)\n")
m = re.search(r"    try:\n        start_dashboard\(.*?\n    except OSError as e:\n        dash = [^\n]*\n", s, re.S)
if not m:
    raise SystemExit("dashboard block not found")
s = s[:m.start()] + ("    hub.ensure()\n    pub, lan = dash_links()\n"
                     "    dash = f\"Dashboard (both bots): {lan}  PIN {DASH_PIN}\" + (f\"\\nPhone anywhere: {pub}\" if pub else \"\")\n") + s[m.end():]
rep("    trader.adopt()\n", "    trader.adopt()\n"
    "    send(f\"[{BOT_ID}] {BOT_TITLE}\\nStrategies: \" + \", \".join(strat_name(k) for k in ('engine',) + tuple(STRATEGIES))\n"
    "         + f\"\\nReal orders: engine grades {', '.join(trader.real_grades)} + lab strategies when their agent mode approves\"\n"
    "         + f\"\\nMax {MAX_OPEN} open (A {GRADE_SLOTS['A']}, B {GRADE_SLOTS['B']}, C {GRADE_SLOTS['C']}), {MAX_TRADES_DAY}/day, daily stop -${DAILY_STOP_USD:g}\"\n"
    "         + \"\\nEvery signal is also recorded as a shadow trade. Weekly report: Saturday after the close.\")\n")

# /close -> all trades; /dash also sends the links; new commands
m = re.search(r"    def _close\(m\):\n(.*?)\n\n", s, re.S)
s = s[:m.start()] + ("    def _close(m):\n        arg = (m.text.split() + [''])[1]\n        with MT5_LOCK:\n"
                     "            if arg.isdigit():\n                ok = trader.close_ticket(int(arg))\n"
                     "                return send(f'Closing #{arg}.' if ok else f'#{arg} is not an open trade.')\n"
                     "            n, strays = trader.close_all('MANUAL')\n"
                     "        send(f'Closing {n} open trade(s).' + (f' Untracked closed: {strays}' if strays else '') if n or strays else 'No open trade.')\n\n") + s[m.end():]
rep("caption=\"XAUUSD M5 agent dashboard\")", "caption=f\"{BOT_ID} dashboard\")\n        pub, lan = dash_links()\n"
    "        send(f\"Live dashboard (both bots, buttons): {lan}  PIN {DASH_PIN}\" + (f\"\\nPhone anywhere: {pub}\" if pub else \"\\nFor phone access outside your Wi-Fi install cloudflared (see top of file)\"))")
rep("    threading.Thread(target=lambda: bot.infinity_polling(skip_pending=True), daemon=True).start()\n",
    """    @bot.message_handler(commands=["positions"])
    @only_me
    def _positions(m):
        with MT5_LOCK:
            send(positions_text(trader.live_views()))

    @bot.message_handler(commands=["report"])
    @only_me
    def _report(m):
        send_weekly(mem, send, manual=True, send_doc=send_doc)

    @bot.message_handler(commands=["shadow"])
    @only_me
    def _shadow(m):
        text, _, _ = weekly_report(mem, days=1, title=BOT_TITLE + " · last 24 h")
        send(text)

    @bot.message_handler(commands=["link"])
    @only_me
    def _link(m):
        pub, lan = dash_links()
        if pub:
            mk = telebot.types.InlineKeyboardMarkup()
            mk.add(telebot.types.InlineKeyboardButton("Open dashboard", url=pub))
            bot.send_message(chat, f"Dashboard (both bots). PIN {DASH_PIN}\\nSame Wi-Fi: {lan}", reply_markup=mk)
        else:
            send(f"Dashboard (both bots): {lan}  PIN {DASH_PIN}\\nPhone on the same Wi-Fi: open that link.\\n"
                 "Phone anywhere: install cloudflared on the laptop (winget install Cloudflare.cloudflared) and restart the bot.")

    @bot.message_handler(commands=["grades"])
    @only_me
    def _grades(m):
        arg = (m.text.split() + [""])[1].upper()
        if arg:
            trader.real_grades[:] = sorted({g for g in arg.split(",") if g in ("A", "B", "C")})
        send(f"Engine grades that place real orders: {', '.join(trader.real_grades) or 'none (shadow only)'}\\nUse /grades A,B,C or /grades A")

    threading.Thread(target=lambda: bot.infinity_polling(skip_pending=True), daemon=True).start()
""")
rep("\\n/pause /resume - alerts", "\\n/positions - all open trades\\n/close - close ALL (or /close <ticket>)\\n/report - weekly report now (+ CSV)\\n"
    "/shadow - shadow trades, last 24 h\\n/link - dashboard link for the phone\\n/grades A,B,C - which engine grades trade for real\\n/pause /resume - alerts")
rep("    while sc.running:\n", """    last_pub = 0.0
    while sc.running:
        try:
            hub.ensure()
            if time.time() - last_pub >= 5:
                last_pub = time.time()
                hub_write(f"{BOT_ID}.json", pro_snapshot(mem, sc, trader, agents))
                run_hub_commands(hub_take_commands(BOT_ID), sc, trader, send, mem)
                if weekly_due(mem):
                    send_weekly(mem, send, send_doc=send_doc)
        except Exception as e:
            print("hub error:", repr(e))
""")
rep("        bot.stop_polling()\n", "        bot.stop_polling()\n        hub.stop()\n")

# header note
rep('"""\nXAUUSD AGENT v3', f'"""\n{P["id"]} - {P["title"]}\n'
    "  PRO PACK: engine grades A/B/C, extra lab strategies, up to 4 trades at once with grade slots,\n"
    "  every signal recorded as a SHADOW trade, weekly Telegram report (Saturday), one dashboard for both bots.\n"
    "  Dashboard: starts by itself with the first bot. Laptop: http://localhost:8765  Phone (same Wi-Fi): http://<laptop-ip>:8765\n"
    "  Phone anywhere: install cloudflared once ->  winget install Cloudflare.cloudflared  (the link arrives on Telegram, /link)\n"
    "  Windows asks once to allow Python on the network: allow 'Private networks'.\n\nXAUUSD AGENT v3")
open(dst, "w", encoding="utf-8").write(s)
print("ok", dst, len(s.splitlines()), "lines")
