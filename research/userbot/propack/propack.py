# ============================================================================
# 6b. PRO PACK - grades, extra strategies, several trades at once, shadow trades,
#     weekly Telegram report, one shared dashboard for both bots (phone friendly)
# ============================================================================
import shutil
import socket
import subprocess
import csv as _csv

STRAT = {
    "engine":      dict(name=f"{TIMEFRAME} setup engine", off=0),
    "macd_h1":     dict(name="MACD cross H1", off=101, mode="dual-strict", sl_atr=1.5, tp_r=2.0, hold_min=1440),
    "ema_h1":      dict(name="EMA 20/50 cross H1", off=102, mode="strict", sl_atr=1.5, tp_r=2.0, hold_min=1440),
    "donchian_h4": dict(name="Donchian-20 breakout H4", off=103, mode="strict", sl_atr=1.5, tp_r=2.0, hold_min=1440),
    "bull_top2":   dict(name="Agent Bull top 2%", off=104, sl_atr=1.0, tp_r=2.0, hold_min=960),
    "bear_sniper": dict(name="Agent Bear sniper", off=105, sl_atr=1.0, tp_r=1.5, hold_min=480),
}
SHADOW_COST = 0.25          # spread 0.20 + slippage 0.05, charged on every shadow trade
GRADE_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "?": 4}


def strat_name(k):
    return STRAT.get(k, {}).get("name", k)


def strat_magic(k):
    return MAGIC + STRAT.get(k, STRAT["engine"])["off"]


def fixed_plan(k):
    return (STRAT[k]["sl_atr"], STRAT[k]["tp_r"], 1e9, 0.0, 1e9, 0.0)


def sday_of(ts):
    """Trading day of a UTC time (the day starts at the 18:00 New York reopen), as YYYY-MM-DD."""
    ny = pd.Timestamp(ts).tz_localize("UTC").tz_convert("America/New_York")
    return str((ny + pd.Timedelta(hours=6)).tz_localize(None).floor("D"))[:10]


def agent_grade(side, pb, pr, thr):
    """A = own agent in its top 35% AND opposite agent below median (dual-strict)
       B = own agent in its top 35% (strict)   C = own agent above its median   D = below median."""
    own, opp = (pb, pr) if side == 1 else (pr, pb)
    ko, kp = ("bull", "bear") if side == 1 else ("bear", "bull")
    if own >= thr[ko][0.65] and opp <= thr[kp][0.5]:
        return "A"
    if own >= thr[ko][0.65]:
        return "B"
    if own >= thr[ko][0.5]:
        return "C"
    return "D"


def _h1_atr(m1):
    x = m1.set_index("time").resample("60min", label="left", closed="left").agg(
        {"high": "max", "low": "min", "close": "last"}).dropna()
    h, l, c = x.high.values, x.low.values, x.close.values
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    return float(pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().iat[-1])


def gate_analyse(gate, bar_time):
    """One scoring pass of Agent Bull / Bear per closed candle. Returns a snapshot with the verdicts
    (for the approval card), agent probabilities of the last 15-minute points, the self-calibrated
    thresholds, strategy indicator states, ATR(H1), and recent M1 bars (for shadow trades)."""
    if not gate.ready:
        return None, f"agents not loaded ({gate.error})"
    key = str(bar_time)
    snap = getattr(gate, "snap", None)
    if snap and snap["key"] == key:
        return snap, None
    try:
        ga, m1 = gate.ga, gate.get_m1()
        Xb = ga.bull.features(m1)
        b, r = ga.bull.score(m1, Xb), ga.bear.score(m1, Xb)
        ga._cal = ga._calibration(Xb)
        sc = {"bull": b, "bear": r}
        verd = {d: ga.check(d, m1, sc) for d in ("BUY", "SELL")}
        tail = Xb.iloc[-5:]
        pb = ga.bull.general.predict_proba(tail[ga.bull.cols])[:, 1]
        pr = ga.bear.general.predict_proba(tail[ga.bear.cols])[:, 1]
        thr = {s: {q: float(np.quantile(ga._cal[s], q)) for q in (0.5, 0.65, 0.95, 0.98)} for s in ("bull", "bear")}
        rows = []
        for k in range(len(tail)):
            w = tail.iloc[k]
            g = lambda c: float(w[c]) if c in tail.columns and pd.notna(w[c]) else np.nan
            rows.append(dict(t=pd.Timestamp(w["decision_time"]), pb=float(pb[k]), pr=float(pr[k]), price=g("close"),
                             ema=np.sign(g("ind_H1_dist_ema50") - g("ind_H1_dist_ema20")),
                             macd=np.sign(g("ind_H1_macd_hist")), dc=g("str_H4_breakout20"), trend=g("reg_trend_blend")))
        snap = dict(key=key, verdicts=verd, rows=rows, thr=thr, atr_h1=_h1_atr(m1.tail(60 * 24 * 10)),
                    price=float(m1.close.iat[-1]), m1=m1.tail(1440 * 4).reset_index(drop=True))
    except Exception as e:
        return None, f"agent scoring failed ({type(e).__name__}: {e})"
    gate.snap, gate.cache = snap, {key: verd}
    return snap, None


def _gate_verdicts(self, bar_time):
    snap, err = gate_analyse(self, bar_time)
    return (snap["verdicts"], None) if snap else (None, err)


AgentGate.verdicts = _gate_verdicts          # the approval card and /agents use the same single scoring pass


def pro_migrate(mem):
    cols = {r[1] for r in mem.db.execute("PRAGMA table_info(trades)")}
    for c, typ in (("strategy", "TEXT"), ("grade", "TEXT"), ("magic", "INT"), ("plan", "TEXT"), ("shadow_id", "INT")):
        if c not in cols:
            mem.db.execute(f"ALTER TABLE trades ADD COLUMN {c} {typ}")
    mem.db.executescript("""
    CREATE TABLE IF NOT EXISTS shadow(id INTEGER PRIMARY KEY, bar_time TEXT, entry_ts TEXT, sday TEXT, strategy TEXT,
        setup TEXT, side INT, grade TEXT, score REAL, entry REAL, risk REAL, sl REAL, tp REAL, plan TEXT, flat_by TEXT,
        cost REAL, decision TEXT, real_ticket INT, last_ts TEXT, best_R REAL DEFAULT 0, worst_R REAL DEFAULT 0,
        stage TEXT DEFAULT 'INIT', closed TEXT, exit REAL, R REAL, reason TEXT,
        UNIQUE(bar_time, strategy, setup, side));
    CREATE INDEX IF NOT EXISTS shadow_open ON shadow(closed);""")
    mem.db.commit()


# ---------------------------------------------------------------- extra strategies (from the lab)
class StrategyPack:
    """MACD H1, EMA 20/50 H1, Donchian-20 H4 (each with its agent approval mode) and the two agents
    trading on their own. Signals come from the agents' own 15-minute feature table, exactly as backtested."""

    def __init__(self, mem, enabled=STRATEGIES):
        self.mem, self.enabled = mem, tuple(enabled)
        self.seen = mem.get("strat_seen") or {}

    def prime(self, snap):
        """At start-up: mark everything already in the table as seen, so no stale signal is traded."""
        if snap:
            for k in self.enabled:
                self.seen[k] = str(snap["rows"][-1]["t"])
            self.mem.put("strat_seen", self.seen)

    def _open_shadow(self, k):
        return self.mem.q("SELECT id FROM shadow WHERE strategy=? AND closed IS NULL LIMIT 1", (k,))

    def signals(self, snap):
        if not snap:
            return []
        rows, thr, atr = snap["rows"], snap["thr"], snap["atr_h1"]
        out = []
        for k in self.enabled:
            st = STRAT[k]
            last = pd.Timestamp(self.seen.get(k, "1970-01-01"))
            for j in range(1, len(rows)):
                w, p = rows[j], rows[j - 1]
                if w["t"] <= last:
                    continue
                side = 0
                if k == "macd_h1" and w["macd"] in (1, -1) and w["macd"] != p["macd"]:
                    side = int(w["macd"])
                elif k == "ema_h1" and w["ema"] in (1, -1) and w["ema"] != p["ema"]:
                    side = int(w["ema"])
                elif k == "donchian_h4" and w["dc"] in (1, -1) and w["dc"] != p["dc"]:
                    side = int(w["dc"])
                elif k == "bull_top2" and w["pb"] >= thr["bull"][0.98] and not self._open_shadow(k):
                    side = 1
                elif k == "bear_sniper" and w["pr"] >= thr["bear"][0.95] and w["trend"] < 0 and not self._open_shadow(k):
                    side = -1
                if not side:
                    continue
                g = agent_grade(side, w["pb"], w["pr"], thr)
                if k in ("bull_top2", "bear_sniper"):
                    approved, g = True, "A"
                else:
                    approved = g == "A" if st["mode"] == "dual-strict" else g in ("A", "B")
                risk = st["sl_atr"] * atr
                entry = snap["price"]
                out.append(dict(strategy=k, setup=k, side=side, grade=g, score=round(w["pb"] if side == 1 else w["pr"], 3),
                                entry=entry, risk=risk, tp_r=st["tp_r"], plan=fixed_plan(k), bar_time=str(w["t"]),
                                entry_ts=str(w["t"]), flat_by=str(w["t"] + pd.Timedelta(minutes=st["hold_min"])),
                                cost=SHADOW_COST, real_ok=approved,
                                why=("" if approved else f"agents {st['mode']} needs grade {'A' if st['mode'] == 'dual-strict' else 'A/B'}, got {g}")))
            self.seen[k] = str(rows[-1]["t"])
        self.mem.put("strat_seen", self.seen)
        return out


# ---------------------------------------------------------------- shadow trades
def shadow_log(mem, c, decision):
    mem.x("""INSERT OR IGNORE INTO shadow(bar_time,entry_ts,sday,strategy,setup,side,grade,score,entry,risk,sl,tp,plan,
             flat_by,cost,decision,last_ts) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
          (c["bar_time"], c["entry_ts"], sday_of(c["entry_ts"]), c["strategy"], c["setup"], c["side"], c["grade"],
           c["score"], c["entry"], c["risk"], c["entry"] - c["side"] * c["risk"], c["entry"] + c["side"] * c["risk"] * c["tp_r"],
           json.dumps(list(c["plan"])), c["flat_by"], c["cost"], decision, c["entry_ts"]))
    r = mem.q("SELECT id FROM shadow WHERE bar_time=? AND strategy=? AND setup=? AND side=?",
              (c["bar_time"], c["strategy"], c["setup"], c["side"]))
    return r[0]["id"] if r else None


def shadow_update(mem, m1):
    """Walk every open shadow trade through the M1 candles since its last update: same stop, target,
    profit lock, trailing stop and time exit as a real trade. Stop is checked before target (cautious)."""
    if m1 is None or not len(m1):
        return 0
    opens = mem.q("SELECT * FROM shadow WHERE closed IS NULL")
    t = m1.time.to_numpy()
    o, h, l, c = (m1[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    done = 0
    for s in opens:
        plan = tuple(json.loads(s["plan"]))
        side, entry, risk = s["side"], s["entry"], s["risk"]
        best, worst, stage = s["best_R"] or 0.0, s["worst_R"] or 0.0, s["stage"]
        i0 = np.searchsorted(t, np.datetime64(pd.Timestamp(s["last_ts"])), side="left")
        flat = np.datetime64(pd.Timestamp(s["flat_by"]))
        exit_px, reason, last = None, None, s["last_ts"]
        for i in range(i0, len(t)):
            if t[i] >= flat:
                exit_px, reason = o[i], "TIME"
                break
            want = plan_stop_R(best, plan)
            stop = entry + side * want * risk if want > -1 else s["sl"]
            adv, fav = (l[i], h[i]) if side == 1 else (h[i], l[i])
            if side * (adv - stop) <= 0:
                exit_px = o[i] if side * (o[i] - stop) < 0 else stop
                reason = {"LOCK": "LOCK", "TRAIL": "TRAIL"}.get(stage, "STOP")
                break
            if side * (fav - s["tp"]) >= 0:
                exit_px, reason = (o[i] if side * (o[i] - s["tp"]) > 0 else s["tp"]), "TARGET"
                break
            best = max(best, side * (fav - entry) / risk)
            worst = min(worst, side * (adv - entry) / risk)
            if best >= plan[4]:
                stage = "TRAIL"
            elif best >= plan[2]:
                stage = "LOCK"
            last = str(pd.Timestamp(t[i]) + pd.Timedelta(minutes=1))
        if exit_px is not None:
            R = side * (exit_px - entry) / risk - (s["cost"] or 0) / risk
            mem.x("UPDATE shadow SET closed=?, exit=?, R=?, reason=?, best_R=?, worst_R=?, stage=?, last_ts=? WHERE id=?",
                  (str(pd.Timestamp(t[i]))[:19], float(exit_px), float(R), reason, best, worst, stage, last, s["id"]))
            done += 1
        elif last != s["last_ts"]:
            mem.x("UPDATE shadow SET best_R=?, worst_R=?, stage=?, last_ts=? WHERE id=?", (best, worst, stage, last, s["id"]))
    return done


# ---------------------------------------------------------------- several trades at once
class MultiTrader(Trader):
    """Up to MAX_OPEN positions. Each strategy has its own MAGIC; grade slots make sure a weaker trade
    can never take the room an A-grade trade needs; no hedging; daily money stop; every trade keeps
    its own exit plan (engine: lock + trail; lab strategies: fixed stop / target / time)."""

    def __init__(self, feed, mem, send, auto=AUTO_TRADE):
        super().__init__(feed, mem, send, auto)
        pro_migrate(mem)
        self.close_try, self.real_grades = {}, list(ENGINE_REAL_GRADES)

    def magics(self):
        return {strat_magic(k) for k in STRAT}

    def positions(self):
        ps = self.mt5.positions_get(symbol=self.symbol)
        return [p for p in (ps or ()) if p.magic in self.magics()]

    def open_trades(self):
        return self.mem.q("SELECT * FROM trades WHERE closed IS NULL ORDER BY id")

    def open_trade(self):
        r = self.open_trades()
        return r[-1] if r else None

    def day_usd(self, sday):
        return sum(r["pnl"] or 0.0 for r in self.mem.q("SELECT pnl FROM trades WHERE sday=? AND closed IS NOT NULL", (sday,)))

    def gate_multi(self, sday, c, new_now):
        if not self.auto:
            return "auto-trading is off"
        ok, why = self.allowed()
        if not ok:
            return why
        if c["strategy"] == "engine" and "approval_status" in globals():
            appr = approval_status(self.mem)
            if not appr["passed"]:
                acc = self.mt5.account_info()
                if acc is None or acc.trade_mode != self.mt5.ACCOUNT_TRADE_MODE_DEMO:
                    return "AI approval: model is UNPROVEN - it never trades a real account"
                if globals().get("APPROVAL_MODE") == "strict":
                    return "AI approval: UNPROVEN"
        opens = self.open_trades()
        if len(opens) >= MAX_OPEN:
            return f"all {MAX_OPEN} slots are full"
        if sum(o["grade"] == c["grade"] for o in opens) >= GRADE_SLOTS.get(c["grade"], 0):
            return f"grade {c['grade']} slots full ({GRADE_SLOTS.get(c['grade'], 0)})"
        if any(o["strategy"] == c["strategy"] and o["setup"] == c["setup"] for o in opens):
            return f"{strat_name(c['strategy']) if c['strategy'] != 'engine' else NAMES.get(c['setup'], c['setup'])} already has a trade open"
        if any(o["side"] != c["side"] for o in opens):
            return "no hedging: a trade in the other direction is open"
        if new_now["all"] >= MAX_NEW_PER_CANDLE:
            return f"max {MAX_NEW_PER_CANDLE} new trades per candle"
        if c["strategy"] == "engine" and new_now["engine"] >= 1:
            return "one engine trade per candle (same idea, no double risk)"
        n = len(self.mem.q("SELECT id FROM trades WHERE sday=?", (sday,)))
        if n >= MAX_TRADES_DAY:
            return f"daily limit of {MAX_TRADES_DAY} trades reached"
        if self.day_usd(sday) <= -DAILY_STOP_USD:
            return f"daily loss stop reached ({money(self.day_usd(sday))})"
        return None

    def _find(self, comment, tries=8):
        for _ in range(tries):
            for p in self.positions():
                if p.comment == comment:
                    return p
            time.sleep(0.25)
        known = {t["ticket"] for t in self.open_trades()}
        new = [p for p in self.positions() if p.ticket not in known]
        return new[0] if len(new) == 1 else None          # some brokers rewrite the comment

    def enter_multi(self, sday, c, sid):
        mt5, side, risk = self.mt5, c["side"], float(c["risk"])
        magic, comment = strat_magic(c["strategy"]), f"XA{c['grade']}{sid}"

        def build():
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None or not tick.ask or not tick.bid:
                return None, None, "no live price from MT5"
            price, spread = (tick.ask if side == 1 else tick.bid), tick.ask - tick.bid
            if spread / risk > MAX_COST_R * (1.5 if c["strategy"] != "engine" else 1):
                return None, None, f"live spread ${spread:.2f} is {spread / risk:.0%} of the risk"
            if abs(price - c["entry"]) / risk > MAX_DRIFT_R:
                return None, None, f"price moved {abs(price - c['entry']):.2f} away from the signal"
            return dict(action=mt5.TRADE_ACTION_DEAL, symbol=self.symbol, volume=FIXED_LOT,
                        type=mt5.ORDER_TYPE_BUY if side == 1 else mt5.ORDER_TYPE_SELL, price=price,
                        sl=round(price - side * risk, self.digits), tp=round(price + side * risk * c["tp_r"], self.digits),
                        deviation=DEVIATION_PTS, magic=magic, comment=comment, type_time=mt5.ORDER_TIME_GTC), spread, None

        req, spread, why = build()
        if why:
            return None, why
        r, err = self._send(req, "open")
        if err and r is not None and r.retcode == mt5.TRADE_RETCODE_INVALID_STOPS:
            r, err = self._send({k: v for k, v in req.items() if k not in ("sl", "tp")}, "open")
        pos = self._find(comment, 8 if not err else 4)
        if pos is None:
            return None, err or "order sent but no position appeared"
        entry = float(pos.price_open)
        sl, tp = entry - side * risk, entry + side * risk * c["tp_r"]
        if not pos.sl or not pos.tp:
            ok, merr = self._modify(pos.ticket, sl, tp)
            if not ok:
                cerr = self._close_position(pos, "NOSTOP")
                self.send(f"⚠️ Order #{pos.ticket} filled but the stop could not be set ({merr}). "
                          + ("Position closed again." if not cerr else f"CLOSE ALSO FAILED ({cerr}) - close it by hand now."))
                return None, f"stop could not be set: {merr}"
        else:
            sl, tp = float(pos.sl), float(pos.tp)
        self.mem.x("""INSERT INTO trades(ticket,signal_id,sday,opened,setup,score,side,lots,entry_plan,entry,sl0,tp,risk,slip,
                      spread,flat_by,stop_now,strategy,grade,magic,plan,shadow_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                   (int(pos.ticket), c.get("signal_id"), sday, utc_now().strftime("%Y-%m-%d %H:%M:%S"), c["setup"], c["score"],
                    side, float(pos.volume), c["entry"], entry, sl, tp, risk, side * (entry - c["entry"]), spread,
                    c["flat_by"], sl, c["strategy"], c["grade"], magic, json.dumps(list(c["plan"])), sid))
        usd = float(pos.volume) * OZ_PER_LOT
        dot = "\U0001F7E2" if side == 1 else "\U0001F534"
        label = strat_name(c["strategy"]) if c["strategy"] != "engine" else NAMES.get(c["setup"], c["setup"])
        n_open = len(self.open_trades())
        self.send(f"{dot} [{BOT_ID}] {'BUY' if side == 1 else 'SELL'} {pos.volume:.2f} {self.symbol}  #{pos.ticket}  · grade {c['grade']}\n"
                  f"Strategy: {label}\nFilled {entry:,.2f} (spread {spread:.2f})\n"
                  f"Stop {sl:,.2f} = -${risk * usd:,.2f}   Target {tp:,.2f} = +${risk * c['tp_r'] * usd:,.2f}\n"
                  + (f"Lock at +{LOCK_AT_R:g} R, trail from +{TRAIL_AT_R:g} R\n" if c["strategy"] == "engine" else "Fixed stop and target\n")
                  + f"Flat by {c['flat_by'][5:16]} UTC · open trades {n_open}/{MAX_OPEN}")
        return int(pos.ticket), None

    def manage(self):
        for tr in self.open_trades():
            try:
                self._manage_one(tr)
            except Exception as e:
                print("manage error", tr.get("ticket"), e)

    def _manage_one(self, tr):
        mt5 = self.mt5
        ps = mt5.positions_get(ticket=int(tr["ticket"]))
        if ps is None:
            return
        if len(ps) == 0:
            return self._reconcile(tr)
        pos, tick = ps[0], mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return
        plan = tuple(json.loads(tr["plan"])) if tr.get("plan") else PLAN
        side, entry, risk = tr["side"], tr["entry"], tr["risk"]
        cur = tick.bid if side == 1 else tick.ask
        fav = side * (cur - entry) / risk
        best, worst = max(tr["best_R"] or 0.0, fav), min(tr["worst_R"] or 0.0, fav)
        want = plan_stop_R(best, plan)
        if want > 0:
            new_sl = round(entry + side * want * risk, self.digits)
            gain = side * (new_sl - float(pos.sl)) if pos.sl else 1e9
            step = 0.05 * risk if tr["stage"] != "INIT" else self.point
            if gain >= step and side * (cur - new_sl) > self.stops_gap:
                ok, err = self._modify(pos.ticket, new_sl, float(pos.tp) or tr["tp"])
                if ok:
                    self.mem.x("UPDATE trades SET stop_now=?, stage=? WHERE id=?",
                               (new_sl, "TRAIL" if best >= plan[4] else "LOCK", tr["id"]))
                    if tr["stage"] == "INIT":
                        self.send(f"\U0001F512 [{BOT_ID}] Profit locked on #{pos.ticket}: stop {new_sl:,.2f} "
                                  f"(+${side * (new_sl - entry) * float(pos.volume) * OZ_PER_LOT:,.2f}).")
        if best - (tr["best_R"] or 0) > 0.01 or (tr["worst_R"] or 0) - worst > 0.01:
            self.mem.x("UPDATE trades SET best_R=?, worst_R=? WHERE id=?", (best, worst, tr["id"]))
        want_close = tr["want_close"] or ("TIME" if utc_now() >= pd.Timestamp(tr["flat_by"]) else None)
        if want_close and time.time() - self.close_try.get(tr["id"], 0) > 10:
            self.close(tr, want_close)

    def close(self, tr, reason):
        self.close_try[tr["id"]] = time.time()
        self.mem.x("UPDATE trades SET want_close=? WHERE id=?", (reason, tr["id"]))
        ps = self.mt5.positions_get(ticket=int(tr["ticket"]))
        if ps:
            return self._close_position(ps[0], reason)
        return None

    def close_all(self, reason="MANUAL"):
        trs = self.open_trades()
        for tr in trs:
            self.close(tr, reason)
        return len(trs), self.close_strays()

    def close_ticket(self, ticket, reason="MANUAL"):
        for tr in self.open_trades():
            if int(tr["ticket"]) == int(ticket):
                self.close(tr, reason)
                return True
        return False

    def close_strays(self):
        known = {t["ticket"] for t in self.mem.q("SELECT ticket FROM trades WHERE closed IS NULL")}
        return [(p.ticket, self._close_position(p, "MANUAL")) for p in self.positions() if p.ticket not in known]

    def live_views(self):
        tick = self.mt5.symbol_info_tick(self.symbol)
        out = []
        for tr in self.open_trades():
            cur = (tick.bid if tr["side"] == 1 else tick.ask) if tick else tr["entry"]
            usd = tr["lots"] * OZ_PER_LOT
            out.append(dict(tr, price=cur, now_R=tr["side"] * (cur - tr["entry"]) / tr["risk"],
                            now_usd=tr["side"] * (cur - tr["entry"]) * usd,
                            stop_R=tr["side"] * (tr["stop_now"] - tr["entry"]) / tr["risk"],
                            label=strat_name(tr.get("strategy") or "engine") if (tr.get("strategy") or "engine") != "engine"
                            else NAMES.get(tr["setup"], tr["setup"])))
        return out

    def live_view(self):
        v = self.live_views()
        return v[-1] if v else None


def positions_text(views):
    if not views:
        return "No open trades."
    return "\n\n".join(f"{'BUY' if v['side'] == 1 else 'SELL'} #{v['ticket']} · grade {v.get('grade') or '?'} · {v['label']}\n"
                       f"Entry {v['entry']:,.2f} now {v['price']:,.2f}  {v['now_R']:+.2f} R {money(v['now_usd'])}\n"
                       f"Stop {v['stop_now']:,.2f} ({v['stage']})  target {v['tp']:,.2f}  flat by {v['flat_by'][5:16]}"
                       for v in views)


# ---------------------------------------------------------------- scanner with grades + shadows + several trades
class ProScanner(Scanner):
    def __init__(self, feed, cfg, send, mem=None, trader=None, agents=None):
        super().__init__(feed, cfg, send, mem, trader, agents)
        if mem is not None:
            pro_migrate(mem)
        self.pack = StrategyPack(mem) if mem is not None else None
        self.last_snap_err = None

    def recent_m1(self, snap):
        if snap is not None:
            return snap["m1"]
        try:
            return self.feed.m1_utc(days=4)
        except Exception:
            return None

    def scan_once(self, force=False, quiet=False):
        if force:
            return super().scan_once(force=True)
        if quiet:
            res = super().scan_once(quiet=True)
            if self.agents is not None and self.pack is not None:
                snap, _ = gate_analyse(self.agents, f"prime {res['time']}")
                self.pack.prime(snap)
            return res
        d = self.frame()
        i = len(d) - 1
        bar_time = d.time.iat[i]
        if bar_time == self.last_bar:
            return None
        retrain = self.model is None or time.time() - self.trained_at > RETRAIN_HOURS * 3600
        res = scan(d, i, int(self.cfg.get("replays", 10_000)), model=None if retrain else self.model, fresh_bars=1)
        if retrain:
            self.model, self.trained_at = res["model"], time.time()
        self.scans += 1
        self.last_report = report(res)
        self.last_bar = bar_time
        min_score = float(self.cfg.get("min_score", 0.0))
        if self.mem:
            self.mem.resolve_signals(d)
            if "approval_status" in globals():
                res["approval"] = approval_status(self.mem)
        snap, err = (gate_analyse(self.agents, bar_time) if (USE_AGENTS and self.agents is not None) else (None, "agents off"))
        self.last_snap_err = err
        if snap:
            res["agents"], res["agents_error"] = snap["verdicts"], None
        else:
            res["agents"], res["agents_error"] = None, err
        entry_ts = str(bar_time + pd.Timedelta(minutes=TF_MIN))
        cands = []
        for s in res["setups"]:
            side = 1 if s["side"] == "BUY" else -1
            if s["score"] <= min_score:
                g = "D"
            elif snap:
                g = agent_grade(side, snap["rows"][-1]["pb"], snap["rows"][-1]["pr"], snap["thr"])
            else:
                g = "?"
            why = ("" if s["score"] > min_score else "model score not above the limit")
            if not why and s["risk_check"] != "PASS":
                why = "; ".join(s["risk_reasons"])
            if not why and g == "?":
                why = err or "agents unavailable"
            if not why and g not in self.trader_grades():
                why = f"grade {g} is shadow-only on this bot"
            cands.append(dict(strategy="engine", setup=s["setup"], side=side, grade=g, score=s["score"], entry=s["entry"],
                              risk=s["risk_per_oz"], tp_r=TP_R, plan=PLAN, bar_time=str(bar_time), entry_ts=entry_ts,
                              flat_by=s["flat_by"], cost=float(res.get("spread_cost") or SHADOW_COST), real_ok=not why,
                              why=why, _s=s))
        if self.pack is not None:
            cands += self.pack.signals(snap)
        cands.sort(key=lambda c: (GRADE_ORDER.get(c["grade"], 9), -c["score"]))
        sday, new_now, cards, grades_of = res["sday"], {"all": 0, "engine": 0}, {}, {}
        for c in cands:
            if not c["real_ok"]:
                decision = "SHADOW: " + c["why"]
            elif self.paused:
                decision = "SHADOW: paused"
            elif self.trader is None:
                decision = "SHADOW: signals only"
            else:
                why = self.trader.gate_multi(sday, c, new_now)
                decision = "SHADOW: " + why if why else "PENDING"
            sid = shadow_log(self.mem, c, decision) if self.mem else None
            if decision == "PENDING":
                if c["strategy"] == "engine":
                    c["signal_id"] = self.mem.log_signal(res, c["_s"], "PENDING")
                ticket, e = self.trader.enter_multi(sday, c, sid)
                decision = f"TRADED #{ticket}" if ticket else f"SHADOW: order failed - {e}"
                new_now["all"] += bool(ticket)
                new_now["engine"] += bool(ticket) and c["strategy"] == "engine"
                self.mem.x("UPDATE shadow SET decision=?, real_ticket=? WHERE id=?", (decision, ticket, sid))
                if c["strategy"] == "engine" and ticket:
                    cards.setdefault(c["_s"]["side"], (c["_s"], decision))
                    grades_of.setdefault(c["_s"]["side"], c["grade"])
            if c["strategy"] == "engine" and self.mem:
                self.mem.log_signal(res, c["_s"], decision)
                if c.get("signal_id"):
                    self.mem.x("UPDATE signals SET decision=? WHERE id=?", (decision, c["signal_id"]))
        for side, (s, decision) in cards.items():
            card = pro_alert(res, side, min_score).replace(" · Grade ", " · model ", 1)
            self.send(card + "\n<b>" + html.escape(f"[{BOT_ID}] agent grade {grades_of.get(side, '?')} · {decision}", quote=False) + "</b>", html=True)
        self.alerts += bool(cards)
        if self.mem:
            shadow_update(self.mem, self.recent_m1(snap))
        res["candidates"] = len(cands)
        return res

    def trader_grades(self):
        return self.trader.real_grades if self.trader is not None and hasattr(self.trader, "real_grades") else list(ENGINE_REAL_GRADES)


# ---------------------------------------------------------------- statistics and the weekly report
def _st(R, pnl=None):
    s = r_stats(list(R))
    s["pf"] = None if s["pf"] == float("inf") else s["pf"]
    if pnl is not None:
        s["usd"] = float(sum(pnl))
    return s


def _pf(s):
    return "-" if not s["n"] else "∞" if s["pf"] is None else f"{s['pf']:.2f}"


def group_stats(rows, key, with_usd=False):
    out = {}
    for r in rows:
        out.setdefault(r[key] or "?", []).append(r)
    return {k: _st([x["R"] for x in v], [x.get("pnl") or 0 for x in v] if with_usd else None) for k, v in sorted(out.items())}


def weekly_report(mem, days=7, title=None):
    """Telegram report: real trades and every shadow signal of the last `days` days, by strategy and grade."""
    until = utc_now()
    since = until - pd.Timedelta(days=days)
    s0 = since.strftime("%Y-%m-%d %H:%M:%S")
    real = mem.q("SELECT * FROM trades WHERE closed IS NOT NULL AND R IS NOT NULL AND closed>=?", (s0,))
    for r in real:
        r["strategy"] = r.get("strategy") or "engine"
        r["grade"] = r.get("grade") or "?"
    sh = mem.q("SELECT * FROM shadow WHERE bar_time>=?", (s0,))
    shc = [x for x in sh if x["R"] is not None]
    L = [f"📊 WEEKLY REPORT · {title or BOT_TITLE}", f"{since:%a %d %b} → {until:%a %d %b %Y} (UTC) · {TIMEFRAME} · magic {MAGIC}+", ""]
    a = _st([r["R"] for r in real], [r["pnl"] or 0 for r in real])
    L += ["REAL TRADES (orders on MT5)",
          f"  {a['n']} trades · win {a['win']:.0%} · {a['total']:+.2f} R · {money(a['usd'])} · PF {_pf(a)} · worst drop {a['maxdd']:.2f} R"
          if a["n"] else "  none this week"]
    if a["n"]:
        for k, s in group_stats(real, "strategy", True).items():
            L.append(f"  · {strat_name(k):<24} {s['n']:>3}  win {s['win']:.0%}  {s['total']:+6.2f} R  {money(s['usd'])}  PF {_pf(s)}")
        L.append("  by grade: " + " | ".join(f"{g} {s['n']}×{s['avg']:+.2f}R" for g, s in group_stats(real, "grade").items()))
        ex = {}
        for r in real:
            ex[r["reason"]] = ex.get(r["reason"], 0) + 1
        L.append("  exits: " + ", ".join(f"{k} {v}" for k, v in sorted(ex.items())))
        best = max(real, key=lambda r: r["R"]); worst = min(real, key=lambda r: r["R"])
        L.append(f"  best {best['R']:+.2f} R ({strat_name(best['strategy'])}) · worst {worst['R']:+.2f} R ({strat_name(worst['strategy'])})")
    b = _st([x["R"] for x in shc])
    nd = max(1, len({x["sday"] for x in sh}))
    L += ["", "SHADOW TRADES (every signal, no order, same exits)",
          f"  {len(sh)} signals ({len(sh) / nd:.1f} per trading day) · {len(shc)} finished · {len(sh) - len(shc)} still running",
          f"  finished: win {b['win']:.0%} · avg {b['avg']:+.3f} R · total {b['total']:+.1f} R · PF {_pf(b)}" if b["n"] else "  none finished yet"]
    if b["n"]:
        L.append("  by grade:")
        for g, s in group_stats(shc, "grade").items():
            L.append(f"    {g}: {s['n']:>4}  win {s['win']:.0%}  avg {s['avg']:+.3f} R  PF {_pf(s)}")
        L.append("  by strategy:")
        for k, s in group_stats(shc, "strategy").items():
            L.append(f"    {strat_name(k):<24} {s['n']:>4}  avg {s['avg']:+.3f} R  PF {_pf(s)}")
        eng = [x for x in shc if x["strategy"] == "engine"]
        if eng:
            bys = sorted(group_stats(eng, "setup").items(), key=lambda kv: -kv[1]["avg"])
            L.append("  best setups: " + ", ".join(f"{NAMES.get(k, k)} {s['avg']:+.2f}R ({s['n']})" for k, s in bys[:3]))
            L.append("  worst setups: " + ", ".join(f"{NAMES.get(k, k)} {s['avg']:+.2f}R ({s['n']})" for k, s in bys[-3:][::-1]))
    miss = [x for x in sh if x["grade"] == "A" and str(x["decision"]).startswith("SHADOW")]
    if miss:
        why = {}
        for x in miss:
            k = str(x["decision"]).replace("SHADOW: ", "").split(" (")[0][:60]
            why[k] = why.get(k, 0) + 1
        L += ["", f"A-GRADE SIGNALS NOT TRADED: {len(miss)}"] + [f"  {v}× {k}" for k, v in sorted(why.items(), key=lambda kv: -kv[1])[:6]]
    L += ["", "WHAT IT SAYS"]
    for g, s in (group_stats(shc, "grade").items() if b["n"] else []):
        if s["n"] < 30:
            L.append(f"  grade {g}: only {s['n']} finished - too early to judge")
        else:
            L.append(f"  grade {g}: {'making money' if s['avg'] > 0.02 else 'about break-even' if s['avg'] > -0.02 else 'losing'} "
                     f"({s['avg']:+.3f} R per signal over {s['n']})")
    L.append("  One week is a small sample. Judge after 4+ weeks and 100+ finished signals per grade.")
    return "\n".join(L), real, sh


def report_csv(real, sh):
    buf = io.StringIO()
    w = _csv.writer(buf)
    w.writerow(["kind", "time", "strategy", "setup", "side", "grade", "score", "entry", "exit", "R", "usd", "reason", "decision"])
    for r in real:
        w.writerow(["REAL", r["opened"], r["strategy"], r["setup"], r["side"], r["grade"], r["score"], r["entry"], r["exit"],
                    r["R"], r["pnl"], r["reason"], f"#{r['ticket']}"])
    for x in sh:
        w.writerow(["SHADOW", x["bar_time"], x["strategy"], x["setup"], x["side"], x["grade"], x["score"], x["entry"], x["exit"],
                    x["R"], "", x["reason"] or "running", x["decision"]])
    return buf.getvalue().encode()


def weekly_due(mem):
    """True once per week, after the Friday close (Saturday from WEEKLY_REPORT_UTC hour)."""
    now = utc_now()
    wd, hr = WEEKLY_REPORT_UTC
    if now.weekday() < wd or (now.weekday() == wd and now.hour < hr):
        return False
    key = f"{now.isocalendar()[0]}-W{now.isocalendar()[1]}"
    if mem.get("weekly_sent") == key:
        return False
    mem.put("weekly_sent", key)
    return True


# ---------------------------------------------------------------- shared dashboard (both bots, phone friendly)
def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def hub_write(name, obj):
    os.makedirs(HUB_DIR, exist_ok=True)
    p = os.path.join(HUB_DIR, name)
    tmp = p + f".{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, default=str)
    os.replace(tmp, p)


def hub_read(name, default=None):
    try:
        with open(os.path.join(HUB_DIR, name), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def hub_take_commands(bot_id):
    p = os.path.join(HUB_DIR, f"{bot_id}.cmd.json")
    if not os.path.exists(p):
        return []
    tmp = p + f".{os.getpid()}.take"
    try:
        os.replace(p, tmp)
        with open(tmp, encoding="utf-8") as f:
            cmds = json.load(f)
        os.remove(tmp)
        return cmds if isinstance(cmds, list) else []
    except (OSError, ValueError):
        return []


def hub_add_command(bot_id, cmd, arg=None):
    p = os.path.join(HUB_DIR, f"{bot_id}.cmd.json")
    cmds = hub_read(f"{bot_id}.cmd.json", []) or []
    cmds.append(dict(cmd=cmd, arg=arg, at=utc_now().strftime("%H:%M:%S")))
    hub_write(f"{bot_id}.cmd.json", cmds)
    return p


def pro_snapshot(mem, sc, trader, agents=None):
    """Everything the shared dashboard shows for THIS bot (written to the hub folder every few seconds)."""
    real = mem.q("SELECT * FROM trades WHERE closed IS NOT NULL AND R IS NOT NULL ORDER BY closed, id")
    for r in real:
        r["strategy"] = r.get("strategy") or "engine"
        r["grade"] = r.get("grade") or "?"
    a = _st([r["R"] for r in real], [r["pnl"] or 0 for r in real])
    cum, curve = 0.0, []
    for r in real[-400:]:
        cum += r["pnl"] or 0
        curve.append([r["closed"][5:16], round(cum, 2)])
    wk = (utc_now() - pd.Timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
    sh7 = mem.q("SELECT strategy, setup, grade, R, sday, decision FROM shadow WHERE bar_time>=?", (wk,))
    sh_all = mem.q("SELECT strategy, grade, R FROM shadow WHERE R IS NOT NULL")
    rows = lambda st: [dict(k=k, n=s["n"], win=s["win"], avg=s["avg"], total=s["total"], pf=s["pf"], usd=s.get("usd")) for k, s in st.items()]
    status = dict(bot=BOT_ID, title=BOT_TITLE, tf=TIMEFRAME, magic=MAGIC, login=ACCOUNT_ID, server=SERVER, symbol=SYMBOL,
                  strategies=[strat_name(k) for k in ("engine",) + tuple(STRATEGIES)], max_open=MAX_OPEN, slots=GRADE_SLOTS,
                  daily_stop=DAILY_STOP_USD, max_day=MAX_TRADES_DAY, auto=False, allowed=False, reason="signals only",
                  paused=False, balance=0.0, equity=0.0, last_bar="-", agents=bool(agents and agents.ready),
                  agents_err=(sc.last_snap_err if sc is not None else None), real_grades=list(ENGINE_REAL_GRADES), pid=os.getpid())
    views, today = [], dict(n=0, usd=0.0)
    if sc is not None:
        status.update(paused=sc.paused, last_bar=str(sc.last_bar), scans=sc.scans)
    if trader is not None:
        with MT5_LOCK:
            ok, why = trader.allowed()
            status.update(auto=trader.auto, allowed=ok, reason=why, balance=trader.feed.balance(), equity=trader.feed.equity(),
                          real_grades=trader.real_grades)
            views = trader.live_views()
        sd = sday_of(utc_now())
        today = dict(n=len(mem.q("SELECT id FROM trades WHERE sday=?", (sd,))), usd=trader.day_usd(sd), sday=sd)
    for v in views:
        v["flat_by"] = str(v["flat_by"])
    nice = lambda k: strat_name(k) if k != "engine" else f"{TIMEFRAME} engine"
    return dict(now=utc_now().strftime("%Y-%m-%d %H:%M:%S"), status=status, today=today, open=views, real=a, curve=curve,
                real_by_strategy=rows({nice(k): v for k, v in group_stats(real, "strategy", True).items()}),
                real_by_grade=rows(group_stats(real, "grade", True)),
                shadow_week=dict(n=len(sh7), per_day=len(sh7) / max(1, len({x["sday"] for x in sh7})),
                                 by_grade=rows(group_stats([x for x in sh7 if x["R"] is not None], "grade")),
                                 by_strategy=rows({nice(k): v for k, v in group_stats([x for x in sh7 if x["R"] is not None], "strategy").items()})),
                shadow_all=dict(by_grade=rows(group_stats(sh_all, "grade")),
                                by_strategy=rows({nice(k): v for k, v in group_stats(sh_all, "strategy").items()})),
                recent_real=[dict(t=r["closed"][5:16], s=nice(r["strategy"]) if r["strategy"] != "engine" else NAMES.get(r["setup"], r["setup"]),
                                  side=r["side"], g=r["grade"], R=r["R"], usd=r["pnl"], why=r["reason"]) for r in real[-15:][::-1]],
                recent_shadow=[dict(t=x["bar_time"][5:16], s=nice(x["strategy"]) if x["strategy"] != "engine" else NAMES.get(x["setup"], x["setup"]),
                                    side=x["side"], g=x["grade"], R=x["R"], d=x["decision"], why=x["reason"])
                               for x in mem.q("SELECT * FROM shadow ORDER BY id DESC LIMIT 25")],
                weekly=mem.get("weekly_text"))


HUB_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#121211"><title>XAUUSD Bots</title><style>
:root{color-scheme:dark;--bg:#121211;--card:#1a1a19;--ink:#f0efec;--sub:#c3c2b7;--mute:#8f8e86;--line:#383835;--blue:#3987e5;--red:#e66767;--good:#0ca30c;--warn:#fab219}
@media (prefers-color-scheme:light){:root{color-scheme:light;--bg:#f3f2ef;--card:#fcfcfb;--ink:#0b0b0b;--sub:#52514e;--mute:#7a7972;--line:#dedcd6;--blue:#2a78d6;--red:#e34948;--good:#008300;--warn:#b77700}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1200px;margin:0 auto;padding:16px 14px 40px}h1{font-size:19px;margin:0}h2{font-size:13px;margin:0 0 8px;color:var(--sub);text-transform:uppercase;letter-spacing:.05em}
.top{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;margin-bottom:12px}.mute{color:var(--mute);font-size:12px}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px}.tab{border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:999px;padding:6px 14px;font-weight:600;cursor:pointer}
.tab.on{background:var(--blue);border-color:var(--blue);color:#fff}
.pill{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);border-radius:999px;padding:3px 10px;font-size:12px;font-weight:600}
.dot{width:8px;height:8px;border-radius:50%;background:var(--mute)}.on .dot{background:var(--good)}.wait .dot{background:var(--warn)}.off .dot{background:var(--red)}
.grid{display:grid;gap:10px}.k{grid-template-columns:repeat(auto-fit,minmax(140px,1fr));margin:10px 0}.two{grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));margin-top:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;min-width:0}
.lab{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--sub)}.val{font-size:21px;font-weight:700;font-variant-numeric:tabular-nums;margin-top:2px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--sub);text-align:left;font-weight:600}
th,td{padding:5px 8px 5px 0;border-bottom:1px solid var(--line);white-space:nowrap}td.n,th.n{text-align:right}.scroll{overflow-x:auto}
.btns{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 2px}button.b{border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:8px;padding:8px 12px;font-weight:600;cursor:pointer;font-size:13px}
button.b:hover{border-color:var(--blue)}button.red{border-color:var(--red);color:var(--red)}button.sm{padding:3px 8px;font-size:12px}
.pos{color:var(--good)}.neg{color:var(--red)}.empty{color:var(--mute);padding:12px 0;text-align:center}svg{display:block;width:100%;height:auto}svg text{fill:var(--sub);font-size:11px}
pre{white-space:pre-wrap;font:12px/1.45 ui-monospace,Consolas,monospace;margin:0;color:var(--sub)}.toast{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 14px;display:none}
.botsum{cursor:pointer}.botsum:hover{border-color:var(--blue)}
</style></head><body><main>
<div class="top"><h1>XAUUSD Bots</h1><span id="upd" class="mute"></span></div>
<div class="tabs" id="tabs"></div><div id="view"></div>
<p class="mute">Demo accounts. Shadow trades = every signal recorded with the same exits but no order. R = one unit of risk. Past results do not predict future results.</p>
</main><div class="toast" id="toast"></div><script>
const $=id=>document.getElementById(id),f=(x,d=2)=>x==null?'-':(x>0?'+':'')+Number(x).toFixed(d),usd=x=>x==null?'-':(x<0?'-':'+')+'$'+Math.abs(x).toFixed(2),
esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])),cls=x=>x==null?'':x>0?'pos':x<0?'neg':'',pf=s=>!s.n?'-':s.pf==null?'∞':s.pf.toFixed(2);
let D={bots:[]},tab=localStorage.getItem('tab')||'ALL';
function toast(t){const e=$('toast');e.textContent=t;e.style.display='block';setTimeout(()=>e.style.display='none',2500)}
async function cmd(bot,c,arg,ask){if(ask&&!confirm(ask))return;const r=await fetch('/api/cmd',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({bot,cmd:c,arg})});toast(r.ok?'Sent to '+bot+': '+c+(arg?' '+arg:''):'Failed')}
function table(head,rows,empty){return rows.length?`<div class="scroll"><table><tr>${head.map(h=>`<th class="${h[1]||''}">${h[0]}</th>`).join('')}</tr>${rows.map(r=>`<tr>${r.map((c,i)=>`<td class="${head[i][1]||''}">${c}</td>`).join('')}</tr>`).join('')}</table></div>`:`<div class="empty">${empty}</div>`}
function stat(rows,empty){return table([['Group'],['N','n'],['Win','n'],['Avg R','n'],['Total R','n'],['PF','n']],rows.map(r=>[esc(r.k),r.n,(r.win*100).toFixed(0)+'%',`<span class="${cls(r.avg)}">${f(r.avg,3)}</span>`,f(r.total,1),pf(r)]),empty)}
function curve(c){if(!c.length)return '<div class="empty">No closed real trades yet</div>';const W=900,H=200,L=50,R=60,T=10,B=22,ys=[0,...c.map(p=>p[1])],lo=Math.min(...ys),hi=Math.max(...ys),sp=(hi-lo)||1,
x=i=>L+(W-L-R)*i/Math.max(1,c.length),y=v=>T+(H-T-B)*(1-(v-lo)/sp),d=ys.map((v,i)=>(i?'L':'M')+x(i).toFixed(1)+' '+y(v).toFixed(1)).join(' ');
return `<svg viewBox="0 0 ${W} ${H}"><line x1="${L}" x2="${W-R}" y1="${y(0)}" y2="${y(0)}" stroke="var(--line)"/><text x="${L-6}" y="${y(hi)+4}" text-anchor="end">${usd(hi)}</text><text x="${L-6}" y="${y(lo)+4}" text-anchor="end">${usd(lo)}</text>
<path d="${d}" fill="none" stroke="var(--blue)" stroke-width="2"/><text x="${x(c.length)+6}" y="${y(ys[ys.length-1])+4}" style="fill:var(--ink);font-weight:600">${usd(ys[ys.length-1])}</text></svg>`}
function mode(s,age){if(age>90)return['off','OFFLINE'];if(s.paused)return['wait','Paused'];if(s.auto&&s.allowed)return['on','Auto-trading'];return['wait','Signals only: '+s.reason]}
function botView(b){const s=b.status,a=b.real,t=b.today,[mc,mt]=mode(s,b.age),id=esc(s.bot);
return `<div class="card"><div class="top" style="margin:0"><b>${esc(s.title)}</b><span class="pill ${mc}"><span class="dot"></span>${esc(mt)}</span>
<span class="mute">${esc(s.symbol)} ${s.tf} · demo ${s.login} · magic ${s.magic}+ · agents ${s.agents?'loaded':'NOT loaded'} · last candle ${esc(s.last_bar)} · updated ${b.age.toFixed(0)}s ago</span></div>
<div class="mute" style="margin-top:4px">Strategies: ${s.strategies.map(esc).join(' · ')} · real orders for engine grades ${s.real_grades.join(', ')} · max ${s.max_open} open (A ${s.slots.A}, B ${s.slots.B}, C ${s.slots.C}) · daily stop -$${s.daily_stop} · max ${s.max_day}/day</div>
<div class="btns">${s.paused?`<button class="b" onclick="cmd('${id}','resume')">▶ Resume</button>`:`<button class="b" onclick="cmd('${id}','pause')">⏸ Pause</button>`}
${s.auto?`<button class="b" onclick="cmd('${id}','auto_off')">Auto OFF</button>`:`<button class="b" onclick="cmd('${id}','auto_on')">Auto ON</button>`}
<button class="b" onclick="cmd('${id}','scan')">🔍 Scan now</button><button class="b" onclick="cmd('${id}','report')">📊 Report now</button>
${['A','B','C'].map(g=>`<button class="b sm" onclick="cmd('${id}','grade','${g}')">${s.real_grades.includes(g)?'✔':'✘'} real ${g}</button>`).join('')}
<button class="b red" onclick="cmd('${id}','close_all',null,'Close ALL open trades of ${id}?')">✖ Close all</button></div></div>
<div class="grid k">${[['Balance','$'+s.balance.toFixed(2)],['Equity','$'+s.equity.toFixed(2)],['Today',`<span class="${cls(t.usd)}">${usd(t.usd)}</span> · ${t.n} trades`],['Open',b.open.length+' / '+s.max_open],
['Real trades',a.n+' · PF '+pf(a)],['Real net',`<span class="${cls(a.usd)}">${usd(a.usd)}</span>`],['Shadow 7d',b.shadow_week.n+' · '+b.shadow_week.per_day.toFixed(1)+'/day']].map(k=>`<div class="card"><div class="lab">${k[0]}</div><div class="val">${k[1]}</div></div>`).join('')}</div>
<div class="card"><h2>Open trades</h2>${table([['Side'],['Grade'],['Strategy'],['Entry','n'],['Now','n'],['R','n'],['$','n'],['Stop','n'],['Stage'],['Flat by'],['']],b.open.map(o=>[o.side==1?'BUY':'SELL',esc(o.grade),esc(o.label),o.entry.toFixed(2),o.price.toFixed(2),`<span class="${cls(o.now_R)}">${f(o.now_R)}</span>`,usd(o.now_usd),o.stop_now.toFixed(2),esc(o.stage),esc(o.flat_by.slice(5,16)),`<button class="b sm red" onclick="cmd('${id}','close','${o.ticket}','Close #${o.ticket}?')">Close</button>`]),'No open trades')}</div>
<div class="card" style="margin-top:10px"><h2>Real account curve ($, closed trades)</h2>${curve(b.curve)}</div>
<div class="grid two"><div class="card"><h2>Real trades by strategy</h2>${stat(b.real_by_strategy,'No real trades yet')}</div><div class="card"><h2>Real trades by grade</h2>${stat(b.real_by_grade,'No real trades yet')}</div>
<div class="card"><h2>Shadow · last 7 days by grade</h2>${stat(b.shadow_week.by_grade,'No finished shadow trades yet')}</div><div class="card"><h2>Shadow · last 7 days by strategy</h2>${stat(b.shadow_week.by_strategy,'No finished shadow trades yet')}</div>
<div class="card"><h2>Shadow · all time by grade</h2>${stat(b.shadow_all.by_grade,'-')}</div><div class="card"><h2>Shadow · all time by strategy</h2>${stat(b.shadow_all.by_strategy,'-')}</div></div>
<div class="card" style="margin-top:10px"><h2>Latest signals (shadow)</h2>${table([['Candle'],['Side'],['Grade'],['Strategy / setup'],['Decision'],['R','n']],b.recent_shadow.map(x=>[esc(x.t),x.side==1?'BUY':'SELL',esc(x.g),esc(x.s),esc(String(x.d).slice(0,70)),x.R==null?'running':`<span class="${cls(x.R)}">${f(x.R)}</span>`]),'No signals yet')}</div>
<div class="card" style="margin-top:10px"><h2>Latest real trades</h2>${table([['Closed'],['Side'],['Grade'],['Strategy'],['Exit'],['R','n'],['$','n']],b.recent_real.map(x=>[esc(x.t),x.side==1?'BUY':'SELL',esc(x.g),esc(x.s),esc(x.why),`<span class="${cls(x.R)}">${f(x.R)}</span>`,usd(x.usd)]),'No real trades yet')}</div>
${b.weekly?`<div class="card" style="margin-top:10px"><h2>Last weekly report</h2><pre>${esc(b.weekly)}</pre></div>`:''}`}
function allView(){if(!D.bots.length)return '<div class="card empty">No bot is running yet. Start a bot file and it appears here.</div>';
return `<div class="grid two">${D.bots.map(b=>{const s=b.status,[mc,mt]=mode(s,b.age),t=b.today;return `<div class="card botsum" onclick="setTab('${esc(s.bot)}')"><div class="top" style="margin:0 0 6px"><b>${esc(s.title)}</b><span class="pill ${mc}"><span class="dot"></span>${esc(mt)}</span></div>
<table><tr><th>Balance</th><td class="n">$${s.balance.toFixed(2)}</td><th>Equity</th><td class="n">$${s.equity.toFixed(2)}</td></tr><tr><th>Today</th><td class="n ${cls(t.usd)}">${usd(t.usd)}</td><th>Trades today</th><td class="n">${t.n}</td></tr>
<tr><th>Open</th><td class="n">${b.open.length}/${s.max_open}</td><th>Real PF</th><td class="n">${pf(b.real)}</td></tr><tr><th>Real net</th><td class="n ${cls(b.real.usd)}">${usd(b.real.usd)}</td><th>Shadow 7d</th><td class="n">${b.shadow_week.n} (${b.shadow_week.per_day.toFixed(1)}/day)</td></tr></table>
<div class="btns" onclick="event.stopPropagation()">${s.paused?`<button class="b sm" onclick="cmd('${esc(s.bot)}','resume')">Resume</button>`:`<button class="b sm" onclick="cmd('${esc(s.bot)}','pause')">Pause</button>`}<button class="b sm" onclick="cmd('${esc(s.bot)}','report')">Report</button><button class="b sm red" onclick="cmd('${esc(s.bot)}','close_all',null,'Close ALL trades of ${esc(s.bot)}?')">Close all</button></div></div>`}).join('')}</div>`}
function setTab(t){tab=t;localStorage.setItem('tab',t);draw()}
function draw(){$('tabs').innerHTML=['ALL',...D.bots.map(b=>b.status.bot)].map(t=>`<button class="tab ${t==tab?'on':''}" onclick="setTab('${esc(t)}')">${t=='ALL'?'Both bots':esc(t)}</button>`).join('');
const b=D.bots.find(x=>x.status.bot==tab);$('view').innerHTML=tab=='ALL'||!b?allView():botView(b)}
async function load(){try{const r=await fetch('/api/state');if(r.status==401){location.reload();return}D=await r.json();$('upd').textContent='updated '+D.now+' UTC'+(D.link?' · phone link active':'');draw()}catch(e){$('upd').textContent='dashboard not reachable'}}
load();setInterval(load,5000);
</script></body></html>"""

LOGIN_HTML = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>XAUUSD Bots</title>
<style>body{margin:0;background:#121211;color:#f0efec;font:15px system-ui,sans-serif;display:grid;place-items:center;height:100vh}form{background:#1a1a19;border:1px solid #383835;border-radius:12px;padding:22px;display:grid;gap:10px;width:260px}
input,button{font:inherit;padding:10px;border-radius:8px;border:1px solid #383835;background:#121211;color:#f0efec}button{background:#3987e5;border-color:#3987e5;color:#fff;font-weight:600}</style></head>
<body><form method="POST" action="/login"><b>XAUUSD Bots</b><input name="pin" type="password" inputmode="numeric" placeholder="Dashboard PIN" autofocus><button>Open</button></form></body></html>"""


class Hub:
    """One dashboard for every bot on this computer. Each bot writes its own status file into HUB_DIR;
    whichever bot starts first serves the page (the other takes over if it stops). Buttons write a
    command file that the right bot picks up within a few seconds."""

    def __init__(self, send=None):
        self.srv, self.send, self.tunnel, self.link, self.last_try = None, send, None, None, 0.0
        os.makedirs(HUB_DIR, exist_ok=True)

    def ensure(self):
        if self.srv is not None or time.time() - self.last_try < 30:
            return
        self.last_try = time.time()
        hub = self

        class H(BaseHTTPRequestHandler):
            def _ok(self):
                ck = self.headers.get("Cookie", "")
                return not DASH_PIN or f"xpin={DASH_PIN}" in ck.replace(" ", "").split(";")

            def _reply(self, code, body, ctype="application/json", extra=()):
                body = body if isinstance(body, bytes) else body.encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                for k, v in extra:
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if not self._ok():
                    return self._reply(401 if self.path.startswith("/api") else 200, LOGIN_HTML, "text/html; charset=utf-8")
                if self.path.startswith("/api/state"):
                    return self._reply(200, json.dumps(hub.state(), default=str))
                return self._reply(200, HUB_HTML, "text/html; charset=utf-8")

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(min(n, 10000)).decode("utf-8", "replace")
                if self.path.startswith("/login"):
                    pin = raw.split("pin=", 1)[-1].split("&")[0]
                    if pin == DASH_PIN:
                        return self._reply(303, "", "text/plain", [("Location", "/"), ("Set-Cookie", f"xpin={DASH_PIN}; Path=/; Max-Age=2592000; SameSite=Lax")])
                    return self._reply(200, LOGIN_HTML.replace("Open</button>", "Open</button><span style='color:#e66767'>Wrong PIN</span>"), "text/html; charset=utf-8")
                if not self._ok():
                    return self._reply(401, "{}")
                try:
                    d = json.loads(raw)
                    if d.get("cmd") not in ("pause", "resume", "auto_on", "auto_off", "scan", "report", "close_all", "close", "grade"):
                        return self._reply(400, '{"error":"unknown command"}')
                    hub_add_command(str(d["bot"]), d["cmd"], d.get("arg"))
                    return self._reply(200, '{"ok":true}')
                except Exception as e:
                    return self._reply(400, json.dumps(dict(error=str(e))))

            def log_message(self, *a):
                pass
        try:
            self.srv = ThreadingHTTPServer(("0.0.0.0" if DASH_PUBLIC else "127.0.0.1", DASH_PORT), H)
        except OSError:
            return                                        # the other bot already serves the page
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        hub_write("_server.json", dict(owner=BOT_ID, port=DASH_PORT, lan=f"http://{lan_ip()}:{DASH_PORT}", started=utc_now()))
        self.start_tunnel()

    def start_tunnel(self):
        """Phone access from anywhere: a free Cloudflare quick tunnel, if cloudflared is installed."""
        exe = shutil.which("cloudflared") or next((p for p in (os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloudflared.exe"),
                                                                r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
                                                                r"C:\Program Files\cloudflared\cloudflared.exe") if os.path.exists(p)), None)
        if not TUNNEL or not exe:
            hub_write("_link.json", dict(public=None, lan=f"http://{lan_ip()}:{DASH_PORT}", owner=BOT_ID))
            return
        try:
            self.tunnel = subprocess.Popen([exe, "tunnel", "--no-autoupdate", "--url", f"http://localhost:{DASH_PORT}"],
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as e:
            print("tunnel failed:", e)
            return

        def watch():
            import re
            for line in self.tunnel.stdout:
                m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
                if m and m.group(0) != self.link:
                    self.link = m.group(0)
                    hub_write("_link.json", dict(public=self.link, lan=f"http://{lan_ip()}:{DASH_PORT}", owner=BOT_ID))
                    if self.send:
                        self.send(f"📱 Dashboard link (phone, anywhere): {self.link}\nPIN: the DASH_PIN in the bot file. "
                                  f"Same Wi-Fi: http://{lan_ip()}:{DASH_PORT}")
        threading.Thread(target=watch, daemon=True).start()

    def state(self):
        bots = []
        for fn in sorted(os.listdir(HUB_DIR)):
            if fn.endswith(".json") and not fn.startswith("_") and not fn.endswith(".cmd.json"):
                b = hub_read(fn)
                if b and "status" in b:
                    try:
                        b["age"] = (utc_now() - datetime.strptime(b["now"], "%Y-%m-%d %H:%M:%S")).total_seconds()
                    except (KeyError, ValueError):
                        b["age"] = 999
                    if b["age"] < 6 * 3600:
                        bots.append(b)
        link = hub_read("_link.json", {}) or {}
        return dict(now=utc_now().strftime("%Y-%m-%d %H:%M:%S"), bots=bots, link=link.get("public"))

    def stop(self):
        if self.tunnel:
            self.tunnel.terminate()


def dash_links():
    link = hub_read("_link.json", {}) or {}
    return link.get("public"), link.get("lan") or f"http://{lan_ip()}:{DASH_PORT}"


def run_hub_commands(cmds, sc, trader, send, mem):
    for c in cmds:
        k, arg = c.get("cmd"), c.get("arg")
        try:
            if k == "pause":
                sc.paused = True; send(f"[{BOT_ID}] Paused from the dashboard: no new trades. Open trades are still managed.")
            elif k == "resume":
                sc.paused = False; send(f"[{BOT_ID}] Resumed from the dashboard.")
            elif k in ("auto_on", "auto_off"):
                trader.auto = k == "auto_on"; send(f"[{BOT_ID}] Auto-trading {'ON' if trader.auto else 'OFF'} (dashboard).")
            elif k == "scan":
                with MT5_LOCK:
                    sc.scan_once(force=True)
            elif k == "report":
                send_weekly(mem, send, manual=True)
            elif k == "close_all":
                with MT5_LOCK:
                    n, strays = trader.close_all("MANUAL")
                send(f"[{BOT_ID}] Closing {n} open trade(s) (dashboard)." + (f" Untracked closed: {strays}" if strays else ""))
            elif k == "close":
                with MT5_LOCK:
                    ok = trader.close_ticket(int(arg))
                send(f"[{BOT_ID}] " + (f"Closing #{arg} (dashboard)." if ok else f"#{arg} is not an open trade."))
            elif k == "grade" and arg in ("A", "B", "C"):
                g = trader.real_grades
                g.remove(arg) if arg in g else g.append(arg)
                g.sort()
                send(f"[{BOT_ID}] Engine grades that place real orders: {', '.join(g) or 'none (shadow only)'}")
        except Exception as e:
            send(f"[{BOT_ID}] dashboard command {k} failed: {e}")


def send_weekly(mem, send, manual=False, send_doc=None):
    text, real, sh = weekly_report(mem)
    mem.put("weekly_text", text)
    send(("(requested) " if manual else "") + text)
    if send_doc:
        try:
            send_doc(report_csv(real, sh), f"{BOT_ID}_week_{utc_now():%Y-%m-%d}.csv")
        except Exception as e:
            print("csv send failed", e)
