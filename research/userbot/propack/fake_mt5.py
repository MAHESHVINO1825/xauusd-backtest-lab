"""Tiny MetaTrader5 stand-in for offline replay: market fills, broker-side SL/TP, deals history."""
import types, itertools
from collections import namedtuple
C = dict(ACCOUNT_TRADE_MODE_DEMO=0, ORDER_FILLING_FOK=0, ORDER_FILLING_IOC=1, ORDER_FILLING_RETURN=2, TRADE_RETCODE_INVALID_FILL=10030,
         TRADE_RETCODE_DONE=10009, TRADE_RETCODE_PLACED=10008, TRADE_RETCODE_DONE_PARTIAL=10010, TRADE_RETCODE_NO_CHANGES=10025,
         TRADE_RETCODE_INVALID_STOPS=10016, TRADE_RETCODE_REQUOTE=10004, TRADE_RETCODE_PRICE_CHANGED=10020, TRADE_RETCODE_PRICE_OFF=10021,
         TRADE_ACTION_DEAL=1, TRADE_ACTION_SLTP=6, ORDER_TYPE_BUY=0, ORDER_TYPE_SELL=1, POSITION_TYPE_BUY=0, POSITION_TYPE_SELL=1,
         ORDER_TIME_GTC=0, DEAL_ENTRY_OUT=1, DEAL_ENTRY_OUT_BY=3, DEAL_REASON_TP=5, DEAL_REASON_SL=4, DEAL_REASON_EXPERT=3)
Pos = namedtuple("Pos", "ticket type volume price_open sl tp magic comment symbol")
Deal = namedtuple("Deal", "entry price volume profit swap commission fee reason")
class FakeMT5:
    def __init__(self, login):
        for k, v in C.items(): setattr(self, k, v)
        self.login, self.bid, self.ask, self.pos, self.deals, self.ids, self.balance = login, 0, 0, {}, {}, itertools.count(1000), 1100.0
    def set_bar(self, o, h, l, c, spread=0.2):
        # walk the bar: check stops on extremes, then settle at close
        for p in self.pos.copy().values():
            buy = p.type == 0
            lo, hi = l, h
            hit_sl = p.sl and ((buy and lo <= p.sl) or (not buy and hi + spread >= p.sl))
            hit_tp = p.tp and ((buy and hi >= p.tp) or (not buy and lo + spread <= p.tp))
            if hit_sl: self._close(p, p.sl, self.DEAL_REASON_SL)
            elif hit_tp: self._close(p, p.tp, self.DEAL_REASON_TP)
        self.bid, self.ask = c, c + spread
    def _close(self, p, px, reason):
        buy = p.type == 0
        pnl = (px - p.price_open if buy else p.price_open - px) * p.volume * 100
        self.balance += pnl
        self.deals[p.ticket] = [Deal(0, p.price_open, p.volume, 0, 0, 0, 0, 0), Deal(1, px, p.volume, pnl, 0, 0, 0, reason)]
        del self.pos[p.ticket]
    def symbol_info(self, s): return types.SimpleNamespace(digits=2, point=0.01, trade_stops_level=0)
    def symbol_info_tick(self, s): return types.SimpleNamespace(bid=self.bid, ask=self.ask, time=0)
    def account_info(self):
        eq = self.balance + sum(((self.bid - p.price_open) if p.type == 0 else (p.price_open - self.ask)) * p.volume * 100 for p in self.pos.values())
        return types.SimpleNamespace(login=self.login, trade_mode=0, trade_allowed=True, balance=self.balance, equity=eq)
    def terminal_info(self): return types.SimpleNamespace(trade_allowed=True)
    def last_error(self): return (0, "ok")
    def positions_get(self, symbol=None, ticket=None):
        if ticket is not None: return tuple(p for p in self.pos.values() if p.ticket == ticket)
        return tuple(self.pos.values())
    def history_deals_get(self, position=None): return tuple(self.deals.get(position, ()))
    def order_send(self, req):
        R = lambda code: types.SimpleNamespace(retcode=code, comment="", order=0)
        if req["action"] == self.TRADE_ACTION_SLTP:
            p = self.pos.get(req["position"])
            if not p: return R(10013)
            self.pos[p.ticket] = p._replace(sl=req["sl"], tp=req["tp"]); return R(self.TRADE_RETCODE_DONE)
        if "position" in req:                       # close
            p = self.pos.get(req["position"])
            if p: self._close(p, self.bid if p.type == 0 else self.ask, self.DEAL_REASON_EXPERT)
            return R(self.TRADE_RETCODE_DONE)
        t = next(self.ids)
        px = self.ask if req["type"] == 0 else self.bid
        self.pos[t] = Pos(t, req["type"], req["volume"], px, req.get("sl", 0), req.get("tp", 0), req["magic"], req["comment"], req["symbol"])
        return R(self.TRADE_RETCODE_DONE)
