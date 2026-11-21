"""Example: add Agent Bull / Agent Bear approval to ANY existing XAUUSD bot on MT5.

Copy the `xauusd-backtest-lab` folder next to your bot (or add it to PYTHONPATH), then:
    pip install -r xauusd-backtest-lab/requirements.txt MetaTrader5

Only three things change in your bot:
    1. create GoldAgents once at start-up
    2. before every order, call ga.check("BUY" | "SELL", m1)
    3. place the order only if verdict["approve"] is True
"""
import sys
import time

import MetaTrader5 as mt5

sys.path.insert(0, r"C:\Users\mahes\Documents\GitHub\xauusd-backtest-lab")   # path to this repo
from agents.plugin import GoldAgents  # noqa: E402

# ---- your MT5 details (kept in code, no .env) ----
MT5_LOGIN = 12345678
MT5_PASSWORD = "your_password"
MT5_SERVER = "MetaQuotes-Demo"
SYMBOL = "XAUUSD"
LOT = 0.01

# Mode: "strict" for trend / breakout bots, "dual-strict" for reversal / scalping bots
ga = GoldAgents(mode="strict")


def my_bot_signal():
    """Your existing strategy goes here. Return "BUY", "SELL" or None."""
    return None


def place_order(direction):
    tick = mt5.symbol_info_tick(SYMBOL)
    req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": LOT,
           "type": mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL,
           "price": tick.ask if direction == "BUY" else tick.bid, "deviation": 20, "magic": 777,
           "comment": "agent-approved"}
    return mt5.order_send(req)


def main():
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        raise SystemExit(f"MT5 init failed: {mt5.last_error()}")
    while True:
        sig = my_bot_signal()
        if sig:
            m1 = GoldAgents.m1_from_mt5(mt5, SYMBOL)       # ~14 months of M1, converted to UTC
            verdict = ga.check(sig, m1)
            print(verdict["reason"])                       # send this to Telegram if you like
            if verdict["approve"]:
                print(place_order(sig))
        time.sleep(60)


if __name__ == "__main__":
    main()
