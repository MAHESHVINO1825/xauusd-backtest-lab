"""Trade simulator. Signals come from timeframe bars; fills and exits are walked
minute by minute on M1 bars, so SL/TP order inside a big bar is resolved from the
real M1 path. If SL and TP fall in the same M1 bar, the SL is assumed first.

Prices in the data are treated as BID. Longs buy at ask (bid + spread) and exit
on bid; shorts sell at bid and exit on ask. Contract size: 100 oz per lot.
"""
from __future__ import annotations

import numpy as np
from numba import njit

CONTRACT = 100.0
REASONS = {0: "SL", 1: "TP", 2: "Time exit", 3: "End of data"}


@njit(cache=True)
def simulate(long_sig, short_sig, entry_m1, atr_tf,
             m_open, m_high, m_low, m_close, m_spread, m_min,
             sl_atr, rr, hold_min, risk_usd, slippage, commission, swap_night,
             fixed_lot, max_forced_risk_usd):
    """fixed_lot > 0: always trade that lot. Otherwise size by risk_usd; when even
    0.01 lot risks more than risk_usd it is still taken if that risk is at most
    max_forced_risk_usd, else the signal is skipped (counted in `skipped`)."""
    n_tf = len(long_sig)
    n_m1 = len(m_open)
    cap = n_tf
    o_ent = np.empty(cap, np.int64)
    o_ext = np.empty(cap, np.int64)
    o_dir = np.empty(cap, np.int8)
    o_lot = np.empty(cap, np.float64)
    o_ep = np.empty(cap, np.float64)
    o_xp = np.empty(cap, np.float64)
    o_pnl = np.empty(cap, np.float64)
    o_r = np.empty(cap, np.float64)
    o_why = np.empty(cap, np.int8)
    k = 0
    skipped = 0
    busy = -1
    for i in range(n_tf - 1):
        d = 0
        if long_sig[i]:
            d = 1
        elif short_sig[i]:
            d = -1
        if d == 0:
            continue
        e = entry_m1[i]
        if e <= busy or e >= n_m1 - 1:
            continue
        a = atr_tf[i]
        if not (a > 0):
            continue
        sl_d = sl_atr * a
        tp_d = sl_d * rr
        if fixed_lot > 0:
            lot = fixed_lot
        else:
            lot = np.floor(risk_usd / (sl_d * CONTRACT) * 100.0 + 1e-9) / 100.0
            if lot < 0.01:
                if 0.01 * sl_d * CONTRACT > max_forced_risk_usd:
                    skipped += 1
                    continue  # account too small for this stop
                lot = 0.01
        if d == 1:
            ep = m_open[e] + m_spread[e] + slippage
            sl = ep - sl_d
            tp = ep + tp_d
        else:
            ep = m_open[e] - slippage
            sl = ep + sl_d
            tp = ep - tp_d
        t0 = m_min[e]
        xp = 0.0
        why = 3
        j = e
        while j < n_m1:
            sp = m_spread[j]
            if d == 1:
                if m_open[j] <= sl and j > e:
                    xp = m_open[j]; why = 0; break
                if m_low[j] <= sl:
                    xp = sl; why = 0; break
                if m_high[j] >= tp:
                    xp = max(tp, m_open[j]) if j > e else tp; why = 1; break
            else:
                if m_open[j] + sp >= sl and j > e:
                    xp = m_open[j] + sp; why = 0; break
                if m_high[j] + sp >= sl:
                    xp = sl; why = 0; break
                if m_low[j] + sp <= tp:
                    xp = min(tp, m_open[j] + sp) if j > e else tp; why = 1; break
            if m_min[j] - t0 >= hold_min:
                xp = m_close[j] if d == 1 else m_close[j] + sp
                why = 2
                break
            j += 1
        if j >= n_m1:
            j = n_m1 - 1
            xp = m_close[j] if d == 1 else m_close[j] + m_spread[j]
            why = 3
        if d == 1:
            xp -= slippage if why != 1 else 0.0
            pnl = (xp - ep) * CONTRACT * lot
        else:
            xp += slippage if why != 1 else 0.0
            pnl = (ep - xp) * CONTRACT * lot
        # nights held across the 22:00 UTC rollover
        nights = ((m_min[j] - 1320) // 1440) - ((t0 - 1320) // 1440)
        pnl -= commission * lot + swap_night * lot * nights
        o_ent[k] = e
        o_ext[k] = j
        o_dir[k] = d
        o_lot[k] = lot
        o_ep[k] = ep
        o_xp[k] = xp
        o_pnl[k] = pnl
        o_r[k] = pnl / (sl_d * CONTRACT * lot)
        o_why[k] = why
        k += 1
        busy = j
    return (o_ent[:k], o_ext[:k], o_dir[:k], o_lot[:k], o_ep[:k], o_xp[:k],
            o_pnl[:k], o_r[:k], o_why[:k], skipped)
