"""XAUUSD Backtest Lab — Streamlit dashboard.  Start with run_backtest_lab.bat"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import data as D
from core import indicators as I
from core import search as S

st.set_page_config(page_title="XAUUSD Backtest Lab", page_icon="🟡", layout="wide")

GOLD, BLUE, ORANGE, GREEN, PURPLE = "#E2B54A", "#4C8DFF", "#F08C3C", "#5BD69A", "#8B6FD6"
PANEL, LINE, MUTED = "#161B22", "#262C36", "#9AA4B2"

st.markdown(f"""
<style>
.block-container {{padding-top: 3.2rem; max-width: 1500px}}
section[data-testid="stSidebar"] {{background: #11151B; border-right: 1px solid {LINE}}}
.kpi {{background:{PANEL}; border:1px solid {LINE}; border-radius:12px; padding:14px 16px; height:100%}}
.kpi .l {{font-size:12px; color:{MUTED}}}
.kpi .v {{font-family:'JetBrains Mono',ui-monospace,monospace; font-size:26px; font-weight:600; margin-top:4px}}
.kpi .s {{font-size:11px; color:{MUTED}; margin-top:2px}}
.panel {{background:{PANEL}; border:1px solid {LINE}; border-radius:12px; padding:14px 18px}}
.badge {{display:inline-block; font-size:12px; padding:3px 8px; border-radius:6px}}
.ok {{background:rgba(91,214,154,.15); color:{GREEN}}}
.warn {{background:rgba(240,140,60,.16); color:#F5A35C}}
.muted {{color:{MUTED}; font-size:13px}}
.gate {{display:flex; justify-content:space-between; border-top:1px solid #222833; padding:6px 0; font-size:13px}}
h1, h2, h3 {{letter-spacing:-.01em}}
</style>
""", unsafe_allow_html=True)


def kpi(col, label, value, sub="", color=None, size=26):
    style = (f"color:{color};" if color else "") + f"font-size:{size}px;white-space:nowrap"
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v" style="{style}">{value}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def fig_base(fig, h=300):
    fig.update_layout(template="plotly_dark", height=h, margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color="#C5CCD6", size=12),
                      legend=dict(orientation="h", y=1.08, x=0), xaxis=dict(gridcolor="#222833"),
                      yaxis=dict(gridcolor="#222833"))
    return fig


def money(x):
    return f"{'+' if x >= 0 else '-'}${abs(x):,.0f}"


# ------------------------------------------------------------------ sidebar
ss = st.session_state
ss.setdefault("page", "Setup & Run")

with st.sidebar:
    st.markdown(f"### 🟡 XAUUSD Backtest Lab")
    st.markdown('<div class="muted">1 · Indicators</div>', unsafe_allow_html=True)
    chosen = []
    defaults = {"EMA cross", "Supertrend", "RSI", "Stoch RSI", "Bollinger", "VWAP", "ATR expansion", "Session filter"}
    for g, names in I.GROUPS.items():
        sel = st.pills(g, names, selection_mode="multi", default=[n for n in names if n in defaults], key=f"pill_{g}",
                       help="\n".join(f"**{n}** — {I.RULES[n]}" for n in names))
        chosen += list(sel or [])
    st.caption(f"{len(chosen)} selected · filters (ATR expansion, Session filter) only restrict timing")

    st.markdown('<div class="muted">2 · Indicators per combination</div>', unsafe_allow_html=True)
    max_k = st.segmented_control("Combination size", [1, 2, 3], default=2, format_func=lambda k: f"up to {k}",
                                 label_visibility="collapsed") or 2
    st.markdown('<div class="muted">3 · Timeframes</div>', unsafe_allow_html=True)
    tfs = st.pills("Timeframes", list(D.TIMEFRAMES), selection_mode="multi", default=["M15", "H1", "H4"],
                   label_visibility="collapsed") or []
    st.markdown('<div class="muted">4 · Trading style</div>', unsafe_allow_html=True)
    style = st.segmented_control("Style", ["Scalping", "Swing"], default="Swing", label_visibility="collapsed") or "Swing"
    st.caption("Scalping: SL 1–1.5 ATR, TP 1–2R, max hold 4 h" if style == "Scalping"
               else "Swing: SL 2–3 ATR, TP 1.5–2.5R, max hold 5 days")
    direction = st.segmented_control("Direction", ["Long", "Short", "Both"], default="Both",
                                     label_visibility="collapsed") or "Both"

    st.markdown('<div class="muted">5 · Account</div>', unsafe_allow_html=True)
    capital = st.slider("Account capital ($)", 500, 2000, 1000, 100)
    lot_mode = st.radio("Position size", ["Fixed lot", "Risk %"], horizontal=True)
    if lot_mode == "Fixed lot":
        fixed_lot = st.number_input("Lot size", 0.01, 1.0, 0.01, 0.01)
        risk_pct = 1.0
    else:
        fixed_lot = 0.01
        risk_pct = st.slider("Risk per trade (%)", 0.25, 3.0, 1.0, 0.25)
        st.caption(f"${capital * risk_pct / 100:,.0f} per trade. If even 0.01 lot risks more than 3% of capital, "
                   "the trade is skipped and counted.")

    with st.expander("Costs & advanced"):
        base_spread = st.number_input("Typical spread ($)", 0.05, 2.0, 0.20, 0.01,
                                      help="Scaled by hour of day using the spread profile measured from Dukascopy ticks")
        slippage = st.number_input("Slippage per side ($)", 0.0, 1.0, 0.05, 0.01)
        commission = st.number_input("Commission per lot round trip ($)", 0.0, 20.0, 0.0, 0.5)
        swap = st.number_input("Swap per lot per night ($)", 0.0, 50.0, 5.0, 0.5)
        finalists = st.slider("Finalists to validate", 10, 80, 40, 5)

    cfg = S.Config(indicators=chosen, max_k=int(max_k), timeframes=tfs, style=style, direction=direction,
                   capital=float(capital), lot_mode=lot_mode, fixed_lot=float(fixed_lot), risk_pct=float(risk_pct),
                   base_spread=float(base_spread), slippage=float(slippage), commission=float(commission),
                   swap_night=float(swap), finalists=int(finalists))
    n_runs = S.count_runs(cfg) if chosen and tfs else 0
    st.markdown(f'<div class="panel" style="margin:8px 0"><div class="muted">Data split (fixed — stops overfitting)</div>'
                f'<div style="display:flex;height:9px;border-radius:5px;overflow:hidden;margin:8px 0">'
                f'<div style="flex:4;background:{BLUE}"></div><div style="flex:1.5;background:{GOLD}"></div>'
                f'<div style="flex:1.4;background:{PURPLE}"></div></div>'
                f'<div style="display:flex;justify-content:space-between;font-size:11px;color:#C5CCD6">'
                f'<span>Train 2015–22</span><span>Test 2023–24</span><span>Holdout 2025–26</span></div></div>',
                unsafe_allow_html=True)
    run_clicked = st.button(f"Run {n_runs:,} backtests", type="primary", width="stretch",
                            disabled=n_runs == 0)
    if n_runs:
        n_c = len(S.combos_for(cfg))
        st.caption(f"{n_c:,} combos × {len(tfs)} TF × 4 exit presets")

# ------------------------------------------------------------------ header / nav
c2, c1 = st.container(), st.container()
with c1:
    try:
        m1 = D.load_m1()
        st.markdown(f"<div class='muted'>Data {m1['time'].iloc[0]:%Y-%m-%d} → {m1['time'].iloc[-1]:%Y-%m-%d} · "
                    f"{len(m1):,} M1 bars · spread profile from Dukascopy ticks</div>", unsafe_allow_html=True)
    except Exception as e:  # noqa
        st.error(f"Could not load data: {e}")
        st.stop()
with c2:
    if "page_next" in ss:
        ss["page"] = ss.pop("page_next")
    page = st.segmented_control("Page", ["Setup & Run", "Leaderboard", "Strategy Detail", "Data Health"],
                                key="page", label_visibility="collapsed") or "Setup & Run"

# ------------------------------------------------------------------ run
if run_clicked:
    bar = st.progress(0.0, "Starting…")
    with st.status("Running backtests…", expanded=True) as status:
        out = S.run_search(cfg, progress=lambda p, t: bar.progress(min(p, 1.0), t), log=st.write)
        status.update(label="Run complete", state="complete")
    ss["run_id"] = os.path.basename(out)
    ss.pop("strategy", None)
    st.rerun()

runs = S.list_runs()
if not runs and page not in ("Data Health",):
    st.markdown("## Welcome")
    st.info("Pick indicators, timeframes, style and capital on the left, then press **Run**. "
            "Results stay saved in the results folder, so you can reopen past runs.")
    st.stop()

if runs:
    ss.setdefault("run_id", runs[0])
    if ss["run_id"] not in runs:
        ss["run_id"] = runs[0]


@st.cache_data(show_spinner=False)
def get_run(run_id):
    js = S.load_run(run_id)
    return js


def run_picker():
    def label(r):
        try:
            c = get_run(r)["meta"]["config"]
            return f"{r[:8]} {r[9:11]}:{r[11:13]} · {c['style']} · {', '.join(c['timeframes'])} · ${c['capital']:,.0f}"
        except Exception:  # noqa
            return r
    ss["run_id"] = st.selectbox("Run", runs, index=runs.index(ss["run_id"]), format_func=label,
                                label_visibility="collapsed")
    return get_run(ss["run_id"])


def fin_table(js):
    rows = []
    for f in js["finalists"]:
        m = f["metrics"]
        all_net = sum(m[p]["net"] for p in D.PERIODS)
        first_fail = next((S.GATE_NAMES[i] for i, g in enumerate(f["gates"]) if not g), "")
        rows.append({"id": f["id"], "Combination": f["combo"], "TF": f["tf"], "Exit": f"SL {f['sl_atr']} ATR · {f['rr']}R", "SL×ATR": f["sl_atr"], "RR": f["rr"],
                     "Trades": sum(m[p]["trades"] for p in D.PERIODS),
                     "Win %": round(m["test"]["win"], 1), "PF train": round(m["train"]["pf"], 2),
                     "PF test": round(m["test"]["pf"], 2), "PF holdout": round(m["holdout"]["pf"], 2),
                     "Net $": round(all_net, 0), "Max DD %": round(max(m[p]["dd"] for p in D.PERIODS), 1),
                     "Gates": f["passed"], "First failed gate": first_fail,
                     "Gate map": "".join("✓" if g else "✕" for g in f["gates"])})
    cols = ["id", "Combination", "TF", "Exit", "SL×ATR", "RR", "Trades", "Win %", "PF train", "PF test", "PF holdout",
            "Net $", "Max DD %", "Gates", "First failed gate", "Gate map"]
    df = pd.DataFrame(rows, columns=cols)
    if len(df):
        df = df.sort_values(["Gates", "PF test"], ascending=[False, False]).reset_index(drop=True)
        df.index += 1
    return df


# ================================================================== pages
if page == "Setup & Run":
    js = run_picker()
    meta, c = js["meta"], js["meta"]["config"]
    st.markdown(f"## Run summary")
    st.markdown(f"<div class='muted'>{c['style']} · {c['direction']} · {', '.join(c['timeframes'])} · "
                f"${c['capital']:,.0f} · {c['lot_mode']}{' ' + str(c['fixed_lot']) if c['lot_mode'] == 'Fixed lot' else ' ' + str(c['risk_pct']) + '%'}"
                f" · {len(c['indicators'])} indicators, up to {c['max_k']} per combo · {meta['seconds']}s</div>",
                unsafe_allow_html=True)
    df = fin_table(js)
    n9 = int((df["Gates"] == 9).sum()) if len(df) else 0
    k = st.columns(4)
    kpi(k[0], "Backtests run", f"{meta['total']:,}")
    kpi(k[1], "Passed fast screen", f"{meta['screened']:,}", "train PF ≥ 1.1, test PF ≥ 1.0, enough trades")
    kpi(k[2], "Passed all 9 gates", f"{n9}", f"of {len(df)} finalists validated", GREEN if n9 else ORANGE)
    best = df[df["Gates"] == 9]["PF test"].max() if n9 else (df["PF test"].max() if len(df) else 0)
    kpi(k[3], "Best test-period PF" + ("" if n9 else " (not validated)"), f"{best:.2f}" if best else "—", "", GOLD)

    st.write("")
    if n9 == 0:
        st.warning("No combination survived every gate in this run. That is a real result, not a bug: "
                   "it means these indicator combos do not hold an edge after costs on this data.")
    st.markdown("#### Top 5")
    if len(df):
        st.dataframe(df.head(5)[["Combination", "TF", "Exit", "Trades", "PF train", "PF test", "PF holdout", "Net $", "Gates", "Gate map"]],
                     width="stretch")
    a, b = st.columns(2)
    with a:
        st.markdown("#### Profit factor distribution (test period, every backtest)")
        sc = js["screen"]
        x = sc.loc[sc["test_trades"] >= 20, "test_pf"].clip(0, 3)
        fig = go.Figure(go.Histogram(x=x, nbinsx=40, marker_color=BLUE))
        fig.add_vline(x=1.3, line_dash="dash", line_color=GOLD, annotation_text="1.3 pass bar")
        fig.add_vline(x=1.0, line_color=MUTED)
        st.plotly_chart(fig_base(fig, 260), width="stretch")
        st.caption(f"Median test PF {x.median():.2f}. A wide spread around 1.0 is what pure luck looks like — "
                   "which is why the gates exist.")
    with b:
        st.markdown("#### Why finalists failed (first gate missed)")
        fails = df[df["Gates"] < 9]["First failed gate"].value_counts().reindex(S.GATE_NAMES).dropna()
        if not len(fails):
            st.caption("Nothing to show — no finalists failed a gate in this run.")
        else:
            fig = go.Figure(go.Bar(x=fails.values, y=fails.index, orientation="h", marker_color=ORANGE))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_base(fig, 260), width="stretch")

elif page == "Leaderboard":
    js = run_picker()
    df = fin_table(js)
    st.markdown("## Leaderboard")
    st.markdown("<div class='muted'>Finalists ranked by gates passed, then <b>test-period</b> profit factor — never by training results.</div>",
                unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
    q = f1.text_input("Search", placeholder="e.g. VWAP")
    only9 = f2.toggle("Passed all gates", value=False)
    tf_f = f3.multiselect("TF", sorted(df["TF"].unique()) if len(df) else [])
    min_tr = f4.number_input("Min trades", 0, 5000, 0, 50)
    v = df.copy()
    if q:
        v = v[v["Combination"].str.contains(q, case=False)]
    if only9:
        v = v[v["Gates"] == 9]
    if tf_f:
        v = v[v["TF"].isin(tf_f)]
    v = v[v["Trades"] >= min_tr]
    st.dataframe(v.drop(columns=["id", "SL×ATR", "RR"]), width="stretch", height=560,
                 column_config={"Gates": st.column_config.ProgressColumn("Gates", min_value=0, max_value=9, format="%d/9"),
                                "Net $": st.column_config.NumberColumn(format="%+.0f")})
    st.download_button("Export CSV", v.to_csv(index=False).encode(), f"leaderboard_{ss['run_id']}.csv", "text/csv")
    if not len(v):
        st.info("No finalists match. Widen the filters, or run with more indicators or timeframes.")
        st.stop()
    pick = st.selectbox("Open strategy", v["id"].tolist(),
                        format_func=lambda i: f"{df.set_index('id').loc[i, 'Combination']} · {df.set_index('id').loc[i, 'TF']} · "
                                              f"SL {df.set_index('id').loc[i, 'SL×ATR']} ATR · RR {df.set_index('id').loc[i, 'RR']}")
    if st.button("View strategy →", type="primary") and pick:
        ss["strategy"] = pick
        ss["page_next"] = "Strategy Detail"
        st.rerun()
    st.markdown("<div class='panel' style='margin-top:10px'><b>Gates</b><br>" +
                "<br>".join(f"{i + 1}. <b>{n}</b> — {h}" for i, (n, h) in enumerate(zip(S.GATE_NAMES, S.GATE_HELP))) +
                "</div>", unsafe_allow_html=True)

elif page == "Strategy Detail":
    js = run_picker()
    fins = {f["id"]: f for f in js["finalists"]}
    if not fins:
        st.info("This run has no finalists — no combination passed the fast screen. Try more indicators, timeframes or the other style.")
        st.stop()
    order = fin_table(js)["id"].tolist()
    default = ss.get("strategy") if ss.get("strategy") in fins else order[0]
    sid = st.selectbox("Strategy", order, index=order.index(default),
                       format_func=lambda i: f"{fins[i]['combo']} · {fins[i]['tf']} · SL {fins[i]['sl_atr']} ATR · RR {fins[i]['rr']} · {fins[i]['passed']}/9")
    f = fins[sid]
    c = js["meta"]["config"]
    tr = pd.read_parquet(os.path.join(js["dir"], f"trades_{sid}.parquet"))
    m, det = f["metrics"], f["detail"]
    st.markdown(f"## {f['combo']}")
    rules = " · ".join(f"{n}: {I.RULES[n]}" for n in f["combo"].split(" + "))
    st.markdown(f"<div class='muted'>{f['tf']} · {c['style']} · {c['direction']} · SL {f['sl_atr']}×ATR(14) · TP {f['rr']}R · "
                f"max hold {'4 h' if c['style'] == 'Scalping' else '5 days'} · ${c['capital']:,.0f}<br>{rules}</div>",
                unsafe_allow_html=True)
    cls = "ok" if f["passed"] == 9 else "warn"
    st.markdown(f"<span class='badge {cls}'>Passed {f['passed']} / 9 gates</span>", unsafe_allow_html=True)

    per = st.segmented_control("Period", ["All", "Train", "Test", "Holdout"], default="All", key="per_sel") or "All"
    t = tr if per == "All" else tr[tr["period"] == per.lower()]
    mm = S.metrics(t["pnl"].values, t["r"].values, c["capital"])
    k = st.columns(8)
    kpi(k[0], "Net profit", money(mm["net"]), "", GREEN if mm["net"] > 0 else ORANGE, size=20)
    kpi(k[1], "Return", f"{mm['ret']:+.1f}%", size=20)
    kpi(k[2], "Profit factor", f"{mm['pf']:.2f}", "", GOLD, size=20)
    kpi(k[3], "Win rate", f"{mm['win']:.1f}%", size=20)
    kpi(k[4], "Max drawdown", f"{mm['dd']:.1f}%", size=20)
    kpi(k[5], "Trades", f"{mm['trades']:,}", f"{(t['side'] == 'BUY').mean() * 100:.0f}% long" if len(t) else "", size=20)
    kpi(k[6], "Expectancy", f"{mm['exp_r']:+.2f} R", size=20)
    kpi(k[7], "Avg win / loss $", f"{mm['avg_win']:.0f} / {mm['avg_loss']:.0f}", size=20)

    st.markdown("#### Equity curve")
    from plotly.subplots import make_subplots
    eq = c["capital"] + tr["pnl"].cumsum()
    ddp = (eq / eq.cummax() - 1) * 100
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)
    x_lo, x_hi = tr["exit_time"].min(), tr["exit_time"].max()
    for (a_, b_, col) in [(x_lo, D.TRAIN_END, BLUE), (D.TRAIN_END, D.TEST_END, GOLD), (D.TEST_END, x_hi, PURPLE)]:
        fig.add_shape(type="rect", xref="x", yref="paper", x0=a_, x1=b_, y0=0, y1=1, fillcolor=col, opacity=0.12, line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=tr["exit_time"], y=eq, name="Equity $", line=dict(color=GOLD, width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=tr["exit_time"], y=ddp, name="Drawdown %", fill="tozeroy",
                             line=dict(color="rgba(240,110,90,.8)", width=1)), 2, 1)
    st.plotly_chart(fig_base(fig, 380), width="stretch")
    st.caption("Blue = train 2015–22 · gold = test 2023–24 · purple = holdout 2025–26. Red = drawdown from peak.")

    a, b = st.columns(2)
    with a:
        st.markdown("#### Monthly returns (% of capital)")
        mo = tr.assign(y=tr["exit_time"].dt.year, mth=tr["exit_time"].dt.month).groupby(["y", "mth"])["pnl"].sum() / c["capital"] * 100
        hm = mo.unstack().reindex(columns=range(1, 13))
        lim = max(1.0, np.nanmax(np.abs(hm.values)))
        fig = go.Figure(go.Heatmap(z=hm.values, x=["J", "F", "M", "A", "M ", "J ", "J  ", "A ", "S", "O", "N", "D"],
                                   y=[str(y) for y in hm.index], zmid=0, zmin=-lim, zmax=lim,
                                   colorscale=[[0, ORANGE], [0.5, PANEL], [1, BLUE]],
                                   text=np.where(np.isnan(hm.values), "", np.vectorize(lambda v: f"{v:.1f}")(np.nan_to_num(hm.values))), texttemplate="%{text}", showscale=False))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_base(fig, 300), width="stretch")
    with b:
        st.markdown("#### Net P/L by entry hour (UTC)")
        hr = tr.groupby(tr["entry_time"].dt.hour)["pnl"].sum().reindex(range(24), fill_value=0)
        fig = go.Figure(go.Bar(x=hr.index, y=hr.values, marker_color=[BLUE if v >= 0 else ORANGE for v in hr.values]))
        st.plotly_chart(fig_base(fig, 300), width="stretch")

    a, b, cc = st.columns(3)
    with a:
        st.markdown("#### Monte Carlo (1,000 reshuffles)")
        mp = os.path.join(js["dir"], f"mc_{sid}.npy")
        if os.path.exists(mp):
            q = np.load(mp)
            x = np.arange(q.shape[1])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=q[2], line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=x, y=q[0], fill="tonexty", fillcolor="rgba(76,141,255,.22)", line=dict(width=0), name="5–95%"))
            fig.add_trace(go.Scatter(x=x, y=q[1], line=dict(color=BLUE, width=2), name="median"))
            st.plotly_chart(fig_base(fig, 240), width="stretch")
        st.caption(f"95% worst drawdown {det['mc_dd95']:.1f}% (train + test trades)")
    with b:
        st.markdown("#### 6-month windows")
        if det["roll"]:
            r = pd.DataFrame(det["roll"], columns=["win", "pf", "net"])
            fig = go.Figure(go.Bar(x=r["win"], y=r["net"], marker_color=[BLUE if v > 0 else ORANGE for v in r["net"]]))
            st.plotly_chart(fig_base(fig, 240), width="stretch")
        st.caption(f"{det['roll_pct']:.0f}% of windows profitable")
    with cc:
        st.markdown("#### Parameter stability")
        nb = det["neighbours"]
        fig = go.Figure(go.Bar(x=["lengths −20%", "lengths +20%", "stop −25%", "stop +25%"], y=nb,
                               marker_color=[GOLD if v >= 1.1 else ORANGE for v in nb]))
        fig.add_hline(y=1.0, line_color=MUTED)
        fig.add_hline(y=m["test"]["pf"], line_dash="dash", line_color=GREEN, annotation_text="this strategy")
        st.plotly_chart(fig_base(fig, 240), width="stretch")
        st.caption("Test PF of nearby settings — similar bars = robust, not a lucky spike")

    a, b = st.columns([1, 2])
    with a:
        st.markdown("#### Validation gates")
        vals = [f"{m['train']['trades']} / {m['test']['trades']}", f"{m['test']['pf']:.2f}", f"PF {det['stress_pf']:.2f}",
                f"{det['roll_pct']:.0f}%", f"median {np.median(det['neighbours']):.2f}", f"{det['year_pct']:.0f}%",
                f"beats {det['rnd_pct']:.0f}%", f"{det['mc_dd95']:.1f}%",
                f"PF {m['holdout']['pf']:.2f} / {m['holdout']['trades']}"]
        html = "".join(f"<div class='gate'><span>{'✅' if g else '❌'} {n}</span><span style='font-family:monospace'>{v}</span></div>"
                       for g, n, v in zip(f["gates"], S.GATE_NAMES, vals))
        st.markdown(f"<div class='panel'>{html}</div>", unsafe_allow_html=True)
        if det.get("skipped"):
            st.warning(f"{det['skipped']} signals skipped — the stop was too wide for this account at Risk % sizing.")
    with b:
        st.markdown("#### Trades")
        st.dataframe(t.sort_values("entry_time", ascending=False), width="stretch", height=360)
        d1, d2 = st.columns(2)
        d1.download_button("Export trades CSV", tr.to_csv(index=False).encode(), f"trades_{f['combo']}_{f['tf']}.csv".replace(" ", ""))
        bot = dict(symbol="XAUUSD", timeframe=f["tf"], direction=c["direction"], style=c["style"],
                   entry_rules={n: I.RULES[n] for n in f["combo"].split(" + ")},
                   entry_trigger="Enter at next bar open when ALL rules switch to true on a closed bar",
                   stop_loss=f"{f['sl_atr']} x ATR(14) of the signal bar", take_profit=f"{f['rr']} x stop distance",
                   max_hold_minutes=S.HOLD_MIN[c["style"]], lot_mode=c["lot_mode"], fixed_lot=c["fixed_lot"],
                   risk_pct=c["risk_pct"], capital=c["capital"], one_position_at_a_time=True,
                   backtest=dict(gates_passed=f["passed"], test_pf=round(m["test"]["pf"], 2), holdout_pf=round(m["holdout"]["pf"], 2)))
        d2.download_button("Export bot settings (JSON)", json.dumps(bot, indent=2).encode(), f"bot_{sid}.json", "application/json")

elif page == "Data Health":
    st.markdown("## Data health")
    h = D.data_health()
    k = st.columns(3)
    kpi(k[0], "First bar", f"{h['first']:%Y-%m-%d}")
    kpi(k[1], "Last bar", f"{h['last']:%Y-%m-%d}")
    kpi(k[2], "M1 bars", f"{h['bars']:,}")
    a, b = st.columns(2)
    with a:
        st.markdown("#### Bars and price range by year")
        st.dataframe(h["by_year"].rename(columns={"bars": "M1 bars", "low": "Low", "high": "High"}), width="stretch")
        st.caption("Source: XAU/USD 1-minute archive 2015–2024 (GitHub parshant65-lang, converted from broker time to UTC) + Dukascopy ticks 2025–2026 (GitHub Touhid-HZ). "
                   "Spread profile: Dukascopy tick data 2026 (GitHub Touhid-HZ), 6 sample weeks.")
    with b:
        st.markdown("#### Spread multiplier by hour (UTC)")
        sh = D.spread_shape()
        fig = go.Figure(go.Bar(x=list(range(24)), y=sh, marker_color=[ORANGE if v > 1.2 else BLUE for v in sh]))
        fig.add_hline(y=1.0, line_color=MUTED)
        st.plotly_chart(fig_base(fig, 280), width="stretch")
        st.caption("Typical spread × this multiplier is charged on every trade. The 21:00 UTC rollover is the most expensive hour.")
    st.markdown("#### Largest weekday gaps (minutes)")
    st.dataframe(pd.DataFrame({"after": h["gap_times"].values, "gap minutes": h["gaps"].values}), width="stretch")
