"""AI Workforce & Demand Planner (Streamlit).

    streamlit run apps/workforce_planner/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # so `cxcore` imports on any host

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cxcore.config import PORTFOLIO_URL, REPO_URL
from cxcore.llm import LLM
from cxcore.workforce import (CHANNELS, Scenario, anomalies, backtest, brief, flat_roster_check, forecast,
                              load_demand, plan_facts, staff_plan)

BLUE, ORANGE, AQUA, YELLOW, GREY = "#3987e5", "#d95926", "#199e70", "#c98500", "#8b98a8"
CH_COLORS = {"phone": BLUE, "chat": AQUA, "email": YELLOW}
LAYOUT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
              margin=dict(l=10, r=10, t=40, b=10), font=dict(size=13), hoverlabel=dict(font_size=13),
              legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom", title_text=""))

st.set_page_config(page_title="AI Workforce & Demand Planner", page_icon="🗓️", layout="wide")


@st.cache_data
def history() -> pd.DataFrame:
    return load_demand()


@st.cache_data
def cached_forecast() -> pd.DataFrame:
    return forecast(history(), weeks=1)


@st.cache_data
def cached_backtest() -> dict:
    bt = backtest(history())
    return {k: v for k, v in bt.items()}


@st.cache_data
def cached_anomalies() -> pd.DataFrame:
    return anomalies(history())


@st.cache_data
def cached_plan(s: Scenario) -> dict:
    return staff_plan(cached_forecast(), s)


@st.cache_resource
def llm() -> LLM:
    return LLM()


hist, fc, bt, anoms = history(), cached_forecast(), cached_backtest(), cached_anomalies()
st.caption("CX AI SUITE · PROJECT 6")
st.title("AI Workforce & Demand Planner")
st.write("Forecasts next week's phone, chat and email volume hour by hour, works out how many agents each hour "
         "needs to hit the service level, and prices every what-if. An AI copilot writes it up for managers. "
         f"Synthetic data for a fictional support centre. [Source]({REPO_URL}) · [Portfolio]({PORTFOLIO_URL})")

# ---- scenario controls -------------------------------------------------------------------
with st.sidebar:
    st.header("What-if")
    vol = st.slider("Contact volume change", -30, 50, 0, 5, format="%d%%",
                    help="e.g. +30% for a product launch or a bad firmware update")
    defl = st.slider("AI assistant deflection (chat + email)", 0, 50, 0, 5, format="%d%%",
                     help="Share of chats and emails the RAG assistant resolves without an agent")
    aht = st.slider("Handle time change", -30, 30, 0, 5, format="%d%%",
                    help="e.g. -10% if the triage agent pre-fills case details")
    sl = st.slider("Service level target (answered in time)", 70, 95, 80, 5, format="%d%%",
                   help="Phone within 20 s, chat within 60 s")
    shrink = st.slider("Shrinkage", 15, 45, 30, 5, format="%d%%", help="Breaks, training, leave and meetings")
    cost = st.number_input("Loaded cost per agent hour", 5.0, 100.0, 15.0, 1.0)
    st.caption("Baseline: phone AHT 7 min, chat 9 min at 2 chats per agent, email 6 min worked 8am-8pm.")

base_s = Scenario(sl_target=sl / 100, shrinkage=shrink / 100, cost_per_hour=cost)
scen_s = Scenario(volume_change=vol / 100, ai_deflection=defl / 100, aht_change=aht / 100,
                  sl_target=sl / 100, shrinkage=shrink / 100, cost_per_hour=cost)
base, scen = cached_plan(base_s), cached_plan(scen_s)
saving = base["weekly_cost"] - scen["weekly_cost"]

m = st.columns(5)
m[0].metric("Contacts next 7 days", f"{int(scen['volume'].sum().sum()):,}",
            f"{scen['volume'].sum().sum() / hist.iloc[-168:].sum().sum() - 1:+.1%} vs last week", delta_color="off")
m[1].metric("Agents needed (FTE)", f"{scen['fte']:.1f}", f"{scen['fte'] - base['fte']:+.1f} vs baseline" if saving else None,
            delta_color="inverse")
m[2].metric("Weekly staffing cost", f"{scen['weekly_cost']:,.0f}", f"{-saving:+,.0f} vs baseline" if saving else None,
            delta_color="inverse")
m[3].metric("Yearly impact", f"{saving * 52:+,.0f}" if saving else "0", help="Weekly difference × 52, positive = saving")
m[4].metric("Peak hour", f"{scen['peak']:%a %H:00}", f"{scen['peak_agents']} agents scheduled", delta_color="off")

tab_brief, tab_fc, tab_staff, tab_trust = st.tabs(["AI brief", "Demand forecast", "Staffing plan", "Accuracy & anomalies"])

# ---- AI brief ----------------------------------------------------------------------------
with tab_brief:
    facts = plan_facts(hist, fc, base, scen, scen_s, bt, anoms)
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Weekly staffing brief")
        # The LLM is only called on a click, never on every slider move, so a free-tier quota lasts.
        q = st.text_input("Ask the planner", placeholder="e.g. Which day is hardest to staff and why?")
        presets = st.columns(4)
        ask = None
        if presets[0].button("Ask", type="primary", width="stretch") and q:
            ask = q
        if presets[1].button("AI weekly brief", width="stretch"):
            ask = "brief"
        if presets[2].button("Biggest risk?", width="stretch"):
            ask = "Where is the biggest service-level risk next week and what should we do about it?"
        if presets[3].button("AI business case", width="stretch"):
            ask = "Write a short business case for the AI assistant using the scenario numbers."
        if ask:
            with st.spinner("Copilot is writing..."):
                st.session_state.answer = brief(facts, None if ask == "brief" else ask, llm())
        text, mode = st.session_state.get("answer") or brief(facts, None, LLM("mock"))
        st.markdown(text)
        st.caption(f"Mode: **{mode}**. The copilot only sees the computed facts on the right, so it can't invent numbers. "
                   "Change a what-if and click again to refresh.")
    with right:
        st.subheader("Facts the copilot sees")
        st.json(facts, expanded=False)
        fr = flat_roster_check(fc, scen_s)
        st.info(f"**Why hour-by-hour planning pays:** spreading roughly the same phone hours evenly across the day "
                f"({fr['flat_agent_hours']:,} vs {fr['plan_agent_hours']:,} agent-hours) drops phone service level from "
                f"{fr['plan_sl']:.0%} to {fr['flat_sl']:.0%}, with {fr['flat_hours_below_target']} hours under target.")

# ---- forecast ----------------------------------------------------------------------------
with tab_fc:
    lookback = hist.iloc[-14 * 24:]
    fig = go.Figure()
    for ch in CHANNELS:
        c = CH_COLORS[ch]
        fig.add_scatter(x=lookback.index, y=lookback[ch], name=f"{ch} actual", line=dict(color=c, width=1.5))
        fig.add_scatter(x=list(fc.index) + list(fc.index[::-1]), y=list(fc[f"{ch}_hi"]) + list(fc[f"{ch}_lo"][::-1]),
                        fill="toself", fillcolor=c, opacity=0.15, line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=fc.index, y=fc[ch] * (1 + vol / 100) * (1 - (defl / 100 if ch != "phone" else 0)),
                        name=f"{ch} forecast", line=dict(color=c, width=2, dash="dot"))
    fig.add_vline(x=fc.index[0], line_dash="dot", line_color=GREY)
    fig.update_layout(**LAYOUT, title="Contacts per hour: last 2 weeks and next 7 days (80% band)", height=440,
                      yaxis_title="Contacts per hour")
    st.plotly_chart(fig, width="stretch")
    st.caption("Method: a robust hour-of-week shape (median of the last 6 weeks, so one bad week can't bend it) "
               "scaled to a weekly level that grows at the median recent rate. The band is the base forecast; the dotted "
               "line includes your what-if.")
    daily = scen["volume"].resample("D").sum()
    daily.index = daily.index.strftime("%a %d %b")
    st.dataframe(daily.astype(int).assign(total=lambda d: d.sum(axis=1)), width="stretch")

# ---- staffing ----------------------------------------------------------------------------
with tab_staff:
    plan = scen["plan"]
    fig = go.Figure()
    for ch in CHANNELS:
        fig.add_bar(x=plan.index, y=plan[ch], name=f"{ch} agents", marker_color=CH_COLORS[ch])
    fig.add_scatter(x=plan.index, y=plan.scheduled, name="scheduled (incl. shrinkage)", line=dict(color=ORANGE, width=2))
    fig.update_layout(**LAYOUT, barmode="stack", bargap=0, height=440, yaxis_title="Agents",
                      title="Agents needed each hour to hit the service level")
    st.plotly_chart(fig, width="stretch")
    st.caption("Phone and chat use Erlang C: the fewest agents that answer the target share of contacts in time without "
               "pushing occupancy above 88%. Chat counts 2 conversations per agent. Email is spread across business hours. "
               "Scheduled = on-floor agents ÷ (1 − shrinkage).")
    heat = plan.scheduled.groupby([plan.index.strftime("%a %d"), plan.index.hour]).sum().unstack()
    heat = heat.reindex(pd.unique(plan.index.strftime("%a %d")))
    fig = go.Figure(go.Heatmap(z=heat.values, x=[f"{h:02d}:00" for h in heat.columns], y=heat.index,
                               colorscale=[[0, "#0d1117"], [1, BLUE]], hovertemplate="%{y} %{x}: %{z} agents<extra></extra>"))
    fig.update_layout(**LAYOUT, title="Rota view: agents to schedule by day and hour", height=340)
    st.plotly_chart(fig, width="stretch")

# ---- accuracy & anomalies ------------------------------------------------------------------
with tab_trust:
    c = st.columns(3)
    c[0].metric("Forecast error (hourly WAPE)", f"{bt['model_wape']:.1%}")
    c[1].metric("Naive baseline error", f"{bt['naive_wape']:.1%}", help="Same hour last week")
    c[2].metric("Improvement", f"{1 - bt['model_wape'] / bt['naive_wape']:.0%} lower error")
    st.write(f"Backtest: the last {bt['holdout_weeks']} weeks are hidden, forecast from the weeks before, then compared "
             "with what actually happened. A forecast is only worth using if it beats the simple guess.")
    t = bt["table"].copy()
    st.dataframe(t, hide_index=True, width="stretch", column_config={
        "model_wape_hourly": st.column_config.NumberColumn("Model error, hourly", format="percent"),
        "naive_wape_hourly": st.column_config.NumberColumn("Naive error, hourly", format="percent"),
        "model_wape_daily": st.column_config.NumberColumn("Model error, daily", format="percent"),
        "band_coverage": st.column_config.NumberColumn("Inside 80% band", format="percent")})
    st.subheader("Unusual days")
    st.write("Days far from the same weekday in nearby weeks (robust z-score ≥ 3.5 and at least 15% change). "
             "These are the events a planner should explain and prepare for next time.")
    daily_total = hist.sum(axis=1).resample("D").sum()
    fig = go.Figure(go.Scatter(x=daily_total.index, y=daily_total, line=dict(color=BLUE, width=1.5), name="contacts per day"))
    if not anoms.empty:
        fig.add_scatter(x=pd.to_datetime(anoms.date), y=anoms.contacts, mode="markers", name="flagged",
                        marker=dict(color=ORANGE, size=11, line=dict(width=2, color="#0d1117")))
    fig.update_layout(**LAYOUT, height=340, title="Daily contacts, 16 weeks", yaxis_title="Contacts per day")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(anoms, hide_index=True, width="stretch",
                 column_config={"change": st.column_config.NumberColumn("vs expected", format="percent")})
    st.caption("In this synthetic history: 31 Aug is a public holiday and 14-16 Sep is a firmware-update surge.")
