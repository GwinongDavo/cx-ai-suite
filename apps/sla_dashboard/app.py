"""SLA & Escalation Risk Dashboard (Streamlit).

    streamlit run apps/sla_dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # so `cxcore` imports on Streamlit Cloud

import pandas as pd
import plotly.express as px
import streamlit as st

from cxcore.config import PORTFOLIO_URL, REPO_URL
from cxcore.sla import NOW, breach_by, kpis, load_tickets, matured, open_risk, train_risk_model

# Dark-mode categorical slots from the suite's validated palette.
BLUE, ORANGE, AQUA, YELLOW = "#3987e5", "#d95926", "#199e70", "#c98500"
STATE_COLORS = {"Breached": "#e66767", "Due < 12h": YELLOW, "High risk": "#9085e9", "On track": AQUA}
LAYOUT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
              margin=dict(l=10, r=10, t=40, b=10), font=dict(size=13), hoverlabel=dict(font_size=13))

st.set_page_config(page_title="SLA & Escalation Risk", page_icon="📈", layout="wide")


@st.cache_data
def data() -> pd.DataFrame:
    return load_tickets()


@st.cache_resource
def model():
    return train_risk_model(data())


df_all = data()
st.caption("CX AI SUITE · PROJECT 4")
st.title("SLA & Escalation Risk Dashboard")
st.write("Service health for a support desk, plus a model that flags open tickets likely to breach SLA "
         f"before they do. Synthetic data for a fictional laptop brand, as of {NOW:%d %b %Y %H:%M}. "
         f"[Source]({REPO_URL}) · [Portfolio]({PORTFOLIO_URL})")

with st.container():
    c1, c2, c3, c4 = st.columns(4)
    days = c1.selectbox("Period", [30, 60, 90, 120], index=3, format_func=lambda d: f"Last {d} days")
    channels = c2.multiselect("Channel", sorted(df_all.channel.unique()), placeholder="All channels")
    tiers = c3.multiselect("Customer tier", sorted(df_all.customer_tier.unique()), placeholder="All tiers")
    products = c4.multiselect("Product", sorted(df_all["product"].unique()), placeholder="All products")

df = df_all[df_all.created_at >= NOW - pd.Timedelta(days=days)]
for col, sel in (("channel", channels), ("customer_tier", tiers), ("product", products)):
    if sel:
        df = df[df[col].isin(sel)]
if df.empty:
    st.warning("No tickets match these filters.")
    st.stop()

k = kpis(df)
m = st.columns(6)
m[0].metric("Tickets", f"{k['tickets']:,}")
m[1].metric("Open now", k["open"])
m[2].metric("Resolution SLA met", f"{k['resolution_sla_met']:.1%}")
m[3].metric("First response SLA met", f"{k['first_response_sla_met']:.1%}")
m[4].metric("Median resolution", f"{k['median_resolution_h']:.0f} h")
m[5].metric("CSAT (1-5)", f"{k['csat']:.2f}")

tab_overview, tab_risk, tab_model = st.tabs(["Overview", "At-risk queue", "Risk model"])

with tab_overview:
    left, right = st.columns([3, 2])
    weekly = (matured(df).groupby("week")
              .agg(tickets=("ticket_id", "count"), breach_rate=("resolution_breached", "mean")).reset_index())
    fig = px.line(weekly, x="week", y="breach_rate", markers=True, title="Resolution SLA breach rate by week",
                  labels={"week": "Week", "breach_rate": "Breach rate"}, color_discrete_sequence=[BLUE],
                  hover_data={"tickets": True})
    fig.update_traces(line_width=2, marker_size=8)
    fig.update_layout(**LAYOUT, yaxis_tickformat=".0%")
    left.plotly_chart(fig, width="stretch")
    left.caption("Only tickets whose SLA deadline has passed are counted (open and overdue = breached), "
                 "so recent weeks aren't flattered by the slow tickets still being open.")

    vol = df.groupby("week").size().reset_index(name="tickets")
    fig = px.bar(vol, x="week", y="tickets", title="Ticket volume by week", color_discrete_sequence=[BLUE],
                 labels={"week": "Week", "tickets": "Tickets"})
    fig.update_layout(**LAYOUT, bargap=0.15)
    right.plotly_chart(fig, width="stretch")
    right.caption("The August spike is a simulated bad firmware push: blue screens and no-power tickets jump, and breaches follow.")

    cols = st.columns(3)
    for col, (field, title) in zip(cols, [("priority", "Breach rate by priority"), ("channel", "Breach rate by channel"),
                                          ("category", "Breach rate by category")]):
        b = breach_by(df, field)
        fig = px.bar(b.sort_values("breach_rate"), x="breach_rate", y=field, orientation="h", title=title,
                     color_discrete_sequence=[BLUE], hover_data={"tickets": True}, text_auto=".0%",
                     labels={"breach_rate": "Breach rate", field: ""})
        fig.update_layout(**LAYOUT, xaxis_tickformat=".0%", height=360 if field != "category" else 420)
        col.plotly_chart(fig, width="stretch")

    agents = (df[df.status == "resolved"].groupby("agent")
              .agg(tickets=("ticket_id", "count"), breach_rate=("resolution_breached", "mean"),
                   median_h=("age_h", "median"), csat=("csat", lambda s: pd.to_numeric(s, errors="coerce").mean()))
              .sort_values("breach_rate", ascending=False).reset_index())
    st.subheader("Agent scorecard")
    st.dataframe(agents, hide_index=True, width="stretch", column_config={
        "breach_rate": st.column_config.ProgressColumn("Breach rate", format="percent", min_value=0, max_value=1),
        "median_h": st.column_config.NumberColumn("Median resolution (h)", format="%.0f"),
        "csat": st.column_config.NumberColumn("CSAT", format="%.2f")})

with tab_risk:
    rm = model()
    o = open_risk(df_all, rm)
    o = o[o.index.isin(df.index)] if (channels or tiers or products) else o
    s = o.state.value_counts()
    c = st.columns(4)
    for i, name in enumerate(["Breached", "Due < 12h", "High risk", "On track"]):
        c[i].metric(name, int(s.get(name, 0)))
    st.write("Open tickets sorted by time left on the resolution SLA. **Risk** is the model's probability "
             "that the ticket breaches, scored from what was known when it was opened.")
    st.dataframe(
        o[["ticket_id", "state", "risk", "hours_to_due", "priority", "customer_tier", "channel", "category", "agent", "subject"]],
        hide_index=True, width="stretch", column_config={
            "risk": st.column_config.ProgressColumn("Breach risk", format="percent", min_value=0, max_value=1),
            "hours_to_due": st.column_config.NumberColumn("Hours to SLA due", format="%.1f"),
            "state": st.column_config.TextColumn("State")})
    fig = px.scatter(o, x="hours_to_due", y="risk", color="state", color_discrete_map=STATE_COLORS,
                     hover_data=["ticket_id", "priority", "category"], title="Open tickets: time left vs breach risk",
                     labels={"hours_to_due": "Hours to SLA due (negative = already breached)", "risk": "Breach risk"})
    fig.update_traces(marker=dict(size=11, line=dict(width=2, color="#0d1117")))
    fig.add_vline(x=0, line_dash="dot", line_color="#8b98a8")
    fig.update_layout(**LAYOUT, yaxis_tickformat=".0%", legend_title_text="")
    st.plotly_chart(fig, width="stretch")

with tab_model:
    rm = model()
    c = st.columns(3)
    c[0].metric("ROC AUC (25% holdout)", f"{rm.auc:.3f}")
    c[1].metric("Average precision", f"{rm.avg_precision:.3f}", help="Area under the precision-recall curve")
    c[2].metric("Base breach rate", f"{rm.base_rate:.1%}", help="What a model that guesses at random would score on average precision")
    st.write("Logistic regression with one-hot categorical features and scaled numeric features "
             "(queue load, hour of day, weekday). Chosen over a tree model because the weights are readable: "
             "a team lead can see *why* a ticket is flagged. Class weights are balanced because breaches are the minority class.")
    d = rm.top_drivers.sort_values("weight")
    d["direction"] = d.weight.map(lambda w: "Raises risk" if w > 0 else "Lowers risk")
    fig = px.bar(d, x="weight", y="feature", orientation="h", color="direction",
                 color_discrete_map={"Raises risk": ORANGE, "Lowers risk": BLUE},
                 title="Top 10 model weights (log-odds)", labels={"weight": "Weight", "feature": ""})
    fig.update_layout(**LAYOUT, legend_title_text="", height=420)
    st.plotly_chart(fig, width="stretch")
    st.caption("Caveat: the data is synthetic, generated with known breach drivers (priority, channel, region staffing, "
               "queue load, agent skill). The model recovering them shows the pipeline works; real ticket data would need "
               "its own validation and drift monitoring.")
