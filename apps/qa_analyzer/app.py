"""Call & Email QA Analyzer (Streamlit).

    streamlit run apps/qa_analyzer/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # so `cxcore` imports on Streamlit Cloud

import pandas as pd
import plotly.express as px
import streamlit as st

from cxcore.config import PORTFOLIO_URL, REPO_URL
from cxcore.qa import RUBRIC, QAScorer, agreement, load_interactions

BLUE, ORANGE = "#3987e5", "#d95926"
LAYOUT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
              margin=dict(l=10, r=10, t=40, b=10), font=dict(size=13))
SCORE_ICON = {2: "✅ Met", 1: "🟡 Partial", 0: "❌ Missed"}

st.set_page_config(page_title="Call & Email QA Analyzer", page_icon="🎧", layout="wide")


@st.cache_resource
def scorer() -> QAScorer:
    return QAScorer()


@st.cache_data(show_spinner="Scoring all sample interactions…")
def batch() -> dict:
    return agreement(scorer())


items = load_interactions()
s = scorer()
st.caption("CX AI SUITE · PROJECT 5")
st.title("Call & Email QA Analyzer")
live = s.llm.live
st.write("Scores support calls, chats and emails against a weighted QA rubric, checks the agent's facts against "
         "the knowledge base, flags critical failures, and writes coaching tips. "
         f"Mode: **{'LLM judge · ' + s.llm.label if live else 'demo mode (keyword rules, no API key)'}**. "
         f"[Source]({REPO_URL}) · [Portfolio]({PORTFOLIO_URL})")

tab_one, tab_batch, tab_rubric = st.tabs(["Score an interaction", "Team view & judge agreement", "Rubric"])

with tab_one:
    left, right = st.columns([1, 1])
    with left:
        choice = st.selectbox("Sample interaction", [f"{i['id']} · {i['channel']} · {i['title']}" for i in items])
        item = items[[f"{i['id']} · {i['channel']} · {i['title']}" for i in items].index(choice)]
        uploaded = st.file_uploader("…or upload a transcript (.txt, lines starting 'Agent:' / 'Customer:')", type=["txt"])
        default = uploaded.read().decode("utf-8", "ignore") if uploaded else item["transcript"]
        text = st.text_area("Transcript", default, height=420, key=f"t-{choice}-{bool(uploaded)}")
        go = st.button("Score interaction", type="primary")
    with right:
        if go or "last" not in st.session_state or st.session_state.get("last_choice") != choice:
            with st.spinner("Scoring…"):
                st.session_state["last"] = s.score(text)
                st.session_state["last_choice"] = choice
        r = st.session_state["last"]
        c = st.columns(2)
        c[0].metric("QA score", f"{r.overall}/100")
        if not uploaded and text == item["transcript"]:
            from cxcore.qa import overall_score
            h = item["human_scores"]
            human = overall_score({k: {"score": h[k]} for k in RUBRIC}, ["x"] if h["policy_compliance"] == 0 else [])
            c[1].metric("Human analyst score", f"{human}/100", delta=f"{r.overall - human:+d} vs human", delta_color="off")
        if r.critical_failures:
            st.error("**Critical failure** (score capped at 40): " + "; ".join(r.critical_failures))
        if r.summary:
            st.caption(r.summary)
        rows = [{"Criterion": RUBRIC[k]["label"], "Weight": f"{RUBRIC[k]['weight']}%", "Result": SCORE_ICON[v["score"]],
                 "Evidence": v["evidence"], "Coaching tip": v["tip"]} for k, v in r.scores.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if not uploaded and text == item["transcript"]:
            st.info("**Human analyst note:** " + item["human_notes"])

with tab_batch:
    a = batch()
    c = st.columns(3)
    c[0].metric("Exact agreement with humans", f"{a['exact_agreement']:.0%}", help="Share of criterion scores (0/1/2) identical to the human analyst's")
    c[1].metric("Within one point", f"{a['within_one']:.0%}")
    c[2].metric("Mean abs. error (overall score)", f"{a['overall_mae']:.1f} pts")
    st.write("Before trusting an automated QA judge, measure it against human analysts on the same interactions. "
             "This is the calibration set; criteria with low agreement are where the prompt or rules need work.")
    df = pd.DataFrame(a["rows"])
    long = df.melt(id_vars=["id", "title"], value_vars=["model_score", "human_score"], var_name="scorer", value_name="score")
    long["scorer"] = long.scorer.map({"model_score": "Automated", "human_score": "Human analyst"})
    fig = px.bar(long, x="id", y="score", color="scorer", barmode="group", hover_data=["title"],
                 color_discrete_map={"Automated": BLUE, "Human analyst": ORANGE},
                 title="QA score per interaction: automated vs human", labels={"id": "", "score": "QA score", "scorer": ""})
    fig.update_layout(**LAYOUT, yaxis_range=[0, 100], bargap=0.25, bargroupgap=0.08)
    st.plotly_chart(fig, width="stretch")
    pc = pd.DataFrame({"Criterion": [RUBRIC[k]["label"] for k in a["per_criterion"]],
                       "Agreement": list(a["per_criterion"].values())})
    st.dataframe(pc, hide_index=True, width="stretch", column_config={
        "Agreement": st.column_config.ProgressColumn("Exact agreement", format="percent", min_value=0, max_value=1)})
    st.dataframe(df.rename(columns={"model_score": "Automated", "human_score": "Human", "critical": "Critical flags"}),
                 hide_index=True, width="stretch")

with tab_rubric:
    st.dataframe(pd.DataFrame([{"Criterion": v["label"], "Weight": f"{v['weight']}%", "What 'met' looks like": v["guide"]}
                               for v in RUBRIC.values()]), hide_index=True, width="stretch")
    st.write("Scores: 0 missed, 1 partial, 2 met. Weighted to 100. A critical failure (unsafe advice, asking for a "
             "password, promising refunds or free repairs outside policy) caps the score at 40.")
