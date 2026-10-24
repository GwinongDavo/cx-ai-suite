"""Ticket Triage Agent: FastAPI app over cxcore.triage, plus the mock warranty API it calls.

    uvicorn apps.triage_agent.main:app --reload
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cxcore import DATA_DIR, ROOT
from cxcore.config import PORTFOLIO_URL, REPO_URL
from cxcore.llm import LLM
from cxcore.triage import TriageAgent

HERE = Path(__file__).resolve().parent
agent = TriageAgent()

app = FastAPI(title="Aurora Ticket Triage Agent",
              description="Reads a support message, checks warranty, searches the KB, applies escalation policy and drafts a reply.")
app.mount("/static", StaticFiles(directory=ROOT / "apps" / "shared_static"), name="static")

SAMPLES = [
    {"label": "Battery won't charge", "channel": "email",
     "text": "Hi, my Aurora Book Pro 16 says plugged in, not charging since this morning. I'm using the charger it came with. Serial DHNN4J3MT5. Thanks, Priya"},
    {"label": "Swollen battery", "channel": "chat",
     "text": "The bottom of my laptop is lifting and the trackpad is pushed up, it's hard to click. Is this safe?? serial NJ5ZLZYPTE"},
    {"label": "Fleet outage (enterprise)", "channel": "phone",
     "text": "This is Farah from Kestrel Logistics IT. About 30 of our Aurora Edge X1 laptops are blue-screening after last night's firmware push. One serial is NEV5WEA5GE. We need someone today."},
    {"label": "Expired warranty", "channel": "email",
     "text": "My Aurora Book 14 screen flickers and shows lines even in the BIOS screen. Serial XZQWDQCW1T. Can you fix it?"},
    {"label": "Angry repeat contact", "channel": "email",
     "text": "This is the third time I'm writing. Wi-Fi STILL disconnects every few minutes on my Aurora Book Pro 16 and nobody has fixed it. Unacceptable. DHNN4J3MT5"},
    {"label": "Missing serial", "channel": "chat",
     "text": "Hello, I spilled some water on my keyboard yesterday and now a few keys don't work. Do I have accidental damage cover?"},
    {"label": "Legal threat", "channel": "email",
     "text": "Returned my laptop 3 weeks ago and still no refund. If this isn't sorted by Friday I'm talking to a lawyer and posting this everywhere. Serial XZQWDQCW1T"},
]


class Ticket(BaseModel):
    text: str = Field(min_length=10, max_length=4000)
    channel: str = Field("email", pattern="^(email|chat|phone)$")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (HERE / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "mode": agent.llm.label, "repo_url": REPO_URL, "portfolio_url": PORTFOLIO_URL}


@app.get("/api/samples")
def samples() -> list[dict]:
    return SAMPLES


@app.post("/api/triage")
def triage(t: Ticket) -> dict:
    return agent.run(t.text, t.channel).to_dict()


@app.get("/mock-api/warranty/{serial}")
def warranty(serial: str) -> dict:
    """The mock OEM warranty service the agent calls as a tool."""
    rec = agent.warranty_api.lookup(serial)
    if not rec:
        raise HTTPException(404, "Serial not found")
    return rec


@lru_cache(maxsize=1)
def _queue() -> dict:
    # Bulk pass uses the rules engine only (no LLM calls), which is how you'd pre-sort a
    # backlog cheaply before an LLM drafts replies one by one.
    rules_agent = TriageAgent(LLM("mock"), agent.retriever)
    df = pd.read_csv(DATA_DIR / "tickets.csv")
    open_t = df[df.status == "open"]
    rows = []
    for _, t in open_t.iterrows():
        r = rules_agent.run(t.body, t.channel, pd.Timestamp(t.created_at).to_pydatetime())
        rows.append({"ticket_id": t.ticket_id, "created_at": t.created_at, "subject": t.subject, "channel": t.channel,
                     "labelled_priority": t.priority, "priority": r.priority, "decision": r.decision, "queue": r.queue,
                     "reason": r.reasons[-1] if r.reasons else ""})
    sample = df.sample(400, random_state=7)
    pred = [rules_agent.run(b).priority for b in sample.body]
    acc = float((sample.priority.values == pd.Series(pred).values).mean())
    return {"rows": sorted(rows, key=lambda r: (r["priority"], r["created_at"])),
            "decisions": Counter(r["decision"] for r in rows), "priority_accuracy": round(acc, 3), "n_eval": len(sample)}


@app.get("/api/queue")
def queue() -> dict:
    return _queue()
