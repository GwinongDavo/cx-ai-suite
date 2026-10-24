"""Support Knowledge Assistant: a FastAPI app over the RAG pipeline in cxcore.rag.

    uvicorn apps.rag_assistant.main:app --reload
"""
from __future__ import annotations

import itertools
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cxcore import ROOT
from cxcore.config import PORTFOLIO_URL, REPO_URL
from cxcore.rag import RagAssistant

HERE = Path(__file__).resolve().parent
assistant = RagAssistant()
_case_ids = itertools.count(880001)

app = FastAPI(title="Aurora Support Knowledge Assistant",
              description="RAG over a support knowledge base with citations and human handoff.")
app.mount("/static", StaticFiles(directory=ROOT / "apps" / "shared_static"), name="static")


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=600, examples=["My battery stops charging at 80%"])


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (HERE / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "mode": assistant.llm.label, "vector_backend": assistant.retriever.backend,
            "chunks": len(assistant.retriever.chunks), "handoff_threshold": assistant.threshold,
            "repo_url": REPO_URL, "portfolio_url": PORTFOLIO_URL}


@app.post("/api/ask")
def ask(q: Question) -> dict:
    result = assistant.answer(q.question).to_dict()
    if result["handoff"]:
        # What a real system would push to the ticketing tool: everything the next agent
        # needs so the customer never repeats themselves (KB-015 handoff note).
        result["handoff_ticket"] = {
            "case_id": f"CS-{next(_case_ids)}",
            "priority": result["priority"] or "P3",
            "summary": q.question,
            "reason": result["handoff_reason"],
            "kb_checked": [h["id"] for h in result["hits"][:3]],
        }
    return result


@app.get("/api/kb/{chunk_id}")
def kb_chunk(chunk_id: str) -> dict:
    for c in assistant.retriever.chunks:
        if c.id == chunk_id:
            return c.to_dict()
    raise HTTPException(404, "Unknown chunk")


@app.get("/evals", response_class=HTMLResponse)
def evals():
    report = ROOT / "evals" / "results" / "report.html"
    if not report.exists():
        return HTMLResponse("<p>No eval report yet. Run <code>python -m evals.run_evals</code>.</p>", 404)
    return FileResponse(report)
