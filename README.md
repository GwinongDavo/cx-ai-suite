# CX AI Suite

Five connected AI tools for a customer support (CX) team at a laptop maker, built on one shared dataset:
a knowledge base, 1,520 support tickets, 420 warranty records and 8 QA-scored calls and emails.

| # | Project | What it shows | Stack |
|---|---------|---------------|-------|
| 1 | [Support Knowledge Assistant](#1-support-knowledge-assistant) | RAG with hybrid search, citations, confidence scoring and human handoff | FastAPI, embeddings, BM25, NumPy vector index |
| 2 | [RAG Eval Harness + CI](#2-rag-eval-harness--ci) | Model evaluation as a quality gate on every push | pytest, GitHub Actions, LLM-as-judge |
| 3 | [Ticket Triage Agent](#3-ticket-triage-agent) | Tool-using agent: extraction, warranty API, KB search, policy rules, reply drafting | FastAPI, LLM JSON mode, rules engine |
| 4 | [SLA & Escalation Risk Dashboard](#4-sla--escalation-risk-dashboard) | Service KPIs plus a model that predicts SLA breaches on open tickets | Streamlit, Plotly, scikit-learn |
| 5 | [Call & Email QA Analyzer](#5-call--email-qa-analyzer) | Rubric scoring with an LLM judge, measured against human analysts | Streamlit, LLM-as-judge |

All data is **synthetic**, for a fictional brand ("Aurora" laptops). No real company or customer data is used.

**Works with any of:** Azure OpenAI, OpenAI, or Google Gemini (set one key). With no key, every app runs in
a **demo mode** that uses deterministic fallbacks (TF-IDF search, extractive answers, rule-based scoring),
so a live link never breaks because a key ran out.

```
cxcore/            shared library: LLM client, retrieval, RAG, triage, SLA model, QA scoring
apps/              the four web apps (+ shared CSS)
evals/             golden set, eval runner, thresholds, generated reports
data/              KB articles (markdown), tickets.csv, warranty.json, qa/interactions.json
scripts/           synthetic data generator
tests/             unit + API tests (run offline)
render.yaml        one-click deploy of all four apps to Render
.github/workflows  CI: tests + eval gate on every push, optional live-LLM evals
```

## Run it locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements-dev.txt
copy .env.example .env          # optional: add a key (macOS/Linux: cp)

uvicorn apps.rag_assistant.main:app --port 8001      # http://localhost:8001
uvicorn apps.triage_agent.main:app --port 8002       # http://localhost:8002
streamlit run apps/sla_dashboard/app.py              # http://localhost:8501
streamlit run apps/qa_analyzer/app.py --server.port 8502

pytest                                  # 13 tests, no API key needed
python -m evals.run_evals --gate        # eval report in evals/results/
```

To use a key locally, set it in your shell before starting an app, e.g. PowerShell:
`$env:GEMINI_API_KEY="..."`. Deploying: see [docs/DEPLOY.md](docs/DEPLOY.md).

---

## 1. Support Knowledge Assistant

A customer asks a question; the assistant answers only from the knowledge base, cites the exact sections it used,
and hands off to a human (with a ticket and context) when it isn't confident.

**How it works**
1. **Chunking:** each KB article is split by section (56 chunks), so citations point to a specific section.
2. **Hybrid retrieval:** BM25 (keyword) and vector search (cosine similarity over embeddings) each rank the chunks;
   the two rankings are merged with **reciprocal rank fusion**. Keywords catch exact terms like error codes
   (`INACCESSIBLE_BOOT_DEVICE`), embeddings catch paraphrases ("won't turn on" vs "does not power on").
3. **Confidence gate:** a score from similarity and query-term coverage. Below the threshold → human handoff, no LLM call.
4. **Grounded generation:** the LLM gets the numbered excerpts and must return JSON with `answerable` and `cited_ids`.
   Citations that weren't in the retrieved set are dropped, so the model can't invent a source.
5. **Safety override:** messages about swollen batteries, smoke or burning are always escalated as P1.

The vector index is a NumPy matrix: at 56 chunks that's faster than any external service. `VectorIndex` is the
single place where Azure AI Search or pgvector would plug in for a large corpus. Embeddings are cached to disk.

## 2. RAG Eval Harness + CI

`evals/golden.jsonl` holds 36 questions with the expected KB article, a fact the answer must contain, and whether
the right behaviour is to hand off (8 out-of-scope or safety questions).

Metrics: retrieval **hit@3** and **MRR**, **citation precision**, **answer key-fact recall**, **handoff precision/recall**,
latency p50/p95, and optional **LLM-as-judge faithfulness** (is every claim supported by the excerpts?).
The report also shows a **threshold trade-off table** (handoff precision/recall at other confidence thresholds),
which is how the threshold was chosen.

CI (`.github/workflows/ci.yml`) runs tests and the offline evals on every push and **fails the build** if any
metric drops below `evals/thresholds.json`. The report is posted to the GitHub Actions job summary and served live
at `/evals` on the assistant. A manual "live evals" run uses the real LLM (kept off normal pushes to save credits).

## 3. Ticket Triage Agent

Paste an email or chat. The agent:
1. **Extracts** serial number, category, sentiment, repeat contact, refund request, legal/media threat, safety risk
   (LLM in JSON mode; regex for the serial; rule-detected red flags can't be removed by the model).
2. **Calls the warranty API** (`/mock-api/warranty/{serial}`): plan, expiry, accidental damage cover.
3. **Searches the KB** for relevant fixes.
4. **Applies the escalation policy** as deterministic rules: priority P1–P4, queue, and reply / ask / escalate.
5. **Drafts the reply** from those facts only, or a **handoff note** for Level 2 so the customer never repeats themselves.

Design choice: the LLM does language (reading messy messages, writing replies); **rules make the decisions**.
Priority and escalation are auditable, and a model mistake can't downgrade a safety case.
The "open queue" view triages the whole backlog with the rules-only pass and reports priority accuracy
against the labelled tickets (~93%).

## 4. SLA & Escalation Risk Dashboard

KPIs (SLA attainment, first response, median resolution, CSAT), weekly breach trend, breach rate by priority,
channel and category, an agent scorecard, and an **at-risk queue**.

The risk model is a logistic regression trained only on fields known when a ticket is opened (priority, tier,
channel, category, country, queue load, hour, weekday), so it can flag tickets **before** they breach.
Holdout ROC AUC ≈ 0.83. Logistic regression was chosen because its weights are readable by a team lead.

Breach rate counts only tickets whose SLA deadline has passed (open and overdue counts as breached), to avoid
survivorship bias where recent weeks look good because the slow tickets are still open.

## 5. Call & Email QA Analyzer

Scores an interaction on six weighted criteria (verification, empathy, diagnosis, resolution accuracy,
policy & safety compliance, closing), each 0/1/2 with an evidence quote and a coaching tip. Critical failures
(unsafe advice, asking for a password, promising free repairs outside policy) cap the score at 40.

With an LLM, the judge gets the relevant KB sections so it can check facts (e.g. quoting a 5-day timeline on a P1).
The **judge agreement** tab compares automated scores with human analyst scores on the same 8 interactions
(exact agreement, within-one, mean absolute error), which is how you'd decide whether to trust an automated QA judge.

## Limitations (on purpose, and what I'd do next)

- The data is synthetic. The SLA model recovers patterns I generated; real tickets would need fresh validation and drift monitoring.
- The golden set is 36 questions and the QA calibration set is 8 interactions: enough to show the method, too small for real sign-off.
- Demo mode is a fallback, not the product: extractive answers and keyword rules are visibly weaker than the LLM path.
- Next steps: Azure AI Search for the index, prompt-injection tests in the eval set, tracing (OpenTelemetry), and feedback buttons that feed the golden set.
