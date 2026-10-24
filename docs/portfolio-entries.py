# Paste these into PROJECTS in your portfolio's data.py (after replacing the URLs),
# and copy docs/screenshots/<slug>.jpg to static/img/projects/<slug>.jpg.
# "link" is the GitHub button your template already shows. Live demo URLs are in the
# descriptions until the template gets a "Live demo" button.
REPO = "https://github.com/GwinongDavo/cx-ai-suite"

CX_PROJECTS = [
    {"slug": "support-assistant", "title": "Support Knowledge Assistant (RAG)", "file": "rag.py", "category": "ai",
     "description": "A support assistant that answers from a knowledge base with citations and hands off to a human when unsure.",
     "long_description": (
         "Hybrid retrieval (BM25 + embeddings, merged with reciprocal rank fusion) over a sectioned knowledge base, "
         "a confidence gate tuned on an eval set, and grounded answers whose citations are validated so the model "
         "can't invent a source. Safety issues are always escalated as P1 with a handoff ticket. "
         "Runs on Azure OpenAI, OpenAI or Gemini."),
     "tags": ["Python", "FastAPI", "RAG", "Embeddings", "Azure OpenAI"], "link": REPO},
    {"slug": "rag-evals", "title": "RAG Eval Harness + CI", "file": "run_evals.py", "category": "ai",
     "description": "Automated evaluation that fails the build when answer quality drops.",
     "long_description": (
         "A 36-question golden set scored on retrieval hit@3 and MRR, citation precision, key-fact recall, "
         "handoff precision and recall, latency, and LLM-as-judge faithfulness. GitHub Actions runs it on every "
         "push as a quality gate and publishes the report; a threshold trade-off table shows how the handoff "
         "threshold was chosen."),
     "tags": ["Python", "Model evaluation", "GitHub Actions", "pytest"], "link": REPO},
    {"slug": "triage-agent", "title": "Ticket Triage Agent", "file": "triage.py", "category": "ai",
     "description": "An agent that reads a support email, checks warranty, applies the escalation policy and drafts the reply.",
     "long_description": (
         "Extracts case details with an LLM in JSON mode, calls a warranty API and knowledge base search as tools, "
         "then applies escalation rules (P1 to P4, routing, reply / ask / escalate) and drafts a grounded reply or a "
         "Level 2 handoff note. The LLM handles language; auditable rules make the decisions. Every step is shown "
         "in an agent trace. Matches labelled ticket priority about 93% of the time."),
     "tags": ["Python", "FastAPI", "AI agents", "Tool calling"], "link": REPO},
    {"slug": "sla-dashboard", "title": "SLA & Escalation Risk Dashboard", "file": "sla.py", "category": "data",
     "description": "Support KPIs plus a model that flags open tickets likely to breach SLA before they do.",
     "long_description": (
         "SLA attainment, first response, resolution time, CSAT, breach trends and an agent scorecard over 1,500 "
         "tickets, with a logistic regression risk model (holdout ROC AUC 0.83) trained only on what's known at "
         "ticket creation. Breach rates count only matured tickets to avoid survivorship bias."),
     "tags": ["Python", "Streamlit", "scikit-learn", "Plotly"], "link": REPO},
    {"slug": "qa-analyzer", "title": "Call & Email QA Analyzer", "file": "qa.py", "category": "ai",
     "description": "Scores support calls and emails against a QA rubric with evidence, coaching tips and critical-failure flags.",
     "long_description": (
         "An LLM judge grounded in the knowledge base scores six weighted criteria with evidence quotes and "
         "coaching tips, and flags critical failures like unsafe advice or asking for a password. Scores are "
         "measured against human analysts (agreement and error) before the judge is trusted."),
     "tags": ["Python", "Streamlit", "LLM-as-judge", "QA"], "link": REPO},
]
