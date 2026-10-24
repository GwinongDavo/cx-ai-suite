# Interview notes: CX AI Suite

How to explain each project in about 60 seconds, the numbers to remember, and questions you're likely to get.
Run each app and read its code once before the interview; every claim below is something you can show.

## The one-line pitch

"I built five connected tools for a laptop maker's support team on one synthetic dataset: a RAG assistant that
cites its sources and knows when to hand off, an eval harness that gates CI on answer quality, a triage agent that
checks warranty and applies the escalation policy, an SLA dashboard that predicts breaches, and a QA analyzer that
I measured against human scores. They run on Azure OpenAI, OpenAI or Gemini, and fall back to a demo mode with no key."

Tie it to your CX experience: the escalation rules, SLA targets and QA rubric are modelled on how real support
operations work (priority levels, first-response targets, verification before discussing a case).

## Numbers to remember

| | |
|---|---|
| Knowledge base | 16 articles → 56 section chunks |
| Golden eval set | 36 questions (28 answerable, 8 should hand off) |
| Offline eval | hit@3 0.97, citation precision 0.90, handoff precision/recall 1.0 |
| Triage priority accuracy | ~93% vs labelled tickets (rules-only pass) |
| SLA risk model | logistic regression, holdout ROC AUC ≈ 0.83, base breach rate 19% |
| QA judge agreement (rules) | 73% exact, 96% within one point, MAE ≈ 8 points |

Re-run the live evals with your key and learn those numbers too, they're the ones that matter.

## 1. Support Knowledge Assistant

**Story:** retrieve → confidence gate → grounded answer with citations → handoff if unsure.

- *Why hybrid search?* Keywords catch exact strings like error codes; embeddings catch paraphrases. Reciprocal rank
  fusion merges the two rankings without having to calibrate their scores against each other.
- *Why chunk by section?* So a citation points to the exact paragraph an agent or customer can check.
- *How do you stop hallucinated citations?* The model must return `cited_ids`; anything not in the retrieved set is
  dropped, and an answer with no valid citation counts as "not answerable" and hands off.
- *How did you pick the handoff threshold?* From the threshold table in the eval report: the lowest threshold that
  keeps handoff recall at 100% (never answer what the KB can't support) with the fewest false handoffs.
- *Why NumPy instead of a vector database?* 56 chunks fit in memory and search in microseconds. `VectorIndex` is the
  seam where Azure AI Search or pgvector would go when the corpus grows.

## 2. Eval harness + CI

- *What do you measure?* Retrieval (hit@3, MRR), citations, key-fact recall, handoff precision/recall, latency,
  and faithfulness via LLM-as-judge.
- *Why gate CI?* A prompt tweak or a new KB article can quietly break retrieval. The build fails if any metric drops
  below its threshold, the same way unit tests protect code.
- *Why are live evals manual?* They cost API credits and are non-deterministic; the offline gate runs on every push.
- *Weakness?* 36 questions is small, and LLM-as-judge has its own bias. Next: grow the set from real handoffs and
  spot-check the judge against human labels (the QA analyzer shows how).

## 3. Ticket triage agent

- *What makes it an agent?* It reads the message, decides what it's missing, calls tools (warranty API, KB search),
  and produces an action: reply, ask for info, or escalate with a handoff note.
- *Why are decisions rules, not the LLM?* Priority and escalation must be auditable and consistent. The LLM can add a
  red flag but can never remove one the rules found, so a safety case can't be downgraded.
- *How would you run it in production?* Behind the ticketing system's webhook, human approval on drafted replies at
  first, and logging every trace step for audit.

## 4. SLA & escalation risk dashboard

- *Features?* Only what's known at ticket creation, so there's no leakage from the outcome.
- *Why logistic regression?* Readable weights a team lead can act on; AUC 0.83 is good enough to sort a queue.
- *Survivorship bias:* breach rate only counts tickets whose deadline has passed; otherwise recent weeks look great
  because slow tickets are still open.
- *Honest caveat:* the data is synthetic with known drivers. Real data needs validation and drift monitoring.

## 5. Call & email QA analyzer

- *How is it scored?* Six weighted criteria, 0/1/2 each, evidence quote and coaching tip per criterion; critical
  failures cap the score at 40.
- *Why compare with humans?* Before trusting an automated judge you measure agreement. Diagnosis has the lowest
  agreement in rules mode, which tells you where the prompt/rules need work.

## Questions to expect

- **"What would you change for Lenovo-scale data?"** Azure AI Search with hybrid + semantic ranking, per-locale KBs,
  caching, rate limiting, PII redaction before prompts, OpenTelemetry tracing, and a feedback loop into the golden set.
- **"How do you handle prompt injection?"** System prompt only allows answers from excerpts; citations are validated;
  decisions are rules. Next step: add injection cases to the eval set.
- **"What did you build vs generate?"** Be straightforward: you designed and built it with an AI coding assistant, and
  you can walk through every module. Make sure that's true by reading each file before the interview.
