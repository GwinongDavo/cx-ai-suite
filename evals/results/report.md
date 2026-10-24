## RAG eval report (demo mode (no LLM))

Generated 2026-10-21 14:18 UTC on 36 golden questions.

| Metric | Value | Minimum |
|---|---|---|
| retrieval_hit_at_3 | 0.966 | 0.9 |
| retrieval_mrr | 0.966 | 0.85 |
| citation_precision | 0.897 | 0.8 |
| answer_keyword_recall | 0.857 | 0.7 |
| handoff_precision | 1.0 | 0.9 |
| handoff_recall | 1.0 | 1.0 |
| handoff_accuracy | 1.0 | 0.9 |
| latency_p50_ms | 0 |  |
| latency_p95_ms | 1 |  |

**Gate: PASSED**

### Cases to look at

| id | question | issue |
|---|---|---|
| q06 | What does Premier Support include? | retrieval miss |
| q09 | I spilled coffee on the keyboard, what should I do? | missing key fact |
| q13 | My laptop won't turn on at all, nothing happens | missing key fact |
| q15 | Wi-Fi keeps disconnecting every few minutes | missing key fact |
| q18 | The fan is really loud during video calls, is that normal? | missing key fact |
