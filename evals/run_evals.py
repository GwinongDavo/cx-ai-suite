"""Evaluate the RAG assistant against the golden set and gate CI on the results.

    python -m evals.run_evals                 # uses whatever provider the env selects
    python -m evals.run_evals --provider mock --gate   # what CI runs on every push
    python -m evals.run_evals --judge         # add LLM-as-judge faithfulness (needs a key)

Writes evals/results/results.json, report.md (used as the GitHub Actions summary)
and report.html (served live by the RAG app at /evals).
"""
from __future__ import annotations

import argparse
import html
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from cxcore.llm import LLM
from cxcore.rag import RagAssistant

HERE = Path(__file__).resolve().parent
JUDGE_PROMPT = """You grade a support assistant's answer for faithfulness.
Given the knowledge base excerpts and the answer, return JSON:
{"faithful": bool, "unsupported_claims": [str]}
"faithful" is true only if every factual claim in the answer is supported by the excerpts."""


def load_golden(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluate(assistant: RagAssistant, golden: list[dict], judge: bool = False) -> dict:
    rows = []
    for g in golden:
        r = assistant.answer(g["question"])
        ranked_docs = list(dict.fromkeys(h["doc_id"] for h in r.hits))
        expected = set(g["expected_docs"])
        rank = next((i + 1 for i, d in enumerate(ranked_docs) if d in expected), None)
        cited_docs = {c["id"].split("#")[0] for c in r.citations}
        row = {
            "id": g["id"], "question": g["question"], "answer": r.answer,
            "confidence": r.confidence, "handoff": r.handoff, "should_handoff": g["should_handoff"],
            "expected_docs": g["expected_docs"], "retrieved_docs": ranked_docs[:3],
            "hit_at_3": (rank is not None and rank <= 3) if expected else None,
            "reciprocal_rank": (1 / rank if rank else 0.0) if expected else None,
            "citation_ok": (bool(cited_docs) and cited_docs <= expected) if expected and r.citations else None,
            "keywords_ok": all(k.lower() in r.answer.lower() for k in g["must_include"]) if g["must_include"] else None,
            "latency_ms": r.latency_ms, "handoff_reason": r.handoff_reason,
        }
        if judge and assistant.llm.live and r.citations:
            context = "\n\n".join(h["text"] for h in r.hits)
            verdict = assistant.llm.chat_json(JUDGE_PROMPT, f"Excerpts:\n{context}\n\nAnswer:\n{r.answer}")
            row["faithful"] = bool(verdict.get("faithful"))
            row["unsupported_claims"] = verdict.get("unsupported_claims", [])
        rows.append(row)
    return {"rows": rows, "metrics": metrics(rows), "curve": threshold_curve(rows)}


def _rate(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def metrics(rows: list[dict]) -> dict:
    tp = sum(r["handoff"] and r["should_handoff"] for r in rows)
    fp = sum(r["handoff"] and not r["should_handoff"] for r in rows)
    fn = sum(not r["handoff"] and r["should_handoff"] for r in rows)
    lat = sorted(r["latency_ms"] for r in rows)
    m = {
        "n": len(rows),
        "retrieval_hit_at_3": _rate([r["hit_at_3"] for r in rows]),
        "retrieval_mrr": _rate([r["reciprocal_rank"] for r in rows]),
        "citation_precision": _rate([r["citation_ok"] for r in rows]),
        "answer_keyword_recall": _rate([r["keywords_ok"] for r in rows]),
        "handoff_precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "handoff_recall": round(tp / (tp + fn), 3) if tp + fn else None,
        "handoff_accuracy": _rate([r["handoff"] == r["should_handoff"] for r in rows]),
        "latency_p50_ms": int(statistics.median(lat)),
        "latency_p95_ms": int(lat[min(len(lat) - 1, int(0.95 * len(lat)))]),
    }
    if any("faithful" in r for r in rows):
        m["faithfulness"] = _rate([r.get("faithful") for r in rows if "faithful" in r])
    return m


def threshold_curve(rows: list[dict]) -> list[dict]:
    """What handoff precision/recall would be at other confidence thresholds (safety cases excluded)."""
    out = []
    for t in [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]:
        rel = [r for r in rows if not (r["handoff_reason"] or "").startswith("Safety")]
        pred = [r["confidence"] < t for r in rel]
        tp = sum(p and r["should_handoff"] for p, r in zip(pred, rel))
        fp = sum(p and not r["should_handoff"] for p, r in zip(pred, rel))
        fn = sum((not p) and r["should_handoff"] for p, r in zip(pred, rel))
        out.append({"threshold": t, "precision": round(tp / (tp + fp), 3) if tp + fp else 1.0,
                    "recall": round(tp / (tp + fn), 3) if tp + fn else 1.0, "false_handoffs": fp})
    return out


def check_gate(m: dict, thresholds: dict) -> list[str]:
    failures = []
    for key, minimum in thresholds.items():
        value = m.get(key)
        if value is not None and value < minimum:
            failures.append(f"{key} = {value} is below the minimum {minimum}")
    return failures


def write_reports(result: dict, mode: str, thresholds: dict, failures: list[str], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    result = {**result, "mode": mode, "generated_at": stamp, "thresholds": thresholds, "gate_failures": failures}
    (out / "results.json").write_text(json.dumps(result, indent=2))

    m = result["metrics"]
    md = [f"## RAG eval report ({mode})", "", f"Generated {stamp} on {m['n']} golden questions.", "",
          "| Metric | Value | Minimum |", "|---|---|---|"]
    md += [f"| {k} | {v} | {thresholds.get(k, '')} |" for k, v in m.items() if k != "n"]
    md += ["", "**Gate: PASSED**" if not failures else "**Gate: FAILED**\n\n" + "\n".join(f"- {f}" for f in failures)]
    misses = [r for r in result["rows"] if r["hit_at_3"] is False or r["handoff"] != r["should_handoff"] or r["keywords_ok"] is False]
    if misses:
        md += ["", "### Cases to look at", "", "| id | question | issue |", "|---|---|---|"]
        for r in misses:
            issue = []
            if r["hit_at_3"] is False: issue.append("retrieval miss")
            if r["handoff"] != r["should_handoff"]: issue.append("wrong handoff")
            if r["keywords_ok"] is False: issue.append("missing key fact")
            md.append(f"| {r['id']} | {r['question']} | {', '.join(issue)} |")
    (out / "report.md").write_text("\n".join(md) + "\n")
    (out / "report.html").write_text(render_html(result))


def render_html(result: dict) -> str:
    m, rows = result["metrics"], result["rows"]
    esc = html.escape
    def pct(v): return "n/a" if v is None else (f"{v * 100:.0f}%" if isinstance(v, float) and v <= 1 else str(v))
    def mark(v): return "" if v is None else ("✅" if v else "❌")
    cards = "".join(f'<div class="kpi"><span>{esc(k.replace("_", " "))}</span><b>{pct(v)}</b></div>'
                    for k, v in m.items() if k != "n")
    curve = "".join(f"<tr><td>{c['threshold']}</td><td>{pct(c['precision'])}</td><td>{pct(c['recall'])}</td><td>{c['false_handoffs']}</td></tr>"
                    for c in result["curve"])
    body = "".join(
        f"<tr><td>{r['id']}</td><td>{esc(r['question'])}</td><td>{r['confidence']:.2f}</td>"
        f"<td>{'handoff' if r['handoff'] else 'answer'}{'' if r['handoff'] == r['should_handoff'] else ' ❌'}</td>"
        f"<td>{mark(r['hit_at_3'])}</td><td>{mark(r['citation_ok'])}</td><td>{mark(r['keywords_ok'])}</td>"
        f"<td>{esc(', '.join(r['retrieved_docs']))}</td></tr>" for r in rows)
    gate = ('<p class="ok">Gate passed</p>' if not result["gate_failures"] else
            '<p class="bad">Gate failed: ' + esc("; ".join(result["gate_failures"])) + "</p>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAG Eval Report</title><link rel="stylesheet" href="/static/suite.css"></head><body><main class="wrap">
<p class="eyebrow"><a href="/">← Support assistant</a></p>
<h1>RAG eval report</h1><p class="muted">Mode: {esc(result['mode'])} · {esc(result['generated_at'])} · {m['n']} golden questions · regenerated by CI on every push</p>
{gate}<div class="kpis">{cards}</div>
<h2>Handoff threshold trade-off</h2><p class="muted">Lower thresholds answer more questions but risk answering ones the KB can't support.</p>
<div class="table"><table><tr><th>Threshold</th><th>Handoff precision</th><th>Handoff recall</th><th>False handoffs</th></tr>{curve}</table></div>
<h2>Per-question results</h2><div class="table"><table><tr><th>id</th><th>Question</th><th>Conf.</th><th>Decision</th><th>Hit@3</th><th>Citations</th><th>Key fact</th><th>Top docs</th></tr>{body}</table></div>
</main></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", help="mock | openai | azure | gemini (default: auto-detect)")
    ap.add_argument("--golden", default=str(HERE / "golden.jsonl"))
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--gate", action="store_true", help="exit 1 if any metric is below thresholds.json")
    ap.add_argument("--judge", action="store_true", help="add LLM-as-judge faithfulness scoring")
    args = ap.parse_args()

    llm = LLM(args.provider) if args.provider else LLM()
    assistant = RagAssistant(llm)
    result = evaluate(assistant, load_golden(Path(args.golden)), judge=args.judge)
    all_thresholds = json.loads((HERE / "thresholds.json").read_text())
    thresholds = all_thresholds["live" if llm.live else "mock"]
    failures = check_gate(result["metrics"], thresholds)
    write_reports(result, llm.label, thresholds, failures, Path(args.out))
    print(json.dumps(result["metrics"], indent=2))
    if failures:
        print("\nGATE FAILED:\n  " + "\n  ".join(failures))
    return 1 if (args.gate and failures) else 0


if __name__ == "__main__":
    sys.exit(main())
