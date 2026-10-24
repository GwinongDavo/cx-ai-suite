"""Call and email QA scoring against a weighted rubric, with an LLM judge grounded in the KB.

Each criterion is scored 0 (missed), 1 (partial) or 2 (met), with a quote from the
transcript as evidence. Some failures are "critical" (a safety miss, asking for a
password, promising something outside policy) and cap the whole score, the way real
QA forms auto-fail a call.

`agreement()` compares the scorer with the human QA scores in data/qa, which is how
you would decide whether to trust an LLM judge before rolling it out.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from . import DATA_DIR
from .llm import LLM
from .retrieval import Retriever

RUBRIC = {
    "greeting_verification": {"weight": 15, "label": "Greeting & verification",
        "guide": "Greets, gives their name, and verifies identity with account email AND serial number before discussing the case (KB-013)."},
    "empathy": {"weight": 15, "label": "Empathy",
        "guide": "Acknowledges the customer's situation or frustration in their own words, not just a scripted 'sorry'."},
    "diagnosis": {"weight": 15, "label": "Diagnosis",
        "guide": "Asks the right questions or uses the details given to identify the actual issue and its severity."},
    "resolution_accuracy": {"weight": 25, "label": "Resolution accuracy",
        "guide": "Steps, policies, timelines and priority given match the knowledge base. Wrong SLA or missing a required step lowers the score."},
    "policy_compliance": {"weight": 20, "label": "Policy & safety compliance",
        "guide": "No unauthorised refunds or free repairs, never asks for passwords or full card numbers, safety issues (swollen battery, smoke) handled as P1 with stop-using advice, escalations follow KB-015."},
    "closing": {"weight": 10, "label": "Closing & next steps",
        "guide": "Gives a case number, clear next steps and timelines, and checks if anything else is needed."},
}
CRITICAL_CAP = 40  # max overall score when a critical failure is found

JUDGE_PROMPT = """You are a senior QA analyst for a laptop brand's customer support team.
Score the interaction against each rubric criterion: 0 = missed, 1 = partially met, 2 = fully met.
Use the knowledge base excerpts to check facts, policies, SLAs and safety handling. Be strict and fair.
For each criterion give a short evidence quote from the transcript (or "none") and a one-line coaching tip.
Flag critical failures: safety risk not escalated or customer told something unsafe; asking for a password or
full card number; promising refunds/credits/free repairs outside policy.

Rubric:
%s

Return JSON:
{"scores": {"<criterion>": {"score": 0|1|2, "evidence": str, "tip": str}},
 "critical_failures": [str], "summary": str}"""


@dataclass
class QAResult:
    scores: dict          # criterion -> {score, evidence, tip}
    critical_failures: list[str]
    overall: int          # 0..100
    summary: str
    mode: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def load_interactions() -> list[dict]:
    return json.loads((DATA_DIR / "qa" / "interactions.json").read_text())


def overall_score(scores: dict, critical: list[str]) -> int:
    total = sum(RUBRIC[c]["weight"] * scores[c]["score"] / 2 for c in RUBRIC)
    return int(round(min(total, CRITICAL_CAP) if critical else total))


class QAScorer:
    def __init__(self, llm: LLM | None = None, retriever: Retriever | None = None):
        self.llm = llm or LLM()
        self.retriever = retriever or Retriever(self.llm)

    def score(self, transcript: str) -> QAResult:
        if self.llm.live:
            try:
                return self._llm_score(transcript)
            except Exception as exc:
                print(f"[llm] call failed, using fallback: {exc!r}", flush=True)
                pass  # fall back to rules rather than fail the demo
        return self._rule_score(transcript)

    def _llm_score(self, transcript: str) -> QAResult:
        customer_text = " ".join(l.split(":", 1)[1] for l in transcript.splitlines() if l.lower().startswith("customer:"))
        kb = self.retriever.search(customer_text or transcript[:600], k=4)
        kb += self.retriever.search("priority SLA escalation policy safety", k=1)
        rubric = "\n".join(f"- {k} ({v['weight']}%): {v['guide']}" for k, v in RUBRIC.items())
        context = "\n\n".join(f"[{h.chunk.id}] {h.chunk.title} > {h.chunk.section}\n{h.chunk.text}" for h in kb)
        data = self.llm.chat_json(JUDGE_PROMPT % rubric, f"Knowledge base:\n{context}\n\nInteraction:\n{transcript}")
        scores = {}
        for c in RUBRIC:
            s = data.get("scores", {}).get(c, {})
            scores[c] = {"score": max(0, min(2, int(s.get("score", 0)))), "evidence": s.get("evidence", ""), "tip": s.get("tip", "")}
        critical = [str(x) for x in data.get("critical_failures", []) if x]
        return QAResult(scores, critical, overall_score(scores, critical), data.get("summary", ""), self.llm.label)

    # ---- demo mode: transparent keyword rules -------------------------------------------
    @staticmethod
    def _rule_score(t: str) -> QAResult:
        agent = "\n".join(l for l in t.splitlines() if not l.lower().startswith("customer:"))
        a, low = agent.lower(), t.lower()
        found = lambda pat, text=a: re.search(pat, text, re.I)  # noqa: E731
        def quote(pat):
            m = found(pat)
            if not m:
                return "none"
            line = next((l for l in agent.splitlines() if re.search(pat, l, re.I)), m.group(0))
            return line.strip()[:160]

        scores, critical = {}, []
        verified = found(r"verif|confirm the email|account email|email on your account")
        serial_asked = found(r"serial")
        named = found(r"my name is|this is \w+|\w+ speaking|i'm \w+ from|, \w+$")
        scores["greeting_verification"] = {"score": (1 if named or verified else 0) + (1 if verified and serial_asked else 0),
            "evidence": quote(r"verif|serial|speaking|my name is|this is"), "tip": "Verify account email and serial before discussing the case."}
        emp = len(re.findall(r"sorry|understand|frustrat|worr|disruptive|awful|stressful|that would", a))
        scores["empathy"] = {"score": min(2, emp), "evidence": quote(r"sorry|understand|frustrat|worr|awful|stressful"),
            "tip": "Acknowledge the impact on the customer in their words before troubleshooting."}
        questions = len(re.findall(r"\?", agent))
        scores["diagnosis"] = {"score": 2 if questions >= 3 else (1 if questions >= 1 else 0), "evidence": quote(r"\?"),
            "tip": "Ask one or two targeted questions to confirm the issue and severity."}

        safety_issue = found(r"bulg|swoll|swell|smoke|burning", low)
        fleet = found(r"\b\d{2,} .*(laptops|devices|edge x1s)|multiple devices", low)
        acc, acc_tip = 2, "Matches the knowledge base."
        if safety_issue and not found(r"stop using|unplug"):
            acc, acc_tip = 0, "Swollen battery / burning smell: stop using, unplug, escalate P1 (KB-001)."
            critical.append("Safety risk not handled as P1")
        if fleet and found(r"5 days|five days|standard timeframe"):
            acc, acc_tip = 0, "Multiple business devices down is P1: 1 hour response and Level 2 escalation (KB-015)."
        if found(r"cracked|dropped|spill", low) and found(r"claim") and not found(r"photo"):
            acc, acc_tip = 1, "ADP claims need a description, date and photos of the damage (KB-003)."
        if found(r"take (the )?(bottom )?panel off|open the (laptop|device)|remove the battery"):
            acc = 0
            critical.append("Told the customer to open the device")
        scores["resolution_accuracy"] = {"score": acc, "evidence": quote(r"\b(go to|open|change|switch|roll back|you're within|covers|update)\b"), "tip": acc_tip}

        comp, comp_tip = 2, "No compliance issues found."
        if found(r"password"):
            comp, comp_tip = 0, "Never ask for a Windows password or full card number (KB-013)."
            critical.append("Asked for the customer's password")
        if found(r"(repair|fix|replace) it for free|free (repair|replacement)|goodwill|full refund anyway") and found(r"expired|warranty ended", low):
            comp, comp_tip = 0, "Free repairs on expired warranty need supervisor approval (KB-002)."
            critical.append("Promised a free repair outside policy")
        if safety_issue and not found(r"stop using|unplug"):
            comp = 0
        if fleet and found(r"5 days|five days|standard timeframe"):
            comp, comp_tip = 1, "P1 cases must be escalated immediately (KB-015)."
        scores["policy_compliance"] = {"score": comp, "evidence": quote(r"password|free|goodwill|fine") if comp < 2 else "none", "tip": comp_tip}

        close = (1 if found(r"cs-\d{6}|case number|case cs") else 0) + (1 if found(r"anything else|if .* again|reply to this|next|within \d|business days") else 0)
        scores["closing"] = {"score": close, "evidence": quote(r"cs-\d{6}|anything else|reply"), "tip": "Give the case number, next steps and timeline, then check for anything else."}
        summary = "Rule-based demo scoring (keyword checks); connect an LLM for judgement-based scoring with KB fact checks."
        return QAResult(scores, critical, overall_score(scores, critical), summary, "demo mode (rules)")


def agreement(scorer: QAScorer, items: list[dict] | None = None) -> dict:
    """How closely the scorer matches human QA analysts."""
    items = items or load_interactions()
    rows, exact, within1, n = [], 0, 0, 0
    per_crit = {c: [] for c in RUBRIC}
    for it in items:
        r = scorer.score(it["transcript"])
        human = it["human_scores"]
        # Analysts mark a critical failure by giving policy_compliance a 0, so apply the same cap.
        human_overall = overall_score({c: {"score": human[c]} for c in RUBRIC}, ["human"] if human["policy_compliance"] == 0 else [])
        for c in RUBRIC:
            d = abs(r.scores[c]["score"] - human[c])
            exact += d == 0; within1 += d <= 1; n += 1
            per_crit[c].append(d == 0)
        rows.append({"id": it["id"], "title": it["title"], "model_score": r.overall, "human_score": human_overall,
                     "critical": "; ".join(r.critical_failures)})
    mae = sum(abs(r["model_score"] - r["human_score"]) for r in rows) / len(rows)
    return {"rows": rows, "exact_agreement": exact / n, "within_one": within1 / n, "overall_mae": mae,
            "per_criterion": {c: sum(v) / len(v) for c, v in per_crit.items()}}
