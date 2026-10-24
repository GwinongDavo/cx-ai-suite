"""Ticket triage agent: extract -> look up warranty -> search KB -> apply policy -> draft reply.

The LLM does the language work (reading a messy email, writing a reply). Priority,
routing and escalation are deterministic rules from the escalation policy (KB-015),
so every decision is explainable and auditable, and a model mistake can't silently
downgrade a safety case.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from . import DATA_DIR
from .llm import LLM
from .retrieval import Retriever

SERIAL = re.compile(r"\b([A-Z]{2}(?=[A-Z]*[0-9])[A-Z0-9]{8})\b")  # needs a digit, so words like ACCIDENTAL never match
TODAY = date(2026, 10, 20)  # the synthetic dataset's "today"

CATEGORY_KEYWORDS = {
    "Battery safety": r"swell|swollen|bulg|lifting|smoke|burning|hiss|fire",
    "Fleet outage": r"\b\d{2,}\s+(of our|devices|laptops)|fleet|our office has|multiple .*laptops",
    "No power": r"stays black|won'?t (turn|power) on|not power on|no power|dead|black screen|nothing happens",
    "Blue screen": r"blue screen|bsod|inaccessible_boot|irql|crash",
    "Battery & charging": r"charg|battery",
    "Wi-Fi & network": r"wi-?fi|wireless|network|disconnect",
    "Display": r"flicker|pixel|lines|display|screen (is|has)",
    "Physical damage": r"drop|crack|spill|water|coffee|liquid",
    "Dock & display out": r"dock|monitor",
    "Returns & refunds": r"return|refund|money back",
    "Warranty question": r"warranty|extend|coverage",
}
CATEGORY_PRIORITY = {"Battery safety": "P1", "Fleet outage": "P1", "No power": "P2"}
FIRST_RESPONSE_H = {"P1": 1, "P2": 4, "P3": 8, "P4": 24}
RESOLUTION_H = {"P1": 24, "P2": 48, "P3": 120, "P4": 168}
NEGATIVE = r"unacceptable|angry|furious|terrible|worst|ridiculous|still no|again|third time|deadline|urgent|asap|disappointed"
LEGAL = r"lawyer|legal action|lawsuit|sue\b|court|consumer tribunal|media|journalist|twitter|post this"

EXTRACT_PROMPT = """You extract structured fields from a customer support message for a laptop brand.
Return JSON with keys:
  serial (string or null; Aurora serials are 2 letters + 8 letters/digits),
  category (one of: %s, Other),
  summary (one sentence, agent-facing),
  sentiment (number from -1 very negative to 1 very positive),
  repeat_contact (bool: the customer says they already contacted support about this),
  wants_refund (bool), legal_or_media_threat (bool), safety_risk (bool),
  multiple_devices_affected (bool)""" % ", ".join(CATEGORY_KEYWORDS)

REPLY_PROMPT = """You write the first reply to a customer for Aurora laptop support.
Use ONLY the facts given (warranty record, decision, knowledge base excerpts). Be warm, concise
(under 150 words), and concrete. Include the case number. If the decision is "escalate",
explain a specialist will take over and when they will respond; do not troubleshoot a safety
issue beyond telling them to stop using and unplug the device. If the decision is "ask",
politely ask for the missing information. Never promise refunds, credits, or free repairs that
the facts don't support. Return JSON: {"reply": str}"""


class WarrantyAPI:
    """Stands in for the OEM's warranty service. The triage app also serves it over HTTP."""

    def __init__(self) -> None:
        self.records = {r["serial"]: r for r in json.loads((DATA_DIR / "warranty.json").read_text())}

    def lookup(self, serial: str) -> dict | None:
        rec = self.records.get(serial.upper())
        if not rec:
            return None
        end = date.fromisoformat(rec["warranty_end"])
        return {**rec, "active": end >= TODAY, "days_left": (end - TODAY).days}


@dataclass
class Step:
    name: str
    detail: str
    data: dict | list | None = None
    status: str = "ok"   # ok | warn | bad
    ms: int = 0


@dataclass
class TriageResult:
    case_id: str
    decision: str        # reply | escalate | ask
    priority: str
    queue: str
    reasons: list[str]
    extracted: dict
    warranty: dict | None
    kb: list[dict]
    sla: dict
    reply: str
    handoff_note: str | None
    mode: str
    trace: list[Step] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["trace"] = [s.__dict__ for s in self.trace]
        return d


class TriageAgent:
    def __init__(self, llm: LLM | None = None, retriever: Retriever | None = None):
        self.llm = llm or LLM()
        self.retriever = retriever or Retriever(self.llm)
        self.warranty_api = WarrantyAPI()
        self._n = 0

    # --- step 1 -----------------------------------------------------------------
    def extract(self, text: str) -> dict:
        rules = self._extract_rules(text)
        if not self.llm.live:
            return rules
        try:
            data = self.llm.chat_json(EXTRACT_PROMPT, text)
        except Exception as exc:
            print(f"[llm] call failed, using fallback: {exc!r}", flush=True)
            return rules
        # Trust regex for the serial (exact pattern) and OR the safety flags, so the model
        # can add signals but never remove one the rules found.
        data["serial"] = rules["serial"] or (data.get("serial") or None)
        for flag in ("safety_risk", "legal_or_media_threat", "multiple_devices_affected", "repeat_contact", "wants_refund"):
            data[flag] = bool(data.get(flag)) or rules[flag]
        if data.get("category") not in CATEGORY_KEYWORDS:
            data["category"] = rules["category"]
        data["sentiment"] = float(data.get("sentiment", rules["sentiment"]))
        return data

    @staticmethod
    def _extract_rules(text: str) -> dict:
        low = text.lower()
        category = next((c for c, pat in CATEGORY_KEYWORDS.items() if re.search(pat, low)), "Other")
        neg = len(re.findall(NEGATIVE, low))
        m = SERIAL.search(text.upper())
        return {
            "serial": m.group(1) if m else None,
            "category": category,
            "summary": text.strip().split("\n")[0][:160],
            "sentiment": round(max(-1.0, 0.1 - 0.35 * neg - 0.2 * text.count("!")), 2),
            "repeat_contact": bool(re.search(r"again|third time|already (contacted|called|emailed)|still no|\b3 weeks\b", low)),
            "wants_refund": bool(re.search(r"refund|money back", low)),
            "legal_or_media_threat": bool(re.search(LEGAL, low)),
            "safety_risk": category == "Battery safety",
            "multiple_devices_affected": category == "Fleet outage",
        }

    # --- step 4 -----------------------------------------------------------------
    @staticmethod
    def policy(ex: dict, w: dict | None) -> tuple[str, str, str, list[str]]:
        reasons: list[str] = []
        enterprise = bool(w and w["plan"] == "Premier Support")
        priority = CATEGORY_PRIORITY.get(ex["category"], "P4" if ex["category"] in ("Warranty question", "Returns & refunds") else "P3")
        if ex["safety_risk"]:
            priority = "P1"; reasons.append("Safety risk (KB-015: P1)")
        if ex["multiple_devices_affected"] and (enterprise or (w and w["segment"] == "business")):
            priority = "P1"; reasons.append("Business customer with multiple devices down (KB-015: P1)")
        if enterprise and priority in ("P3", "P4") and ex["category"] in ("No power", "Display", "Blue screen"):
            priority = "P2"; reasons.append("Premier Support customer with a device down (KB-015: P2)")

        if ex["legal_or_media_threat"]:
            return "escalate", priority, "Escalations manager", reasons + ["Legal or media threat goes to the escalations manager"]
        if priority == "P1":
            return "escalate", priority, "Safety & critical response" if ex["safety_risk"] else "Level 2 · critical", reasons
        if not ex["serial"]:
            return "ask", priority, "Level 1", reasons + ["No serial number: needed to check warranty (KB-002)"]
        if w is None:
            return "ask", priority, "Level 1", reasons + ["Serial not found in warranty system; confirm it"]
        if ex["repeat_contact"] and ex["sentiment"] < -0.3:
            return "escalate", priority, "Level 2", reasons + ["Repeat contact with negative sentiment (KB-015)"]
        if ex["wants_refund"] and ex["category"] == "Returns & refunds" and ex["repeat_contact"]:
            return "escalate", priority, "Level 2", reasons + ["Refund chase: needs supervisor review (KB-011)"]
        if not w["active"] and ex["category"] not in ("Warranty question", "Returns & refunds"):
            reasons.append(f"Warranty expired {-w['days_left']} days ago: paid repair or extension only (KB-002)")
        reasons.append("Covered by the knowledge base: Level 1 can resolve")
        return "reply", priority, "Level 1", reasons

    # --- step 5 -----------------------------------------------------------------
    def draft(self, case_id: str, text: str, decision: str, priority: str, ex: dict, w: dict | None, kb: list) -> str:
        if self.llm.live:
            facts = {"case_id": case_id, "decision": decision, "priority": priority,
                     "first_response_within_hours": FIRST_RESPONSE_H[priority],
                     "warranty": w, "extracted": ex,
                     "kb_excerpts": [f"[{h.chunk.id}] {h.chunk.section}: {h.chunk.text}" for h in kb]}
            try:
                return self.llm.chat_json(REPLY_PROMPT, f"Customer message:\n{text}\n\nFacts:\n{json.dumps(facts, default=str)}")["reply"]
            except Exception as exc:
                print(f"[llm] call failed, using fallback: {exc!r}", flush=True)
                pass
        return self._template_reply(case_id, decision, priority, ex, w, kb)

    @staticmethod
    def _template_reply(case_id, decision, priority, ex, w, kb) -> str:
        name = (w or {}).get("owner", "").split(" ")[0] or "there"
        if decision == "ask":
            return (f"Hi {name},\n\nThanks for contacting Aurora Support (case {case_id}). So I can check your warranty "
                    "and get the right fix, could you send your device's 10-character serial number? It's on the label "
                    "under the laptop, or in Aurora Vantage > Device.\n\nKind regards,\nAurora Support")
        if decision == "escalate":
            safety = ("Please stop using the device and unplug it now. Don't try to open it or remove the battery. "
                      if ex["safety_risk"] else "")
            return (f"Hi {name},\n\nThank you for letting us know, and I'm sorry for the trouble. {safety}I've escalated "
                    f"case {case_id} as {priority} to a specialist team, who will contact you within "
                    f"{FIRST_RESPONSE_H[priority]} hour{'s' if FIRST_RESPONSE_H[priority] > 1 else ''}. They already have "
                    "your details, so you won't need to repeat anything.\n\nKind regards,\nAurora Support")
        status = ""
        if w:
            status = (f"Your {w['model']} is covered by {w['plan']} until {w['warranty_end']}. " if w["active"] else
                      f"Your {w['model']}'s warranty ended on {w['warranty_end']}, so repairs would be chargeable, "
                      "but I can arrange a quote or an extension. ")
        tip = ""
        if kb:
            sent = re.split(r"(?<=[.!?])\s+", kb[0].chunk.text.replace("\n", " "))
            tip = f"From our guide \"{kb[0].chunk.title}\": " + " ".join(sent[:2]) + " "
        return (f"Hi {name},\n\nThanks for contacting Aurora Support (case {case_id}). {status}\n\n{tip}\n\n"
                "If that doesn't fix it, just reply to this email and we'll book a repair.\n\nKind regards,\nAurora Support")

    # --- orchestration --------------------------------------------------------------
    def run(self, text: str, channel: str = "email", created_at: datetime | None = None) -> TriageResult:
        self._n += 1
        case_id = f"CS-{770000 + self._n}"
        created_at = created_at or datetime.now()
        trace: list[Step] = []

        t = time.perf_counter()
        ex = self.extract(text)
        trace.append(Step("Extract case details", f"{ex['category']} · serial {ex['serial'] or 'missing'} · sentiment {ex['sentiment']:+.2f}",
                          ex, "ok" if ex["serial"] else "warn", int((time.perf_counter() - t) * 1000)))

        t = time.perf_counter()
        w = self.warranty_api.lookup(ex["serial"]) if ex["serial"] else None
        if ex["serial"]:
            detail = (f"{w['model']} · {w['plan']} · {'active' if w['active'] else 'expired'} until {w['warranty_end']}"
                      f"{' · ADP' if w['adp'] else ''}") if w else "Serial not found"
            trace.append(Step("Tool: warranty_lookup", detail, w, "ok" if w and w["active"] else "warn", int((time.perf_counter() - t) * 1000)))
        else:
            trace.append(Step("Tool: warranty_lookup", "Skipped: no serial number in message", None, "warn"))

        t = time.perf_counter()
        hint = {"Fleet outage": "blue screen after update escalation", "Battery safety": "swollen battery safety"}.get(ex["category"], ex["category"])
        kb = self.retriever.search(f"{hint}. {text}", k=3)
        trace.append(Step("Tool: kb_search", ", ".join(h.chunk.id for h in kb), [h.to_dict() for h in kb], "ok", int((time.perf_counter() - t) * 1000)))

        decision, priority, queue, reasons = self.policy(ex, w)
        trace.append(Step("Apply escalation policy", f"{decision.upper()} · {priority} · {queue}", reasons,
                          {"reply": "ok", "ask": "warn", "escalate": "bad"}[decision]))

        sla = {"first_response_due": (created_at + timedelta(hours=FIRST_RESPONSE_H[priority])).isoformat(timespec="minutes"),
               "resolution_due": (created_at + timedelta(hours=RESOLUTION_H[priority] / (2 if w and w["plan"] == "Premier Support" else 1))).isoformat(timespec="minutes")}

        t = time.perf_counter()
        reply = self.draft(case_id, text, decision, priority, ex, w, kb)
        trace.append(Step("Draft reply", "LLM drafted from facts + KB" if self.llm.live else "Template reply (demo mode)", None, "ok", int((time.perf_counter() - t) * 1000)))

        handoff = None
        if decision == "escalate":
            handoff = (f"[{priority}] {ex['summary']}\nSerial: {ex['serial'] or 'unknown'} | Warranty: "
                       f"{(w['plan'] + (' active' if w['active'] else ' EXPIRED')) if w else 'unknown'}\n"
                       f"Why escalated: {'; '.join(reasons)}\nKB checked: {', '.join(h.chunk.id for h in kb)}\n"
                       f"Channel: {channel} | Sentiment: {ex['sentiment']:+.2f}")
        return TriageResult(case_id, decision, priority, queue, reasons, ex, w, [h.to_dict() for h in kb], sla,
                            reply, handoff, self.llm.label, trace)
