"""RAG answer pipeline: retrieve -> decide if answerable -> grounded answer with citations.

The assistant hands off to a human when:
  1. retrieval confidence is below the threshold (the KB probably does not cover it),
  2. the model itself says the context does not answer the question, or
  3. the question contains a safety signal (swollen battery, smoke...), which is
     always escalated as P1 even when the KB has advice to give.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

from .llm import LLM
from .retrieval import Hit, Retriever, tokenize

HANDOFF_THRESHOLD = float(os.getenv("CX_HANDOFF_THRESHOLD", "0.30"))

SAFETY_PATTERNS = re.compile(
    r"swell|swollen|bulg|smok|burning smell|burnt|fire|flame|spark|hiss|explod|melt", re.I)

# KB articles mix customer advice with internal agent guidance. Demo mode is extractive,
# so it drops internal lines and rewrites "the customer" into the second person.
INTERNAL_LINE = re.compile(r"escalat|agents? (must|can|should)|supervisor|never ask the customer|internal", re.I)


def to_customer_voice(sentence: str) -> str:
    s = re.sub(r"\b(tell|ask) the customer (to|for) ", "", sentence, flags=re.I)
    s = re.sub(r"\bthe customer's\b", "your", s)
    s = re.sub(r"\b(the )?customers?\b", "you", s)
    return s[:1].upper() + s[1:]


SYSTEM_PROMPT = """You are a customer support assistant for Aurora laptops.
Answer ONLY from the numbered knowledge base excerpts provided. Rules:
- If the excerpts do not contain the answer, set "answerable" to false and do not guess.
- Keep the answer under 120 words, in plain, friendly language, with numbered steps when there are steps.
- Cite every excerpt you used by its id, e.g. "KB-001#3".
- Never promise refunds, credits, or free repairs that the excerpts do not allow.
Return JSON: {"answerable": bool, "answer": str, "cited_ids": [str]}"""


@dataclass
class RagResult:
    question: str
    answer: str
    citations: list[dict]
    confidence: float
    handoff: bool
    handoff_reason: str | None
    priority: str | None
    mode: str
    latency_ms: int
    hits: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class RagAssistant:
    def __init__(self, llm: LLM | None = None, retriever: Retriever | None = None,
                 threshold: float = HANDOFF_THRESHOLD):
        self.llm = llm or LLM()
        self.retriever = retriever or Retriever(self.llm)
        self.threshold = threshold

    def answer(self, question: str, k: int = 4) -> RagResult:
        start = time.perf_counter()
        hits = self.retriever.search(question, k=k)
        confidence = self.retriever.confidence(question, hits)
        safety = bool(SAFETY_PATTERNS.search(question))

        answerable, answer, cited = False, "", []
        if confidence >= self.threshold or safety:
            if self.llm.live:
                answerable, answer, cited = self._generate(question, hits)
            else:
                answerable, answer, cited = self._extractive(question, hits)

        handoff, reason, priority = False, None, None
        if safety:
            handoff, reason, priority = True, "Safety risk detected: escalated as P1 per KB-015.", "P1"
            answer = (answer + "\n\n" if answer else "") + (
                "For your safety, please stop using the device and unplug it now. "
                "I've flagged this as urgent and a specialist will contact you.")
        elif confidence < self.threshold:
            handoff, reason = True, f"Low retrieval confidence ({confidence:.2f} < {self.threshold:.2f})."
        elif not answerable:
            handoff, reason = True, "The knowledge base excerpts do not answer this question."

        if handoff and not answer:
            answer = ("I don't want to guess on this one. I'm passing your question to a support "
                      "specialist with the details so you won't need to repeat yourself.")

        by_id = {h.chunk.id: h for h in hits}
        citations = [{"id": cid, "citation": by_id[cid].chunk.citation} for cid in cited if cid in by_id]
        return RagResult(
            question=question, answer=answer.strip(), citations=citations, confidence=confidence,
            handoff=handoff, handoff_reason=reason, priority=priority, mode=self.llm.label,
            latency_ms=int((time.perf_counter() - start) * 1000), hits=[h.to_dict() for h in hits])

    def _generate(self, question: str, hits: list[Hit]) -> tuple[bool, str, list[str]]:
        context = "\n\n".join(f"[{h.chunk.id}] {h.chunk.title} > {h.chunk.section}\n{h.chunk.text}" for h in hits)
        try:
            data = self.llm.chat_json(SYSTEM_PROMPT, f"Knowledge base excerpts:\n{context}\n\nCustomer question: {question}")
        except Exception as exc:  # a bad provider response should degrade, not crash the demo
            print(f"[llm] call failed, using fallback: {exc!r}", flush=True)
            return self._extractive(question, hits)
        valid = {h.chunk.id for h in hits}
        cited = [c for c in data.get("cited_ids", []) if c in valid]  # drop invented citations
        answerable = bool(data.get("answerable")) and bool(cited)
        return answerable, str(data.get("answer", "")), cited

    @staticmethod
    def _extractive(question: str, hits: list[Hit], max_sentences: int = 3) -> tuple[bool, str, list[str]]:
        """Demo-mode answer: the KB sentences that best match the question, with citations."""
        terms = set(tokenize(question))
        # Stay inside the best-matching article (plus any chunk that is nearly as strong
        # lexically) so the answer doesn't stitch together unrelated policies.
        top_lex = max(h.lexical for h in hits) or 1.0
        hits = [h for h in hits if h.chunk.doc_id == hits[0].chunk.doc_id or h.lexical >= 0.8 * top_lex]
        scored = []
        for rank, h in enumerate(hits):
            for sent in re.split(r"(?<=[.!?])\s+|\n", h.chunk.text):
                sent = sent.strip()
                if len(sent) < 25 or INTERNAL_LINE.search(sent):
                    continue
                sent = to_customer_voice(sent)
                overlap = len(terms & set(tokenize(sent)))
                section_match = len(terms & set(tokenize(h.chunk.section)))
                if overlap or section_match:
                    scored.append((overlap + 0.6 * section_match - rank * 0.3, sent, h.chunk.id))
        if not scored:
            return False, "", []
        best = sorted(scored, key=lambda s: s[0], reverse=True)[:max_sentences]
        cited = list(dict.fromkeys(cid for _, _, cid in best))
        body = " ".join(f"{re.sub(r'^[0-9]+[.)] |^- ', '', s)} [{cid}]" for _, s, cid in best)
        return True, "Here's what our support guides say: " + body, cited
