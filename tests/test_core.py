import pytest

from cxcore.llm import LLM, parse_json
from cxcore.qa import QAScorer, agreement, load_interactions
from cxcore.rag import RagAssistant
from cxcore.retrieval import Retriever, load_chunks
from cxcore.sla import kpis, load_tickets, open_risk, train_risk_model
from cxcore.triage import SERIAL, TriageAgent


@pytest.fixture(scope="module")
def retriever():
    return Retriever(LLM("mock"))


def test_kb_chunks_have_ids_and_text():
    chunks = load_chunks()
    assert len({c.doc_id for c in chunks}) == 16
    assert all(c.id.startswith("KB-") and c.text for c in chunks)


@pytest.mark.parametrize("question,doc", [
    ("battery stuck at 80 percent", "KB-001"),
    ("dead pixel replacement policy", "KB-009"),
    ("bitlocker recovery key after bios update", "KB-007"),
])
def test_retrieval_finds_the_right_article(retriever, question, doc):
    assert retriever.search(question)[0].chunk.doc_id == doc


def test_rag_cites_only_retrieved_chunks(retriever):
    r = RagAssistant(LLM("mock"), retriever).answer("How long is the return window?")
    retrieved = {h["id"] for h in r.hits}
    assert r.citations and {c["id"] for c in r.citations} <= retrieved
    assert not r.handoff


def test_rag_hands_off_out_of_scope_and_safety(retriever):
    a = RagAssistant(LLM("mock"), retriever)
    assert a.answer("What's the weather in Kuala Lumpur?").handoff
    safety = a.answer("there is smoke coming out of my laptop")
    assert safety.handoff and safety.priority == "P1"


def test_parse_json_handles_code_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_serial_pattern_needs_a_digit():
    assert SERIAL.search("PF3A9K2L01")
    assert not SERIAL.search("ACCIDENTAL")


def test_triage_policy(retriever):
    agent = TriageAgent(LLM("mock"), retriever)
    assert agent.run("My laptop case is swelling up, serial NJ5ZLZYPTE").decision == "escalate"
    assert agent.run("Wi-Fi keeps dropping, no serial handy").decision == "ask"
    legal = agent.run("Still no refund, I'm calling my lawyer. Serial XZQWDQCW1T")
    assert legal.queue == "Escalations manager"


def test_sla_model_beats_chance():
    df = load_tickets()
    model = train_risk_model(df)
    assert model.auc > 0.7
    assert 0 < kpis(df)["resolution_sla_met"] < 1
    assert open_risk(df, model).risk.between(0, 1).all()


def test_qa_rules_catch_critical_failures():
    items = {i["id"]: i for i in load_interactions()}
    scorer = QAScorer(LLM("mock"))
    assert scorer.score(items["QA-05"]["transcript"]).critical_failures  # asked for password
    assert scorer.score(items["QA-01"]["transcript"]).overall >= 90
    assert agreement(scorer)["within_one"] >= 0.8
