from fastapi.testclient import TestClient

from apps.rag_assistant.main import app as rag_app
from apps.triage_agent.main import app as triage_app

rag = TestClient(rag_app)
triage = TestClient(triage_app)


def test_rag_api():
    assert rag.get("/api/health").json()["status"] == "ok"
    r = rag.post("/api/ask", json={"question": "Can I return a laptop after 20 days?"}).json()
    assert r["answer"] and "confidence" in r
    h = rag.post("/api/ask", json={"question": "best pizza near me"}).json()
    assert h["handoff"] and h["handoff_ticket"]["case_id"].startswith("CS-")
    assert rag.post("/api/ask", json={"question": ""}).status_code == 422


def test_triage_api_and_mock_warranty():
    assert triage.get("/mock-api/warranty/NEV5WEA5GE").json()["plan"] == "Premier Support"
    assert triage.get("/mock-api/warranty/ZZ00000000").status_code == 404
    r = triage.post("/api/triage", json={"text": "Battery not charging. Serial DHNN4J3MT5", "channel": "email"}).json()
    assert r["decision"] in {"reply", "ask", "escalate"} and len(r["trace"]) == 5
    q = triage.get("/api/queue").json()
    assert q["rows"] and q["priority_accuracy"] > 0.8
