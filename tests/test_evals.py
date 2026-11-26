from docs_agent.agent import ROOT, AgentResult
from docs_agent.retrieval import load_chunks
from evals.run_evals import load_questions, score


def result(answer="", citations=(), found=True, escalated=False):
    return AgentResult(answer=answer, citations=list(citations), found_in_docs=found, escalated=escalated)


def test_question_file_is_well_formed():
    qs = load_questions()
    assert len(qs) == 30
    assert len({q["id"] for q in qs}) == 30
    docs = {c.doc for c in load_chunks(ROOT / "policies")}
    for q in qs:
        assert q["expect"] in {"answer", "unknown", "escalate", "any"}
        assert set(q["expected_docs"]) <= docs, q["id"]
        if q["expect"] == "answer":
            assert q["expected_docs"], q["id"]


Q = {"id": "t", "category": "lookup", "question": "?", "expect": "answer",
     "must_include": ["30 days|thirty days"], "must_not_include": ["365"], "expected_docs": ["returns"]}


def test_correct_answer_passes():
    assert score(Q, result("You have thirty days.", ["returns#return-window"])) == []


def test_wrong_source_fails():
    assert any("expected" in p for p in score(Q, result("30 days", ["shipping#standard-shipping"])))


def test_forbidden_text_fails():
    assert "contains '365'" in score(Q, result("30 days, or 365 for some", ["returns#return-window"]))


def test_unknown_question_must_not_claim_an_answer():
    q = {**Q, "expect": "unknown", "must_include": [], "expected_docs": []}
    assert score(q, result("I don't know.", found=False)) == []
    assert score(q, result("Yes, 10% off.", ["stores#price-matching"])) != []


def test_unneeded_ticket_fails_and_needed_one_passes():
    assert "opened a ticket it didn't need" in score(Q, result("30 days", ["returns#return-window"], escalated=True))
    q = {**Q, "expect": "escalate", "must_include": [], "expected_docs": []}
    assert score(q, result("Ticket opened.", found=False, escalated=True)) == []
    assert score(q, result("Sorry.", found=False)) == ["did not open a ticket"]
