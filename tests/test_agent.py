import json

from docs_agent.agent import NUDGE, Agent, check_answer
from docs_agent.llm import ScriptedModel


def call(name, i=1, **args):
    return {"id": f"c{i}", "name": name, "arguments": json.dumps(args)}


def submit(answer, citations, found=True, i=9):
    return call("submit_answer", i, answer=answer, citations=citations, found_in_docs=found)


def make(replies, tmp_path, **kw):
    model = ScriptedModel(replies)
    return Agent(model, tickets_path=tmp_path / "t.jsonl", **kw), model


def test_search_then_answer(tmp_path):
    agent, _ = make([
        {"tool_calls": [call("search_docs", query="return window")]},
        {"tool_calls": [submit("You have 30 days.", ["returns#return-window"])]},
    ], tmp_path)
    r = agent.ask("How long to return?")
    assert r.answer == "You have 30 days." and r.found_in_docs and not r.gave_up
    assert [s.tool for s in r.steps] == ["search_docs", "submit_answer"]
    assert r.model_calls == 2


def test_made_up_citation_is_rejected_and_model_retries(tmp_path):
    agent, model = make([
        {"tool_calls": [call("search_docs", query="return window")]},
        {"tool_calls": [submit("30 days", ["returns#made-up"])]},
        {"tool_calls": [submit("30 days", ["returns#return-window"])]},
    ], tmp_path)
    r = agent.ask("How long to return?")
    assert r.citations == ["returns#return-window"]
    rejection = model.calls[2][-1]
    assert rejection["role"] == "tool" and rejection["content"].startswith("Rejected")


def test_cannot_cite_without_searching(tmp_path):
    assert "not returned by search_docs" in check_answer(
        {"answer": "x", "citations": ["returns#return-window"], "found_in_docs": True}, set())


def test_found_in_docs_needs_a_citation():
    assert "no citations" in check_answer({"answer": "x", "citations": [], "found_in_docs": True}, set())


def test_unknown_answer_without_citations_is_fine():
    assert check_answer({"answer": "I don't know.", "citations": [], "found_in_docs": False}, set()) is None


def test_plain_text_reply_gets_a_nudge(tmp_path):
    agent, model = make([
        {"content": "We have no student discount."},
        {"tool_calls": [submit("I don't know of a student discount.", [], found=False)]},
    ], tmp_path)
    r = agent.ask("Student discount?")
    assert not r.found_in_docs and not r.gave_up
    assert model.calls[1][-1] == {"role": "user", "content": NUDGE}


def test_gives_up_after_max_steps(tmp_path):
    agent, _ = make([{"tool_calls": [call("search_docs", i=n, query="x")]} for n in range(3)],
                    tmp_path, max_steps=3)
    r = agent.ask("loop forever")
    assert r.gave_up and "couldn't find" in r.answer


def test_tool_errors_go_back_to_the_model(tmp_path):
    agent, model = make([
        {"tool_calls": [call("calculate", expression="__import__('os')")]},
        {"tool_calls": [submit("Sorry, I couldn't work that out.", [], found=False)]},
    ], tmp_path)
    agent.ask("what is os")
    assert model.calls[1][-1]["content"].startswith("Error:")


def test_bad_json_arguments_are_reported(tmp_path):
    agent, model = make([
        {"tool_calls": [{"id": "c1", "name": "search_docs", "arguments": "{not json"}]},
        {"tool_calls": [submit("I don't know.", [], found=False)]},
    ], tmp_path)
    agent.ask("x")
    assert "not valid JSON" in model.calls[1][-1]["content"]


def test_escalation_opens_ticket(tmp_path):
    agent, _ = make([
        {"tool_calls": [call("escalate_to_human", team="privacy", summary="Wants account deleted")]},
        {"tool_calls": [submit("I've opened a ticket with our privacy team.", [], found=False)]},
    ], tmp_path)
    r = agent.ask("Delete my account")
    assert r.escalated
    assert json.loads((tmp_path / "t.jsonl").read_text())["team"] == "privacy"


def test_conversation_is_kept_between_questions_until_reset(tmp_path):
    agent, model = make([
        {"tool_calls": [submit("Hi!", [], found=False, i=1)]},
        {"tool_calls": [submit("Hi again!", [], found=False, i=2)]},
        {"tool_calls": [submit("New chat.", [], found=False, i=3)]},
    ], tmp_path)
    agent.ask("hello")
    agent.ask("hello again")
    assert [m["content"] for m in model.calls[1] if m["role"] == "user"] == ["hello", "hello again"]
    agent.reset()
    agent.ask("fresh")
    assert [m["content"] for m in model.calls[2] if m["role"] == "user"] == ["fresh"]
