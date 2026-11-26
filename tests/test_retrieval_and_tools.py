import pytest

from docs_agent.agent import ROOT
from docs_agent.retrieval import BM25Index, load_chunks, tokenize
from docs_agent.tools import ToolBox, calculate


@pytest.fixture(scope="module")
def index():
    return BM25Index(load_chunks(ROOT / "policies"))


def test_chunks_have_unique_ids_and_text():
    chunks = load_chunks(ROOT / "policies")
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(c.text for c in chunks)
    assert "warranty#tents-and-backpacks" in ids


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("What's the warranty on a tent?") == ["s", "warranty", "tent"]
    assert tokenize("Tents and points, express") == ["tent", "point", "express"]


@pytest.mark.parametrize("query,expected", [
    ("tent warranty years", "warranty#tents-and-backpacks"),
    ("express shipping cutoff time same day", "shipping#express-shipping"),
    ("when do points expire", "loyalty#points-expiry"),
    ("delete my account", "privacy#deleting-your-account"),
])
def test_search_ranks_the_right_section_first(index, query, expected):
    assert index.search(query)[0][0].id == expected


def test_search_with_no_matching_words_returns_nothing(index):
    assert index.search("zzzz qqqq") == []


@pytest.mark.parametrize("expr,value", [
    ("120 - 8.95", 111.05), ("60 + 9.95", 69.95), ("320 * 2", 640), ("640 / 100 * 5", 32),
    ("$1,000 / 4", 250), ("-3 + 5", 2),
])
def test_calculate(expr, value):
    assert calculate(expr) == pytest.approx(value)


@pytest.mark.parametrize("expr", ["__import__('os').system('ls')", "open('x')", "2 ** 1000", "a + 1"])
def test_calculate_refuses_anything_but_arithmetic(expr):
    with pytest.raises(ValueError):
        calculate(expr)


def test_toolbox_tracks_seen_sections_and_writes_tickets(index, tmp_path):
    tb = ToolBox(index, tmp_path / "t.jsonl")
    out = tb.run("search_docs", {"query": "return window"})
    assert "returns#return-window" in tb.seen_ids and "<section" in out
    assert tb.run("escalate_to_human", {"team": "billing", "summary": "charged twice"}).startswith("Ticket T-0001")
    assert tb.run("escalate_to_human", {"team": "privacy", "summary": "delete"}).startswith("Ticket T-0002")
    assert len((tmp_path / "t.jsonl").read_text().splitlines()) == 2
