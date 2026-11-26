"""Score the agent on the 30 questions in questions.jsonl.

    python -m evals.run_evals --retrieval-only   # search quality only, no API key needed
    python -m evals.run_evals                    # full agent run against the model
"""

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from docs_agent.agent import ROOT, Agent, AgentResult
from docs_agent.retrieval import BM25Index, load_chunks

HERE = Path(__file__).resolve().parent


def load_questions(path: Path = HERE / "questions.jsonl") -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _norm(text: str) -> str:
    return text.lower().replace("’", "'")


def score(q: dict, r: AgentResult) -> list[str]:
    """Returns the reasons a result failed. An empty list means it passed."""
    problems = []
    answer = _norm(r.answer)
    if r.error:
        problems.append(f"model error: {r.error[:80]}")
    elif r.gave_up:
        problems.append("gave up without an answer")
    for item in q["must_include"]:
        if not any(_norm(alt) in answer for alt in item.split("|")):
            problems.append(f"missing '{item}'")
    for bad in q["must_not_include"]:
        if _norm(bad) in answer:
            problems.append(f"contains '{bad}'")
    if q["expect"] == "answer":
        if not r.found_in_docs:
            problems.append("said the documents don't answer it")
        cited_docs = {c.split("#")[0] for c in r.citations}
        if q["expected_docs"] and not cited_docs & set(q["expected_docs"]):
            problems.append(f"cited {sorted(cited_docs) or 'nothing'}, expected {q['expected_docs']}")
    elif q["expect"] == "unknown":
        if r.found_in_docs:
            problems.append("claimed an answer the documents don't have")
    elif q["expect"] == "escalate":
        if not r.escalated:
            problems.append("did not open a ticket")
    if q["expect"] != "escalate" and r.escalated:
        problems.append("opened a ticket it didn't need")
    return problems


def retrieval_check(questions: list[dict], k: int = 4) -> None:
    index = BM25Index(load_chunks(ROOT / "policies"))
    rows = [q for q in questions if q["expected_docs"]]
    hits = 0
    for q in rows:
        found = {c.doc for c, _ in index.search(q["question"], k=k)}
        ok = bool(found & set(q["expected_docs"]))
        hits += ok
        if not ok:
            print(f"  miss {q['id']}: {q['question']!r} -> {sorted(found)}")
    print(f"Search finds a right document in the top {k} for {hits}/{len(rows)} questions.")


def full_run(questions: list[dict]) -> None:
    from docs_agent.llm import OpenAIChatModel

    model = OpenAIChatModel()
    tickets = Path(tempfile.mkdtemp()) / "tickets.jsonl"  # don't pollute the real ticket file
    rows, by_cat = [], defaultdict(lambda: [0, 0])
    for q in questions:
        agent = Agent(model, tickets_path=tickets)  # fresh conversation per question
        try:
            r = agent.ask(q["question"])
            problems = score(q, r)
        except Exception as e:  # e.g. a timeout: count it as a failure and keep going
            r = AgentResult(answer="", citations=[], found_in_docs=False, escalated=False, gave_up=True)
            problems = [f"error: {type(e).__name__}"]
        by_cat[q["category"]][0] += not problems
        by_cat[q["category"]][1] += 1
        mark = "PASS" if not problems else "FAIL"
        print(f"{mark} {q['id']} [{q['category']}] {q['question']}")
        if problems:
            print(f"     {'; '.join(problems)}\n     answer: {r.answer!r}")
        rows.append({"question": q, "passed": not problems, "problems": problems, "result": asdict(r)})

    passed = sum(x["passed"] for x in rows)
    n = len(rows)
    print(f"\nPassed {passed}/{n} ({passed / n:.0%})")
    for cat, (p, t) in by_cat.items():
        print(f"  {cat:12s} {p}/{t}")
    res = [x["result"] for x in rows]
    print(f"Average model calls {sum(r['model_calls'] for r in res) / n:.1f}, "
          f"average time {sum(r['seconds'] for r in res) / n:.1f} s, "
          f"total tokens {sum(r['prompt_tokens'] + r['completion_tokens'] for r in res):,}")

    out = HERE / "results" / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"Saved details to {out.relative_to(ROOT)}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--retrieval-only", action="store_true")
    p.add_argument("--only", help="comma-separated question ids, e.g. q03,q19")
    args = p.parse_args(argv)
    questions = load_questions()
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    retrieval_check(questions, k=1)
    retrieval_check(questions, k=4)
    if not args.retrieval_only:
        print()
        full_run(questions)


if __name__ == "__main__":
    sys.exit(main())
