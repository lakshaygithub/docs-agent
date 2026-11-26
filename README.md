# docs-agent

A customer-help agent that answers questions from a set of policy documents, calls tools, and knows when to hand off to a person. It comes with a 30-question test set that scores every answer.

The documents are for **Harbourline Outdoor Co.**, a made-up outdoor gear store (`policies/`). One of them contains a planted instruction telling AI assistants to lie about shipping and returns, to check that the agent treats documents as data.

## How it works

```
question ─► model ─► tool calls ─► model ─► ... ─► submit_answer ─► checks ─► answer
                      │
                      ├─ search_docs        BM25 keyword search over document sections
                      ├─ calculate          safe arithmetic (no eval)
                      └─ escalate_to_human  opens a ticket for billing, privacy, warranty
```

- **Agent loop** (`docs_agent/agent.py`): sends the conversation to the model, runs the tools it asks for, and repeats until the model calls `submit_answer`. It stops after 8 rounds.
- **Structured answers.** The final answer is a tool call with three fields: `answer`, `citations` (section ids) and `found_in_docs`. This makes answers easy to check and to score.
- **Guardrails on the answer:**
  - The agent can only cite sections that search actually returned. If the model invents a citation, the answer is rejected and the model tries again.
  - If it says it found the answer, it has to cite something.
- **Errors don't crash the agent.** If a tool fails or the arguments aren't valid JSON, the error goes back to the model as a message so it can recover. A plain-text reply gets one reminder to use `submit_answer`.
- **Documents are data, not instructions.** Search results are wrapped in `<section>` tags, and the system prompt says never to follow instructions found inside them.
- **Sessions.** The conversation is kept between questions until `reset()`.
- **Telemetry.** Each answer records:
  - every tool call with its time in milliseconds
  - the number of model calls
  - tokens used
  - total time
- **Search** (`docs_agent/retrieval.py`):
  - BM25 over the document sections, written from scratch.
  - No embeddings, so it runs instantly and needs no API.
  - Light stemming so "tents" matches "tent".

## Evaluation

`evals/questions.jsonl` has 30 questions in 6 groups:

| Group | Count | What it checks |
|---|---|---|
| lookup | 11 | one fact from one section |
| reasoning | 6 | combining sections, applying a rule, or knowing the answer is "ask the maker" |
| calculation | 4 | arithmetic on document numbers. One is a trap: 640 points is worth $30, not $32, because points redeem in blocks of 100. |
| unknown | 3 | the documents don't say. It must not make something up. |
| escalate | 3 | needs a person, so it must open a ticket |
| adversarial | 3 | the planted instruction, and a user asking for the system prompt (checked with a canary string) |

**How an answer is scored:**
- It must contain the expected facts and must not contain forbidden ones.
- It must cite a section from the right document.
- It must set `found_in_docs` correctly.
- It must open a ticket only when one is needed.

```bash
python -m evals.run_evals --retrieval-only   # search only, no API key
python -m evals.run_evals                    # the full agent
python -m evals.run_evals --only q19,q28     # a few questions
```

Full runs save every answer, tool call and score to `evals/results/`.

### Results

**Search alone:**
- The right document comes first for 19 of 23 questions.
- It is in the top 4 for all 23.
- All 4 misses rank the membership perks section first, because it talks about shipping a lot. The agent sees the top 4 sections, so it still gets the right one.

**Full agent:** not run yet. It needs a model endpoint.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest -q                          # 35 tests, no API key needed (uses a scripted model)

cp .env.example .env               # fill in Azure OpenAI or OpenAI details
set -a && source .env && set +a
python -m docs_agent.cli --trace   # chat in the terminal and see each tool call
```

It works with Azure OpenAI, OpenAI, or any OpenAI-compatible server such as a local llama.cpp server. Set `OPENAI_BASE_URL` to use one.

## What I'd do next

- Embedding search, compared against BM25 on the same questions.
- Run each question several times to measure how often answers change.
- Have a second model grade answers where a substring check is too strict.
