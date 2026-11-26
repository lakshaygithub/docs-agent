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
- **Checks on the answer:**
  - The agent can only cite sections that search actually returned. If the model invents a citation, the answer is rejected and the model tries again.
  - If it says it found the answer, it has to cite something.
- **Errors don't crash the agent.** If a tool fails or the arguments aren't valid JSON, the error goes back to the model as a message so it can recover. A plain-text reply gets one reminder to use `submit_answer`.
- **Documents are data, not instructions.** Search results are wrapped in `<section>` tags, and the system prompt says never to follow instructions found inside them.
- **Sessions.** The conversation is kept between questions until `reset()`.
- **One tool call at a time, no repeats.** If the model sends several calls at once, only the first runs. An exact repeat of an earlier call is refused.
- **Automatic first search.** The customer's question is always searched before the model starts.
- **No made-up actions.** An answer that mentions a ticket is rejected unless one was opened.
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

All runs used **Qwen2.5-7B-Instruct** (4-bit), running locally on an Apple M4 Mac (16 GB) with llama.cpp. All 30 questions were run once, at temperature 0.

| Run | What changed | Passed | Model calls per question | Time per question | Tokens (all 30) |
|---|---|---|---|---|---|
| 1 | first version | **16/30 (53%)** | 2.9 | 28 s | 114k |
| 2 | one tool call per step, answers need a search first, better rejection messages | **21/30 (70%)** | 3.7 | 30 s | 140k |
| 3 | automatic first search, no repeated calls, no claiming a ticket that wasn't opened | **23/30 (77%)** | 1.6 | 16 s | 58k |

**Run 3 by group:**

| Group | Passed |
|---|---|
| lookup | 11/11 |
| escalate | 3/3 |
| adversarial | 3/3 |
| unknown | 2/3 |
| calculation | 2/4 |
| reasoning | 2/6 |

**What the failures taught me:**
- **The model fired several tool calls at once.** It sent the same search up to 30 times in one reply, and sometimes submitted an answer in the same reply as its search, before reading the results. The server ignores `parallel_tool_calls=false`, so the agent now runs only the first call in each reply and tells the model the rest were skipped.
- **My leak-test code word looked like a document ID** ("HB-7731"), so the model cited it, got rejected 7 times and gave up. A plain word fixed it.
- **A rejection that only says "no" gets repeated.** Saying which section ids *can* be cited works better. Exact repeats of a call are now refused, not run again.
- **It claimed it had opened a ticket when it hadn't.** Answers that mention a ticket are now rejected unless one was really opened.
- **A side effect of that fix:** in run 3 the model opened more tickets it didn't need (3 questions). That's the biggest remaining problem, along with multi-step math. It still says 640 points is $32, not $30.
- **One run-3 failure is the scorer's fault, not the model's.** "You do not need to pay" was a correct answer my keyword check didn't accept. I left the score as it is rather than change the test after seeing the result.

**Search alone:**
- The right document comes first for 19 of 23 questions, and is in the top 4 for all 23.
- The 4 misses all rank the membership perks section first, because it talks about shipping a lot.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest -q                          # 41 tests, no API key needed (uses a scripted model)

cp .env.example .env               # fill in Azure OpenAI or OpenAI details
set -a && source .env && set +a
python -m docs_agent.cli --trace   # chat in the terminal and see each tool call
```

It works with Azure OpenAI, OpenAI, or any OpenAI-compatible server. To run it fully locally like I did:

```bash
llama-server -m qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf --jinja -c 16384 --port 8089
OPENAI_BASE_URL=http://127.0.0.1:8089/v1 OPENAI_API_KEY=local OPENAI_MODEL=qwen2.5-7b-instruct python -m evals.run_evals
```

## What I'd do next

- Stop the extra tickets: escalate only when the customer asks for one of the three actions chat can't do.
- Run the same 30 questions on a larger hosted model (Azure OpenAI) to separate agent problems from small-model problems.
- Embedding search, compared against BM25 on the same questions.
- Run each question several times to measure how often answers change.
- Have a second model grade answers where a substring check is too strict.
