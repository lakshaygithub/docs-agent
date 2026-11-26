"""Chat with the agent in the terminal:  python -m docs_agent.cli  [--trace]"""

import sys

from .agent import Agent
from .llm import OpenAIChatModel


def main():
    trace = "--trace" in sys.argv
    agent = Agent(OpenAIChatModel())
    print("Harbourline help (type 'new' to start over, 'quit' to exit)")
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in ("quit", "exit"):
            break
        if q == "new":
            agent.reset()
            continue
        if not q:
            continue
        r = agent.ask(q)
        if trace:
            for s in r.steps:
                print(f"  [{s.tool}] {s.arguments} -> {s.result[:120]!r} ({s.ms:.0f} ms)")
        print(f"\nbot> {r.answer}")
        if r.citations:
            print(f"     sources: {', '.join(r.citations)}")
        print(f"     ({r.model_calls} model calls, {r.prompt_tokens + r.completion_tokens} tokens, {r.seconds:.1f} s)")


if __name__ == "__main__":
    main()
