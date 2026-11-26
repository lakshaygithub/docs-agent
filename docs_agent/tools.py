"""The tools the model can call, as JSON schemas plus the Python that runs them."""

import ast
import json
import operator
from datetime import datetime, timezone
from pathlib import Path

from .retrieval import BM25Index

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search the Harbourline policy documents. Returns the best-matching sections with ids.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "What to look for, in plain words."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate an arithmetic expression such as '120 - 8.95' or '320 * 2'.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Open a ticket for a person on the support team, for things chat cannot do.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team": {"type": "string", "enum": ["warranty", "privacy", "billing", "general"]},
                    "summary": {"type": "string", "description": "One or two sentences on what the customer needs."},
                },
                "required": ["team", "summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Give the final answer to the customer. Always end your turn with this.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "The reply to the customer."},
                    "citations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ids of the document sections the answer is based on.",
                    },
                    "found_in_docs": {
                        "type": "boolean",
                        "description": "False if the documents do not answer the question.",
                    },
                },
                "required": ["answer", "citations", "found_in_docs"],
            },
        },
    },
]

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def calculate(expression: str) -> float:
    """Safe arithmetic: walks the parsed expression and allows only numbers and + - * / **."""

    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            if isinstance(node.op, ast.Pow) and abs(ev(node.right)) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("only numbers and + - * / ** are allowed")

    result = ev(ast.parse(expression.replace("$", "").replace(",", ""), mode="eval").body)
    return round(result, 10)


class ToolBox:
    """Runs tool calls. Keeps track of which sections were shown to the model."""

    def __init__(self, index: BM25Index, tickets_path: Path):
        self.index = index
        self.tickets_path = tickets_path
        self.seen_ids: set[str] = set()
        self.tickets: list[dict] = []

    def run(self, name: str, args: dict) -> str:
        if name == "search_docs":
            hits = self.index.search(args["query"])
            self.seen_ids.update(c.id for c, _ in hits)
            if not hits:
                return "No matching sections."
            # Document text is wrapped and labelled so the model treats it as data.
            return "\n\n".join(
                f'<section id="{c.id}" title="{c.title}">\n{c.text}\n</section>' for c, _ in hits
            )
        if name == "calculate":
            return str(calculate(args["expression"]))
        if name == "escalate_to_human":
            ticket = {
                "id": f"T-{len(self._read_tickets()) + 1:04d}",
                "team": args["team"],
                "summary": args["summary"],
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            self.tickets_path.parent.mkdir(parents=True, exist_ok=True)
            with self.tickets_path.open("a") as f:
                f.write(json.dumps(ticket) + "\n")
            self.tickets.append(ticket)
            return f"Ticket {ticket['id']} opened for the {ticket['team']} team."
        raise ValueError(f"unknown tool: {name}")

    def _read_tickets(self) -> list[str]:
        if not self.tickets_path.exists():
            return []
        return self.tickets_path.read_text().splitlines()
