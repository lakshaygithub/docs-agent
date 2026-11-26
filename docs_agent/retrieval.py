"""Split the policy documents into sections and search them with BM25.

BM25 is a keyword-ranking formula: a section scores higher when it contains
the query's words, especially rare words, and the score is adjusted so long
sections don't win just for being long. No embeddings or API calls needed.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

STOPWORDS = set(
    "a an and are as at be by can do does for from how i if in is it my of on or "
    "the to what when which who will with you your".split()
)


@dataclass
class Chunk:
    id: str  # e.g. "returns#return-window"
    doc: str  # e.g. "returns"
    title: str  # e.g. "Returns and Refunds > Return window"
    text: str


def stem(word: str) -> str:
    """Very light stemming: 'tents' -> 'tent', 'points' -> 'point', but 'express' stays."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def slugify(heading: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")


def load_chunks(folder: Path) -> list[Chunk]:
    """One chunk per '## ' section of every markdown file in the folder."""
    chunks = []
    for path in sorted(folder.glob("*.md")):
        doc_title = path.stem
        sections: list[tuple[str, list[str]]] = []  # (heading, lines)
        for line in path.read_text().splitlines():
            if line.startswith("# "):
                doc_title = line[2:].strip()
            elif line.startswith("## "):
                sections.append((line[3:].strip(), []))
            elif line.strip() and sections:
                sections[-1][1].append(line.strip())
        for heading, lines in sections:
            if lines:
                chunks.append(Chunk(
                    id=f"{path.stem}#{slugify(heading)}",
                    doc=path.stem,
                    title=f"{doc_title} > {heading}",
                    text=" ".join(lines),
                ))
    return chunks


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        # The section title counts as part of the text, so "Tents and backpacks" helps.
        self.docs = [tokenize(c.title + " " + c.text) for c in chunks]
        self.avg_len = sum(map(len, self.docs)) / len(self.docs)
        df = Counter(w for d in self.docs for w in set(d))
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        scored = []
        for chunk, tf, doc in zip(self.chunks, self.tf, self.docs, strict=True):
            score = 0.0
            for w in terms:
                if w in tf:
                    f = tf[w]
                    norm = f + self.k1 * (1 - self.b + self.b * len(doc) / self.avg_len)
                    score += self.idf[w] * f * (self.k1 + 1) / norm
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]
