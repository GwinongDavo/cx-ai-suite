"""Hybrid retrieval over the knowledge base: BM25 + vector search, fused with RRF.

Vector backend:
  - with a live LLM provider: real embeddings (OpenAI / Azure / Gemini), cached to disk
  - in demo mode: TF-IDF vectors (scikit-learn), so search still works with no key

The index is a plain NumPy matrix with cosine similarity. At this size (~80 chunks)
that is faster than any external store; `VectorIndex` is the seam where Azure AI
Search or pgvector would plug in for a real corpus.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import DATA_DIR
from .llm import LLM

STOPWORDS = set(
    """a an and are as at be but by can do does for from has have how i if in is it its my
    me of on or our so that the their them then there this to was we what when where which
    who why will with you your yours not no just get got still also any all after before
    about into out up down off over again more most very too than i'm it's don't can't
    hi hello thanks thank please help need want would could should
    laptop device computer aurora""".split()
)

TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


@dataclass
class Chunk:
    id: str          # e.g. "KB-001#3"
    doc_id: str      # e.g. "KB-001"
    title: str
    section: str
    text: str

    @property
    def citation(self) -> str:
        return f"{self.doc_id} · {self.title} › {self.section}"

    def to_dict(self) -> dict:
        return {"id": self.id, "doc_id": self.doc_id, "title": self.title,
                "section": self.section, "text": self.text}


def load_chunks(kb_dir: Path = DATA_DIR / "kb") -> list[Chunk]:
    """Split each markdown article into one chunk per `##` section."""
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("kb-*.md")):
        doc_id = "KB-" + path.stem.split("-")[1]
        raw = path.read_text(encoding="utf-8")
        title = raw.splitlines()[0].lstrip("# ").strip()
        parts = re.split(r"^## ", raw, flags=re.M)
        for n, part in enumerate(parts[1:], start=1):
            section, _, body = part.partition("\n")
            chunks.append(Chunk(f"{doc_id}#{n}", doc_id, title, section.strip(), body.strip()))
    return chunks


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.avgdl = sum(map(len, docs)) / max(len(docs), 1)
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tfs = [Counter(d) for d in docs]

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for i, (tf, d) in enumerate(zip(self.tfs, self.docs)):
            norm = self.k1 * (1 - self.b + self.b * len(d) / self.avgdl)
            out[i] = sum(self.idf.get(t, 0) * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in query if t in tf)
        return out


class VectorIndex:
    """Cosine-similarity index over unit vectors."""

    def __init__(self, vectors: np.ndarray):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.matrix = vectors / np.clip(norms, 1e-9, None)

    def search(self, query_vec: np.ndarray) -> np.ndarray:
        q = query_vec / max(np.linalg.norm(query_vec), 1e-9)
        return self.matrix @ q


@dataclass
class Hit:
    chunk: Chunk
    score: float       # fused RRF score
    dense: float       # cosine similarity
    lexical: float     # BM25 score

    def to_dict(self) -> dict:
        return {**self.chunk.to_dict(), "citation": self.chunk.citation,
                "score": round(self.score, 4), "dense": round(self.dense, 3), "lexical": round(self.lexical, 3)}


class Retriever:
    def __init__(self, llm: LLM | None = None, chunks: list[Chunk] | None = None,
                 cache_dir: Path = DATA_DIR / ".cache"):
        self.llm = llm or LLM()
        self.chunks = chunks or load_chunks()
        texts = [f"{c.title}. {c.section}. {c.text}" for c in self.chunks]
        self.bm25 = BM25([tokenize(t) for t in texts])
        # TF-IDF is always built: it's the demo-mode backend and the fallback if the
        # embedding API is down or rate limited, so search never takes the app down.
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._tfidf = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2), sublinear_tf=True)
        self._tfidf_index = VectorIndex(self._tfidf.fit_transform(texts).toarray())
        self.backend, self.index = "tfidf", self._tfidf_index
        if self.llm.live:
            try:
                self.index = VectorIndex(self._cached_embeddings(texts, cache_dir))
                self.backend = "embeddings"
            except Exception as exc:  # bad key, quota, network
                print(f"[retrieval] embeddings unavailable, using TF-IDF: {exc}")

    def _cached_embeddings(self, texts: list[str], cache_dir: Path) -> np.ndarray:
        key = hashlib.sha256((self.llm.provider + self.llm.embed_model + "".join(texts)).encode()).hexdigest()[:16]
        path = cache_dir / f"emb-{key}.npy"
        if path.exists():
            return np.load(path)
        vecs = np.array(self.llm.embed(texts), dtype=np.float32)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, vecs)
        return vecs

    def _embed_query(self, query: str) -> np.ndarray:
        if self.backend == "embeddings":
            return np.array(self.llm.embed([query])[0], dtype=np.float32)
        return self._tfidf.transform([query]).toarray()[0]

    def search(self, query: str, k: int = 4, rrf_k: int = 60) -> list[Hit]:
        self.last_backend = self.backend
        try:
            dense = self.index.search(self._embed_query(query))
        except Exception:  # embedding call failed for this query: use TF-IDF instead
            self.last_backend = "tfidf"
            dense = self._tfidf_index.search(self._tfidf.transform([query]).toarray()[0])
        lexical = self.bm25.scores(tokenize(query))
        dense_rank = {i: r for r, i in enumerate(np.argsort(-dense))}
        lex_rank = {i: r for r, i in enumerate(np.argsort(-lexical))}
        fused = {i: 1 / (rrf_k + dense_rank[i]) + (1 / (rrf_k + lex_rank[i]) if lexical[i] > 0 else 0)
                 for i in range(len(self.chunks))}
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        # How far the best chunk stands out from the rest of the corpus for this query.
        # Raw cosine values differ a lot between embedding models; a z-score doesn't.
        self.last_dense_z = float((dense.max() - dense.mean()) / (dense.std() + 1e-9))
        return [Hit(self.chunks[i], fused[i], float(dense[i]), float(lexical[i])) for i in top]

    def confidence(self, query: str, hits: list[Hit]) -> float:
        """0..1 score for "the knowledge base can answer this".

        Blends a similarity signal with how many of the question's key terms the best
        chunks contain. The similarity signal depends on the backend:
          - TF-IDF: the raw cosine of the best chunk (sparse vectors, so it's well scaled)
          - embeddings: how far the best chunk stands out from the whole corpus (z-score),
            because raw cosine ranges differ a lot between embedding models
        The handoff threshold is tuned on the eval set (see the threshold curve in the report).
        """
        if not hits:
            return 0.0
        if getattr(self, "last_backend", self.backend) == "tfidf":
            dense = min(max((hits[0].dense - 0.05) / 0.35, 0.0), 1.0)
        else:
            dense = min(max((getattr(self, "last_dense_z", 0.0) - 2.0) / 3.0, 0.0), 1.0)
        terms = set(tokenize(query))
        if not terms:
            return 0.0
        top_text = set(tokenize(" ".join(f"{h.chunk.title} {h.chunk.section} {h.chunk.text}" for h in hits[:2])))
        coverage = len(terms & top_text) / len(terms)
        return round(0.5 * dense + 0.5 * coverage, 3)


def kb_titles() -> dict[str, str]:
    return {c.doc_id: c.title for c in load_chunks()}


if __name__ == "__main__":
    r = Retriever()
    for q in ["my battery stops charging at 80%", "how many ADP claims per year", "pizza recipe"]:
        hits = r.search(q)
        print(q, "->", r.confidence(q, hits), json.dumps([h.chunk.id for h in hits]))
