"""Corpus retrieval — the agent's first, cheapest tool.

Lifted from Week 2's RAG pipeline (chunk -> embed -> hybrid dense+BM25 retrieve).
The agent calls `retrieve()` before ever touching the web: if the existing
60-article corpus already answers the question, there is no need to search.
"""
import json
import os
import re

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

load_dotenv()

CORPUS_PATH = "corpus/articles.jsonl"
INDEX_PATH = "corpus/index.json"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
MIN_COMBINED_SCORE = 0.62  # below this, corpus coverage is insufficient -> agent goes to the web

EMBED_MODEL = os.environ.get("NEBIUS_EMBED_MODEL", "Qwen/Qwen3-Embedding-8B")

client = OpenAI(
    api_key=os.environ["NEBIUS_API_KEY"],
    base_url=os.environ.get("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1/"),
)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c for c in chunks if len(c.strip()) > 100]


def load_articles() -> list[dict]:
    if not os.path.exists(CORPUS_PATH):
        return []
    with open(CORPUS_PATH) as f:
        return [json.loads(line) for line in f]


def embed(texts: list[str]) -> np.ndarray:
    vectors = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend([d.embedding for d in resp.data])
    return np.array(vectors, dtype=np.float32)


def _chunk_article(article: dict) -> list[dict]:
    return [
        {
            "chunk_id": f"{article['id']}_{i}",
            "company": article["company"],
            "industry": article["industry"],
            "date": article["date"],
            "url": article["url"],
            "text": chunk,
        }
        for i, chunk in enumerate(chunk_text(article["text"]))
    ]


def build_index(force: bool = False) -> dict:
    if os.path.exists(INDEX_PATH) and not force:
        with open(INDEX_PATH) as f:
            data = json.load(f)
        data["embeddings"] = np.array(data["embeddings"], dtype=np.float32)
        return data

    articles = load_articles()
    meta = [chunk for article in articles for chunk in _chunk_article(article)]
    embeddings = embed([m["text"] for m in meta]) if meta else np.zeros((0, 1), dtype=np.float32)
    data = {"embeddings": embeddings, "meta": meta}
    with open(INDEX_PATH, "w") as f:
        json.dump({"embeddings": embeddings.tolist(), "meta": meta}, f)
    return data


def add_article_to_index(article: dict, index: dict) -> dict:
    """Incrementally embed one new article's chunks and append to the live index + on-disk corpus.

    Used after a human approves a web-researched article for permanent inclusion —
    this is how the agent grows Week 2's corpus over time instead of rebuilding it from scratch.
    """
    with open(CORPUS_PATH, "a") as f:
        f.write(json.dumps(article) + "\n")

    new_meta = _chunk_article(article)
    new_embeddings = embed([m["text"] for m in new_meta])

    if index["embeddings"].size == 0:
        index["embeddings"] = new_embeddings
    else:
        index["embeddings"] = np.vstack([index["embeddings"], new_embeddings])
    index["meta"] = index["meta"] + new_meta

    with open(INDEX_PATH, "w") as f:
        json.dump({"embeddings": index["embeddings"].tolist(), "meta": index["meta"]}, f)
    return index


# Without stopword removal, BM25 rewards documents packed with common function words
# ("what", "at", "in", "the") the query happens to share, regardless of topical relevance —
# e.g. an irrelevant "capital of France?" query out-scored a real in-corpus match on raw
# BM25 alone. Stripping stopwords keeps the signal on the words that actually carry meaning
# (company names, industries, dates).
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from",
    "had", "has", "have", "how", "i", "in", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "was", "were", "what", "when", "where", "which", "who", "why", "will",
    "with", "you", "your",
    # Domain-generic: every article in this corpus is about a layoff, so these words
    # appear almost everywhere and carry no discriminating signal for routing.
    "layoff", "layoffs", "happened", "company",
}


_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t for t in _WORD_RE.findall(text.lower()) if t not in _STOPWORDS]


def _cosine_scores(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-8)
    return doc_norms @ query_norm


def retrieve(query: str, index: dict, k: int = TOP_K) -> tuple[list[dict], float]:
    meta = index["meta"]
    if not meta:
        return [], 0.0

    query_vec = embed([query])[0]
    dense_scores = _cosine_scores(query_vec, index["embeddings"])

    tokenized_corpus = [_tokenize(m["text"]) for m in meta]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = np.array(bm25.get_scores(_tokenize(query)))
    bm25_norm = bm25_scores / (bm25_scores.max() + 1e-8)

    combined = 0.6 * dense_scores + 0.4 * bm25_norm
    top_idx = np.argsort(combined)[::-1][:k]

    results = [meta[i] | {"score": float(combined[i])} for i in top_idx]
    top_combined_score = float(combined[top_idx[0]]) if len(top_idx) else 0.0
    return results, top_combined_score
