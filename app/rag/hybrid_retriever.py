import re

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.db.session import DB_URL
from app.rag.embeddings import embed_text

_reranker = None

CONFIDENCE_THRESHOLD = 0.0


def _tokenize(text: str):
    """Lowercase word tokens, punctuation stripped -- 'damaged,' and 'damaged'
    must match, or BM25 silently misses the exact word that matters most."""
    return re.findall(r"\w+", text.lower())


def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def _load_all_chunks():
    with psycopg.connect(DB_URL) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id, doc_title, section_title, content, embedding FROM policy_chunks"
            )
            rows = cur.fetchall()
    return [
        {
            "chunk_id": r[0],
            "doc_title": r[1],
            "section_title": r[2],
            "content": r[3],
            "embedding": np.array(r[4].to_list()) if r[4] is not None else None,
        }
        for r in rows
    ]


def _cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _reciprocal_rank_fusion(bm25_ranking, dense_ranking, k=60):
    scores = {}
    for rank, chunk_id in enumerate(bm25_ranking):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    for rank, chunk_id in enumerate(dense_ranking):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def retrieve(query: str, top_k_for_rerank: int = 8):
    chunks = _load_all_chunks()
    if not chunks:
        return None, None

    tokenized_corpus = [_tokenize(c["content"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus, b=0.4)
    bm25_scores = bm25.get_scores(_tokenize(query))
    bm25_ranking = [chunks[i]["chunk_id"] for i in np.argsort(bm25_scores)[::-1]]

    query_embedding = np.array(embed_text(query))
    dense_scores = [
        (_cosine_sim(query_embedding, c["embedding"]), c["chunk_id"]) for c in chunks
    ]
    dense_ranking = [cid for _, cid in sorted(dense_scores, reverse=True)]

    fused = _reciprocal_rank_fusion(bm25_ranking, dense_ranking)
    top_candidate_ids = [cid for cid, _ in fused[:top_k_for_rerank]]
    candidates = [c for c in chunks if c["chunk_id"] in top_candidate_ids]

    reranker = _get_reranker()
    pairs = [[query, c["content"]] for c in candidates]
    rerank_scores = list(reranker.predict(pairs))

    # Small boost when the section title itself shares words with the query --
    # our chunks have deliberately descriptive titles, and the base cross-encoder
    # occasionally under-ranks a longer, substantive chunk against a shorter one
    # that merely mentions the same words in passing.
    query_tokens = set(_tokenize(query))
    TITLE_BOOST = 1.5
    for i, c in enumerate(candidates):
        title_tokens = set(_tokenize(c["section_title"]))
        overlap = len(query_tokens & title_tokens)
        rerank_scores[i] += overlap * TITLE_BOOST

    best_idx = int(np.argmax(rerank_scores))
    best_score = float(rerank_scores[best_idx])
    best_chunk = candidates[best_idx]

    if best_score < CONFIDENCE_THRESHOLD:
        return None, best_score

    return best_chunk, best_score