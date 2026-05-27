"""Evaluate the hybrid retriever against questions_test.parquet.

Computes Recall@k, MRR@k, nDCG@k for k in {1, 3, 5, 10}.
This is a Python re-implementation of the same fusion logic used by the
TypeScript tool — keep them in lockstep when tuning weights.

Usage:
    # Fusion only (baseline):
    python indexing/eval_retrieval.py \
        --db        data/index/rag.db \
        --questions data/raw/questions_test.parquet \
        --top-k     10

    # Fusion + cross-encoder reranker (requires reranker service running):
    python indexing/eval_retrieval.py \
        --db        data/index/rag.db \
        --questions data/raw/questions_test.parquet \
        --top-k     10 \
        --rerank
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from embed import DIM, embed_one  # noqa: E402
import httpx  # noqa: E402

TOP_K_PER_BRANCH = 16
KW_W = 0.7
VEC_W = 0.3
SCALE = 4
THRESHOLD = 0.30
RERANK_POOL = TOP_K_PER_BRANCH * 2  # mirrors fusion.ts: TOP_K_PER_BRANCH * 2

RERANKER_URL = os.environ.get("RERANKER_URL", "http://127.0.0.1:8001")


_STRIP_WORDS = {
    "what", "who", "when", "where", "how", "which", "why",
    "is", "are", "was", "were", "do", "does", "did",
    "can", "could", "should", "would", "will", "have", "has", "had",
    "the", "a", "an", "of", "to", "in", "for", "on", "at", "by",
    "from", "with", "and", "or", "but", "not", "that", "this", "these",
    "be", "been", "being", "get", "got", "make", "made", "used", "use",
    "also", "any", "some", "into", "than", "then", "its", "their",
}


def strip_to_key_terms(query: str) -> str | None:
    words = [w for w in re.sub(r"[^\w\s]", " ", query.lower()).split()
             if len(w) > 2 and w not in _STRIP_WORDS]
    if len(words) < 3:
        return None
    stripped = " ".join(words)
    if stripped == query.lower().strip():
        return None
    return stripped


def fts_query(q: str) -> str:
    # Replace punctuation with spaces first, then split — mirrors fusion.ts ftsQuery
    # so hyphenated terms like "INV-2026-11-331" expand to separate tokens rather
    # than collapsing into one broken token.
    words = [w for w in re.sub(r"[^\w\s]", " ", q.lower()).split() if len(w) > 1]
    if not words:
        return '""'
    return " OR ".join(f'"{w}"' for w in words)


def load_matrix(con: sqlite3.Connection):
    rows = con.execute(
        "SELECT chunk_id, embedding FROM chunks WHERE embedding IS NOT NULL ORDER BY chunk_id"
    ).fetchall()
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    mat = np.zeros((len(rows), DIM), dtype=np.float32)
    for i, (_, blob) in enumerate(rows):
        mat[i] = np.frombuffer(blob, dtype=np.float32)
    return mat, ids


def chunk_to_doc(con: sqlite3.Connection):
    return {r[0]: r[1] for r in con.execute("SELECT chunk_id, doc_id FROM chunks").fetchall()}


def call_reranker(
    reranker: httpx.Client, query: str, passages: list[str]
) -> list[float] | None:
    """POST to the cross-encoder service. Returns None on any failure."""
    try:
        r = reranker.post(
            f"{RERANKER_URL}/rerank",
            json={"query": query, "passages": passages},
            timeout=8.0,
        )
        if r.status_code == 200:
            data = r.json()
            if isinstance(data.get("scores"), list):
                return data["scores"]
    except Exception:
        pass
    return None


def search(
    con: sqlite3.Connection,
    client: httpx.Client,
    matrix: np.ndarray,
    ids: np.ndarray,
    c2d: dict,
    query: str,
    top_n: int = 10,
    reranker: httpx.Client | None = None,
):
    def top_k_vec(qv: np.ndarray) -> dict[int, float]:
        sims = matrix @ qv
        order = np.argsort(-sims)[:TOP_K_PER_BRANCH]
        return {int(ids[i]): float(sims[i]) * SCALE for i in order}

    qvec = embed_one(client, query)
    vec_scores = top_k_vec(qvec)

    # Multi-query: also embed stripped key terms for long queries and merge
    # candidates, keeping the best score per chunk across both passes.
    if len(query.split()) > 6:
        stripped = strip_to_key_terms(query)
        if stripped:
            qvec2 = embed_one(client, stripped)
            for cid, score in top_k_vec(qvec2).items():
                if score > vec_scores.get(cid, -float("inf")):
                    vec_scores[cid] = score

    rows = con.execute(
        "SELECT c.chunk_id FROM chunks_fts f JOIN chunks c ON c.chunk_id = f.rowid "
        "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
        (fts_query(query), TOP_K_PER_BRANCH),
    ).fetchall()
    kw_scores: dict[int, float] = {}
    for rank, (cid,) in enumerate(rows, start=1):
        kw_scores[cid] = (1 / (1 + rank)) * SCALE

    all_ids = set(vec_scores) | set(kw_scores)
    fused = []
    for cid in all_ids:
        vec = vec_scores.get(cid, 0.0)
        kw = kw_scores.get(cid, 0.0)
        final = VEC_W * vec + KW_W * kw
        if final >= THRESHOLD:
            fused.append((cid, final))
    fused.sort(key=lambda x: -x[1])

    # Take the candidate pool (mirrors fusion.ts behaviour).
    pool: list[tuple[int, float]] = fused[:RERANK_POOL]

    # Cross-encoder reranking: re-score each (query, chunk_text) pair jointly.
    if reranker is not None and pool:
        chunk_ids = [cid for cid, _ in pool]
        placeholders = ",".join("?" * len(chunk_ids))
        text_rows = con.execute(
            f"SELECT chunk_id, text FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        id_to_text = {r[0]: r[1] for r in text_rows}
        passages = [id_to_text.get(cid, "") for cid, _ in pool]
        scores = call_reranker(reranker, query, passages)
        if scores is not None and len(scores) == len(pool):
            pool = sorted(
                [(cid, score) for (cid, _), score in zip(pool, scores)],
                key=lambda x: -x[1],
            )

    # Deduplicate by doc_id, keeping the best-scoring chunk per document.
    seen: list[str] = []
    for cid, _ in pool:
        d = c2d.get(cid)
        if d and d not in seen:
            seen.append(d)
        if len(seen) >= top_n:
            break
    return seen


def parse_expected(s) -> list[str]:
    if s is None:
        return []
    if isinstance(s, list):
        return [str(x) for x in s]
    text = str(s).strip()
    # Convert Python-list-as-string to JSON.
    text = text.replace("'", '"')
    try:
        v = json.loads(text)
        if isinstance(v, list):
            return [str(x) for x in v]
    except Exception:
        pass
    return []


def ndcg(predicted: list[str], expected: set[str], k: int) -> float:
    dcg = 0.0
    for i, d in enumerate(predicted[:k]):
        if d in expected:
            dcg += 1 / math.log2(i + 2)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(expected))))
    return dcg / ideal if ideal > 0 else 0.0


def main() -> int:
    global KW_W, VEC_W
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder reranking (requires reranker service on RERANKER_URL).",
    )
    ap.add_argument("--kw-weight", type=float, default=None,
                    help=f"Override KW_W (default {KW_W}).")
    ap.add_argument("--vec-weight", type=float, default=None,
                    help=f"Override VEC_W (default {VEC_W}).")
    args = ap.parse_args()
    if args.kw_weight is not None:
        KW_W = args.kw_weight
    if args.vec_weight is not None:
        VEC_W = args.vec_weight
    print(f"[weights] vec={VEC_W}  kw={KW_W}")

    con = sqlite3.connect(args.db)
    print("[load] embedding matrix")
    matrix, ids = load_matrix(con)
    c2d = chunk_to_doc(con)
    print(f"[load] {len(ids)} chunks, dim={DIM}")

    indexed = {r[0] for r in con.execute("SELECT doc_id FROM documents").fetchall()}

    qt = pq.read_table(args.questions).to_pandas()
    before = len(qt)
    qt = qt[
        qt.apply(lambda r: any(d in indexed for d in parse_expected(r.expected_doc_ids)), axis=1)
    ].reset_index(drop=True)
    print(f"[filter] {len(qt)} / {before} questions have answer docs in the index")
    if args.limit:
        qt = qt.head(args.limit)

    ks = [1, 3, 5, args.top_k]
    hits_at = {k: 0 for k in ks}
    mrr_at = {k: 0.0 for k in ks}
    ndcg_at = {k: 0.0 for k in ks}

    reranker_client: httpx.Client | None = None
    if args.rerank:
        reranker_client = httpx.Client(timeout=10)
        try:
            reranker_client.get(f"{RERANKER_URL}/health", timeout=3).raise_for_status()
            print(f"[rerank] connected to {RERANKER_URL}")
        except Exception as e:
            print(f"[rerank] WARNING: reranker not reachable at {RERANKER_URL} ({e})")
            print("[rerank] continuing without reranking — results will be fusion-only")
            reranker_client = None

    t0 = time.time()
    with httpx.Client(timeout=120) as client:
        for i, row in enumerate(qt.itertuples(index=False)):
            expected = set(parse_expected(row.expected_doc_ids))
            if not expected:
                continue
            predicted = search(
                con, client, matrix, ids, c2d, row.question,
                top_n=args.top_k, reranker=reranker_client,
            )
            for k in ks:
                top = predicted[:k]
                if expected & set(top):
                    hits_at[k] += 1
                for idx, d in enumerate(top, start=1):
                    if d in expected:
                        mrr_at[k] += 1.0 / idx
                        break
                ndcg_at[k] += ndcg(top, expected, k)
            if (i + 1) % 25 == 0:
                dt = time.time() - t0
                print(f"  {i+1}/{len(qt)}  ({(i+1)/dt:.1f} q/s)")

    if reranker_client is not None:
        reranker_client.close()

    mode = f"fusion + cross-encoder reranker ({RERANKER_URL})" if args.rerank else "fusion only"
    n = len(qt)
    print()
    print(f"mode: {mode}")
    print(f"{'k':>4} {'Recall':>8} {'MRR':>8} {'nDCG':>8}")
    for k in ks:
        print(f"{k:>4} {hits_at[k]/n:>8.3f} {mrr_at[k]/n:>8.3f} {ndcg_at[k]/n:>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
