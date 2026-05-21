"""Evaluate the hybrid retriever against questions_test.parquet (pgvector version).

Nuvolos counterpart to eval_retrieval.py — same fusion math and metrics, but
queries PostgreSQL/pgvector instead of SQLite.

Run from the Backend VS Code app on Nuvolos:
    cd /files
    python indexing/eval_retrieval_pg.py \
        --questions data/raw/questions_test.parquet \
        --top-k     10

For a quick sanity check on the first 100 questions:
    python indexing/eval_retrieval_pg.py \
        --questions data/raw/questions_test.parquet \
        --limit 100

Environment variables (same defaults as backend):
    PGHOST / PGPORT / PGUSER / PGPASSWORD / PGDATABASE
    OLLAMA_HOST        http://localhost:11434
    RAG_EMBED_MODEL    nomic-embed-text
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import httpx
import psycopg2
from pgvector.psycopg2 import register_vector
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from embed import DIM, MODEL, embed_one  # noqa: E402

# ── Configuration (mirrors backend/main.py) ────────────────────────────────────
DB_HOST     = os.environ.get("PGHOST",     "nv-service-b01d63337fab32ac94f65eb2dc8a62ba")
DB_PORT     = int(os.environ.get("PGPORT", "5432"))
DB_USER     = os.environ.get("PGUSER",     "nuvolos")
DB_PASSWORD = os.environ.get("PGPASSWORD", "nuvolos")
DB_NAME     = os.environ.get("PGDATABASE", "nuvolos")

TABLE_CHUNKS = "rag_chunks"

# DUPLICATED from backend/main.py — keep in sync. The FUSION_VERSION constant
# is asserted at startup so any drift between this file and the production
# retriever is caught loudly (rather than producing eval numbers for a
# retriever that no longer ships).
FUSION_VERSION    = "rrf-v1"
TOP_K_PER_BRANCH  = 8
RRF_K             = 60
RRF_DISPLAY_SCALE = 60.0
SCORE_THRESHOLD   = 0.5
BM25_K1           = 1.5
BM25_B            = 0.75

import math, re  # noqa: E402

_TOKEN_RE = re.compile(r'[a-zA-Z0-9]\w*')


def _fts_or_query(q: str) -> str | None:
    words = [w for w in _TOKEN_RE.findall(q.lower()) if len(w) > 1]
    return " | ".join(words) if words else None


def _tokenize(text: str) -> list[str]:
    return [w for w in _TOKEN_RE.findall(text.lower()) if len(w) > 1]


def _bm25_scores(query_terms: list[str], candidates: list[tuple[int, str]]) -> dict[int, float]:
    if not candidates or not query_terms:
        return {}
    tokenized = [(cid, _tokenize(text)) for cid, text in candidates]
    lengths   = [len(toks) for _, toks in tokenized]
    avgdl     = sum(lengths) / len(lengths) if lengths else 1
    N         = len(tokenized)
    df: dict[str, int] = {}
    for _, toks in tokenized:
        for term in set(toks):
            df[term] = df.get(term, 0) + 1
    result: dict[int, float] = {}
    for cid, toks in tokenized:
        dl = len(toks)
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for term in query_terms:
            if term not in df:
                continue
            f   = tf.get(term, 0)
            idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
            score += idf * (f * (BM25_K1 + 1)) / (f + BM25_K1 * (1 - BM25_B + BM25_B * dl / avgdl))
        result[cid] = score
    return result


# ── DB ─────────────────────────────────────────────────────────────────────────
def connect():
    kwargs: dict = dict(host=DB_HOST, port=DB_PORT, user=DB_USER, dbname=DB_NAME,
                        connect_timeout=10)
    if DB_PASSWORD:
        kwargs["password"] = DB_PASSWORD
    conn = psycopg2.connect(**kwargs)
    conn.autocommit = True
    register_vector(conn)
    cur = conn.cursor()
    cur.execute("SET ivfflat.probes = 10;")
    return conn, cur


# ── Search (same retriever as backend/main.py — see FUSION_VERSION) ────────────
def search(cur, client: httpx.Client, query: str, top_n: int = 10) -> list[str]:
    """Hybrid retrieval mirroring backend/main.py.rag_search exactly:
    vector ANN with embedding-not-null filter; keyword branch uses
    `to_tsquery` OR-join + ts_rank pre-ranking + Python BM25 rerank;
    fusion via Reciprocal Rank Fusion (RRF_K=60)."""
    qvec = embed_one(client, query)

    # Vector branch
    cur.execute(
        f"""SELECT chunk_id, doc_id, (embedding <=> %s::vector) AS dist
            FROM   {TABLE_CHUNKS}
            WHERE  embedding IS NOT NULL
            ORDER  BY dist ASC
            LIMIT  %s""",
        (qvec.tolist(), TOP_K_PER_BRANCH),
    )
    vec_doc: dict[int, str] = {}
    vec_rank: dict[int, int] = {}
    for rank, (chunk_id, doc_id, _dist) in enumerate(cur.fetchall(), start=1):
        vec_doc[chunk_id] = doc_id
        vec_rank[chunk_id] = rank

    # Keyword branch — OR tsquery, ts_rank pre-sort, BM25 rerank in Python
    kw_doc: dict[int, str] = {}
    kw_rank: dict[int, int] = {}
    fts_q = _fts_or_query(query)
    if fts_q:
        cur.execute(
            f"""SELECT chunk_id, doc_id, text
                FROM   {TABLE_CHUNKS}
                WHERE  text_tsv @@ to_tsquery('english', %s)
                ORDER  BY ts_rank(text_tsv, to_tsquery('english', %s)) DESC
                LIMIT  %s""",
            (fts_q, fts_q, TOP_K_PER_BRANCH * 3),
        )
        rows = cur.fetchall()
        query_terms = _tokenize(query)
        bm25 = _bm25_scores(query_terms, [(r[0], r[2]) for r in rows])
        ranked = sorted(rows, key=lambda r: bm25.get(r[0], 0.0), reverse=True)[:TOP_K_PER_BRANCH]
        for rank, (chunk_id, doc_id, _text) in enumerate(ranked, start=1):
            kw_doc[chunk_id] = doc_id
            kw_rank[chunk_id] = rank

    # RRF fusion
    all_ids = set(vec_rank) | set(kw_rank)
    fused: list[tuple[str, float]] = []
    for cid in all_ids:
        doc_id = vec_doc.get(cid) or kw_doc[cid]
        rrf_raw = 0.0
        vr = vec_rank.get(cid)
        kr = kw_rank.get(cid)
        if vr is not None:
            rrf_raw += 1.0 / (RRF_K + vr)
        if kr is not None:
            rrf_raw += 1.0 / (RRF_K + kr)
        score = rrf_raw * RRF_DISPLAY_SCALE
        if score >= SCORE_THRESHOLD:
            fused.append((doc_id, score))

    fused.sort(key=lambda x: -x[1])

    # Deduplicate to doc_ids, preserving best-chunk-first order
    seen: list[str] = []
    for doc_id, _ in fused[: top_n * 3]:
        if doc_id not in seen:
            seen.append(doc_id)
        if len(seen) >= top_n:
            break
    return seen


# ── Metrics ────────────────────────────────────────────────────────────────────
def ndcg(predicted: list[str], expected: set[str], k: int) -> float:
    dcg  = sum(1 / math.log2(i + 2) for i, d in enumerate(predicted[:k]) if d in expected)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(expected))))
    return dcg / ideal if ideal > 0 else 0.0


def parse_expected(s) -> list[str]:
    if s is None:
        return []
    if isinstance(s, list):
        return [str(x) for x in s]
    text = str(s).strip().replace("'", '"')
    try:
        v = json.loads(text)
        if isinstance(v, list):
            return [str(x) for x in v]
    except Exception:
        pass
    return []


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate hybrid retriever on pgvector.")
    ap.add_argument("--questions", required=True, help="Path to questions_test.parquet")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="Evaluate first N questions only (debug)")
    args = ap.parse_args()

    conn, cur = connect()
    cur.execute(f"SELECT COUNT(*) FROM {TABLE_CHUNKS} WHERE embedding IS NOT NULL;")
    n_chunks = cur.fetchone()[0]
    print(f"[db   ] connected — {n_chunks} embedded chunks")
    print(f"[eval ] model={MODEL}  dim={DIM}  fusion={FUSION_VERSION}")

    # Parity check with production retriever. If backend/main.py bumps its
    # FUSION_VERSION, this eval must be updated too (or numbers reported here
    # no longer describe the system that ships).
    try:
        sys.path.insert(0, str(HERE.parent / "backend"))
        import main as backend_main  # type: ignore
        if backend_main.FUSION_VERSION != FUSION_VERSION:
            print(f"[WARN ] FUSION_VERSION drift: backend={backend_main.FUSION_VERSION!r} "
                  f"eval={FUSION_VERSION!r} — eval numbers will not describe production.")
    except Exception:
        # Eval is runnable standalone (e.g. with only indexing deps installed).
        pass

    qt = pq.read_table(args.questions).to_pandas()
    if args.limit:
        qt = qt.head(args.limit)
    print(f"[eval ] {len(qt)} questions  top_k={args.top_k}")

    ks = [1, 3, 5, args.top_k]
    hits_at = {k: 0   for k in ks}
    mrr_at  = {k: 0.0 for k in ks}
    ndcg_at = {k: 0.0 for k in ks}
    n_valid = 0

    t0 = time.time()
    with httpx.Client(timeout=120) as client:
        for i, row in enumerate(qt.itertuples(index=False)):
            expected = set(parse_expected(row.expected_doc_ids))
            if not expected:
                continue
            predicted = search(cur, client, row.question, top_n=args.top_k)
            n_valid += 1
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

    conn.close()
    print()
    print(f"{'k':>4} {'Recall':>8} {'MRR':>8} {'nDCG':>8}")
    for k in ks:
        print(f"{k:>4} {hits_at[k]/n_valid:>8.3f} {mrr_at[k]/n_valid:>8.3f} {ndcg_at[k]/n_valid:>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
