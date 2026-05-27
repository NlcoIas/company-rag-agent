"""Unified evaluation harness:
  - Andre's reranker integration (--rerank flag + RERANK_POOL + fts_query fix)
  - Our parse_expected fix (handles numpy.ndarray gold IDs that pandas surfaces
    from parquet list<string> columns — without this, 93 of 500 questions are
    silently dropped, see docs/eval_baseline_findings)
  - Our --denominator {all|scored} flag (honest baseline excludes unparsed rows)
  - Our per-question-type breakdown (poster section: R@k by category)

This file lives alongside the upstream eval_retrieval.py rather than replacing
it, so the diff is reviewable and we don't conflict with the team's eval.

Usage:
    # fusion-only baseline
    python indexing/eval_retrieval_fixed.py \\
        --db data/index/rag.db \\
        --questions data/raw/questions_test.parquet \\
        --top-k 10 --denominator scored

    # with cross-encoder reranking (requires reranker service on RERANKER_URL)
    python indexing/eval_retrieval_fixed.py \\
        --db data/index/rag.db \\
        --questions data/raw/questions_test.parquet \\
        --top-k 10 --denominator scored --rerank
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

# Mirrors src/rag/fusion.ts constants on upstream/main (f154b10).
# Sync with upstream/indexing/eval_retrieval.py when Andre tunes these.
TOP_K_PER_BRANCH = 16
KW_W = 0.3
VEC_W = 0.7
SCALE = 4
THRESHOLD = 0.30
RERANK_POOL = TOP_K_PER_BRANCH * 2  # 32

RERANKER_URL = os.environ.get("RERANKER_URL", "http://127.0.0.1:8001")


def fts_query(q: str) -> str:
    # Replace punctuation with spaces, then split — matches fusion.ts ftsQuery
    # so hyphenated terms like "INV-2026-11-331" expand to separate tokens
    # rather than collapsing into one broken token.
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


def call_reranker(client: httpx.Client, query: str, passages: list[str]) -> list[float] | None:
    """POST to cross-encoder service. Returns None on any failure."""
    try:
        r = client.post(
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


def search(con, ollama, matrix, ids, c2d, query, top_n=10, reranker=None, mode="hybrid"):
    """mode:
       - "hybrid"     fused 0.7·vec + 0.3·kw (the live system)
       - "bm25-only"  pull RERANK_POOL FTS5/BM25 candidates only
       - "vec-only"   pull RERANK_POOL dense-cosine candidates only
    """
    if mode == "vec-only":
        qvec = embed_one(ollama, query)
        sims = matrix @ qvec
        order = np.argsort(-sims)[:RERANK_POOL]
        pool = [(int(ids[idx]), float(sims[idx]) * SCALE) for idx in order]
    elif mode == "bm25-only":
        rows = con.execute(
            "SELECT c.chunk_id FROM chunks_fts f JOIN chunks c ON c.chunk_id = f.rowid "
            "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
            (fts_query(query), RERANK_POOL),
        ).fetchall()
        pool = [(cid, (1 / (1 + rank)) * SCALE) for rank, (cid,) in enumerate(rows, start=1)]
    else:
        qvec = embed_one(ollama, query)
        sims = matrix @ qvec
        order = np.argsort(-sims)[:TOP_K_PER_BRANCH]
        vec_scores = {int(ids[idx]): float(sims[idx]) * SCALE for idx in order}

        rows = con.execute(
            "SELECT c.chunk_id FROM chunks_fts f JOIN chunks c ON c.chunk_id = f.rowid "
            "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
            (fts_query(query), TOP_K_PER_BRANCH),
        ).fetchall()
        kw_scores = {cid: (1 / (1 + rank)) * SCALE for rank, (cid,) in enumerate(rows, start=1)}

        all_ids = set(vec_scores) | set(kw_scores)
        fused = []
        for cid in all_ids:
            vec = vec_scores.get(cid, 0.0)
            kw = kw_scores.get(cid, 0.0)
            final = VEC_W * vec + KW_W * kw
            if final >= THRESHOLD:
                fused.append((cid, final))
        fused.sort(key=lambda x: -x[1])

        pool = fused[:RERANK_POOL]

    # Cross-encoder reranking — pull chunk text, score (query, chunk) pairs.
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

    # Dedupe by doc_id (keep best-scoring chunk per doc).
    seen: list[str] = []
    for cid, _ in pool:
        d = c2d.get(cid)
        if d and d not in seen:
            seen.append(d)
        if len(seen) >= top_n:
            break
    return seen


def parse_expected(s) -> list[str]:
    """Robust parser: handles numpy.ndarray (pandas surfaces parquet list<string>
    columns as this), Python list, and python-list-as-string. The upstream eval
    drops the ndarray case silently — 93 of 500 gold rows get scored 0/0."""
    if s is None:
        return []
    if hasattr(s, "tolist") and not isinstance(s, (str, bytes)):
        try:
            return [str(x) for x in s.tolist()]
        except Exception:
            pass
    if isinstance(s, list):
        return [str(x) for x in s]
    text = str(s).strip()
    text = text.replace("'", '"')
    try:
        v = json.loads(text)
        if isinstance(v, list):
            return [str(x) for x in v]
    except Exception:
        pass
    return []


def ndcg(predicted, expected, k):
    dcg = 0.0
    for i, d in enumerate(predicted[:k]):
        if d in expected:
            dcg += 1 / math.log2(i + 2)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(expected))))
    return dcg / ideal if ideal > 0 else 0.0


def main() -> int:
    global VEC_W, KW_W  # must precede any read of these constants below
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
    ap.add_argument(
        "--mode",
        choices=["hybrid", "bm25-only", "vec-only"],
        default="hybrid",
        help="Retrieval ablation mode (default: hybrid).",
    )
    ap.add_argument("--vec-weight", type=float, default=None,
                    help="Override vec weight in hybrid fusion (default: %.2f)" % VEC_W)
    ap.add_argument("--kw-weight", type=float, default=None,
                    help="Override kw weight in hybrid fusion (default: %.2f)" % KW_W)
    ap.add_argument(
        "--denominator",
        choices=["all", "scored"],
        default="scored",
        help="all = divide by total rows (matches upstream eval, deflated by parser bug); "
             "scored = divide by rows with parseable gold (honest).",
    )
    args = ap.parse_args()

    # Allow weight sweep via CLI without forking the search() function.
    if args.vec_weight is not None:
        VEC_W = args.vec_weight
    if args.kw_weight is not None:
        KW_W = args.kw_weight
    print(f"[info] fusion weights: vec={VEC_W:.2f}  kw={KW_W:.2f}")

    con = sqlite3.connect(args.db)
    print("[load] embedding matrix")
    matrix, ids = load_matrix(con)
    c2d = chunk_to_doc(con)
    print(f"[load] {len(ids)} chunks, dim={DIM}")

    qt = pq.read_table(args.questions).to_pandas()
    if args.limit:
        qt = qt.head(args.limit)

    ks = [1, 3, 5, args.top_k]
    hits_at = {k: 0 for k in ks}
    mrr_at = {k: 0.0 for k in ks}
    ndcg_at = {k: 0.0 for k in ks}
    per_type: dict[str, dict] = {}

    def _bump(qtype, rh, rm, rd):
        if qtype not in per_type:
            per_type[qtype] = {
                "scored": 0,
                "hits_at": {k: 0 for k in ks},
                "mrr_at": {k: 0.0 for k in ks},
                "ndcg_at": {k: 0.0 for k in ks},
            }
        per_type[qtype]["scored"] += 1
        for k in ks:
            per_type[qtype]["hits_at"][k] += rh[k]
            per_type[qtype]["mrr_at"][k] += rm[k]
            per_type[qtype]["ndcg_at"][k] += rd[k]

    scored = 0
    dropped = 0

    reranker_client: httpx.Client | None = None
    if args.rerank:
        reranker_client = httpx.Client(timeout=10)
        try:
            reranker_client.get(f"{RERANKER_URL}/health", timeout=3).raise_for_status()
            print(f"[rerank] connected to {RERANKER_URL}")
        except Exception as e:
            print(f"[rerank] WARNING: reranker not reachable at {RERANKER_URL} ({e})")
            print("[rerank] aborting — refusing to silently fall back to fusion-only.")
            reranker_client.close()
            return 2

    t0 = time.time()
    with httpx.Client(timeout=120) as ollama:
        for i, row in enumerate(qt.itertuples(index=False)):
            expected = set(parse_expected(row.expected_doc_ids))
            if not expected:
                dropped += 1
                continue
            scored += 1
            qtype = str(getattr(row, "question_type", "unknown") or "unknown")
            predicted = search(
                con, ollama, matrix, ids, c2d, row.question,
                top_n=args.top_k, reranker=reranker_client, mode=args.mode,
            )
            rh = {k: 0 for k in ks}
            rm = {k: 0.0 for k in ks}
            rd = {k: 0.0 for k in ks}
            for k in ks:
                top = predicted[:k]
                if expected & set(top):
                    hits_at[k] += 1
                    rh[k] = 1
                for idx, d in enumerate(top, start=1):
                    if d in expected:
                        mrr_at[k] += 1.0 / idx
                        rm[k] = 1.0 / idx
                        break
                v = ndcg(top, expected, k)
                ndcg_at[k] += v
                rd[k] = v
            _bump(qtype, rh, rm, rd)
            if scored % 25 == 0:
                dt = time.time() - t0
                print(f"  scored {scored} ({scored/dt:.1f} q/s)")

    if reranker_client is not None:
        reranker_client.close()

    n = len(qt) if args.denominator == "all" else scored
    mode = f"fusion + cross-encoder ({RERANKER_URL})" if args.rerank else "fusion only"
    print()
    print(f"[info] mode: {mode}")
    print(f"[info] total rows in slice: {len(qt)}")
    print(f"[info] rows with gold IDs (scored): {scored}")
    print(f"[info] rows dropped (no parseable gold IDs): {dropped}")
    print(f"[info] denominator mode: {args.denominator}  (n = {n})")
    print(f"[info] elapsed: {time.time()-t0:.1f}s")
    print()
    print("OVERALL")
    print(f"{'k':>4} {'Recall':>8} {'MRR':>8} {'nDCG':>8}")
    for k in ks:
        print(f"{k:>4} {hits_at[k]/n:>8.3f} {mrr_at[k]/n:>8.3f} {ndcg_at[k]/n:>8.3f}")

    print()
    print("BY QUESTION_TYPE")
    print(f"{'type':<28} {'R@1':>7} {'R@10':>7} {'MRR@10':>7} {'nDCG@10':>7} {'n':>5}")
    top_k = args.top_k
    for qt_name in sorted(per_type, key=lambda t: -per_type[t]["scored"]):
        row = per_type[qt_name]
        ns = row["scored"]
        if ns == 0:
            continue
        r1 = row["hits_at"][1] / ns
        rk = row["hits_at"][top_k] / ns
        mk = row["mrr_at"][top_k] / ns
        dk = row["ndcg_at"][top_k] / ns
        print(f"{qt_name:<28} {r1:>7.3f} {rk:>7.3f} {mk:>7.3f} {dk:>7.3f} {ns:>5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
