"""End-to-end accuracy evaluation: retrieval → LLM → fact scoring.

Selects N questions whose answer documents are present in the local RAG
index, runs the full search → fetch → LLM → score pipeline, and reports
retrieval and answer-quality metrics.

Usage:
    # Fusion only:
    python indexing/eval_e2e.py \
        --db        data/index/rag.db \
        --questions data/raw/questions_test.parquet \
        --n         25

    # With cross-encoder reranker (start it first):
    python indexing/eval_e2e.py \
        --db        data/index/rag.db \
        --questions data/raw/questions_test.parquet \
        --n         25 \
        --rerank

Environment variables honoured:
    OLLAMA_HOST          Ollama base URL      (default http://127.0.0.1:11434)
    LLM_MODEL            Model for answering  (default qwen2.5:9b)
    LLM_DISABLE_THINKING If "1", appends /no_think (for qwen3 family)
    RAG_EMBED_MODEL      Embedding model      (default nomic-embed-text)
    RERANKER_URL         Reranker service URL (default http://127.0.0.1:8001)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from embed import DIM, embed_one  # noqa: E402

# ── retrieval constants (keep in sync with fusion.ts / eval_retrieval.py) ────
TOP_K_PER_BRANCH = 16
KW_W = 0.7
VEC_W = 0.3
SCALE = 4
THRESHOLD = 0.30
RERANK_POOL = TOP_K_PER_BRANCH * 2

# ── service URLs ──────────────────────────────────────────────────────────────
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.5-9b-32k")
RERANKER_URL = os.environ.get("RERANKER_URL", "http://127.0.0.1:8001")

MAX_DOC_CHARS = 4000  # cap document text sent to the LLM

# words ignored when scoring facts
_STOP = {
    "that", "this", "with", "from", "have", "been", "were", "they", "their",
    "what", "when", "where", "which", "about", "into", "will", "than", "then",
    "them", "also", "some", "would", "could", "should", "these", "those",
}


# ── retrieval ─────────────────────────────────────────────────────────────────

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
    words = [w for w in re.sub(r"[^\w\s]", " ", q.lower()).split() if len(w) > 1]
    return " OR ".join(f'"{w}"' for w in words) if words else '""'


def load_matrix(con: sqlite3.Connection):
    rows = con.execute(
        "SELECT chunk_id, embedding FROM chunks WHERE embedding IS NOT NULL ORDER BY chunk_id"
    ).fetchall()
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    mat = np.zeros((len(rows), DIM), dtype=np.float32)
    for i, (_, blob) in enumerate(rows):
        mat[i] = np.frombuffer(blob, dtype=np.float32)
    return mat, ids


def chunk_to_doc(con: sqlite3.Connection) -> dict[int, str]:
    return {r[0]: r[1] for r in con.execute("SELECT chunk_id, doc_id FROM chunks").fetchall()}


def call_reranker(client: httpx.Client, query: str, passages: list[str]) -> list[float] | None:
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


def search(
    con: sqlite3.Connection,
    embed_client: httpx.Client,
    matrix: np.ndarray,
    ids: np.ndarray,
    c2d: dict[int, str],
    query: str,
    top_n: int = 6,
    reranker: httpx.Client | None = None,
) -> list[str]:
    def top_k_vec(qv: np.ndarray) -> dict[int, float]:
        sims = matrix @ qv
        order = np.argsort(-sims)[:TOP_K_PER_BRANCH]
        return {int(ids[i]): float(sims[i]) * SCALE for i in order}

    qvec = embed_one(embed_client, query)
    vec_scores = top_k_vec(qvec)

    if len(query.split()) > 6:
        stripped = strip_to_key_terms(query)
        if stripped:
            qvec2 = embed_one(embed_client, stripped)
            for cid, score in top_k_vec(qvec2).items():
                if score > vec_scores.get(cid, -float("inf")):
                    vec_scores[cid] = score

    rows = con.execute(
        "SELECT c.chunk_id FROM chunks_fts f JOIN chunks c ON c.chunk_id = f.rowid "
        "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
        (fts_query(query), TOP_K_PER_BRANCH),
    ).fetchall()
    kw_scores = {cid: (1 / (1 + rank)) * SCALE for rank, (cid,) in enumerate(rows, 1)}

    all_ids = set(vec_scores) | set(kw_scores)
    fused = sorted(
        [
            (cid, VEC_W * vec_scores.get(cid, 0.0) + KW_W * kw_scores.get(cid, 0.0))
            for cid in all_ids
        ],
        key=lambda x: -x[1],
    )
    pool = [(cid, sc) for cid, sc in fused[:RERANK_POOL] if sc >= THRESHOLD]

    if reranker is not None and pool:
        chunk_ids = [cid for cid, _ in pool]
        ph = ",".join("?" * len(chunk_ids))
        texts = {r[0]: r[1] for r in con.execute(
            f"SELECT chunk_id, text FROM chunks WHERE chunk_id IN ({ph})", chunk_ids
        ).fetchall()}
        passages = [texts.get(cid, "") for cid, _ in pool]
        scores = call_reranker(reranker, query, passages)
        if scores and len(scores) == len(pool):
            pool = sorted(zip([cid for cid, _ in pool], scores), key=lambda x: -x[1])

    seen: list[str] = []
    for cid, _ in pool:
        doc = c2d.get(cid)
        if doc and doc not in seen:
            seen.append(doc)
        if len(seen) >= top_n:
            break
    return seen


# ── LLM answer generation ─────────────────────────────────────────────────────

def ask_llm(client: httpx.Client, question: str, doc_text: str) -> str:
    context = doc_text[:MAX_DOC_CHARS]
    if len(doc_text) > MAX_DOC_CHARS:
        context += "\n[...truncated]"

    disable_think = os.environ.get("LLM_DISABLE_THINKING") == "1"
    system = (
        "You are a company knowledge assistant. "
        "Answer the question using only the provided document. "
        "Be concise and specific."
    )
    prompt = f"Document:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    body: dict = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"num_predict": 500},
    }
    if disable_think:
        body["think"] = False

    r = client.post(f"{OLLAMA_HOST}/api/chat", json=body, timeout=180)
    r.raise_for_status()
    content = r.json()["message"]["content"]
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content


# ── fact scoring ──────────────────────────────────────────────────────────────

def score_fact(answer: str, fact: str) -> float:
    """Fraction of significant words in `fact` that appear in `answer`."""
    answer_lower = answer.lower()
    words = [
        w for w in re.findall(r"\b\w+\b", fact.lower())
        if len(w) > 3 and w not in _STOP
    ]
    if not words:
        return 1.0
    return sum(1 for w in words if w in answer_lower) / len(words)


def parse_list_field(v) -> list[str]:
    """Parse a field that may be a list, a JSON string, or a Python-repr string."""
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if x is not None]
    text = str(v).strip().replace("'", '"')
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return [text] if text else []


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--n", type=int, default=25, help="Questions to evaluate (default 25).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default 42).")
    ap.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking.")
    ap.add_argument(
        "--fact-threshold", type=float, default=0.6,
        help="Min word-coverage for a fact to count as found (default 0.6).",
    )
    args = ap.parse_args()

    con = sqlite3.connect(args.db)

    # Docs present in our index
    indexed = {r[0] for r in con.execute("SELECT doc_id FROM documents").fetchall()}

    print("[load] embedding matrix …")
    matrix, ids = load_matrix(con)
    c2d = chunk_to_doc(con)
    print(f"[load] {len(ids)} chunks across {len(indexed)} documents")

    # Load questions, keep only those answerable from our subset
    qt = pq.read_table(args.questions).to_pandas()
    qt = qt[
        qt.apply(
            lambda r: any(d in indexed for d in parse_list_field(r.expected_doc_ids)),
            axis=1,
        )
    ].reset_index(drop=True)
    print(f"[filter] {len(qt)} / {len(pq.read_table(args.questions))} questions "
          f"have answer docs in the index")

    if len(qt) == 0:
        print("[error] no eligible questions — check that the DB and questions file match")
        return 1
    n = min(args.n, len(qt))
    if n < args.n:
        print(f"[warn] only {n} eligible questions available, evaluating all of them")

    sample = qt.sample(n=n, random_state=args.seed).reset_index(drop=True)

    # Optional reranker
    reranker_client: httpx.Client | None = None
    if args.rerank:
        reranker_client = httpx.Client(timeout=10)
        try:
            reranker_client.get(f"{RERANKER_URL}/health", timeout=3).raise_for_status()
            print(f"[rerank] connected to {RERANKER_URL}")
        except Exception as e:
            print(f"[rerank] WARNING: not reachable ({e}) — running without reranker")
            reranker_client = None

    print(f"\n[eval ] model={LLM_MODEL}  n={n}  seed={args.seed}  "
          f"rerank={'yes' if reranker_client else 'no'}\n")

    results = []
    t0 = time.time()

    with httpx.Client(timeout=120) as embed_client, \
         httpx.Client(timeout=180) as llm_client:

        for i, row in enumerate(sample.itertuples(index=False)):
            question = str(row.question)
            expected = set(parse_list_field(row.expected_doc_ids))
            facts = parse_list_field(row.answer_facts)

            # Retrieve
            retrieved = search(
                con, embed_client, matrix, ids, c2d, question,
                top_n=6, reranker=reranker_client,
            )
            hit_at_6 = bool(expected & set(retrieved))
            hit_at_1 = bool(retrieved) and retrieved[0] in expected

            # Fetch top-3 documents as LLM context (mirrors live agent behaviour)
            doc_parts = []
            for doc_id in retrieved[:3]:
                row_db = con.execute(
                    "SELECT content FROM documents WHERE doc_id = ?", (doc_id,)
                ).fetchone()
                if row_db and row_db[0]:
                    doc_parts.append(f"[{doc_id}]\n{row_db[0]}")
            doc_text = "\n\n---\n\n".join(doc_parts)

            # Generate answer
            try:
                answer = ask_llm(llm_client, question, doc_text)
            except Exception as e:
                answer = f"[LLM ERROR: {e}]"

            # Score facts
            fact_scores = [score_fact(answer, f) for f in facts]
            facts_found = sum(1 for s in fact_scores if s >= args.fact_threshold)
            fact_recall = facts_found / len(facts) if facts else 0.0

            results.append({
                "q": i + 1,
                "question": question,
                "hit_at_1": hit_at_1,
                "hit_at_6": hit_at_6,
                "facts_found": facts_found,
                "total_facts": len(facts),
                "fact_recall": fact_recall,
                "answer": answer,
            })

            mark = "✓" if hit_at_1 else ("~" if hit_at_6 else "✗")
            print(f"  [{i+1:02d}/{n}] {mark}  facts {facts_found}/{len(facts)} "
                  f"({fact_recall:.0%})  {question[:65]}")

    if reranker_client:
        reranker_client.close()

    elapsed = time.time() - t0
    n_res = len(results)

    hit1 = sum(r["hit_at_1"] for r in results)
    hit6 = sum(r["hit_at_6"] for r in results)
    avg_fr = sum(r["fact_recall"] for r in results) / n_res

    mode = "fusion + reranker" if args.rerank else "fusion only"
    print(f"\n{'─' * 58}")
    print(f"  mode            {mode}")
    print(f"  questions       {n_res}   seed={args.seed}")
    print(f"  elapsed         {elapsed:.0f}s  ({elapsed / n_res:.1f}s / question)")
    print(f"{'─' * 58}")
    print(f"  Retrieval Hit@1  {hit1 / n_res:.3f}  ({hit1}/{n_res})")
    print(f"  Retrieval Hit@6  {hit6 / n_res:.3f}  ({hit6}/{n_res})")
    print(f"  Avg Fact Recall  {avg_fr:.3f}")
    print(f"{'─' * 58}")

    # Show low-scoring answers for inspection
    low = [r for r in results if r["fact_recall"] < 0.5]
    if low:
        print(f"\n  Low-scoring answers ({len(low)}) — check retrieval or LLM quality:")
        for r in low:
            tag = "retrieval miss" if not r["hit_at_6"] else "LLM miss"
            print(f"    Q{r['q']:02d} [{tag}] {r['question'][:70]}")
            print(f"         facts={r['facts_found']}/{r['total_facts']}  "
                  f"answer: {r['answer'][:100]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
