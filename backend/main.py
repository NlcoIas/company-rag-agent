"""
backend/main.py — Company RAG Agent API (Nuvolos)
==================================================
FastAPI agentic backend running on the Backend VS Code app.

Architecture:
  Frontend (Gradio, port 7860)
      │  HTTP POST /query
      ▼
  Backend (FastAPI, port 8500)     ← this file
      │  psycopg2 + pgvector       (hybrid retrieval)
      │  Ollama /api/embeddings    (nomic-embed-text — same model as local)
      │  Ollama /v1                (Qwen 3 8B — same model as local)
      ▼
  Database (PostgreSQL + pgvector, port 5432)

Run:
    # Terminal 1 — keep Ollama running
    OLLAMA_MODELS=/space_mounts/pars/ollama_models ollama serve

    # Terminal 2 — start the API
    cd /files/backend
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8500

Environment variables (set in Nuvolos Backend app CONFIGURE):
    OLLAMA_HOST     http://localhost:11434      (default)
    LLM_MODEL       qwen3-8b-32k               (default — matches Modelfile)
    EMBED_MODEL     nomic-embed-text            (default — same as local pipeline)
    PGHOST          nv-service-b01d63337fab32ac94f65eb2dc8a62ba  (default)
    PGPORT          5432
    PGUSER          nuvolos
    PGPASSWORD      nuvolos
    PGDATABASE      nuvolos
"""

import hashlib
import math
import os
import re
import json
import logging
import secrets
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import Optional

import httpx
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from openai import OpenAI
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL   = os.environ.get("LLM_MODEL",   "qwen3:8b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

DB_HOST     = os.environ.get("PGHOST",     "nv-service-b01d63337fab32ac94f65eb2dc8a62ba")
DB_PORT     = int(os.environ.get("PGPORT", "5432"))
DB_USER     = os.environ.get("PGUSER",     "nuvolos")
DB_PASSWORD = os.environ.get("PGPASSWORD", "nuvolos")
DB_NAME     = os.environ.get("PGDATABASE", "nuvolos")

TOP_K_PER_BRANCH = 8
# Reciprocal Rank Fusion (RRF) replaces the prior weighted-sum that gave the
# vector branch a ~5x scale advantage. RRF_K=60 is the standard TREC default.
# Displayed score = raw_rrf * RRF_DISPLAY_SCALE so the prompt heuristic stays
# in a human-friendly range — a top-of-one-branch hit ≈ 1.0, top-of-both ≈ 1.97.
RRF_K              = 60
RRF_DISPLAY_SCALE  = 60.0
SCORE_THRESHOLD    = 0.5      # on the displayed scale (raw ≈ 0.0083)
BM25_K1            = 1.5
BM25_B             = 0.75
# Tracks the fusion algorithm version so the eval harness can assert parity
# with the production retriever (see indexing/eval_retrieval_pg.py).
FUSION_VERSION     = "rrf-v1"
MAX_AGENT_TURNS  = 6
MAX_HISTORY_TURNS = 10   # keep last N conversation turns; bounds context growth
MAX_NEW_TOKENS   = 2048
TEMPERATURE      = 0.1
MAX_EMBED_CHARS  = 6000   # mirrors embed.py — prevents Ollama 500s on long inputs
TABLE_DOCS       = "rag_documents"
TABLE_CHUNKS     = "rag_chunks"

BASH_DENY = [
    r"\brm\s+-rf\b",
    r"\bsudo\b",
    r"\bdd\s+if=",
    r"\bmkfs\b",
    r"\b>\s*/dev/",
    r"\bshutdown\b",
    r"\breboot\b",
]

# Security gates (opt-in by env). Defaults: read-only filesystem, no bash, no
# host writes — safe for Nuvolos proxy URLs which are reachable by anyone with
# the link. Set ENABLE_*=1 in the Backend app CONFIGURE to opt in.
ENABLE_BASH_TOOL      = os.environ.get("ENABLE_BASH_TOOL",      "0") == "1"
ENABLE_FS_WRITE_TOOLS = os.environ.get("ENABLE_FS_WRITE_TOOLS", "0") == "1"
ENABLE_FS_READ_TOOL   = os.environ.get("ENABLE_FS_READ_TOOL",   "1") == "1"
AGENT_FS_ROOT         = os.path.realpath(os.path.expanduser(
    os.environ.get("AGENT_FS_ROOT", os.getcwd())
))
AGENT_API_KEY         = os.environ.get("AGENT_API_KEY", "").strip()
LLM_TIMEOUT_SEC       = float(os.environ.get("LLM_TIMEOUT_SEC", "60"))

# Matches Qwen3 <think>…</think> blocks (enabled when the model reasons aloud)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# ── Global singletons ──────────────────────────────────────────────────────────
llm_client: Optional[OpenAI] = None
db_conn = None
# RLock (reentrant) so db_edit_document → fetch_document → cursor re-entry works
# without deadlocking. Per the audit, the prior shared global cursor was a
# correctness bug under FastAPI's threadpool; we now open a short-lived cursor
# per call inside this lock. Throughput limited to ≈ 1 concurrent DB op, which
# is correct for the 1-3 user course-demo workload.
db_lock = threading.RLock()


@contextmanager
def db_cursor():
    """Short-lived autocommit cursor, serialized via db_lock."""
    with db_lock:
        with db_conn.cursor() as cur:
            yield cur


@contextmanager
def db_write_txn():
    """Explicit transaction for multi-statement writes. Rolls back on any error
    so add_document / edit_document never leave the DB in a half-updated state.
    Autocommit is temporarily disabled inside the lock and restored on exit."""
    with db_lock:
        db_conn.autocommit = False
        cur = db_conn.cursor()
        try:
            yield cur
            db_conn.commit()
        except Exception:
            try:
                db_conn.rollback()
            except Exception:
                log.exception("rollback failed")
            raise
        finally:
            cur.close()
            try:
                db_conn.autocommit = True
            except Exception:
                log.exception("could not restore autocommit")

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a company knowledge assistant. You help users retrieve information from
company documents, emails, and chats (sources: confluence, google_drive, jira,
linear, hubspot, github, fireflies, gmail, slack).

Primary workflow:
1. Call `search` ONCE with a natural-language query. Each result includes doc_id,
   source_type, a preview, and a fused score. Use optional filters when relevant:
   - source_types: when the user specifies a channel (email, slack, jira, etc.)
   - date_from / date_to (YYYY-MM-DD): only when the user is asking about WHEN
     something was communicated — e.g. "emails sent in November" or "last week's
     meeting notes". Do NOT add date filters when a time word describes the topic
     rather than the send date — e.g. "November invoice spike" means an event that
     happened in November, but the document discussing it may have been sent at any
     time (days, weeks, or months later). Let the search query keywords find it.
   - participant: when the user mentions a specific person.
2. Look at the top results. If the highest-scoring hit's preview clearly addresses
   the question, call `open_document` on its doc_id and answer from the full text.
   Score >= 1.0 means the chunk ranked at the top of at least one retrieval branch
   (vector or keyword) — almost always a strong match. Score >= 1.5 means it ranked
   highly in both branches (very strong). Do not keep re-searching above 1.0.
3. Only call `search` a SECOND time if (a) the opened document is clearly off-topic,
   or (b) you need a different piece of information from a different source.
   Never run more than 3 searches total for one question.
4. Always cite the doc_id(s) you used or created at the end of your answer, on a
   line like "Source: dsid_..." — copy the id verbatim, never invent one.
   After add_document or edit_document, always include the returned doc_id so the
   user can open the document directly.
5. If nothing relevant is found, say so plainly — do not fabricate.

You can also write to and edit the knowledge base:
- `add_document`: create a new document under any source type (gmail, slack,
  confluence, jira, etc.) and index it immediately for search.
- `edit_document`: update an existing document by doc_id and re-index it.

Use these when the user asks to record, draft, save, or update something in the
knowledge base — e.g. "write a gmail about the 29.05 meeting" or "edit that
Confluence page to add the new API endpoint".

IMPORTANT for write tasks: call `add_document` in your FIRST tool call — do NOT
narrate your plan or write a preamble first. Keep the document content concise
(1–3 short paragraphs, under 150 words) unless the user explicitly asks for more.
Writing on CPU is slow; brevity is essential.

You also have read, write, edit, and bash tools on the Backend container
filesystem (/files/). Use them only when the user is clearly asking about local
files, not for knowledge base operations.

Keep responses concise. Quote relevant excerpts from documents rather than
paraphrasing when accuracy matters.
"""

# ── Tool definitions ───────────────────────────────────────────────────────────
# The full menu — security gates filter this list below before exposing to the
# LLM. `bash`, `read`, `write`, `edit` are all opt-in via ENABLE_* env vars.
_ALL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search the company knowledge base (documents, emails, chats) using hybrid "
                "BM25 keyword + dense vector retrieval. Returns ranked chunks with a short "
                "preview so you can decide which document to open in full. Use optional "
                "filters to narrow by source, date range, or participant."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language search query."},
                    "source_types": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Optional list of source types to restrict the search to. One or more of: slack, gmail, linear, jira, confluence, google_drive, hubspot, github, fireflies.",
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Optional ISO-8601 date lower bound. Only chunks whose ts_to is null or >= date_from are considered (mostly meaningful for gmail — filters by when the message was sent, not by what it discusses).",
                    },
                    "date_to": {
                        "type": "string",
                        "description": "Optional ISO-8601 date upper bound (mostly meaningful for gmail — filters by when the message was sent, not by what it discusses).",
                    },
                    "participant": {"type": "string", "description": "Optional substring to match against participants (email or Slack handle). Use when the user asks who said or sent something."},
                    "top_n": {"type": "integer", "description": "How many fused hits to return (default 6, max 20)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_document",
            "description": "Fetch the full text of a document by its doc_id (as returned by `search`). Use this after `search` when a preview looks relevant and you need the complete content to answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "The doc_id string from a search result."},
                },
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_document",
            "description": (
                "Create a new document in the company knowledge base and index it for search. "
                "Use this when the user wants to add a note, memo, email draft, meeting summary, "
                "or any content to the knowledge base under a specific source category."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_type": {
                        "type": "string",
                        "description": "Category for the document. One of: gmail, slack, confluence, google_drive, jira, linear, hubspot, github, fireflies.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Title of the document (e.g. email subject, page title, ticket name).",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full text content to store and index.",
                    },
                    "participants": {
                        "type": "string",
                        "description": "Optional comma-separated list of participants (email addresses or names).",
                    },
                    "date": {
                        "type": "string",
                        "description": "Optional ISO-8601 date for the document, e.g. 2026-05-29.",
                    },
                },
                "required": ["source_type", "title", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_document",
            "description": (
                "Update the content of an existing document in the knowledge base and re-index it. "
                "Use new_content to replace the entire document, or old_string + new_string for a targeted edit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "The doc_id of the document to edit (from a search result).",
                    },
                    "new_content": {
                        "type": "string",
                        "description": "Replacement for the entire document content.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Exact text to find and replace (must appear exactly once in the document).",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text for old_string.",
                    },
                },
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a UTF-8 text file from the Backend container filesystem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write UTF-8 text to a file, overwriting any existing content. Creates parent directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Replace the first exact occurrence of old_string with new_string in a file. Fails if old_string is not found or appears more than once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":       {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command on the Backend container via /bin/bash. Returns stdout, stderr, and exit code. Refuses dangerous patterns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to run."},
                },
                "required": ["command"],
            },
        },
    },
]

# Apply security gates: filesystem/bash tools are off by default. Filter the
# advertised TOOLS list so the LLM never sees disabled tools — keeps the
# context window cleaner and removes the temptation entirely.
_GATE_BY_TOOL = {
    "bash":  ENABLE_BASH_TOOL,
    "read":  ENABLE_FS_READ_TOOL,
    "write": ENABLE_FS_WRITE_TOOLS,
    "edit":  ENABLE_FS_WRITE_TOOLS,
}
TOOLS = [t for t in _ALL_TOOLS if _GATE_BY_TOOL.get(t["function"]["name"], True)]


# ── Bearer-token auth (optional) ──────────────────────────────────────────────
def require_api_key(authorization: str | None = Header(None)):
    """If AGENT_API_KEY is set, require `Authorization: Bearer <key>` on
    protected endpoints. If unset, all endpoints are open (default — matches
    prior behavior; backwards-compatible). Uses constant-time compare."""
    if not AGENT_API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")
    token = authorization.split(" ", 1)[1].strip()
    if not secrets.compare_digest(token, AGENT_API_KEY):
        raise HTTPException(status_code=403, detail="Invalid API key")


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_client, db_conn

    log.info(f"Connecting to pgvector @ {DB_HOST}:{DB_PORT}/{DB_NAME}")
    kwargs: dict = dict(host=DB_HOST, port=DB_PORT, user=DB_USER, dbname=DB_NAME,
                        connect_timeout=10)
    if DB_PASSWORD:
        kwargs["password"] = DB_PASSWORD
    db_conn = psycopg2.connect(**kwargs)
    db_conn.autocommit = True
    register_vector(db_conn)
    with db_conn.cursor() as cur:
        cur.execute("SET ivfflat.probes = 10;")  # improves ANN recall (default=1)
        cur.execute(f"SELECT COUNT(*) FROM {TABLE_CHUNKS};")
        n_chunks = cur.fetchone()[0]
    log.info(f"pgvector connected — {n_chunks} chunks indexed.")

    log.info(f"LLM: {OLLAMA_HOST}  model={LLM_MODEL}  embed={EMBED_MODEL}  "
             f"timeout={LLM_TIMEOUT_SEC}s")
    llm_client = OpenAI(
        base_url=f"{OLLAMA_HOST}/v1",
        api_key="ollama",
        timeout=LLM_TIMEOUT_SEC,
    )
    log.info(f"Tools enabled: bash={ENABLE_BASH_TOOL} fs_read={ENABLE_FS_READ_TOOL} "
             f"fs_write={ENABLE_FS_WRITE_TOOLS} fs_root={AGENT_FS_ROOT}")
    log.info(f"Auth: {'bearer-token' if AGENT_API_KEY else 'disabled (open access)'}")
    log.info("Application startup complete.")

    yield

    if db_conn:
        try:
            db_conn.close()
        except Exception:
            log.exception("db close failed during shutdown")


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Company RAG Agent API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────
class HistoryMessage(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    question: str
    history: list[HistoryMessage] = []
    thinking_mode: bool = False

class Source(BaseModel):
    doc_id: str
    source_type: str
    title: Optional[str]
    score: float
    preview: str
    opened: bool = False   # True when the agent read the full document text

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: float
    tool_traces: list[str] = []
    thinking_steps: list[str] = []


# ── Embedding (Ollama, same model as local pipeline) ───────────────────────────
def _embed(text: str) -> list[float]:
    resp = httpx.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text[:MAX_EMBED_CHARS]},
        timeout=60.0,
    )
    resp.raise_for_status()
    vec = np.array(resp.json()["embedding"], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec.tolist()


# ── BM25 keyword scoring ───────────────────────────────────────────────────────
# Token pattern accepts alphanumeric — queries like "Q3 2025" or "ticket 4521"
# keep their distinctive numeric tokens. Audit found the prior alpha-only regex
# silently dropped these for a corpus with many ticket IDs and dates.
_TOKEN_RE = re.compile(r'[a-zA-Z0-9]\w*')


def _fts_or_query(q: str) -> str | None:
    """Tokenize and OR-join words for a PostgreSQL tsquery.
    Mirrors src/rag/fusion.ts ftsQuery: any-word matching for maximum recall."""
    words = [w for w in _TOKEN_RE.findall(q.lower()) if len(w) > 1]
    return " | ".join(words) if words else None


def _tokenize(text: str) -> list[str]:
    return [w for w in _TOKEN_RE.findall(text.lower()) if len(w) > 1]


def _bm25_scores(query_terms: list[str], candidates: list[tuple[int, str]]) -> dict[int, float]:
    """Compute BM25 scores for (chunk_id, text) pairs.
    Mirrors the BM25 ranking used by SQLite FTS5 in the local TypeScript agent."""
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


# ── Retrieval ──────────────────────────────────────────────────────────────────
def _build_filters(
    source_types: list[str] | None,
    date_from: str | None,
    date_to: str | None,
    participant: str | None,
) -> tuple[list[str], list]:
    clauses: list[str] = []
    params: list = []
    if source_types:
        ph = ",".join(["%s"] * len(source_types))
        clauses.append(f"source_type IN ({ph})")
        params.extend(source_types)
    # Date filters apply only to chunks with timestamps set (effectively gmail
    # per chunkers.py). Prior behavior `ts_to IS NULL OR ...` let every
    # non-dated chunk through, making the date filter a near no-op.
    if date_from:
        clauses.append("(ts_to IS NOT NULL AND ts_to >= %s)")
        params.append(date_from)
    if date_to:
        clauses.append("(ts_from IS NOT NULL AND ts_from <= %s)")
        params.append(date_to)
    if participant:
        clauses.append("participants_json LIKE %s")
        params.append(f"%{participant}%")
    return clauses, params


def rag_search(
    query: str,
    source_types: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    participant: str | None = None,
    top_n: int = 6,
) -> list[dict]:
    """Hybrid retrieval: vector ANN + BM25 keyword, fused via Reciprocal Rank
    Fusion (RRF). Per-branch ranks are combined as 1/(RRF_K+rank); the prior
    weighted-sum had the vector score on a ~5x larger scale than keyword.

    Eval parity: indexing/eval_retrieval_pg.py uses the same FUSION_VERSION.
    """
    top_n = max(1, min(20, top_n))
    qvec = _embed(query)
    filter_clauses, filter_params = _build_filters(source_types, date_from, date_to, participant)

    # Vector branch — produces ranked (chunk_id, row_data) list
    vec_clauses = filter_clauses + ["embedding IS NOT NULL"]
    vec_where = "WHERE " + " AND ".join(vec_clauses)
    vec_rows: list[tuple] = []
    with db_cursor() as cur:
        cur.execute(
            f"""SELECT chunk_id, doc_id, source_type, text, ts_from, ts_to,
                       (embedding <=> %s::vector) AS dist
                FROM   {TABLE_CHUNKS} {vec_where}
                ORDER  BY dist ASC LIMIT %s""",
            [qvec] + filter_params + [TOP_K_PER_BRANCH],
        )
        vec_rows = cur.fetchall()

    # vec_data[cid] -> (doc_id, source_type, text, ts_from, ts_to); vec_rank[cid] -> 1-based rank
    vec_data: dict[int, tuple] = {}
    vec_rank: dict[int, int] = {}
    for rank, (chunk_id, doc_id, source_type, text, ts_from, ts_to, _dist) in enumerate(vec_rows, 1):
        vec_data[chunk_id] = (doc_id, source_type, text, ts_from, ts_to)
        vec_rank[chunk_id] = rank

    # Keyword branch — pre-rank by ts_rank in SQL to get a meaningful 24-row
    # sample, then Python BM25 rerank to TOP_K_PER_BRANCH.
    kw_data: dict[int, tuple] = {}
    kw_rank: dict[int, int] = {}
    fts_q = _fts_or_query(query)
    if fts_q:
        try:
            kw_clauses = filter_clauses + ["text_tsv @@ to_tsquery('english', %s)"]
            kw_where = "WHERE " + " AND ".join(kw_clauses)
            with db_cursor() as cur:
                cur.execute(
                    f"""SELECT chunk_id, doc_id, source_type, text, ts_from, ts_to
                        FROM   {TABLE_CHUNKS} {kw_where}
                        ORDER  BY ts_rank(text_tsv, to_tsquery('english', %s)) DESC
                        LIMIT  %s""",
                    filter_params + [fts_q, fts_q, TOP_K_PER_BRANCH * 3],
                )
                kw_rows = cur.fetchall()
            query_terms = _tokenize(query)
            bm25 = _bm25_scores(query_terms, [(r[0], r[3]) for r in kw_rows])
            ranked = sorted(kw_rows, key=lambda r: bm25.get(r[0], 0.0), reverse=True)[:TOP_K_PER_BRANCH]
            for rank, (chunk_id, doc_id, source_type, text, ts_from, ts_to) in enumerate(ranked, 1):
                kw_data[chunk_id] = (doc_id, source_type, text, ts_from, ts_to)
                kw_rank[chunk_id] = rank
        except Exception as e:
            log.warning(f"Keyword branch failed (vector-only fallback): {e}")

    all_chunk_ids = set(vec_data) | set(kw_data)
    if not all_chunk_ids:
        return []

    # Fetch titles for the union of doc_ids
    all_doc_ids = list({(vec_data.get(cid) or kw_data[cid])[0] for cid in all_chunk_ids})
    ph = ",".join(["%s"] * len(all_doc_ids))
    with db_cursor() as cur:
        cur.execute(f"SELECT doc_id, title FROM {TABLE_DOCS} WHERE doc_id IN ({ph})", all_doc_ids)
        titles: dict[str, str | None] = {row[0]: row[1] for row in cur.fetchall()}
    missing = set(all_doc_ids) - set(titles)
    if missing:
        log.warning(f"Missing titles for doc_ids: {sorted(missing)[:5]}{' …' if len(missing)>5 else ''}")

    # RRF fusion
    fused: list[dict] = []
    for cid in all_chunk_ids:
        data = vec_data.get(cid) or kw_data[cid]
        vr = vec_rank.get(cid)
        kr = kw_rank.get(cid)
        rrf_raw = 0.0
        if vr is not None:
            rrf_raw += 1.0 / (RRF_K + vr)
        if kr is not None:
            rrf_raw += 1.0 / (RRF_K + kr)
        score = rrf_raw * RRF_DISPLAY_SCALE
        if score < SCORE_THRESHOLD:
            continue
        fused.append({
            "chunk_id":    cid,
            "doc_id":      data[0],
            "source_type": data[1],
            "title":       titles.get(data[0]),
            "score":       round(score, 3),
            "vec_rank":    vr,
            "kw_rank":     kr,
            "preview":     data[2][:320].replace("\n", " ").strip(),
            "ts_from":     data[3],
            "ts_to":       data[4],
        })

    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused[:top_n]


def fetch_document(doc_id: str) -> dict | None:
    with db_cursor() as cur:
        cur.execute(
            f"SELECT doc_id, source_type, title, content FROM {TABLE_DOCS} WHERE doc_id = %s",
            (doc_id,),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {"doc_id": row[0], "source_type": row[1], "title": row[2], "content": row[3]}


# ── Knowledge-base write helpers ──────────────────────────────────────────────
def _new_doc_id() -> str:
    return "dsid_" + hashlib.md5(uuid.uuid4().bytes).hexdigest()


def _simple_chunk(text: str, chunk_size: int = 2000, overlap: int = 200) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


def _insert_chunks(cur, doc_id: str, source_type: str, title: str, content: str,
                   participants: str | None, ts_from: str | None, ts_to: str | None):
    """Insert chunked, embedded rows. `text_tsv` is a GENERATED ALWAYS column
    (see indexing/schema_pg.sql) — Postgres populates it automatically from
    `text`. Writing to it directly raises 'cannot insert into column' and
    every add_document / edit_document silently failed before this fix."""
    participants_json = (
        json.dumps([p.strip() for p in participants.split(",")]) if participants else "[]"
    )
    for chunk_text in _simple_chunk(content):
        header_parts = [f"[source: {source_type}]", f"[title: {title}]"]
        if participants:
            header_parts.append(f"[participants: {participants}]")
        if ts_from:
            date_str = ts_from + (f" -> {ts_to}" if ts_to and ts_to != ts_from else "")
            header_parts.append(f"[dates: {date_str}]")
        full_text = " ".join(header_parts) + "\n\n" + chunk_text
        embedding = _embed(full_text)
        cur.execute(
            f"""INSERT INTO {TABLE_CHUNKS}
                (doc_id, source_type, text, ts_from, ts_to, participants_json, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s::vector)""",
            (doc_id, source_type, full_text, ts_from, ts_to, participants_json, embedding),
        )


def db_add_document(source_type: str, title: str, content: str,
                    participants: str | None = None, date: str | None = None) -> str:
    if not (content and content.strip()):
        raise ValueError("Document content must not be empty.")
    doc_id = _new_doc_id()
    ts = date or None
    # Wrap doc-row INSERT and per-chunk inserts in one transaction so a
    # mid-batch _embed failure rolls back to no doc + no chunks, rather than
    # leaving an orphan doc row (the prior bug, hidden by autocommit=True).
    with db_write_txn() as cur:
        cur.execute(
            f"INSERT INTO {TABLE_DOCS} (doc_id, source_type, title, content) VALUES (%s, %s, %s, %s)",
            (doc_id, source_type, title, content),
        )
        _insert_chunks(cur, doc_id, source_type, title, content, participants, ts, ts)
    return doc_id


def db_edit_document(doc_id: str, new_content: str | None = None,
                     old_string: str | None = None, new_string: str | None = None) -> str:
    doc = fetch_document(doc_id)
    if not doc:
        return f"Document not found: {doc_id}"
    if new_content is not None:
        updated = new_content
    elif old_string is not None and new_string is not None:
        count = doc["content"].count(old_string)
        if count == 0:
            return f"old_string not found in {doc_id}"
        if count > 1:
            return f"old_string appears {count} times — make it more specific"
        updated = doc["content"].replace(old_string, new_string, 1)
    else:
        return "Provide either new_content or both old_string and new_string."
    if not updated.strip():
        return "Refusing to write empty content."
    # Same transactional guarantee as add: UPDATE → DELETE old chunks → INSERT
    # new chunks all happen together or not at all.
    with db_write_txn() as cur:
        cur.execute(f"UPDATE {TABLE_DOCS} SET content = %s WHERE doc_id = %s", (updated, doc_id))
        cur.execute(f"DELETE FROM {TABLE_CHUNKS} WHERE doc_id = %s", (doc_id,))
        _insert_chunks(cur, doc_id, doc["source_type"], doc["title"] or "",
                       updated, None, None, None)
    return f"Updated {doc_id} and re-indexed."


# ── Filesystem sandbox ────────────────────────────────────────────────────────
def _safe_path(path: str) -> str:
    """Resolve and validate that `path` lives under AGENT_FS_ROOT.
    `realpath` resolves symlinks, so a symlink inside the sandbox pointing
    outside resolves to its true target and is rejected. Raises
    PermissionError on escape; raise text is intentionally generic to avoid
    leaking the sandbox root to the LLM (full path is logged server-side)."""
    if not path or "\x00" in path:
        raise PermissionError("Invalid path")
    expanded = os.path.realpath(os.path.expanduser(path))
    if expanded != AGENT_FS_ROOT and not expanded.startswith(AGENT_FS_ROOT + os.sep):
        log.warning(f"sandbox escape blocked: {path!r} -> {expanded}")
        raise PermissionError("Path not permitted by sandbox policy")
    return expanded


_DISABLED_TOOLS = {
    "bash":  not ENABLE_BASH_TOOL,
    "read":  not ENABLE_FS_READ_TOOL,
    "write": not ENABLE_FS_WRITE_TOOLS,
    "edit":  not ENABLE_FS_WRITE_TOOLS,
}


# ── Tool execution ─────────────────────────────────────────────────────────────
def execute_tool(name: str, args: dict) -> tuple[str, list[dict]]:
    """Returns (text_for_llm, search_hits_for_sources)."""

    # Belt-and-suspenders: even if a disabled tool was somehow registered or
    # replayed from history, refuse it here.
    if _DISABLED_TOOLS.get(name):
        return (f"Tool '{name}' is disabled on this deployment. "
                f"Set the corresponding ENABLE_* env var to enable."), []

    if name == "search":
        hits = rag_search(
            query=args["query"],
            source_types=args.get("source_types"),
            date_from=args.get("date_from"),
            date_to=args.get("date_to"),
            participant=args.get("participant"),
            top_n=args.get("top_n", 6),
        )
        if not hits:
            return "No results above the fusion threshold. Try broadening the query or removing filters.", []
        lines = []
        for h in hits:
            ts_from = h.get("ts_from") or ""
            ts_to   = h.get("ts_to") or ""
            if ts_from and ts_to and ts_to != ts_from:
                time_str = f" [{ts_from} → {ts_to}]"
            elif ts_from:
                time_str = f" [{ts_from}]"
            else:
                time_str = ""
            vr = h.get("vec_rank")
            kr = h.get("kw_rank")
            rank_str = f"vec_rank={vr if vr is not None else '-'} kw_rank={kr if kr is not None else '-'}"
            lines.append(
                f"#{h['chunk_id']} (doc={h['doc_id']}, source={h['source_type']}{time_str}, "
                f"score={h['score']} {rank_str})\n"
                f"title: {h['title'] or ''}\npreview: {h['preview']}"
            )
        return "\n\n".join(lines), hits

    if name == "open_document":
        doc = fetch_document(args["doc_id"])
        if not doc:
            return f"No document found with doc_id={args['doc_id']}", []
        source_hit = {
            "chunk_id": -1,
            "doc_id": doc["doc_id"],
            "source_type": doc["source_type"],
            "title": doc["title"],
            "score": 0.0,
            "vec_rank": None,
            "kw_rank": None,
            "preview": doc["content"][:320].replace("\n", " ").strip(),
            "ts_from": None,
            "ts_to": None,
            "opened": True,
        }
        return (
            f"doc_id: {doc['doc_id']}\nsource: {doc['source_type']}\n"
            f"title: {doc['title'] or ''}\n\n{doc['content']}"
        ), [source_hit]

    if name == "add_document":
        doc_id = db_add_document(
            source_type=args["source_type"],
            title=args["title"],
            content=args["content"],
            participants=args.get("participants"),
            date=args.get("date"),
        )
        source_hit = {
            "chunk_id": -1, "doc_id": doc_id,
            "source_type": args["source_type"], "title": args["title"],
            "score": 0.0, "vec_rank": None, "kw_rank": None,
            "preview": args["content"][:320].replace("\n", " ").strip(),
            "ts_from": args.get("date"), "ts_to": args.get("date"), "opened": True,
        }
        return (
            f"Created and indexed {doc_id} "
            f"(source={args['source_type']}, title={args['title']!r}). "
            f"Include {doc_id} in your answer."
        ), [source_hit]

    if name == "edit_document":
        result = db_edit_document(
            doc_id=args["doc_id"],
            new_content=args.get("new_content"),
            old_string=args.get("old_string"),
            new_string=args.get("new_string"),
        )
        doc = fetch_document(args["doc_id"])
        source_hit = {
            "chunk_id": -1, "doc_id": args["doc_id"],
            "source_type": doc["source_type"] if doc else "",
            "title": doc["title"] if doc else args["doc_id"],
            "score": 0.0, "vec_rank": None, "kw_rank": None,
            "preview": (doc["content"][:320].replace("\n", " ").strip()) if doc else "",
            "ts_from": None, "ts_to": None, "opened": True,
        }
        return result + f" Include {args['doc_id']} in your answer.", [source_hit]

    if name == "read":
        path = _safe_path(args["path"])
        with open(path, "r", encoding="utf-8") as f:
            return f.read(), []

    if name == "write":
        path = _safe_path(args["path"])
        # Sandbox-resolved path is guaranteed to live under AGENT_FS_ROOT before
        # we create any parent directories — prior code makedirs'd first, which
        # was itself a sandbox-escape primitive (mkdir -p /etc/cron.d/...).
        os.makedirs(os.path.dirname(path) or path, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(args["content"])
        return f"Wrote {len(args['content'])} bytes to {os.path.basename(path)}", []

    if name == "edit":
        path = _safe_path(args["path"])
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        old_s, new_s = args["old_string"], args["new_string"]
        count = text.count(old_s)
        if count == 0:
            return f"Error: old_string not found in {os.path.basename(path)}", []
        if count > 1:
            return f"Error: old_string appears {count} times — make it more specific", []
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.replace(old_s, new_s, 1))
        return f"Edited {os.path.basename(path)}", []

    if name == "bash":
        # ENABLE_BASH_TOOL must be opt-in. The BASH_DENY regex denylist is
        # known-bypassable (\rm -rf /, base64-decode-then-exec, etc.); treat
        # enabling this tool as conferring full RCE in the container.
        command = args["command"]
        for pattern in BASH_DENY:
            if re.search(pattern, command):
                return f"Refused: command matches deny pattern '{pattern}'. Run it manually if needed.", []
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=60, executable="/bin/bash",
            )
        except subprocess.TimeoutExpired:
            return "Tool error: bash command timed out after 60 seconds", []
        output = f"exit={result.returncode}\n--- stdout ---\n{result.stdout}"
        if result.stderr:
            output += f"\n--- stderr ---\n{result.stderr}"
        return output, []

    return f"Unknown tool: {name}", []


# ── Thinking helpers ───────────────────────────────────────────────────────────
def _extract_thinking(content: str) -> tuple[str, str]:
    """Split Qwen3 <think>…</think> blocks from visible content.
    Returns (thinking_text, clean_content). If no <think> blocks, thinking_text is ''."""
    thoughts: list[str] = []
    clean = THINK_RE.sub(lambda m: thoughts.append(m.group(1).strip()) or "", content)
    return "\n\n".join(thoughts), clean.strip()


# ── Trace helpers ──────────────────────────────────────────────────────────────
def _trace_args(name: str, args: dict) -> str:
    if name == "search":
        q = args.get("query", "")[:60]
        extras: list[str] = []
        if args.get("source_types"):
            extras.append(f"source={args['source_types']}")
        if args.get("participant"):
            extras.append(f"participant={args['participant']}")
        if args.get("date_from") or args.get("date_to"):
            extras.append(f"date={args.get('date_from', '')}..{args.get('date_to', '')}")
        return f'"{q}"' + (f" ({', '.join(extras)})" if extras else "")
    if name == "open_document":
        return args.get("doc_id", "")
    if name == "add_document":
        return f"{args.get('source_type', '')} / {args.get('title', '')[:50]}"
    if name == "edit_document":
        return args.get("doc_id", "")
    if name in ("read", "write", "edit"):
        return args.get("path", "")
    if name == "bash":
        return args.get("command", "")[:80]
    return ""


def _trace_result(name: str, result_text: str, hits: list) -> str:
    if name == "search":
        return f"{len(hits)} result(s)" if hits else "no results"
    if name == "open_document":
        return "not found" if result_text.startswith("No document") else f"{len(result_text):,} chars"
    if name == "add_document":
        m = re.search(r'dsid_[a-f0-9]+', result_text)
        return m.group(0) if m else "created"
    if name == "edit_document":
        return "updated" if "Updated" in result_text else result_text[:40]
    if name == "bash":
        m = re.match(r"exit=(\d+)", result_text)
        return f"exit={m.group(1)}" if m else "done"
    if result_text.startswith("Error"):
        return result_text[:60]
    return "done"


# ── Agent loop ─────────────────────────────────────────────────────────────────
def run_agent(
    question: str, history: list[HistoryMessage], thinking_mode: bool = False
) -> tuple[str, list[dict], list[str], list[str]]:
    system_content = "/think\n\n" + SYSTEM_PROMPT if thinking_mode else SYSTEM_PROMPT
    messages: list[dict] = [{"role": "system", "content": system_content}]
    for h in history[-MAX_HISTORY_TURNS * 2:]:   # each turn = 2 messages (user + assistant)
        messages.append({"role": h.role, "content": h.content})
    messages.append({"role": "user", "content": question})

    all_sources: list[dict] = []
    traces: list[str] = []
    thinking_steps: list[str] = []

    for _ in range(MAX_AGENT_TURNS):
        resp = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            extra_body={"options": {"num_ctx": 32768, **({"think": True} if thinking_mode else {})}},
        )
        msg = resp.choices[0].message
        raw_content = msg.content or ""

        # Extract <think>…</think> blocks — present when Qwen3 reasoning is active
        think_text, visible_content = _extract_thinking(raw_content)
        # Newer Ollama versions may return thinking in model_extra["thinking"] instead of
        # inline <think> tags when using the OpenAI-compatible endpoint
        if not think_text and thinking_mode:
            model_extra = getattr(msg, 'model_extra', None) or {}
            extra_think = model_extra.get('thinking') or model_extra.get('think_content')
            if extra_think:
                think_text = str(extra_think)
                visible_content = raw_content  # content is already clean (no <think> tags)
        if think_text:
            thinking_steps.append(think_text)

        if not msg.tool_calls:
            answer = visible_content.strip()
            if not answer and thinking_steps:
                # Model put everything in <think> — surface last reasoning block as answer
                answer = thinking_steps[-1]
            elif not answer:
                answer = "The model did not produce an answer. Please try again."
            return answer, all_sources, traces, thinking_steps  # noqa

        # Capture inter-turn reasoning text (what the model says before calling tools)
        if visible_content.strip():
            thinking_steps.append(visible_content.strip())

        messages.append({
            "role":       "assistant",
            "content":    raw_content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            args_parsed: dict = {}
            try:
                # json.loads is INSIDE the try so a malformed tool-call
                # argument (qwen3 thinking mode occasionally emits these) is
                # surfaced to the model as a tool error rather than crashing
                # the request. Also guarantees we still append a tool message
                # with this tc.id, so Ollama's tool_call/tool_result pairing
                # contract holds and the next agent turn doesn't 400.
                args_parsed = json.loads(tc.function.arguments)
                result_text, hits = execute_tool(tc.function.name, args_parsed)
                all_sources.extend(hits)
                summary = _trace_result(tc.function.name, result_text, hits)
            except PermissionError:
                # Sandbox / auth rejection — generic message, full path is
                # logged inside _safe_path; don't echo paths into the LLM.
                result_text = "Tool error: operation not permitted by sandbox policy"
                summary = "error: not permitted"
            except json.JSONDecodeError as e:
                result_text = f"Tool error: invalid JSON arguments ({e})"
                summary = "error: malformed args"
            except Exception as e:
                log.exception("tool '%s' failed", tc.function.name)
                # Surface the exception class only — full traceback logged
                # server-side; raw exception message could leak host paths.
                result_text = f"Tool error ({type(e).__name__})"
                summary = f"error: {type(e).__name__}"
            traces.append(
                f"[{tc.function.name}] {_trace_args(tc.function.name, args_parsed)} → {summary}"
            )
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})

    return "Agent reached the maximum number of turns without a final answer.", all_sources, traces, thinking_steps


# ── Endpoints ──────────────────────────────────────────────────────────────────
# /health stays open (no auth) so the Frontend's startup probe can detect
# backend readiness even before the user has the API key. Everything else
# accepts an optional bearer token via require_api_key.
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": LLM_MODEL,
        "embed": EMBED_MODEL,
        "ollama": OLLAMA_HOST,
        "auth": "required" if AGENT_API_KEY else "disabled",
        "fusion_version": FUSION_VERSION,
    }


@app.get("/stats", dependencies=[Depends(require_api_key)])
def stats():
    with db_cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {TABLE_CHUNKS};")
        n_chunks = cur.fetchone()[0]
        cur.execute(f"SELECT COUNT(*) FROM {TABLE_DOCS};")
        n_docs = cur.fetchone()[0]
        cur.execute(
            f"SELECT source_type, COUNT(*) FROM {TABLE_CHUNKS} "
            f"GROUP BY source_type ORDER BY COUNT(*) DESC;"
        )
        by_source = {r[0]: r[1] for r in cur.fetchall()}
    return {"chunks": n_chunks, "documents": n_docs, "by_source": by_source}


@app.get("/document/{doc_id}", dependencies=[Depends(require_api_key)])
def get_document(doc_id: str):
    doc = fetch_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
    return doc


@app.post("/query", response_model=QueryResponse,
          dependencies=[Depends(require_api_key)])
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    t0 = time.perf_counter()
    answer, raw_sources, traces, thinking = run_agent(req.question, req.history, req.thinking_mode)
    latency = round((time.perf_counter() - t0) * 1000, 1)

    # Keep best search hit per doc_id; opened documents are always included.
    # Guard the elif with `not opened` — otherwise a later search that returns
    # the same doc_id with score > 0 would clobber the opened=True flag.
    best: dict[str, dict] = {}
    for s in raw_sources:
        if s.get("opened"):
            best[s["doc_id"]] = s          # opened always wins — it was actually read
        elif (s["doc_id"] not in best
              or (not best[s["doc_id"]].get("opened")
                  and s["score"] > best[s["doc_id"]]["score"])):
            best[s["doc_id"]] = s

    # Opened docs first, then search hits sorted by score descending
    opened  = [s for s in best.values() if s.get("opened")]
    found   = sorted([s for s in best.values() if not s.get("opened")],
                     key=lambda x: x["score"], reverse=True)

    return QueryResponse(
        answer=answer,
        sources=[
            Source(doc_id=s["doc_id"], source_type=s["source_type"],
                   title=s.get("title"), score=s["score"], preview=s["preview"],
                   opened=s.get("opened", False))
            for s in opened + found
        ],
        latency_ms=latency,
        tool_traces=traces,
        thinking_steps=thinking,
    )
