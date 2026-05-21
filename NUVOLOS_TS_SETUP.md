# Ship to Nuvolos — practical guide

End-to-end recipe for pulling this repo onto Nuvolos and getting both stacks
running. Last verified 2026-05-21 against branch `main` of
`NlcoIas/company-rag-agent` (fast-forwards Andre's `Nuvolos` branch with the
audit-fix commit on top).

Two stacks ship from the same checkout:

- **Path A — Python (Andre's full Nuvolos build)**: FastAPI backend + Gradio
  frontend + pgvector. The team's primary deliverable.
- **Path B — TypeScript (TUI agent)**: hand-rolled CLI agent over SQLite.
  Reference implementation; runs in a Nuvolos VS Code terminal.

Both share the same Ollama install on the Backend app. They differ only on
the data store (pgvector for A, local SQLite for B).

---

## 0. Once per Nuvolos workspace

Open the Backend VS Code app. Then:

```bash
# Rootless Ollama install — system installer needs sudo, this doesn't.
OLLAMA_VERSION=$(curl -fsSL https://api.github.com/repos/ollama/ollama/releases/latest \
                  | grep '"tag_name"' | cut -d'"' -f4)
mkdir -p ~/.local
curl -fsSL "https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst" \
     -o /tmp/ollama.tar.zst
tar -x --zstd -f /tmp/ollama.tar.zst -C ~/.local

# Persist PATH + Ollama config on shared LFS so models survive container
# restarts and KEEP_ALIVE=-1 prevents the 134-second cold-load penalty.
cat >> ~/.bashrc <<'EOF'
export PATH="$HOME/.local/bin:$PATH"
export OLLAMA_MODELS=/space_mounts/pars/ollama_models
export OLLAMA_KEEP_ALIVE=-1
EOF
source ~/.bashrc

# Start the daemon + pull models (~5.5 GB total)
ollama serve &
sleep 2
ollama pull nomic-embed-text   # 274 MB
ollama pull qwen3:8b           # 5.2 GB
# (NO `ollama create` needed — the backend passes num_ctx=32768 via the API)

# Clone the fork (replace with team URL if different)
cd /files
git clone https://github.com/NlcoIas/company-rag-agent.git
cd company-rag-agent
```

If you already cloned an earlier copy: `git pull` is enough — every file in
this guide is already in `main`.

---

## Path A — Python stack on Nuvolos

### A.1 Start the Database app (Nuvolos UI)

In Nuvolos, click into your team space and start the **Database** app.
pgvector is pre-installed. The first start may take 30–60s.

### A.2 Configure env vars (Backend app CONFIGURE panel)

Open the Backend app's CONFIGURE panel and set:

| Var | Value | Notes |
|---|---|---|
| `AGENT_API_KEY` | a long random string | Required to enable bearer-token auth — set before exposing the proxy URL to anyone. `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `ENABLE_BASH_TOOL` | `0` | Leave off. Enabling = full RCE in the container (the regex denylist is bypassable). |
| `ENABLE_FS_READ_TOOL` | `0` | Leave off. Container env carries `PGPASSWORD` and other secrets — `read` is a leak primitive even when sandboxed. |
| `ENABLE_FS_WRITE_TOOLS` | `0` | Leave off unless the team explicitly wants the agent to write to disk. |
| `AGENT_FS_ROOT` | unset (default `<cwd>/agent_workspace`) | If FS tools are enabled, sets the sandbox root. Default is safe: not the source tree. |
| `LLM_TIMEOUT_SEC` | `60` | Default is fine. Bump if you see timeouts on slow `/think` queries. |
| `PGHOST` / `PGPORT` / `PGUSER` / `PGPASSWORD` / `PGDATABASE` | inherited from Nuvolos service | Defaults match Andre's branch; only change if the Database app's hostname changes (visible in Applications → Database → CONFIGURE). |

> The frontend will need `BACKEND_URL` pointing at the Backend app's
> hostname on port 8500 (visible in Nuvolos Applications → Backend
> → CONFIGURE → Network info). Default in `frontend/app.py` is set for the
> current workspace; override via `BACKEND_URL` env var on the Frontend app
> if it changes.

### A.3 Build the pgvector index — Backend app, one-time

```bash
cd /files/company-rag-agent
pip install -r backend/requirements.txt pyarrow pandas
python indexing/build_index_pg.py --input data/raw/documents_subset.parquet
```

~15–30 min on Nuvolos. The script is **resumable** — if it dies, re-run the
same command and it picks up at the last 200-chunk checkpoint.

Done when you see: `[done] indexing complete — Backend API is ready to serve queries.`

### A.4 Start the API — Backend app, every session

```bash
# Terminal 1 — keep Ollama up
ollama serve

# Terminal 2 — the API
cd /files/company-rag-agent/backend
uvicorn main:app --host 0.0.0.0 --port 8500
```

Expected log lines (post-audit-fix):

```
INFO: Connecting to pgvector @ nv-service-...:5432/nuvolos
INFO: pgvector connected — 35344 chunks indexed.
INFO: LLM: http://localhost:11434  model=qwen3:8b  embed=nomic-embed-text  timeout=60.0s
INFO: Tools enabled: bash=False fs_read=False fs_write=False fs_root=/files/company-rag-agent/backend/agent_workspace
INFO: Auth: bearer-token
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8500
```

### A.5 Start the UI — Frontend app, every session

```bash
cd /files/company-rag-agent/frontend
pip install -r requirements.txt   # first time only
python app.py
```

UI: `https://<hash>.proxy-eu1.nuvolos.cloud/proxy/7860/`

If you set `AGENT_API_KEY` in step A.2, also set it as the Frontend's
`BACKEND_API_KEY` env (frontend reads it and forwards on every request).
Without this, the frontend will get 401 on every call.

### A.6 Optional — run the retrieval eval

```bash
cd /files/company-rag-agent
python indexing/eval_retrieval_pg.py --questions data/raw/questions_test.parquet --top-k 10
```

Watch the first line: `[eval] model=nomic-embed-text dim=768 fusion=rrf-v1`.
If you see `[WARN] FUSION_VERSION drift`, the eval and the backend are not
running the same retriever — fix before reporting numbers.

---

## Path B — TypeScript agent on Nuvolos

This stack is **undocumented in the original README**. Steps below are the
team-tested recipe.

### B.1 Node 22 (rootless via nvm)

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm install 22
node --version   # v22.x.x
```

### B.2 Build the SQLite index

The TS agent uses `data/index/rag.db` (SQLite + FTS5 + JSON-encoded dense
vectors), NOT pgvector. The file is gitignored — build it on Nuvolos.

```bash
cd /files/company-rag-agent
python -m venv data/.venv          # isolated from backend/requirements.txt
source data/.venv/bin/activate
pip install pyarrow numpy httpx pandas
python indexing/build_index.py \
    --input data/raw/documents_subset.parquet \
    --out   data/index/rag.db
deactivate
```

~5–10 min on Nuvolos (the embedder is small and Ollama batches well).
Final size ~318 MB. Resumable.

### B.3 Sanity-check retrieval (no LLM)

```bash
cd /files/company-rag-agent
npm install        # ~1 min
npx tsx src/rag/smoke.ts "who complained about the November invoice spike from HybridAI?"
```

Expect the gmail thread *"Unexpected spike on November invoice - HybridAI
migration tokens"* as the top hit, score ≈ 2.6.

### B.4 Launch the TUI

```bash
cd /files/company-rag-agent
npm start
```

`pi-tui` prompt appears. Type questions, Enter to send. `/quit` or Ctrl+C
exits. Tool calls (`read`, `write`, `edit`, `bash`) prompt for permission
before running — see `src/main.ts:30` for the auto-allow set
(`search`, `open_document`, `read`).

The TUI works in a Nuvolos VS Code terminal because that's a real raw-mode
terminal.

---

## What the audit fixes changed (so you know what to expect)

The push you're working from includes the audit-fix commit (`75a40ed`).
Behavioral changes vs Andre's prior `Nuvolos` branch tip:

1. **`add_document` / `edit_document` actually work now.** Before, every
   call silently failed at the schema layer (INSERT into a `GENERATED
   ALWAYS` column). If you tested the agent's "write a gmail" path before
   and it appeared to succeed but search couldn't find the doc — that was
   this bug.
2. **All tool-surface tools (`bash`, `read`, `write`, `edit`) are off by
   default.** They're opt-in via env vars. Without them, the agent can only
   `search`, `open_document`, `add_document`, `edit_document` — which is
   the full knowledge-base agentic loop and what the demo needs.
3. **The /query, /stats, /document endpoints accept an optional
   `Authorization: Bearer <AGENT_API_KEY>` header.** No env set = no auth =
   prior open behavior. Setting the env enables auth across all three
   endpoints simultaneously.
4. **Score numbers in search results look different.** Old scale: 0–4 (top
   hit ~2.6). New scale (RRF * 60): 0–2 (top hit ~1.5–1.9 for a strong
   match). The system prompt has been updated to reflect this — if you see
   the agent re-searching when scores look high, check the prompt has the
   new heuristic ("Score >= 1.0 strong, >= 1.5 very strong").
5. **Date filters now actually filter dates.** If you used `date_from` on
   non-gmail queries before, it was a no-op; now it returns only chunks
   with timestamps in range (which means mostly gmail and fireflies).
6. **`/query` errors are sanitized.** If a tool fails, the LLM sees
   `Tool error (TypeError)` instead of the raw exception. Full traceback
   goes to the server log (`log.exception`). Set logging to DEBUG if you
   need to chase one.

If anything in the demo regresses vs. before the fixes, the most likely
cause is one of:

- Frontend not forwarding the bearer token (gets 401 on every /query)
- Frontend reading old score thresholds from cached state
- `BACKEND_URL` pointing at the wrong service hostname after a Nuvolos
  workspace reprovision

---

## Quick troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| First query 30s+, then fast | Cold model load on first request | Expected. `OLLAMA_KEEP_ALIVE=-1` keeps it warm for subsequent calls. |
| All queries 60s+ | Ollama model not on GPU, or daemon under load | `nvidia-smi` to check; `ollama ps` to confirm model is loaded. |
| 401 from /query | `AGENT_API_KEY` set in backend but frontend doesn't send it | Set `BACKEND_API_KEY` env on Frontend app to the same value, restart `app.py`. |
| 500 from /query | Check Backend logs for the actual exception | `log.exception` writes the full traceback server-side. |
| Eval prints `FUSION_VERSION drift` | Eval and backend out of sync | Re-pull, both files have `FUSION_VERSION = "rrf-v1"` after audit fixes. |
| `add_document` returns "Tool error" | DB schema doesn't match — likely an old index | Re-run `build_index_pg.py --rebuild` to drop & recreate schema. |
| TS smoke test errors importing `tsx` | Run from project root, not parent | `cd /files/company-rag-agent && npx tsx src/rag/smoke.ts ...` |

---

## Known not-fixed (deferred from audit)

The audit identified more findings than this push addresses. Deferred:

- **Second-order prompt injection via corpus documents.** A malicious doc
  retrieved by `search` could try to instruct the model. Mitigation
  requires either a defensive system prompt clause or wrapping retrieved
  content in clear delimiters. Worth a follow-up; not blocking the demo.
- **Corpus-wide BM25 IDF.** Current implementation derives IDF from the
  24 returned rows per query. Fix requires precomputing corpus DF/avgdl
  at startup (~30-60s warmup). Skipped this sprint because RRF replaces
  the broken fusion math — BM25 absolute magnitude no longer affects fusion.
- **Per-Gradio-session cancel state.** Stop button uses a module-global
  `threading.Event`, so two concurrent users clobber each other's cancel
  flag. Low priority unless the demo has multiple simultaneous testers.
- **Connection pool.** Replaced with a single RLock for now; throughput
  limited to ~1 concurrent DB op. Fine for course demo.

See the audit transcript in the chat session for the full punch list.
