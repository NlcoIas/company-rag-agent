# Ship the TypeScript agent to Nuvolos

End-to-end recipe for running the TS TUI agent (`src/main.ts`) on Nuvolos.
This is **Edoardo's TS stack**, not Andre's Python backend — different file
tree, different data store, different runtime.

Reference: branch `main` of `NlcoIas/company-rag-agent` (your fork — has
Andre's TS correctness fix `qwen3.5-9b-32k` → `qwen3-8b-32k` baked in;
Edoardo's upstream `main` *won't run* without that one-line fix).

If you'd rather use Edoardo's upstream directly: clone it, then manually
edit `src/model.ts:4-5` to swap `qwen3.5-9b-32k` for `qwen3-8b-32k` and
the name string accordingly.

---

## 0. Nuvolos workspace setup

### Pick the right image (Nuvolos app catalog)

- **Editor / Backend app** → `VSCode 1.108.0 + Py3.13 + UZH LLMs` (2026-03-05).
  Try this one first — if it ships with Ollama + qwen3 pre-baked, skip the
  install/pull steps below. Check with `which ollama` and `ollama list`
  before you start. **Fallback**: `VSCode 1.117.0 with Py3.13` (latest plain).
- The TS agent doesn't need the **Database** app (that's pgvector for Andre's
  Python stack). You can leave it stopped.
- **Frontend** app is also not needed — the TS agent is a CLI TUI, you
  interact with it in the same terminal where you started it.

### Install Ollama (skip if `which ollama` already returns a path)

Rootless install — Nuvolos containers don't have sudo:

```bash
OLLAMA_VERSION=$(curl -fsSL https://api.github.com/repos/ollama/ollama/releases/latest \
                  | grep '"tag_name"' | cut -d'"' -f4)
mkdir -p ~/.local
curl -fsSL "https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst" \
     -o /tmp/ollama.tar.zst
tar -x --zstd -f /tmp/ollama.tar.zst -C ~/.local
```

### Persist PATH and Ollama config

`OLLAMA_MODELS` puts the ~5 GB of weights on the shared LFS so they survive
container restart. `OLLAMA_KEEP_ALIVE=-1` prevents the 134-second cold-load
penalty after Ollama's default 5-min idle unload.

```bash
cat >> ~/.bashrc <<'EOF'
export PATH="$HOME/.local/bin:$PATH"
export OLLAMA_MODELS=/space_mounts/pars/ollama_models
export OLLAMA_KEEP_ALIVE=-1
EOF
source ~/.bashrc
```

### Start the daemon and pull both models (~5.5 GB total)

```bash
ollama serve &
sleep 2
ollama pull nomic-embed-text   # 274 MB, for retrieval embeddings
ollama pull qwen3:8b           # 5.2 GB, the chat model the TS agent uses
```

If the Cloudflare CDN closes the connection mid-pull (it sometimes does for
concurrent pulls): re-run the same `ollama pull` — both are resumable.

---

## 1. Clone your fork

```bash
cd /files
git clone https://github.com/NlcoIas/company-rag-agent.git
cd company-rag-agent
```

(If you want only the TS code without the unused Python files, that's a
cosmetic ask — they're harmless. `npm start` ignores `backend/`,
`frontend/`, and `indexing/*_pg.*`.)

---

## 2. Create the `qwen3-8b-32k` custom model

The TS agent's `src/model.ts` references the model id `qwen3-8b-32k`,
which doesn't exist out-of-the-box. The `Modelfile` in the repo root
creates it (FROM `qwen3:8b` + `PARAMETER num_ctx 32768`):

```bash
cd /files/company-rag-agent
ollama create qwen3-8b-32k -f Modelfile
ollama list   # confirm qwen3-8b-32k:latest appears
```

This is fast — it just adds a 32k-context layer on top of the existing
`qwen3:8b` weights. No additional download.

---

## 3. Install Node 22 (rootless via nvm)

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm install 22
node --version   # v22.x.x
```

---

## 4. Build the SQLite index (`data/index/rag.db`)

The TS agent uses SQLite + FTS5 + JSON-encoded dense vectors — not
pgvector. `rag.db` is gitignored and must be built once on Nuvolos.

```bash
cd /files/company-rag-agent
python -m venv data/.venv
source data/.venv/bin/activate
pip install pyarrow numpy httpx pandas
python indexing/build_index.py \
    --input data/raw/documents_subset.parquet \
    --out   data/index/rag.db
deactivate
```

Expected time on Nuvolos: ~5–10 minutes (the embedder is small, Ollama
batches well, GPU helps). Final file is ~318 MB. Resumable — re-run the
same command if it dies mid-build.

---

## 5. Sanity-check retrieval (no LLM, fast)

```bash
cd /files/company-rag-agent
npm install        # ~1 min, installs pi-agent-core + pi-tui + deps
npx tsx src/rag/smoke.ts "who complained about the November invoice spike from HybridAI?"
```

Expect the gmail thread *"Unexpected spike on November invoice - HybridAI
migration tokens"* as the top hit with score ≈ 2.6. If the top hit looks
right, the retriever is wired correctly.

---

## 6. Launch the TUI

```bash
cd /files/company-rag-agent
npm start
```

The `pi-tui` chat prompt appears. Type a question and press Enter.

| Key / command | What it does |
|---|---|
| `Enter` | Send the question to the agent |
| `Ctrl+C` or `/quit` | Exit |
| `/reset` | Clear the conversation transcript |

The agent has six tools: `search`, `open_document`, `read` (auto-allowed),
plus `write`, `edit`, `bash` (each prompts you to allow once / always /
deny on first use — see `src/main.ts:30`).

The TUI works in a Nuvolos VS Code terminal because that's a real
raw-mode terminal. **Do not** try to run `npm start` over a tool wrapper
that captures stdout — it needs the live TTY.

---

## What to expect from latencies

- **First query**: ~30 seconds. One-time model-load cost. Subsequent queries
  stay warm because `OLLAMA_KEEP_ALIVE=-1` is set.
- **Warm queries**: depends on the answer length. Pure retrieval (smoke
  test) is ~2 seconds. A full agent turn (search + reason + answer) is
  typically 5–30 seconds on Nuvolos GPU.
- **Reasoning mode**: not exposed in the TS TUI (`src/main.ts:39` sets
  `thinkingLevel: "off"`). To enable, change to `"high"` and rebuild.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Error: model 'qwen3.5-9b-32k' not found` | Using Edoardo's upstream un-patched, or the Modelfile create step was skipped | Either (a) patch `src/model.ts:4-5` to `qwen3-8b-32k`, or (b) run `ollama create qwen3-8b-32k -f Modelfile`. Both. |
| `Error: connect ECONNREFUSED 127.0.0.1:11434` | Ollama daemon not running | `ollama serve &` in the same shell (or a separate terminal) |
| Smoke test returns no hits | Index not built or wrong path | `ls -lh data/index/rag.db` (should be ~318 MB); rebuild if missing |
| First query takes 100+ seconds | Cold-load from `/space_mounts/pars` (~87 s LFS read + 47 s CUDA init) | Expected. Subsequent queries are fast thanks to `KEEP_ALIVE=-1`. If it stays slow, check `nvidia-smi` — model should occupy ~4.5 GiB VRAM. |
| `npx tsx` says "Cannot find module …/src/rag/smoke.ts" | Running from parent dir | `cd /files/company-rag-agent` first |
| TUI shows garbage characters / arrow keys don't work | Running through a terminal wrapper that doesn't pass raw input | Use the Nuvolos VS Code terminal directly, not a sub-shell |

---

## Differences from Edoardo's local Mac/Linux setup

Edoardo's README assumes macOS/Linux local dev. On Nuvolos:

- **Sudo is unavailable** → Ollama install is rootless (the `tar.zst` path
  above), not the system installer Edoardo's README suggests.
- **`/space_mounts/pars`** is shared LFS — use it for Ollama models so
  they survive container restart and so teammates' instances can share
  the same weights.
- **GPU is available** — Tesla T4 with 14.6 GiB free VRAM. `qwen3:8b`
  fully offloads (37/37 layers, ~4.5 GiB VRAM). Warm inference ≈ 4s for
  short responses.
- **Edoardo's model id (`qwen3.5-9b-32k`) doesn't exist on stock Ollama** —
  your fork has Andre's fix to `qwen3-8b-32k`. If you ever rebase off
  Edoardo's upstream `main`, re-apply this change to `src/model.ts:4-5`.

---

## Not used here (but in the same repo)

If anyone on the team wants to run **Andre's Python/Gradio/pgvector
stack** instead of the TS TUI, see `README.md` section "Running on
Nuvolos" — same Ollama install, but uses the **Database** Nuvolos app
(pgvector) and starts FastAPI + Gradio across the Backend + Frontend
apps. The audit-fix commit (`75a40ed`) on this branch added env-gated
security flags to that stack — see commit message for details.
