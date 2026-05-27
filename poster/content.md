# Poster content — paste-ready draft

A0 portrait, UZH scientific-poster template. Numbers are authoritative as of
2026-05-24 evening (upstream/main @ f154b10, parser-fixed eval on 470 gold
questions, 30/30 consistency).

Replace `[[TBD: ...]]` placeholders before printing.

---

## 1 · Header (top strip)

**Title** (largest text on poster, ~60 mm in mm-true rendering):

> Skills turn a generic RAG into a workflow agent

**Subtitle** (smaller, one line):

> Hand-rolled hybrid retrieval + cross-encoder reranking + workflow skills
> over a 10,000-document corporate knowledge corpus

**Authors:**

> Nicolas Schaerer · Andre [[TBD: lastname]] · Edoardo Schiatti

**Course / affiliation:**

> UZH FS2026 — RAG · Group [[TBD: #]]

**QR code:** placeholder, ~80×80 mm — encodes the Nuvolos demo URL.

---

## 2 · Motivation (why this isn't "just another RAG")

A retrieval-only RAG fails in three distinct ways that better retrieval alone
doesn't fix:

1. **Loses chronology.** Even with the right docs in hand, a default agent
   summarises the highest-scoring document and skips the time dimension.
2. **Loses outcomes.** Asked "what did we decide", a default agent recaps the
   discussion — the decision often lives in one later message that gets
   averaged away.
3. **Single-angle.** Onboarding questions need multiple framings (status,
   team, blockers); a default agent does one search and stops.

We added **three skills** — short workflow prompts the user invokes by clicking
a button — that re-route the same hybrid retriever into structured outputs
the user can act on. The retriever is unchanged; the skill prompt shapes what
the LLM does with the results.

| Skill | What it produces | Example question |
|---|---|---|
| **Trace** | Dated timeline of an event across sources | *"Trace the office aquarium leak incident."* |
| **Decide** | Four-line structured block — Decision (verbatim) / Made by / When / Source | *"What did we decide about the gocritic paramTypeCombine rule?"* |
| **Onboard** | Five-section brief — What it is / Goal / Status / Key people / Open issues | *"Onboard me onto the Verbier Q4 ski retreat."* |

---

## 3 · System architecture

```
  ┌────────────┐    skill prompt    ┌─────────────────┐
  │   User     │ ─── prepended ───▶ │  Qwen3-8B agent │
  └────────────┘                    │  search/open    │
                                    └────┬────────┬───┘
                                         │        │
                          ┌──────────────┴───┐   ┌▼────────────────┐
                          │ Hybrid retrieve  │   │ Document fetch  │
                          │ BM25 (FTS5)      │   └─────────────────┘
                          │   +              │
                          │ Dense (nomic-768)│
                          │   ↓ fuse         │
                          │ Cross-encoder    │
                          │ MS-MARCO-MiniLM  │
                          └────┬─────────────┘
                               │
                  ┌────────────▼────────────┐
                  │  SQLite + FTS5          │
                  │  35 344 chunks · 10 k   │
                  │  documents · 9 sources  │
                  └─────────────────────────┘
```

**Stack:**

- TypeScript agent (Node, no integrated RAG framework)
- SQLite + FTS5 — sparse retrieval
- nomic-embed-text via Ollama — dense embeddings (768-dim, 8192 ctx)
- Python sidecar (`ms-marco-MiniLM-L-6-v2`, CPU-pinned) — cross-encoder rerank
- Qwen3-8B via Ollama — generation
- Vanilla HTML/CSS/JS frontend, no build step

**Hand-rolled primitives (no frameworks):** BM25 via SQLite FTS5, dense via
in-memory Float32Array matmul, fusion via weighted sum (0.7·vec + 0.3·kw),
cross-encoder rerank, structured pre-filters.

---

## 4 · Evaluation (the centerpiece)

**Dataset:** EnterpriseRAG-Bench (Onyx, MIT) — 10,000 documents across 9
source types (Slack, Gmail, Confluence, Jira, Linear, HubSpot, GitHub, Google
Drive, Fireflies) yielding 35,344 indexed chunks. 500 gold questions with
expected doc IDs and answer facts.

**Methodology note:** the upstream eval harness silently dropped 93 of 500
gold questions because pandas surfaces parquet `list<string>` columns as
`numpy.ndarray`, which the original `parse_expected` parser couldn't handle.
Dropped rows were concentrated in *harder* categories (project_related,
intra_document_reasoning). We patched the parser; the table below scores
the honest 470 representative questions.

### 4.1 Headline — retrieval lift from cross-encoder reranking

| Metric | Fusion only | + Cross-encoder | Δ |
|---|---|---|---|
| **Recall@1** | 0.649 | **0.772** | **+12.3 pp** |
| Recall@3 | 0.685 | 0.857 | +17.2 pp |
| Recall@5 | 0.691 | 0.872 | +18.1 pp |
| Recall@10 | 0.706 | 0.889 | +18.3 pp |
| MRR@10 | 0.670 | 0.817 | +14.7 pp |
| nDCG@10 | 0.644 | 0.805 | +16.1 pp |

### 4.2 Where the reranker actually moves the needle

Recall@1 by question type (470 questions, parser-fixed):

| Type | n | Baseline | + Rerank | Δ |
|---|---|---|---|---|
| basic | 175 | 0.669 | **0.834** | +16.5 |
| **semantic** | **125** | **0.304** | **0.448** | **+14.4** |
| intra_document_reasoning | 40 | 0.725 | **1.000** | +27.5 |
| project_related | 40 | 0.925 | 0.950 | +2.5 |
| conflicting_info | 20 | 0.950 | **1.000** | +5.0 |
| completeness | 20 | 0.800 | 0.850 | +5.0 |
| constrained | 30 | 1.000 | 0.933 | already perfect (n=30 noise) |
| miscellaneous | 20 | 0.950 | 0.900 | already perfect (n=20 noise) |

**The semantic category — 27% of all questions and the bi-encoder's known
blind spot — moves +14.4 pp at Recall@1 and +29.6 pp at Recall@10
(0.368 → 0.664). Every weaker category improves; the two small regressions
are on already-perfect categories.**

### 4.3 End-to-end answer quality (retrieval → LLM → fact scoring)

50-question random sample (seed 42), Qwen3-8B as the answering LLM, fact
scored as "found" if ≥60 % of significant words appear in the answer.

| Metric | Fusion only | + Cross-encoder | Δ |
|---|---|---|---|
| Retrieval Hit@1 | 0.640 | **0.680** | +4.0 pp |
| Retrieval Hit@6 | 0.740 | **0.860** | +12.0 pp |
| **Avg Fact Recall** | 0.460 | 0.420 | −4.0 pp |
| Latency / question | 4.7 s | 5.0 s | +6 % |

**Honest read on the Fact Recall delta:** rerank reorders the top-3 chunks
sent to the LLM. The LLM picks answers from different chunks — sometimes
better (rerank wins +1 fact on Q18, Q23, Q46), sometimes worse (rerank
loses on Q05, Q10). On n=50 with ~1 fact per question, ±2 facts is ±4 pp,
which is within sampling noise.

**Interpretation:** the cross-encoder closes the retrieval gap convincingly,
but answer quality is currently LLM-limited on the 8 B Qwen3 — not
retrieval-limited. The natural next bottleneck is generation, which the
course's ≤8 B constraint deliberately caps.

### 4.4 Demo reliability

| Skill | Doc-hit rate | p50 latency | p95 latency |
|---|---|---|---|
| Trace | **10 / 10** | 13.1 s | 21.9 s |
| Decide | **10 / 10** | 10.9 s | 23.5 s |
| Onboard | **10 / 10** | 21.5 s | 26.6 s |

**30 demo runs, 30 correct retrievals.** Reproducible via the consistency
harness in this repo.

---

## 5 · Methodology highlights (for "project management" grade)

- **Honest baseline.** Replicated Edoardo's published R@1=0.67 on the first
  100 gold rows to within three decimals, then discovered that the published
  number is on the *easiest* 20% of the gold set. True baseline on 470
  representative rows is 0.649 (fusion only).
- **Eval bug we caught.** Parser bug silently dropped 93 of 500 gold rows.
  Fix is a 3-line addition to `parse_expected` that handles `numpy.ndarray`.
- **Cross-platform fix.** Windows `\r\n` line endings broke the YAML
  frontmatter parser in `src/server/skills.ts` — JS regex `$` doesn't match
  before `\r`. 4-line fix lets the project build on Windows too.
- **Minimal-footprint contribution.** Code added on top of upstream: 4 lines
  on `skills.ts`, ~10-line prompt edit on `skills/onboard.md`, one new eval
  file kept side-by-side with upstream so diffs are reviewable. No new
  frameworks, no architectural changes.

---

## 6 · Cost / latency

| Phase | CPU cost | Wall time |
|---|---|---|
| Index build (35 k chunks, embed) | one-shot | 7 min (RTX 5070 Ti) |
| Retrieval baseline (per query) | ~250 ms | — |
| Retrieval + rerank (per query) | +150 ms (CPU cross-encoder) | — |
| Demo p50 latency (TRACE/DECIDE) | — | ~11 s |
| Demo p50 latency (ONBOARD, 3 searches) | — | ~22 s |

---

## 7 · Limitations and future work

- Reranker is CPU-bound — GPU would cut +150 ms to ~30 ms.
- Date filter is effectively gmail-only (other source chunkers don't
  populate `ts_from`/`ts_to`).
- 30 gold questions are *info_not_found* and similar — excluded from the
  honest denominator.
- LLM (Qwen3-8B) latency dominates the demo (8–25 s typical, occasional
  35 s outliers); a smaller LLM would help if function-calling fidelity
  holds.

---

## 8 · Acknowledgments

- **Dataset:** Onyx EnterpriseRAG-Bench (MIT)
- **Models:** nomic-embed-text, ms-marco-MiniLM-L-6-v2, Qwen3-8B — all
  Apache-2.0, served via Ollama
- **No integrated RAG framework** (no LangChain, no FAISS, no RAGFlow) per
  the course brief; every retrieval primitive hand-implemented
- AI assistance used for code & copy

---

## 9 · Repository + demo

- **Repo:** https://github.com/EDEN757/company-rag-agent
- **Demo (Nuvolos):** [[TBD: live demo URL]]
- **Branch / commit:** `main` @ f154b10
