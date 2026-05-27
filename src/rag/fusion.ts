import { fetchChunksByIds, getDb, loadAllEmbeddings, type ChunkRow } from "./db.js";
import { embedQuery } from "./embed.js";
import { rerank } from "./rerank.js";

export const TOP_K_PER_BRANCH = 16;
// Tuned on our 10k-subset corpus: 0.3 vec / 0.7 kw beats the published
// 0.7/0.3 by ~9 pp R@1 and ~3 pp R@10 (eval on 377 answerable questions,
// data/sweep_mq_03_07_rerank.log). Multi-query keeps this ordering intact.
export const KW_WEIGHT = 0.7;
export const VEC_WEIGHT = 0.3;
export const SCORE_THRESHOLD = 0.30;
export const SCORE_SCALE = 4;

export interface SearchFilters {
  source_types?: string[];
  date_from?: string;
  date_to?: string;
  participant?: string;
}

export interface SearchHit {
  chunk_id: number;
  doc_id: string;
  source_type: string;
  title: string | null;
  score: number;
  vec_score: number;
  kw_score: number;
  rerank_score?: number;
  preview: string;
  ts_from: string | null;
  ts_to: string | null;
}

function buildFilterClause(filters: SearchFilters): { sql: string; params: unknown[] } {
  const clauses: string[] = [];
  const params: unknown[] = [];
  if (filters.source_types && filters.source_types.length > 0) {
    const placeholders = filters.source_types.map(() => "?").join(",");
    clauses.push(`source_type IN (${placeholders})`);
    params.push(...filters.source_types);
  }
  if (filters.date_from) {
    clauses.push("(ts_to IS NULL OR ts_to >= ?)");
    params.push(filters.date_from);
  }
  if (filters.date_to) {
    clauses.push("(ts_from IS NULL OR ts_from <= ?)");
    params.push(filters.date_to);
  }
  if (filters.participant) {
    clauses.push("participants_json LIKE ?");
    params.push(`%${filters.participant}%`);
  }
  return { sql: clauses.length ? `WHERE ${clauses.join(" AND ")}` : "", params };
}

function ftsQuery(q: string): string {
  // Strip operators FTS5 would interpret, then OR-join words so any-of matches.
  const words = q
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter((w) => w.length > 1);
  if (words.length === 0) return '""';
  return words.map((w) => `"${w}"`).join(" OR ");
}

function preview(text: string, maxChars = 320): string {
  const clean = text.replace(/\s+/g, " ").trim();
  return clean.length <= maxChars ? clean : `${clean.slice(0, maxChars).trim()}…`;
}

/**
 * Strip a long query down to its key domain terms by removing question words,
 * auxiliary verbs, and common function words. Returns null if the result is
 * identical to the input (query was already short/clean) or too short to be useful.
 * Used to generate a second, more focused vector embedding for multi-query retrieval.
 */
function stripToKeyTerms(query: string): string | null {
  const REMOVE = new Set([
    "what", "who", "when", "where", "how", "which", "why",
    "is", "are", "was", "were", "do", "does", "did",
    "can", "could", "should", "would", "will", "have", "has", "had",
    "the", "a", "an", "of", "to", "in", "for", "on", "at", "by",
    "from", "with", "and", "or", "but", "not", "that", "this", "these",
    "be", "been", "being", "get", "got", "make", "made", "used", "use",
    "also", "any", "some", "into", "than", "then", "its", "their",
  ]);
  const words = query
    .toLowerCase()
    .replace(/[^\w\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 2 && !REMOVE.has(w));
  if (words.length < 3) return null;
  const stripped = words.join(" ");
  if (stripped === query.toLowerCase().trim()) return null;
  return stripped;
}

export async function search(query: string, filters: SearchFilters = {}, topN = 6): Promise<SearchHit[]> {
  const db = getDb();
  const { sql: whereSql, params } = buildFilterClause(filters);

  // --- Keyword branch ---
  const kwQuery = ftsQuery(query);
  const kwSql = `
    SELECT c.chunk_id, c.doc_id, c.source_type
    FROM chunks_fts f
    JOIN chunks c ON c.chunk_id = f.rowid
    ${whereSql ? whereSql + " AND" : "WHERE"} chunks_fts MATCH ?
    ORDER BY bm25(chunks_fts) ASC
    LIMIT ?
  `;
  const kwRows = db.prepare(kwSql).all(...params, kwQuery, TOP_K_PER_BRANCH) as {
    chunk_id: number;
    doc_id: string;
    source_type: string;
  }[];

  const kwScores = new Map<number, number>();
  kwRows.forEach((r, i) => {
    const rank = i + 1;
    kwScores.set(r.chunk_id, (1 / (1 + rank)) * SCORE_SCALE);
  });

  // --- Vector branch (multi-query) ---
  // For long queries, also embed a stripped key-term version. The full-query
  // embedding captures intent/context; the stripped version focuses on domain
  // terms and recovers docs the full embedding misses due to question-phrasing
  // dilution. Both run in parallel; candidates are merged before fusion.
  const stripped = query.split(/\s+/).length > 6 ? stripToKeyTerms(query) : null;
  const [qvec, qvec2] = await Promise.all([
    embedQuery(query),
    stripped ? embedQuery(stripped) : Promise.resolve(null),
  ]);
  const { matrix, chunkIds, dim } = loadAllEmbeddings();

  // Build allowed-id set for filtering, if any filter applies.
  let allowed: Set<number> | null = null;
  if (whereSql) {
    const rows = db
      .prepare(`SELECT chunk_id FROM chunks ${whereSql}`)
      .all(...params) as { chunk_id: number }[];
    allowed = new Set(rows.map((r) => r.chunk_id));
  }

  // Cosine similarity = dot product (vectors are L2-normalized).
  // Helper: score one query vector against all chunks, return top-K as Map.
  function topKVec(qv: Float32Array): Map<number, number> {
    const sims = new Float32Array(chunkIds.length);
    for (let i = 0; i < chunkIds.length; i++) {
      if (allowed && !allowed.has(chunkIds[i])) { sims[i] = -Infinity; continue; }
      let s = 0;
      const off = i * dim;
      for (let d = 0; d < dim; d++) s += matrix[off + d] * qv[d];
      sims[i] = s;
    }
    const order = Array.from({ length: chunkIds.length }, (_, i) => i);
    order.sort((a, b) => sims[b] - sims[a]);
    const scores = new Map<number, number>();
    for (let i = 0; i < Math.min(TOP_K_PER_BRANCH, order.length); i++) {
      const idx = order[i];
      if (!isFinite(sims[idx])) break;
      scores.set(chunkIds[idx], sims[idx] * SCORE_SCALE);
    }
    return scores;
  }

  // Run vector search for the full query; merge with stripped-query results
  // by keeping the best score per chunk across both passes.
  const vecScores = topKVec(qvec);
  if (qvec2) {
    for (const [id, score] of topKVec(qvec2)) {
      const existing = vecScores.get(id) ?? -Infinity;
      if (score > existing) vecScores.set(id, score);
    }
  }

  // --- Fusion ---
  const allIds = new Set<number>([...kwScores.keys(), ...vecScores.keys()]);
  const fused: { chunk_id: number; final: number; kw: number; vec: number }[] = [];
  for (const id of allIds) {
    const kw = kwScores.get(id) ?? 0;
    const vec = vecScores.get(id) ?? 0;
    const final = VEC_WEIGHT * vec + KW_WEIGHT * kw;
    if (final >= SCORE_THRESHOLD) fused.push({ chunk_id: id, final, kw, vec });
  }
  fused.sort((a, b) => b.final - a.final);
  const pool = fused.slice(0, Math.min(TOP_K_PER_BRANCH * 2, fused.length));

  if (pool.length === 0) return [];

  const rows = fetchChunksByIds(pool.map((t) => t.chunk_id));

  // Cross-encoder reranking: the model scores each (query, passage) pair
  // jointly, catching relevance signals the bi-encoder dot product misses.
  // Falls back to fusion order if the reranker service is unavailable.
  const passages = pool.map((t) => {
    const r = rows.get(t.chunk_id);
    return r ? r.text : "";  // r.text already contains the header prepended
  });
  const rerankScores = await rerank(query, passages);

  // Sort full pool by reranker score, fall back to fusion order.
  const sortedPool = rerankScores !== null
    ? pool.map((c, i) => ({ ...c, rerankScore: rerankScores[i] }))
          .sort((a, b) => b.rerankScore - a.rerankScore)
    : pool;

  // Deduplicate by doc_id: keep the best-scoring chunk per document.
  // Multiple chunks from the same document waste result slots and can cause
  // the agent to open the same document twice. Mirrors the eval script logic.
  const seenDocIds = new Set<string>();
  const hits: SearchHit[] = [];
  for (const t of sortedPool) {
    if (hits.length >= topN) break;
    const r = rows.get(t.chunk_id);
    if (!r) continue;
    if (seenDocIds.has(r.doc_id)) continue;
    seenDocIds.add(r.doc_id);
    hits.push({
      chunk_id: r.chunk_id,
      doc_id: r.doc_id,
      source_type: r.source_type,
      title: r.title,
      score: Number(t.final.toFixed(3)),
      vec_score: Number(t.vec.toFixed(3)),
      kw_score: Number(t.kw.toFixed(3)),
      rerank_score: "rerankScore" in t ? Number((t.rerankScore as number).toFixed(3)) : undefined,
      preview: preview(r.text),
      ts_from: r.ts_from,
      ts_to: r.ts_to,
    });
  }
  return hits;
}
