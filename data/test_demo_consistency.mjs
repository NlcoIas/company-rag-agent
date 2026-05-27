// Consistency test: runs the 3 demo questions N times each.
// Reports per-question hit rate and latency stats.
// Use to prove "demo questions work reliably 30/30" for the poster.
//
// Run: node data/test_demo_consistency.mjs [N]   (default N=10)

import http from 'node:http';
import { writeFileSync } from 'node:fs';

const HOST = '127.0.0.1';
const PORT = 3000;
const N = Number(process.argv[2] ?? 10);

const DEMOS = [
  { skill: 'trace',   q: 'Trace the office aquarium leak incident.' },
  { skill: 'decide',  q: 'What did we decide about the gocritic paramTypeCombine linter rule?' },
  { skill: 'onboard', q: 'Onboard me onto the Verbier Q4 ski retreat.' },
];

const EXPECTED = {
  trace:   ['demo_aquarium_kickoff', 'demo_aquarium_leak', 'demo_aquarium_resolution'],
  decide:  ['demo_gocritic_decision', 'demo_gocritic_thread'],
  onboard: ['demo_verbier_overview', 'demo_verbier_status', 'demo_verbier_committee', 'demo_verbier_vendors'],
};

function askDemo(skill, question) {
  return new Promise((resolveOuter, rejectOuter) => {
    const sessionId = `consistency-${skill}-${Date.now()}-${Math.random()}`;
    const payload = JSON.stringify({ session_id: sessionId, skill_name: skill, message: question });
    const req = http.request({
      host: HOST, port: PORT, method: 'POST', path: '/api/chat',
      headers: {
        'content-type': 'application/json',
        'content-length': Buffer.byteLength(payload),
        'accept': 'text/event-stream',
      },
    }, (res) => {
      if (res.statusCode !== 200) {
        let buf = ''; res.on('data', (c) => (buf += c));
        res.on('end', () => rejectOuter(new Error(`HTTP ${res.statusCode}: ${buf}`)));
        return;
      }
      const t0 = Date.now();
      let firstByteAt = null;
      let buffer = '';
      let fullText = '';
      const toolCalls = [];
      res.setEncoding('utf8');
      res.on('data', (chunk) => {
        if (!firstByteAt) firstByteAt = Date.now();
        buffer += chunk;
        const events = buffer.split('\n\n');
        buffer = events.pop() || '';
        for (const raw of events) {
          const line = raw.split('\n').find((l) => l.startsWith('data:'));
          if (!line) continue;
          let ev;
          try { ev = JSON.parse(line.slice(5).trim()); } catch { continue; }
          if (ev.type === 'text') fullText += ev.delta || '';
          else if (ev.type === 'tool_start') toolCalls.push({ id: ev.id, name: ev.name, args: ev.args });
          else if (ev.type === 'tool_end') {
            const idx = toolCalls.findIndex((c) => c.id === ev.id);
            if (idx >= 0) toolCalls[idx].details = ev.details;
          } else if (ev.type === 'done') {
            resolveOuter({ skill, question, fullText, toolCalls, ttfbMs: firstByteAt - t0, totalMs: Date.now() - t0 });
          }
        }
      });
      res.on('error', rejectOuter);
    });
    req.on('error', rejectOuter);
    req.write(payload);
    req.end();
  });
}

function extractDocIds(toolCalls) {
  const found = new Set();
  for (const c of toolCalls) {
    if (!c.details) continue;
    if (c.details.results && Array.isArray(c.details.results)) {
      for (const r of c.details.results) if (r.doc_id) found.add(r.doc_id);
    }
    if (c.details.doc_id) found.add(c.details.doc_id);
  }
  return [...found];
}

function pct(arr) {
  if (!arr.length) return { p50: 0, p95: 0, max: 0 };
  const s = [...arr].sort((a, b) => a - b);
  return { p50: s[Math.floor(s.length * 0.5)], p95: s[Math.floor(s.length * 0.95)], max: s[s.length - 1] };
}

(async () => {
  console.log(`--- consistency test: ${N} runs × 3 questions = ${N * 3} total ---`);
  console.log(`server: http://${HOST}:${PORT}`);
  const runs = [];
  for (let i = 0; i < N; i++) {
    for (const { skill, q } of DEMOS) {
      process.stdout.write(`[${String(i + 1).padStart(2)}/${N}] ${skill} ... `);
      try {
        const r = await askDemo(skill, q);
        const docs = extractDocIds(r.toolCalls);
        const expected = EXPECTED[skill];
        const hits = expected.filter((d) => docs.includes(d)).length;
        runs.push({ i: i + 1, skill, ttfb: r.ttfbMs, total: r.totalMs, hits, expected: expected.length, retrievedDocs: docs, answerLen: r.fullText.length });
        console.log(`${r.totalMs}ms  hits=${hits}/${expected.length}`);
      } catch (err) {
        runs.push({ i: i + 1, skill, error: err.message });
        console.log(`FAIL: ${err.message}`);
      }
    }
  }
  console.log();
  console.log('--- summary per skill ---');
  for (const skill of ['trace', 'decide', 'onboard']) {
    const xs = runs.filter((r) => r.skill === skill);
    const ok = xs.filter((r) => !r.error);
    const totals = ok.map((r) => r.total);
    const expected = EXPECTED[skill].length;
    const fullHits = ok.filter((r) => r.hits === expected).length;
    const partialOrFull = ok.filter((r) => r.hits > 0).length;
    const lat = pct(totals);
    console.log(`${skill.padEnd(9)} runs=${xs.length}  ok=${ok.length}  full-hits=${fullHits}/${ok.length}  any-hit=${partialOrFull}/${ok.length}  lat p50=${lat.p50}ms p95=${lat.p95}ms`);
  }
  writeFileSync('data/consistency_runs.json', JSON.stringify(runs, null, 2));
  console.log('\nDetailed per-run log: data/consistency_runs.json');
})();
