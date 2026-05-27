// End-to-end test of the 3 demo questions against Edoardo's TS server.
// Captures: total latency, tool calls (with retrieved docs), final answer text.
// Run: node data/test_demo.mjs

import http from 'node:http';
import { randomUUID } from 'node:crypto';

const HOST = '127.0.0.1';
const PORT = 3000;

const DEMOS = [
  { skill: 'trace',   q: 'Trace the office aquarium leak incident.' },
  { skill: 'decide',  q: 'What did we decide about the gocritic paramTypeCombine linter rule?' },
  { skill: 'onboard', q: 'Onboard me onto the Verbier Q4 ski retreat.' },
];

// Expected demo doc_ids per skill (from data/demo_docs/QA.md).
const EXPECTED = {
  trace:   ['demo_aquarium_kickoff', 'demo_aquarium_leak', 'demo_aquarium_resolution'],
  decide:  ['demo_gocritic_decision', 'demo_gocritic_thread'],
  onboard: ['demo_verbier_overview', 'demo_verbier_status', 'demo_verbier_committee', 'demo_verbier_vendors'],
};

function askDemo(skill, question) {
  return new Promise((resolveOuter, rejectOuter) => {
    const sessionId = `test-${skill}-${Date.now()}`;
    const payload = JSON.stringify({ session_id: sessionId, skill_name: skill, message: question });
    const req = http.request({
      host: HOST,
      port: PORT,
      method: 'POST',
      path: '/api/chat',
      headers: {
        'content-type': 'application/json',
        'content-length': Buffer.byteLength(payload),
        'accept': 'text/event-stream',
      },
    }, (res) => {
      if (res.statusCode !== 200) {
        let buf = '';
        res.on('data', (c) => (buf += c));
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
        // SSE: events separated by \n\n
        const events = buffer.split('\n\n');
        buffer = events.pop() || '';
        for (const raw of events) {
          const line = raw.split('\n').find((l) => l.startsWith('data:'));
          if (!line) continue;
          let ev;
          try { ev = JSON.parse(line.slice(5).trim()); } catch { continue; }
          if (ev.type === 'text') {
            fullText += ev.delta || '';
          } else if (ev.type === 'tool_start') {
            toolCalls.push({ name: ev.name, args: ev.args, t: Date.now() - t0 });
          } else if (ev.type === 'tool_end') {
            const lastIdx = toolCalls.findLastIndex((c) => c.id === ev.id || c.name === ev.name);
            if (lastIdx >= 0) {
              toolCalls[lastIdx].summary = ev.summary;
              toolCalls[lastIdx].details = ev.details;
              toolCalls[lastIdx].isError = ev.isError;
            }
          } else if (ev.type === 'error') {
            console.error(`[${skill}] server error event:`, ev.message);
          } else if (ev.type === 'done') {
            const totalMs = Date.now() - t0;
            resolveOuter({ skill, question, fullText, toolCalls, ttfbMs: firstByteAt ? firstByteAt - t0 : null, totalMs });
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
    const det = c.details;
    // Search tool returns { results: [{ doc_id, ... }] }
    if (det.results && Array.isArray(det.results)) {
      for (const r of det.results) {
        if (r.doc_id) found.add(r.doc_id);
      }
    }
    // open_document: { doc_id }
    if (det.doc_id) found.add(det.doc_id);
  }
  return [...found];
}

(async () => {
  console.log('--- demo question end-to-end test ---');
  console.log(`server: http://${HOST}:${PORT}`);
  console.log();

  const results = [];
  for (const { skill, q } of DEMOS) {
    process.stdout.write(`[${skill}] asking: ${q}\n`);
    try {
      const r = await askDemo(skill, q);
      const docIds = extractDocIds(r.toolCalls);
      const expected = EXPECTED[skill];
      const hits = docIds.filter((d) => expected.includes(d));
      const missed = expected.filter((d) => !docIds.includes(d));
      r.hitDocs = hits;
      r.missedDocs = missed;
      r.allRetrievedDocs = docIds;
      results.push(r);
      console.log(`  ttfb: ${r.ttfbMs}ms  total: ${r.totalMs}ms  tools: ${r.toolCalls.length}`);
      console.log(`  retrieved: ${docIds.join(', ') || '(none)'}`);
      console.log(`  expected:  ${expected.join(', ')}`);
      console.log(`  hits:      ${hits.length}/${expected.length}  missed: ${missed.join(', ') || 'none'}`);
      console.log(`  answer (${r.fullText.length} chars):`);
      console.log('  ' + r.fullText.split('\n').join('\n  '));
      console.log();
    } catch (err) {
      console.error(`[${skill}] FAILED:`, err.message);
      console.log();
    }
  }

  console.log('--- summary ---');
  for (const r of results) {
    const expected = EXPECTED[r.skill];
    console.log(`${r.skill.padEnd(9)} ${r.totalMs}ms · tools=${r.toolCalls.length} · docs ${r.hitDocs.length}/${expected.length}`);
  }
})();
