import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

export interface Skill {
  name: string;
  description: string;
  suggested_question: string;
  body: string;
}

const REQUIRED_FIELDS: (keyof Skill)[] = ["name", "description", "suggested_question"];

function stripQuotes(s: string): string {
  const t = s.trim();
  if (t.length >= 2 && (t.startsWith('"') || t.startsWith("'")) && t[0] === t[t.length - 1]) {
    return t.slice(1, -1);
  }
  return t;
}

function parseList(raw: string): string[] {
  const t = raw.trim();
  if (!t.startsWith("[") || !t.endsWith("]")) {
    return t ? [stripQuotes(t)] : [];
  }
  const inner = t.slice(1, -1).trim();
  if (!inner) return [];
  return inner.split(",").map((p) => stripQuotes(p)).filter(Boolean);
}

/** Tiny YAML-subset parser: scalar strings + bracket lists. Returns [fields, body]. */
function parseFrontmatter(rawText: string): { fields: Record<string, string | string[]>; body: string } {
  // Normalize CRLF → LF so regex anchors and split("\n") work identically on Windows.
  // Without this, JS's `$` won't match before a trailing `\r` and every frontmatter line throws.
  const text = rawText.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  if (!text.startsWith("---")) {
    throw new Error("file has no leading '---' frontmatter delimiter");
  }
  const end = text.indexOf("\n---", 3);
  if (end === -1) throw new Error("frontmatter has no closing '---'");
  const header = text.slice(3, end).replace(/^\n+|\n+$/g, "");
  const body = text.slice(end + "\n---".length).replace(/^\n+/, "");

  const fields: Record<string, string | string[]> = {};
  for (const line of header.split("\n")) {
    if (!line.trim() || line.trimStart().startsWith("#")) continue;
    const m = line.match(/^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$/);
    if (!m) throw new Error(`frontmatter line not 'key: value': ${JSON.stringify(line)}`);
    const key = m[1];
    const raw = m[2].trim();
    fields[key] = raw.startsWith("[") ? parseList(raw) : stripQuotes(raw);
  }
  return { fields, body };
}

export function loadSkills(rootDir: string): Map<string, Skill> {
  const out = new Map<string, Skill>();
  let entries: string[];
  try {
    entries = readdirSync(rootDir);
  } catch {
    return out;
  }
  for (const name of entries) {
    if (!name.endsWith(".md")) continue;
    const path = join(rootDir, name);
    const text = readFileSync(path, "utf8");
    let parsed;
    try {
      parsed = parseFrontmatter(text);
    } catch (e) {
      throw new Error(`skill ${path}: ${(e as Error).message}`);
    }
    const missing = REQUIRED_FIELDS.filter((k) => {
      const v = parsed.fields[k];
      return v === undefined || (typeof v === "string" && v === "");
    });
    if (missing.length) {
      throw new Error(`skill ${path}: frontmatter missing required fields ${missing.join(", ")}`);
    }
    const skill: Skill = {
      name: String(parsed.fields.name),
      description: String(parsed.fields.description),
      suggested_question: String(parsed.fields.suggested_question),
      body: parsed.body.trim(),
    };
    if (out.has(skill.name)) {
      throw new Error(`duplicate skill name '${skill.name}' (${path})`);
    }
    out.set(skill.name, skill);
  }
  return out;
}
