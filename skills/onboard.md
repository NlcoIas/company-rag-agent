---
name: onboard
description: Produce a five-minute onboarding brief for a project, team, or topic. Use when the user is new to something and wants a structured summary, not raw search results.
suggested_question: Onboard me onto the Verbier Q4 ski retreat.
---

# Onboard skill

Workflow for the user's project, team, or topic name:

1. Call `search` THREE times in sequence — do not skip any, do not merge them:
   - First: the topic name alone (overview).
   - Second: the topic plus "status" or "current" (state).
   - Third: the topic plus "team", "vendors", or "committee" (people).

2. After each search, open the single top hit with `open_document`.

3. Produce the brief in exactly five sections, one short paragraph each — no bullet lists inside sections:

   - **What it is.** A one-sentence definition.
   - **Goal.** Why it exists, the business outcome it targets.
   - **Status.** Where it currently stands, with the most recent date mentioned in any opened document.
   - **Key people.** Names, roles, and contact handles, drawn from participants and bylines.
   - **Open issues.** Anything described as a blocker, risk, complaint, or in-flight work.

4. Each section cites at least one `doc_id` inline. Use the EXACT doc_id string from the search results — copy it verbatim. Never invent prefixes like `dsid_`, never shorten, never paraphrase the doc_id.

5. End with one final line: `Sources: <distinct doc_ids you cited, comma-separated>` — again exact verbatim doc_id strings.

If a section truly has no supporting content in the retrieved documents, write "Not found in the indexed corpus." rather than guessing.
