"""Build the UZH-branded poster by filling the official A0 portrait template
with our content. python-pptx writes text into the named placeholders; the
template's logo, colours, fonts, footer, and partner logos are left intact.

The architecture diagram lives at poster/uzh/architecture.svg — drag it
into the bottom-left image placeholder (idx=13) after opening the PPTX.

Run:
    data/.venv/Scripts/python.exe poster/build_poster.py
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Mm, Pt

# ── paths ───────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "uzh" / "uzh-portrait-a0.potx"
TEMPLATE_PPTX = HERE / "uzh" / "uzh-portrait-a0.pptx"  # converted at build time
OUTPUT = HERE / "uzh" / "company-rag-poster.pptx"


def potx_to_pptx(src: Path, dst: Path) -> None:
    """Re-pack a .potx as .pptx by rewriting the content-type entry.

    python-pptx refuses templates (.potx) — same ZIP structure but the
    presentation part's content type is 'presentationml.template.main+xml'
    instead of 'presentationml.presentation.main+xml'. We copy every entry
    verbatim and patch just that one string in [Content_Types].xml.
    """
    TEMPLATE_CT = b"application/vnd.openxmlformats-officedocument.presentationml.template.main+xml"
    PRESENTATION_CT = b"application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"
    with zipfile.ZipFile(src, "r") as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "[Content_Types].xml":
                data = data.replace(TEMPLATE_CT, PRESENTATION_CT)
            zout.writestr(item, data)

# ── content ─────────────────────────────────────────────────────────────────

ORG_UNIT = "UZH FS2026 — Retrieval-Augmented Generation course"

TITLE = "Skills turn a generic RAG into a workflow agent"

SUBTITLE = (
    "Cross-encoder reranking and three workflow skills, over a "
    "10,000-document corporate corpus. Hand-rolled, no frameworks."
)

# Column 1 top — Lead (bold, 48pt). Plain language so non-experts get the win.
LEAD = (
    "With multi-query retrieval, tuned weights and cross-encoder reranking, "
    "the correct document is the top result in 73% of 377 answerable test "
    "questions, up from 63% for the published baseline. Fact recall jumped "
    "from 42% to 52% after fixing Qwen3's thinking-token suppression."
)

# Column 2 top (idx=10) — answers Rosan's poster-passerby questions:
# "Why is it there?" (The problem), "What is it?" (The fix + TRACE/DECIDE/ONBOARD).
COL2_TOP = [
    ("The problem", True),
    ("", False),
    ("Search returns documents. Users still have to read, summarise, and trace.", False),
    ("", False),
    ("The fix", True),
    ("", False),
    ("Three skill buttons prepend a prompt that shapes the output: a timeline, "
     "a decision, or an onboarding brief. Same retriever, structured answer.", False),
    ("", False),
    ("Sources: Slack, Gmail, Confluence, Jira, Linear, HubSpot, Google Drive, "
     "GitHub, Fireflies.", False),
    ("", False),
    ("TRACE", True),
    ("Dated timeline of an event in chronological order. Every line cites "
     "the document it came from.", False),
    ("", False),
    ("DECIDE", True),
    ("Final call quoted word-for-word. Who decided it, when, and which "
     "document the decision lives in.", False),
    ("", False),
    ("ONBOARD", True),
    ("Five-section brief: what it is, goal, status, key people, open issues.", False),
]

# Column 2 bottom (idx=12) — Methodology + "Why hand-rolled?" rationale
# that addresses Rosan's "Why did they choose this/that?" question.
COL2_BOT = [
    ("Methodology highlights", True),
    ("", False),
    ("• Tuned fusion weights 0.7 / 0.3 → 0.3 / 0.7. R@1 jumped from 0.63 to 0.74.", False),
    ("• Added multi-query retrieval: long questions are also embedded as "
     "key-terms-only to recover dilution-affected docs.", False),
    ("• Fixed Qwen3 thinking-token suppression (think:false). "
     "Fact recall: 0.42 → 0.52.", False),
    ("• Filtered eval to the 377 questions whose answer docs live in the index.", False),
    ("", False),
    ("Why hand-rolled?", True),
    ("", False),
    ("No LangChain or LlamaIndex by design. Writing every piece — fusion, "
     "reranker, skills — is how we found 0.3 / 0.7 weights beat the published "
     "0.7 / 0.3, and how we caught the Qwen3 think-suppression bug.", False),
    ("", False),
    ("Stack: Qwen3-8B · nomic-embed-text · SQLite FTS5 · "
     "ms-marco-MiniLM-L-6-v2 · TypeScript agent.", False),
]

# Column 3 is rebuilt from scratch as a stack of text-boxes + native tables.
# See build_col3_native() for the layout.

# Ablation table — every value measured on the same 470-question gold set
# All values measured on the same 377-question answerable subset
# (data/sweep_mq_*.log) — Andre's eval_retrieval filter only counts
# questions whose expected_doc_ids are actually in the 10k-doc index.
# Multi-query retrieval enabled (long queries also embedded as key-terms).
# Finding: the deployed 0.7/0.3 weights are reversed for this corpus.
# Re-tuned 0.3/0.7 alone beats them at R@1 by ~11 pp without reranking.
ABLATION_TABLE = [
    ["Variant",                "R@1",   "R@5",   "R@10"],
    ["BM25 only",              "0.716", "0.883", "0.915"],
    ["Dense only",             "0.411", "0.549", "0.629"],
    ["Hybrid 0.7/0.3",         "0.631", "0.687", "0.698"],
    ["Hybrid 0.3/0.7 (tuned)", "0.743", "0.854", "0.859"],
    ["+ Cross-encoder",        "0.732", "0.865", "0.899"],
]

# Live-demo latency (from data/consistency_f154b10.log).
LATENCY_TABLE = [
    ["Skill",   "median", "p95"],
    ["Trace",   "13 s",   "22 s"],
    ["Decide",  "11 s",   "24 s"],
    ["Onboard", "22 s",   "27 s"],
]

PERTYPE_TABLE = [
    ["Type",          "n",   "Fusion", "+Rerank"],
    ["basic",         "175", "0.669",  "0.834"],
    ["semantic",      "125", "0.304",  "0.448"],
    ["intra-doc",     "40",  "0.725",  "1.000"],
    ["project",       "40",  "0.925",  "0.950"],
    ["conflicting",   "20",  "0.950",  "1.000"],
    ["completeness",  "20",  "0.800",  "0.850"],
    ["constrained",   "30",  "1.000",  "0.933"],
    ["misc",          "20",  "0.950",  "0.900"],
]

E2E_TABLE = [
    ["Metric",     "Fusion", "+Rerank"],
    ["Hit@1",      "0.640",  "0.680"],
    ["Hit@6",      "0.740",  "0.880"],
    ["Fact recall","0.400",  "0.520"],
]


# ── helpers ─────────────────────────────────────────────────────────────────

def write_paragraphs(shape, lines):
    """Replace the text frame contents with our paragraphs.

    `lines` is a list of (text, is_heading) tuples. Headings get bold +
    accent colour; the empty-string entries become blank paragraph spacers.
    """
    tf = shape.text_frame
    tf.clear()
    for i, (text, is_heading) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        run = p.add_run()
        run.text = text
        if is_heading:
            run.font.bold = True
            run.font.size = Pt(32)
        # Body sizing left to placeholder default (30pt) for the rest.


def find_ph(slide, idx):
    """Return the placeholder shape with the given idx, or None."""
    for ph in slide.placeholders:
        if ph.placeholder_format.idx == idx:
            return ph
    return None


def set_title(slide, text):
    """Title placeholder has type='title' rather than a numeric idx."""
    for ph in slide.placeholders:
        if ph.placeholder_format.idx == 0:
            tf = ph.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            r = p.add_run()
            r.text = text
            return
    raise RuntimeError("no title placeholder found")


def add_textbox(slide, text, left, top, width, height, size_pt=18, bold=False, color=None):
    """Drop-in helper for a styled text box (not a placeholder)."""
    shape = slide.shapes.add_textbox(left, top, width, height)
    tf = shape.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size_pt)
    run.font.bold = bold
    run.font.name = "Source Sans Pro"
    if color is not None:
        run.font.color.rgb = color
    return shape


def add_native_table(slide, data, left, top, width, height,
                     header_accent=True, highlight_winner=None,
                     value_start_col=1, cell_size_pt=20):
    """Add a native PowerPoint table populated from `data` (list of lists).

    highlight_winner: how to mark the best value(s)
        None       — no extra highlighting beyond the header
        "per_row"  — bold + UZH-blue the max numeric cell in each data row
                     (use when columns are alternatives, rows are items)
        "per_col"  — bold + UZH-blue the max numeric cell in each data column
                     (use when rows are alternatives, columns are metrics)
    value_start_col: first column index considered numeric (default 1, skip
        the label column 0). Use 2 for tables with an extra count column.
    """
    rows = len(data)
    cols = len(data[0]) if rows else 0
    if rows == 0 or cols == 0:
        return None
    shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    table = shape.table
    UZH_BLUE = RGBColor(0x00, 0x28, 0xA5)
    LIGHT_GREY = RGBColor(0xF4, 0xF4, 0xF0)

    # Compute winner cells based on the numeric data BEFORE filling the table
    winners = set()  # set of (r, c) coords to highlight
    if highlight_winner == "per_row":
        for r in range(1, rows):
            best_v = -float("inf")
            best_cs = []
            for c in range(value_start_col, cols):
                try:
                    v = float(data[r][c])
                    if v > best_v:
                        best_v = v
                        best_cs = [c]
                    elif v == best_v:
                        best_cs.append(c)
                except (ValueError, TypeError):
                    pass
            for c in best_cs:
                winners.add((r, c))
    elif highlight_winner == "per_col":
        for c in range(value_start_col, cols):
            best_v = -float("inf")
            best_rs = []
            for r in range(1, rows):
                try:
                    v = float(data[r][c])
                    if v > best_v:
                        best_v = v
                        best_rs = [r]
                    elif v == best_v:
                        best_rs.append(r)
                except (ValueError, TypeError):
                    pass
            for r in best_rs:
                winners.add((r, c))

    for r, row_data in enumerate(data):
        for c, val in enumerate(row_data):
            cell = table.cell(r, c)
            cell.text = str(val)
            cell.margin_left = Mm(2)
            cell.margin_right = Mm(2)
            cell.margin_top = Mm(1)
            cell.margin_bottom = Mm(1)
            tf = cell.text_frame
            for p in tf.paragraphs:
                if c > 0:
                    p.alignment = PP_ALIGN.RIGHT
                for run in p.runs:
                    run.font.name = "Source Sans Pro"
                    run.font.size = Pt(cell_size_pt)
                    if r == 0:
                        run.font.bold = True
                        if header_accent:
                            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                    elif (r, c) in winners:
                        run.font.bold = True
                        run.font.color.rgb = UZH_BLUE
            if r == 0 and header_accent:
                cell.fill.solid()
                cell.fill.fore_color.rgb = UZH_BLUE
            elif r > 0 and r % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT_GREY
    return shape


def build_col3_native(slide):
    """Replace col 3 placeholder (idx=11) with a stack of text-boxes
    and native PowerPoint tables for clean alignment."""
    ph = find_ph(slide, 11)
    if ph is not None:
        sp = ph._element
        sp.getparent().remove(sp)

    x = Mm(564)
    w = Mm(248)
    y_mm = 460   # was 616 — hero shrunk, body starts earlier

    # Unified type scale: section headings 32pt, body/captions 28pt.
    # Padding bumped to 6 mm so sections don't touch each other.
    def heading(text, h_mm=20, size=32):
        nonlocal y_mm
        add_textbox(slide, text, x, Mm(y_mm), w, Mm(h_mm),
                    size_pt=size, bold=True,
                    color=RGBColor(0x00, 0x28, 0xA5))
        y_mm += h_mm + 6

    def caption(text, h_mm=15, size=28):
        nonlocal y_mm
        # Auto-grow box if text is long enough to wrap (>60 chars ≈ 2 lines).
        if h_mm < 28 and len(text) > 60:
            h_mm = 28
        add_textbox(slide, text, x, Mm(y_mm), w, Mm(h_mm),
                    size_pt=size, color=RGBColor(0x44, 0x44, 0x44))
        y_mm += h_mm + 6

    def table(data, h_mm):
        nonlocal y_mm
        add_native_table(slide, data, x, Mm(y_mm), w, Mm(h_mm))
        y_mm += h_mm + 3

    def table_hi(data, h_mm, highlight_winner=None, value_start_col=1):
        nonlocal y_mm
        add_native_table(slide, data, x, Mm(y_mm), w, Mm(h_mm),
                         highlight_winner=highlight_winner,
                         value_start_col=value_start_col)
        y_mm += h_mm + 8

    # ── results block (col 3) ───────────────────────────────────────────────
    heading("Results — retrieval ablation")
    caption("377 answerable questions. Multi-query retrieval enabled. "
            "Same pool, varied retrieval stage.")
    table_hi(ABLATION_TABLE, h_mm=88, highlight_winner="per_col")
    caption("Re-tuned 0.3 / 0.7 weights win at R@1. BM25 alone still wins "
            "R@10. Cross-encoder is the deployed safety net.")

    heading("End-to-end answer quality  (n=50, seed=42)")
    table_hi(E2E_TABLE, h_mm=65, highlight_winner="per_row")
    caption("After fixing Qwen3 thinking suppression (think:false), "
            "the reranker now lifts fact recall by 12 pp instead of hurting it.")

    heading("Demo reliability across 30 runs")
    add_textbox(slide, "30 of 30 demo runs found every expected document.",
                x, Mm(y_mm), w, Mm(18), size_pt=28, bold=True,
                color=RGBColor(0x00, 0x28, 0xA5))
    y_mm += 26

    # Demo prompts at 28pt — three short lines.
    heading("Try these at the laptop", h_mm=18, size=28)
    add_textbox(
        slide,
        "• TRACE — \"the office aquarium leak\"\n"
        "• DECIDE — \"the gocritic linter rule\"\n"
        "• ONBOARD — \"Verbier Q4 ski retreat\"",
        x, Mm(y_mm), w, Mm(46), size_pt=28, color=RGBColor(0x22, 0x22, 0x22),
    )


def set_single_line(shape, text, size_pt=None, bold=None, color=None):
    """Single-paragraph replacement, preserving placeholder's font style."""
    tf = shape.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = text
    if size_pt is not None:
        r.font.size = Pt(size_pt)
    if bold is not None:
        r.font.bold = bold
    if color is not None:
        r.font.color.rgb = color


# ── main ────────────────────────────────────────────────────────────────────

def main():
    if not TEMPLATE.exists():
        raise SystemExit(f"template missing: {TEMPLATE}")

    # python-pptx only opens .pptx, so re-pack the .potx as one.
    if not TEMPLATE_PPTX.exists() or TEMPLATE_PPTX.stat().st_mtime < TEMPLATE.stat().st_mtime:
        print(f"[convert] {TEMPLATE.name} → {TEMPLATE_PPTX.name}")
        potx_to_pptx(TEMPLATE, TEMPLATE_PPTX)

    print(f"[load] {TEMPLATE_PPTX}")
    prs = Presentation(str(TEMPLATE_PPTX))

    # The UZH template ships with two slides (Variante 1 and Variante 2).
    # We only use Variante 1 — drop everything after it so the PDF export is
    # a clean single A0 page instead of two.
    sld_id_lst = prs.slides._sldIdLst
    sld_ids = list(sld_id_lst)
    for extra in sld_ids[1:]:
        r_id = extra.rId
        prs.part.drop_rel(r_id)
        sld_id_lst.remove(extra)
        print(f"[trim] removed extra slide (rId={r_id})")

    slide = prs.slides[0]  # Poster Variante 1 — the only slide now

    # Print discovered placeholders for debugging
    print("[scan] placeholders on slide 1:")
    for ph in slide.placeholders:
        idx = ph.placeholder_format.idx
        name = ph.name
        try:
            sample = ph.text_frame.text[:60].replace("\n", " | ")
        except Exception:
            sample = "(no text frame)"
        print(f"  idx={idx:>3}  name={name:<28}  text='{sample}'")

    # Write content
    print("[write] title + subtitle + unit")
    set_title(slide, TITLE)

    subtitle = find_ph(slide, 17)
    if subtitle:
        # Shrunk hero → subtitle moves up to y=325 (was 395).
        subtitle.left = Mm(29)
        subtitle.top = Mm(325)
        subtitle.width = Mm(498)
        subtitle.height = Mm(70)
        subtitle.fill.solid()
        subtitle.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        tf = subtitle.text_frame
        tf.word_wrap = True
        tf.margin_top = Mm(6)
        tf.margin_bottom = Mm(6)
        tf.margin_left = Mm(8)
        tf.margin_right = Mm(8)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        set_single_line(subtitle, SUBTITLE,
                        size_pt=44, color=RGBColor(0x00, 0x00, 0x00))
        for p in tf.paragraphs:
            p.alignment = PP_ALIGN.CENTER

    # Shrunk hero (y=116-440, h=324) → title moves up. Title still ~165 mm
    # tall to accommodate the 3-line wrap of "Skills turn a generic RAG
    # into a workflow agent".
    title_ph = find_ph(slide, 0)
    if title_ph:
        title_ph.left = Mm(29)
        title_ph.top = Mm(140)
        title_ph.width = Mm(498)
        title_ph.height = Mm(180)
        ttf = title_ph.text_frame
        ttf.margin_top = Mm(8)
        ttf.margin_bottom = Mm(8)
        ttf.margin_left = Mm(15)
        ttf.margin_right = Mm(15)
        ttf.vertical_anchor = MSO_ANCHOR.MIDDLE

    unit = find_ph(slide, 16)
    if unit:
        set_single_line(unit, ORG_UNIT)

    print("[write] lead (col 1 top) — moved up since hero is smaller")
    lead = find_ph(slide, 1)
    if lead:
        # Hero shrunk → body starts at y=460 (was 616).
        lead.left = Mm(29)
        lead.top = Mm(460)
        lead.width = Mm(248)
        lead.height = Mm(140)
        set_single_line(lead, LEAD, size_pt=48, bold=True)

    print("[write] col 2 top (motivation)")
    col2_top = find_ph(slide, 10)
    if col2_top:
        # Extended height to fit the new "The problem" + "The fix" headers.
        col2_top.left = Mm(297)
        col2_top.top = Mm(460)
        col2_top.width = Mm(248)
        col2_top.height = Mm(355)
        write_paragraphs(col2_top, COL2_TOP)

    print("[write] col 2 bottom (methodology)")
    col2_bot = find_ph(slide, 12)
    if col2_bot:
        # Pushed down to make room for the bigger top block. Height bumped
        # for the new "Why hand-rolled?" subsection.
        col2_bot.left = Mm(297)
        col2_bot.top = Mm(825)
        col2_bot.width = Mm(248)
        col2_bot.height = Mm(220)
        write_paragraphs(col2_bot, COL2_BOT)

    print("[write] col 3 (results) — native tables")
    build_col3_native(slide)

    # Sweep stray template stubs. Recurses into groups — the template
    # tucks the "Text" caption under the QR slot inside a group shape,
    # which slide.shapes (top-level only) doesn't reach.
    print("[sweep] removing template stubs (recurses into groups)")
    placeholder_ids = {ph.shape_id for ph in slide.placeholders}
    STUB_NEEDLES = (
        "Lorem", "ipsum", "Fig.",
        "Authors (", "Contact (", "Author 1", "Author 2", "Author 3",
        "1 University unit", "2 University unit",
        "First name Surname", "mail@adresse",
        "+00 00 000",
    )

    def iter_all_shapes(container):
        for shape in container.shapes:
            # MSO_SHAPE_TYPE.GROUP == 6
            if shape.shape_type == 6:
                yield from iter_all_shapes(shape)
            else:
                yield shape

    to_remove = []
    for shape in iter_all_shapes(slide):
        if shape.is_placeholder and shape.shape_id in placeholder_ids:
            continue
        if getattr(shape, "has_chart", False):
            to_remove.append(("chart", shape))
            continue
        if getattr(shape, "has_table", False):
            cells = [cell.text.strip() for row in shape.table.rows for cell in row.cells]
            stub_hits = sum(1 for t in cells if t in ("Title", "Text", "00"))
            if stub_hits >= 2:
                to_remove.append(("stub-table", shape))
            continue
        if not shape.has_text_frame:
            continue
        txt = shape.text_frame.text.strip()
        if txt == "Text" or any(n in txt for n in STUB_NEEDLES):
            to_remove.append(("text-stub", shape))
    for tag, shape in to_remove:
        sample = ""
        if shape.has_text_frame:
            sample = shape.text_frame.text[:40].replace("\n", " | ")
        print(f"  removed {tag:11} id={shape.shape_id} '{sample}'")
        shape._element.getparent().remove(shape._element)

    # ── replace hero image (idx=14) with banana-generated knowledge graph ──
    # Prefer the cropped 5:3 version (no stretch). Fall back to 16:9 original.
    hero_cropped = HERE / "uzh" / "hero_cropped.png"
    hero_orig = HERE / "uzh" / "hero.png"
    hero_png = hero_cropped if hero_cropped.exists() else hero_orig
    if hero_png.exists():
        hero_ph = find_ph(slide, 14)
        if hero_ph:
            sp = hero_ph._element
            sp.getparent().remove(sp)
            print("[image] removed stock hero placeholder idx=14")
        # SHRUNK hero per Andre's feedback — was h=473 (too dominant),
        # now h=324. Frees ~150 mm of vertical space for the body grid.
        slide.shapes.add_picture(
            str(hero_png), Mm(29), Mm(116), width=Mm(784), height=Mm(324)
        )
        # Move hero behind title/subtitle by setting it as the first child
        # of spTree (lowest z-order — drawn first, others on top).
        sp_tree = slide.shapes._spTree
        last = sp_tree[-1]
        sp_tree.remove(last)
        sp_tree.insert(2, last)  # index 2 = after the group-shape boilerplate
        print("[image] inserted hero.png and sent to back")

    # ── insert architecture diagram + caption (col 1 middle/bottom) ──────
    arch_png = HERE / "uzh" / "architecture.png"
    if arch_png.exists():
        img_ph = find_ph(slide, 13)
        if img_ph:
            sp = img_ph._element
            sp.getparent().remove(sp)
            print("[image] removed stock image placeholder idx=13")
        # Architecture moved up + slightly taller (body real estate freed
        # by the shrunk hero). Caption font bumped 12 → 16 pt per feedback.
        # Architecture: 248×200 mm matches new SVG aspect 980:800 (1.225)
        # — h = 248/1.225 = 202.4 mm. Set 200 ≈ natural, no stretch.
        slide.shapes.add_picture(
            str(arch_png), Mm(29), Mm(610), width=Mm(248), height=Mm(200)
        )
        # Caption at 28 pt; box height generous so it doesn't clip if it wraps.
        add_textbox(
            slide,
            "BM25 + dense embeddings fused 30 / 70. "
            "Cross-encoder re-scores the top-32 candidates.",
            Mm(29), Mm(820), Mm(248), Mm(40),
            size_pt=28, color=RGBColor(0x1a, 0x1a, 0x1a),
        )
        print(f"[image] architecture (200mm, no stretch) + 28pt caption")

    # ── ablation bar chart (col 1 bottom — fills dead space) ──────────────
    chart_png = HERE / "uzh" / "ablation_chart.png"
    if chart_png.exists():
        # Chart at 188 mm tall — fits between caption (ends y≈870) and footer.
        slide.shapes.add_picture(
            str(chart_png), Mm(29), Mm(880), width=Mm(248), height=Mm(188)
        )
        print(f"[image] inserted ablation_chart.png at (29, 880) — 248×188 mm")
    else:
        print(f"[image] WARNING: {arch_png} missing — drag in by hand")

    # ── insert QR code (col 3 bottom — moved down so demo queries don't collide)
    qr_png = HERE / "uzh" / "qr.png"
    if qr_png.exists():
        # 70 mm QR fits between label (y=1010) and footer (y=1100).
        qr_size = 70
        qr_x = 564 + (248 - qr_size) / 2  # = 653
        # Label bumped to 24 pt to match unified scale.
        label_box = add_textbox(
            slide,
            "Scan to try the live demo  →",
            Mm(564), Mm(1010), Mm(248), Mm(16),
            size_pt=24, bold=True, color=RGBColor(0x00, 0x28, 0xA5),
        )
        for p in label_box.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
        slide.shapes.add_picture(
            str(qr_png), Mm(qr_x), Mm(1028), width=Mm(qr_size), height=Mm(qr_size)
        )
        print(f"[image] inserted qr.png + label at ({qr_x:.0f}, 1028) — {qr_size}×{qr_size} mm centred")

    # Save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUTPUT))
    print(f"[done] {OUTPUT}")
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"       {size_kb:.1f} KB")

    # Post-process: XML-level cleanup for stub tables python-pptx can't see
    # (e.g. graphicFrames nested in a way that doesn't surface via
    # slide.shapes / .has_table). See poster/clean_stubs_xml.py.
    print("[clean] running XML stub cleaner")
    import subprocess
    cleaner = HERE / "clean_stubs_xml.py"
    if cleaner.exists():
        subprocess.run([__import__("sys").executable, str(cleaner)], check=True)


if __name__ == "__main__":
    main()
