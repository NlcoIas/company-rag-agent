"""Post-process the built PPTX to surgically remove stub tables that
python-pptx misses (those nested inside graphicFrame shapes that
.has_table doesn't expose at the slide-shapes level).

Strategy: parse slide1.xml directly, find every <p:graphicFrame> whose
cell text runs are mostly the literal word "Text", and remove the whole
graphicFrame element. Then rewrite the .pptx zip in place.

Run after build_poster.py."""

import shutil
import zipfile
from pathlib import Path
from lxml import etree

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}

HERE = Path(__file__).resolve().parent
PPTX = HERE / "uzh" / "company-rag-poster.pptx"
TMP = HERE / "uzh" / "company-rag-poster.tmp.pptx"

with zipfile.ZipFile(PPTX, "r") as zin:
    entries = [(item, zin.read(item.filename)) for item in zin.infolist()]

removed_count = 0
new_entries = []
for item, data in entries:
    if item.filename != "ppt/slides/slide1.xml":
        new_entries.append((item, data))
        continue
    tree = etree.fromstring(data)
    parent_map = {child: parent for parent in tree.iter() for child in parent}
    # Find every graphicFrame element (tables, charts)
    for gf in tree.findall(".//p:graphicFrame", NS):
        text_runs = [t.text for t in gf.findall(".//a:t", NS) if t.text]
        text_count = text_runs.count("Text")
        title_count = text_runs.count("Title")
        # Heuristic: at least 2 "Text" stub cells AND fewer than 3 real numeric/word cells
        real_cells = [t for t in text_runs if t not in ("Text", "Title", "00", "1", "2", "3", "")]
        if text_count >= 2 and len(real_cells) < 4:
            parent = parent_map[gf]
            parent.remove(gf)
            removed_count += 1
            print(f"removed stub graphicFrame (text_runs={text_runs[:8]})")
    new_data = etree.tostring(tree, xml_declaration=True, encoding="UTF-8", standalone=True)
    new_entries.append((item, new_data))

with zipfile.ZipFile(TMP, "w", zipfile.ZIP_DEFLATED) as zout:
    for item, data in new_entries:
        zout.writestr(item, data)

shutil.move(TMP, PPTX)
print(f"done — removed {removed_count} stub graphicFrame(s) from {PPTX.name}")
