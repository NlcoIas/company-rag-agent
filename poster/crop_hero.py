"""Crop banana 16:9 hero to 5:3 to match the UZH hero slot exactly.

Banana generates 1760×990 (16:9). The hero slot is 784×473 mm (~5:3 = 1.667).
Currently python-pptx stretches the image to fit, causing ~6% horizontal
squish. Cropping to 5:3 first preserves the original geometry.

We crop from the LEFT (keeping the busy upper-right where the visual interest
lives, sacrificing the already-quiet left edge) — the brief asked for lighter
density in the lower-left for the title card.
"""
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
SRC = HERE / "uzh" / "hero.png"
DST = HERE / "uzh" / "hero_cropped.png"

img = Image.open(SRC)
w, h = img.size
target_aspect = 5 / 3
new_w = int(round(h * target_aspect))
left = w - new_w  # crop from the left (right-aligned)
cropped = img.crop((left, 0, w, h))
cropped.save(DST)
print(f"cropped {w}x{h} -> {new_w}x{h}  ({DST})")
