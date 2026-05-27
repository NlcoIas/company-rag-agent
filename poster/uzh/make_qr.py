"""Generate a QR code for the demo URL.

Uses qrcode + Pillow (pure-Python QR encoder + PNG writer).
Defaults to the GitHub repo URL since the Nuvolos demo URL isn't fixed yet.
"""
import sys
from pathlib import Path
import qrcode

URL = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/EDEN757/company-rag-agent"
OUT = Path(__file__).resolve().parent / "qr.png"

qr = qrcode.QRCode(
    version=None,                           # auto-size based on data
    error_correction=qrcode.constants.ERROR_CORRECT_M,
    box_size=12,                            # px per module → ~370px QR
    border=2,
)
qr.add_data(URL)
qr.make(fit=True)
img = qr.make_image(fill_color="black", back_color="white")
img.save(OUT)
print(f"QR for {URL} → {OUT} ({OUT.stat().st_size} bytes)")
