"""Render the brief to a print-ready A4 PDF."""
from playwright.sync_api import sync_playwright
import pathlib
src = pathlib.Path("brief/brief.html").resolve()
out = pathlib.Path("brief/GodsEye_Brief.pdf").resolve()
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome")
    pg = b.new_page()
    pg.goto(src.as_uri(), wait_until="load")
    pg.wait_for_timeout(2500)
    pg.emulate_media(media="print")
    pg.pdf(path=str(out), format="A4", print_background=True,
           margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
    b.close()
print("wrote", out)
