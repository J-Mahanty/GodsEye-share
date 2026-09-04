"""Re-capture the burst panel, clipped around the per-frame strip."""
from playwright.sync_api import sync_playwright
import pathlib
OUT = pathlib.Path("brief/figs")

with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome")
    ctx = b.new_context(viewport={"width": 1500, "height": 1750},
                        color_scheme="light", device_scale_factor=2)
    pg = ctx.new_page()
    pg.goto("http://localhost:8501", wait_until="domcontentloaded", timeout=90000)
    pg.wait_for_selector("text=ANPR engine", timeout=90000)
    pg.wait_for_timeout(8000)
    pg.get_by_text("ANPR engine", exact=True).first.click()
    pg.wait_for_timeout(5000)
    pg.get_by_text("Burst", exact=True).first.click()
    pg.wait_for_timeout(4000)
    pg.get_by_text("Capture & read", exact=True).first.click()
    pg.wait_for_selector("text=What each frame saw", timeout=120000)
    pg.wait_for_timeout(4000)

    # clip from just above the verdict line down past the per-frame strip
    verdict = pg.locator("text=ground truth").first
    strip = pg.locator("text=frames read this plate correctly").first
    pg.evaluate("window.scrollTo(0, 0)")
    pg.wait_for_timeout(1500)
    vb = verdict.bounding_box(); sb = strip.bounding_box()
    print("verdict box", vb, "strip box", sb)
    if vb and sb:
        top = max(vb["y"] - 60, 0)
        height = (sb["y"] + sb["height"] + 6) - top
        pg.screenshot(path=str(OUT / "burst_read.png"), clip={"x": 250, "y": top, "width": 1245, "height": height})
    else:
        pg.screenshot(path=str(OUT / "burst_read.png"))
    print("recaptured burst_read.png")
    b.close()
