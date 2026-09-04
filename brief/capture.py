"""Capture light-theme screenshots of the two newest features for the brief."""
from playwright.sync_api import sync_playwright
import time, pathlib

OUT = pathlib.Path("brief/figs")
OUT.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome")
    ctx = b.new_context(viewport={"width": 1500, "height": 1100},
                        color_scheme="light", device_scale_factor=2)
    pg = ctx.new_page()
    pg.goto("http://localhost:8501", wait_until="domcontentloaded", timeout=90000)
    pg.wait_for_selector("text=Track a plate", timeout=90000)
    pg.wait_for_timeout(8000)

    # ---------- A. clone caught at search time ----------
    pg.get_by_text("Track a plate", exact=True).first.click()
    pg.wait_for_timeout(5000)
    box = pg.get_by_placeholder("KA 05 MJ 1234 — partial or misread is fine").first
    box.click(); box.fill("WB56NC5568"); box.press("Tab")
    pg.wait_for_timeout(16000)
    pg.wait_for_selector("text=Duplicate registration detected", timeout=60000)
    banner = pg.locator("text=Duplicate registration detected").first
    banner.scroll_into_view_if_needed(); pg.wait_for_timeout(1500)
    pg.screenshot(path=str(OUT / "clone_search.png"),
                  clip={"x": 250, "y": 60, "width": 1150, "height": 620})
    print("captured clone_search.png")

    # ---------- B. burst read + history ----------
    pg.get_by_text("ANPR engine", exact=True).first.click()
    pg.wait_for_timeout(5000)
    pg.get_by_text("Burst", exact=True).first.click()
    pg.wait_for_timeout(4000)
    pg.get_by_text("Capture & read", exact=True).first.click()
    pg.wait_for_timeout(30000)
    try:
        pg.wait_for_selector("text=What each frame saw", timeout=60000)
        pg.locator("text=What each frame saw").first.scroll_into_view_if_needed()
        pg.wait_for_timeout(2000)
        pg.screenshot(path=str(OUT / "burst_read.png"),
                      clip={"x": 250, "y": 40, "width": 1150, "height": 640})
        print("captured burst_read.png")
    except Exception as e:
        pg.screenshot(path=str(OUT / "burst_read.png"), full_page=False)
        print("burst fallback shot:", str(e)[:90])
    b.close()
