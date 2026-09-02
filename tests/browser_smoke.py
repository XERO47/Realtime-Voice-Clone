"""Real browser smoke test using fake microphone devices."""

import os
import re
import sys
from pathlib import Path

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from playwright.sync_api import expect, sync_playwright


BASE_URL = "http://127.0.0.1:8000"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
ROOT = Path(__file__).resolve().parents[1]


def register(page, user_id: str) -> None:
    page.goto(BASE_URL)
    page.locator("#identityInput").fill(user_id)
    page.locator("#goOnline").click()
    page.locator("#readyView").wait_for(state="visible")


def run() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=CHROME,
            headless=True,
            args=["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"],
        )
        context_a = browser.new_context(permissions=["microphone"])
        context_b = browser.new_context(permissions=["microphone"])
        monitor_context = browser.new_context()
        caller_a = context_a.new_page()
        caller_b = context_b.new_page()
        dashboard = monitor_context.new_page()

        register(caller_a, "ALPHA-3101")
        register(caller_b, "BETA-4202")
        dashboard.goto(f"{BASE_URL}/dashboard")

        caller_a.locator("#targetInput").fill("BETA-4202")
        caller_a.locator("#callButton").click()
        caller_b.locator("#incomingModal").wait_for(state="visible")
        caller_b.locator("#acceptButton").click()

        expect(caller_a.locator("#callState")).to_have_text("Connected", timeout=20_000)
        expect(caller_b.locator("#callState")).to_have_text("Connected", timeout=20_000)
        expect(caller_a.locator("#detailMirror")).to_have_text("PCM streaming", timeout=10_000)
        expect(caller_b.locator("#detailMirror")).to_have_text("PCM streaming", timeout=10_000)
        expect(caller_a.locator("#localMicStatus")).to_contain_text("PCM streaming", timeout=10_000)
        session_row = dashboard.locator(".vg-scard", has_text="ALPHA-3101")
        session_row.first.wait_for(timeout=10_000)
        session_row.first.click()
        dashboard.get_by_text(re.compile(r"Voice detected|Quiet, audio healthy")).first.wait_for(timeout=10_000)

        dashboard.locator('[data-tap="caller_a"]').click()
        dashboard.locator("#tapState").wait_for(state="visible")
        dashboard.locator('[data-action="verify"]').click()
        caller_a.locator("#interventionBanner").wait_for(state="visible")

        dashboard.on("dialog", lambda dialog: dialog.accept())
        dashboard.locator('[data-action="end"]').click()
        caller_a.locator("#readyView").wait_for(state="visible", timeout=10_000)

        browser.close()
    print("Browser smoke test passed: two-party WebRTC, audio mirrors, monitor tap, verification, and teardown.")


if __name__ == "__main__":
    run()
