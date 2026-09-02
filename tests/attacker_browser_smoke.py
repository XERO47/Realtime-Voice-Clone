"""End-to-end authorized attacker-console test with synthetic WebRTC audio."""

from __future__ import annotations

import os
from pathlib import Path

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from playwright.sync_api import expect, sync_playwright


BASE_URL = "http://127.0.0.1:8000"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
ROOT = Path(__file__).resolve().parents[1]


def run() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=CHROME,
            headless=True,
            args=["--autoplay-policy=no-user-gesture-required", "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"],
        )
        attacker_context = browser.new_context()
        victim_context = browser.new_context(permissions=["microphone"])
        monitor_context = browser.new_context()
        attacker = attacker_context.new_page()
        victim = victim_context.new_page()
        monitor = monitor_context.new_page()

        victim.goto(BASE_URL)
        victim.locator("#identityInput").fill("VICTIM-2202")
        victim.locator("#goOnline").click()
        victim.locator("#readyView").wait_for(state="visible")

        attacker.goto(f"{BASE_URL}/attacker")
        attacker.wait_for_function("document.querySelectorAll('#attackVoice option').length > 1")
        attacker.locator("#attackScript").fill("This is an authorized synthetic voice transmission test.")
        attacker.locator("#generateAttackSpeech").click()
        attacker.locator("#generatedClip").wait_for(state="visible", timeout=30_000)
        attacker.locator("#attackerId").fill("REDTEAM-1101")
        attacker.locator("#goOnlineAttack").click()
        expect(attacker.locator("#attackServiceStatus span")).to_contain_text("Online as", timeout=10_000)

        monitor.goto(f"{BASE_URL}/dashboard")
        attacker.locator("#attackTarget").fill("VICTIM-2202")
        attacker.locator("#startAttackCall").click()
        victim.locator("#incomingModal").wait_for(state="visible")
        victim.locator("#acceptButton").click()

        expect(attacker.locator("#attackCallState")).to_have_text("Connected to protected user", timeout=20_000)
        expect(victim.locator("#callState")).to_have_text("Connected", timeout=20_000)
        test_session = monitor.locator(".vg-scard", has_text="REDTEAM-1101")
        test_session.wait_for(timeout=10_000)
        test_session.click()
        expect(test_session).to_contain_text("red-team run")

        attacker.locator("#injectSpeech").click()
        attacker.locator("#transmitProgress").wait_for(state="visible")
        monitor.wait_for_timeout(1_000)
        expect(monitor.locator("#onlineA")).not_to_have_text("Waiting for audio")

        attacker.locator('[data-source="clone"]').click()
        expect(attacker.locator("#cloneWorkbench")).to_be_visible()
        attacker.locator("#cloneUpload").set_input_files(ROOT / "artifacts" / "tts_test.wav")
        expect(attacker.locator("#cloneClip")).to_be_visible()
        expect(attacker.locator("#activeSourceLabel")).to_have_text("Voice clone clip")
        attacker.locator("#injectClone").click()
        expect(attacker.locator("#transmitLabel")).to_have_text("TRANSMITTING VOICE CLONE")
        expect(attacker.locator("#transmitProgress")).to_be_visible()
        expect(monitor.locator("#scoreA")).not_to_have_text("--", timeout=60_000)

        attacker.locator("#attackEnd").click()
        victim.locator("#readyView").wait_for(state="visible", timeout=10_000)
        browser.close()

    print("Attacker smoke test passed: TTS and clone-file WebRTC injection, monitoring, detection, and teardown.")


if __name__ == "__main__":
    run()
