"""Headless Streamlit screenshot capture for demo artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


def _wait_streamlit_ready(proc: subprocess.Popen[str], timeout_sec: int) -> tuple[bool, list[str]]:
    """Wait until Streamlit prints startup URL or timeout is reached."""
    lines: list[str] = []
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        line = proc.stdout.readline() if proc.stdout else ""
        if line:
            lines.append(line.rstrip("\n"))
            if (
                "You can now view your Streamlit app in your browser" in line
                or "Local URL:" in line
                or "Network URL:" in line
            ):
                return True, lines
        else:
            if proc.poll() is not None:
                break
            time.sleep(0.2)
    return False, lines


def main() -> None:
    """Run Streamlit, capture screenshot, and persist result metadata."""
    parser = argparse.ArgumentParser(description="Capture UI screenshot from Streamlit app")
    parser.add_argument("--app", type=str, default="streamlit_app.py")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--port", type=int, default=8512)
    parser.add_argument("--startup-timeout-sec", type=int, default=45)
    parser.add_argument("--page-timeout-ms", type=int, default=60000)
    args = parser.parse_args()

    artifacts = Path(args.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_path = artifacts / f"ui_screenshot_{stamp}.png"
    meta_path = artifacts / "ui_screenshot_result.json"
    log_path = artifacts / "ui_screenshot_streamlit.log"

    payload = {
        "ok": False,
        "screenshot": None,
        "reason": None,
        "action_hint": None,
        "port": args.port,
        "app": args.app,
    }

    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        payload["reason"] = f"playwright_import_failed: {exc}"
        payload["action_hint"] = (
            "Install Playwright first: "
            "`pip install -r requirements-playwright.txt` and "
            "`playwright install chromium`."
        )
        meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return

    proc = subprocess.Popen(
        [
            "streamlit",
            "run",
            args.app,
            "--server.headless",
            "true",
            "--server.port",
            str(args.port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        ready, lines = _wait_streamlit_ready(proc, timeout_sec=args.startup_timeout_sec)
        log_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if not ready:
            payload["reason"] = "streamlit_not_ready"
            return

        with sync_playwright() as p:  # pragma: no cover - runtime environment dependent
            try:
                browser = p.chromium.launch()
            except Exception as exc:
                payload["reason"] = f"playwright_browser_failed: {exc}"
                payload["action_hint"] = (
                    "Install browser binaries: `playwright install chromium`."
                )
                return

            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            page.goto(f"http://localhost:{args.port}", wait_until="networkidle", timeout=args.page_timeout_ms)
            time.sleep(2)
            page.screenshot(path=str(screenshot_path), full_page=True)
            browser.close()

        payload["ok"] = True
        payload["screenshot"] = str(screenshot_path)
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        payload["reason"] = f"capture_failed: {exc}"
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()

        meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
