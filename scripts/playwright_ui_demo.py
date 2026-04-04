#!/usr/bin/env python3
"""Automate Streamlit UI demo actions for presentation recording."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _activate_browser_app() -> None:
    """Bring the headed browser app to foreground for reliable screencast focus."""
    candidates = ["Google Chrome for Testing", "Google Chrome", "Chromium"]
    for app_name in candidates:
        try:
            subprocess.run(
                ["osascript", "-e", f'tell application "{app_name}" to activate'],
                check=True,
                capture_output=True,
                text=True,
            )
            return
        except Exception:
            continue


def _submit_query(page, question_box, text: str, timeout_ms: int) -> None:
    """Fill question text, apply it in Streamlit, and wait until a new answer appears."""
    question_box.click(timeout=timeout_ms)
    question_box.fill(text, timeout=timeout_ms)
    page.keyboard.press("Meta+Enter")
    page.wait_for_timeout(500)
    page.wait_for_function(
        """() => {
            const btn = [...document.querySelectorAll('button')]
              .find(el => (el.textContent || '').trim() === 'Спросить');
            return !!btn && !btn.disabled;
        }""",
        timeout=timeout_ms,
    )
    before_answers = page.get_by_role("heading", name="Ответ").count()
    page.get_by_role("button", name="Спросить").click(timeout=timeout_ms)
    page.get_by_role("heading", name="Ответ").nth(before_answers).wait_for(timeout=timeout_ms)
    page.wait_for_timeout(2500)
    page.get_by_role("heading", name="Ответ").nth(before_answers).scroll_into_view_if_needed()
    page.wait_for_timeout(600)


def run_ui_demo(args: argparse.Namespace) -> dict:
    started = time.time()
    payload: dict = {
        "ok": False,
        "url": args.url,
        "steps": {
            "page_loaded": False,
            "selfcheck_clicked": False,
            "selfcheck_ok_visible": False,
            "ask_in_done": False,
            "ask_out_done": False,
            "ask_followup_done": False,
            "sources_opened": False,
            "trace_opened": False,
            "history_scrolled": False,
        },
        "screenshot": None,
        "error": None,
        "duration_sec": None,
    }

    try:
        from playwright.sync_api import TimeoutError as PWTimeout
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        payload["error"] = f"playwright_import_failed: {exc}"
        payload["duration_sec"] = round(time.time() - started, 3)
        return payload

    try:
        with sync_playwright() as p:
            browser_args = [f"--window-size={args.width},{args.height}"]
            if args.headed:
                browser_args.append("--start-maximized")
            browser = p.chromium.launch(headless=not args.headed, slow_mo=args.slow_mo_ms, args=browser_args)
            context = browser.new_context(viewport={"width": args.width, "height": args.height})
            page = context.new_page()

            page.goto(args.url, wait_until="domcontentloaded", timeout=args.timeout_ms)
            page.wait_for_timeout(2000)
            if args.headed:
                _activate_browser_app()
                page.wait_for_timeout(800)
            payload["steps"]["page_loaded"] = True

            # 1) Selfcheck button
            page.get_by_role("button", name="Selfcheck API").click(timeout=args.timeout_ms)
            payload["steps"]["selfcheck_clicked"] = True
            try:
                page.get_by_text("Selfcheck: OK").first.wait_for(timeout=args.timeout_ms)
                payload["steps"]["selfcheck_ok_visible"] = True
            except PWTimeout:
                # Keep demo going; final report will capture it.
                payload["steps"]["selfcheck_ok_visible"] = False

            # 2) Ask in-corpus
            question_box = page.get_by_label("Ваш вопрос")
            _submit_query(page, question_box, args.query_in, args.timeout_ms)
            payload["steps"]["ask_in_done"] = True

            # 3) Ask out-of-corpus
            _submit_query(page, question_box, args.query_out, args.timeout_ms)
            payload["steps"]["ask_out_done"] = True

            # 4) Follow-up in-corpus query to show history and retention
            _submit_query(page, question_box, args.query_followup, args.timeout_ms)
            payload["steps"]["ask_followup_done"] = True

            # 5) Expand details
            sources = page.locator("summary", has_text="Источники").first
            if sources.count() > 0:
                sources.click(timeout=5000)
                payload["steps"]["sources_opened"] = True
                page.wait_for_timeout(1000)

            trace = page.locator("summary", has_text="Debug trace").first
            if trace.count() > 0:
                trace.click(timeout=5000)
                payload["steps"]["trace_opened"] = True
                page.wait_for_timeout(1000)

            # 6) Scroll through the page to visibly show multi-question history.
            page.mouse.wheel(0, 1600)
            page.wait_for_timeout(900)
            page.mouse.wheel(0, 1600)
            page.wait_for_timeout(900)
            page.mouse.wheel(0, -2400)
            page.wait_for_timeout(900)
            payload["steps"]["history_scrolled"] = True

            if args.screenshot:
                screenshot_path = Path(args.screenshot)
                screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(screenshot_path), full_page=True)
                payload["screenshot"] = str(screenshot_path)

            context.close()
            browser.close()

    except Exception as exc:
        payload["error"] = f"ui_demo_failed: {exc}"

    payload["duration_sec"] = round(time.time() - started, 3)
    required_steps = [
        "page_loaded",
        "selfcheck_clicked",
        "ask_in_done",
        "ask_out_done",
        "ask_followup_done",
        "sources_opened",
        "trace_opened",
        "history_scrolled",
    ]
    payload["ok"] = all(payload["steps"].get(step, False) for step in required_steps)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Playwright UI demo flow for Streamlit app")
    parser.add_argument("--url", default="http://localhost:8512")
    parser.add_argument("--output", required=True)
    parser.add_argument("--screenshot", default=None)
    parser.add_argument("--headed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--slow-mo-ms", type=int, default=120)
    parser.add_argument("--timeout-ms", type=int, default=180000)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument(
        "--query-in",
        default="Какие ключевые темы в отчете Сбера за 2024 год?",
    )
    parser.add_argument(
        "--query-out",
        default="Расскажи про ядерный реактор на Луне в отчете Сбера 2015.",
    )
    parser.add_argument(
        "--query-followup",
        default="Какие ключевые финансовые результаты Сбера за 2023 год?",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    output_path = Path(args.output)
    payload = run_ui_demo(args)
    _write_json(output_path, payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
