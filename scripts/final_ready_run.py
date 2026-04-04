"""Run full demo readiness flow with stall monitoring and auto-verdict."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class StepSpec:
    """One orchestration step."""

    name: str
    cmd: list[str]
    stdout_copy_to: str | None = None


@dataclass(slots=True)
class StepResult:
    """Result of one executed step."""

    name: str
    exit_code: int
    duration_sec: int
    outcome: str
    out_path: Path
    err_path: Path
    command: str


def _sanitize_slug(text: str) -> str:
    """Build filesystem-safe slug from free text."""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_") or "step"


def _read_stream(stream, target: Path, stamp_cb) -> None:
    """Pump subprocess stream into file and update output heartbeat."""
    with target.open("w", encoding="utf-8") as out:
        for line in iter(stream.readline, ""):
            out.write(line)
            out.flush()
            stamp_cb()


def _run_step(step: StepSpec, *, cwd: Path, cmd_dir: Path, stall_sec: int) -> StepResult:
    """Execute step with stall monitor based on stream inactivity."""
    out_path = cmd_dir / f"{_sanitize_slug(step.name)}.out"
    err_path = cmd_dir / f"{_sanitize_slug(step.name)}.err"
    started = time.monotonic()
    heartbeat_lock = threading.Lock()
    heartbeat = {"ts": started}

    def _touch() -> None:
        """Update heartbeat timestamp when new subprocess output arrives."""
        with heartbeat_lock:
            heartbeat["ts"] = time.monotonic()

    proc = subprocess.Popen(
        step.cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    t_out = threading.Thread(target=_read_stream, args=(proc.stdout, out_path, _touch), daemon=True)
    t_err = threading.Thread(target=_read_stream, args=(proc.stderr, err_path, _touch), daemon=True)
    t_out.start()
    t_err.start()

    aborted = False
    while proc.poll() is None:
        time.sleep(1.0)
        with heartbeat_lock:
            idle_sec = time.monotonic() - heartbeat["ts"]
        if idle_sec > float(stall_sec):
            aborted = True
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            break

    exit_code = proc.wait() if not aborted else 130
    t_out.join(timeout=1)
    t_err.join(timeout=1)

    outcome = "aborted_after_stall" if aborted else ("ok" if exit_code == 0 else "failed")
    duration_sec = int(round(time.monotonic() - started))
    return StepResult(
        name=step.name,
        exit_code=int(exit_code),
        duration_sec=duration_sec,
        outcome=outcome,
        out_path=out_path,
        err_path=err_path,
        command=" ".join(step.cmd),
    )


def _try_load_json(path: Path) -> dict[str, Any] | None:
    """Best-effort JSON loading."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_trace(ask_output_path: Path) -> dict[str, Any] | None:
    """Parse TRACE JSON from ask command text output."""
    if not ask_output_path.exists():
        return None
    text = ask_output_path.read_text(encoding="utf-8")
    marker = "=== TRACE ==="
    idx = text.find(marker)
    if idx < 0:
        return None
    tail = text[idx + len(marker) :].strip()
    brace = tail.find("{")
    if brace < 0:
        return None
    payload = tail[brace:].strip()
    try:
        return json.loads(payload)
    except Exception:
        return None


def _citations_count(ask_output_path: Path) -> int:
    """Count citations lines like [1] source=..."""
    if not ask_output_path.exists():
        return 0
    cnt = 0
    for line in ask_output_path.read_text(encoding="utf-8").splitlines():
        if re.match(r"^\[\d+\]\s+source=", line.strip()):
            cnt += 1
    return cnt


def _citations_block_empty(ask_output_path: Path) -> bool:
    """Check whether citations section has no source rows."""
    if not ask_output_path.exists():
        return False
    text = ask_output_path.read_text(encoding="utf-8")
    left = text.find("=== CITATIONS ===")
    right = text.find("=== TRACE ===")
    if left < 0 or right < 0 or right <= left:
        return False
    block = text[left:right]
    return "source=" not in block


def _check(results_by_name: dict[str, StepResult], name: str) -> bool:
    """Return true when step finished successfully."""
    row = results_by_name.get(name)
    return bool(row and row.outcome == "ok" and row.exit_code == 0)


def _build_verdict(
    *,
    run_dir: Path,
    results_by_name: dict[str, StepResult],
    data_dir: Path,
    goldset_path: Path,
) -> tuple[dict[str, Any], str]:
    """Build verdict.json payload and verdict.md text."""
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []

    def add_check(name: str, expected: str, actual: str, ok: bool, evidence: str) -> None:
        """Append normalized check row for final verdict payload."""
        checks.append(
            {
                "check": name,
                "expected": expected,
                "actual": actual,
                "ok": bool(ok),
                "evidence": evidence,
            }
        )

    expected_pdf_count = len([p for p in data_dir.rglob("*.pdf") if p.is_file()])
    goldset_lines = len([ln for ln in goldset_path.read_text(encoding="utf-8").splitlines() if ln.strip()])

    selfcheck = _try_load_json(run_dir / "selfcheck.json") or {}
    selfcheck_ok = bool(selfcheck.get("ok") is True)
    add_check(
        "selfcheck",
        "ok=true + chat/embeddings ok",
        f"ok={selfcheck.get('ok')}",
        selfcheck_ok and _check(results_by_name, "selfcheck"),
        "selfcheck.json",
    )

    prewarm_ok = _check(results_by_name, "prewarm")
    add_check("prewarm", "command success", f"exit={results_by_name['prewarm'].exit_code if 'prewarm' in results_by_name else 'n/a'}", prewarm_ok, "cmd/prewarm.out")

    index_payload = _try_load_json(run_dir / "index_sber_fast.json") or {}
    index_ok = (
        _check(results_by_name, "index")
        and int(index_payload.get("indexed_files") or 0) == expected_pdf_count
        and int(index_payload.get("failed_files") or 0) == 0
    )
    add_check(
        "index_full_corpus",
        f"indexed_files={expected_pdf_count} and failed_files=0",
        f"indexed_files={index_payload.get('indexed_files')} failed_files={index_payload.get('failed_files')}",
        index_ok,
        "index_sber_fast.json",
    )

    ask_in_trace = _extract_trace(run_dir / "ask_full.txt") or {}
    ask_in_citations = _citations_count(run_dir / "ask_full.txt")
    ask_in_ok = (
        _check(results_by_name, "ask_in")
        and ask_in_trace.get("grounded_refusal") is False
        and ask_in_citations > 0
    )
    add_check(
        "ask_in_corpus",
        "grounded_refusal=false and citations>0",
        f"grounded_refusal={ask_in_trace.get('grounded_refusal')} citations={ask_in_citations}",
        ask_in_ok,
        "ask_full.txt",
    )

    ask_out_trace = _extract_trace(run_dir / "ask_out_of_corpus.txt") or {}
    ask_out_ok = (
        _check(results_by_name, "ask_out")
        and ask_out_trace.get("grounded_refusal") is True
        and _citations_block_empty(run_dir / "ask_out_of_corpus.txt")
    )
    add_check(
        "ask_out_of_corpus",
        "grounded_refusal=true and citations=[]",
        f"grounded_refusal={ask_out_trace.get('grounded_refusal')} citations_empty={_citations_block_empty(run_dir / 'ask_out_of_corpus.txt')}",
        ask_out_ok,
        "ask_out_of_corpus.txt",
    )

    benchmark_ok = _check(results_by_name, "benchmark_rerank")
    add_check(
        "benchmark_rerank",
        "command success",
        f"exit={results_by_name['benchmark_rerank'].exit_code if 'benchmark_rerank' in results_by_name else 'n/a'}",
        benchmark_ok,
        "benchmark_rerank.json",
    )

    pdf_payload = _try_load_json(run_dir / "pdf_regression_docling.json") or {}
    pdf_ok = (
        _check(results_by_name, "pdf_regression")
        and int(pdf_payload.get("total_files") or 0) == expected_pdf_count
        and int(pdf_payload.get("failed_files") or 0) == 0
    )
    add_check(
        "pdf_regression",
        f"total_files={expected_pdf_count} and failed_files=0",
        f"total_files={pdf_payload.get('total_files')} failed_files={pdf_payload.get('failed_files')}",
        pdf_ok,
        "pdf_regression_docling.json",
    )

    qg_payload = _try_load_json(run_dir / "quality_gate.json") or {}
    qg_retrieval = qg_payload.get("retrieval") or {}
    qg_ok = bool(
        _check(results_by_name, "quality_gate")
        and qg_payload.get("ok") is True
        and float(qg_retrieval.get("mean_recall_at_k") or 0.0) >= 0.55
        and float(qg_retrieval.get("mean_mrr") or 0.0) >= 0.45
        and float(qg_retrieval.get("mean_ndcg_at_k") or 0.0) >= 0.50
    )
    add_check(
        "quality_gate_thresholds",
        "ok=true and recall/mrr/ndcg >= thresholds",
        f"ok={qg_payload.get('ok')} retrieval={qg_retrieval}",
        qg_ok,
        "quality_gate.json",
    )

    qg_stage_ok = bool(
        (qg_payload.get("metric_stage") == "retriever_candidates")
        and len(qg_payload.get("queries") or []) == goldset_lines
    )
    add_check(
        "quality_gate_stage",
        f"metric_stage=retriever_candidates and queries={goldset_lines}",
        f"metric_stage={qg_payload.get('metric_stage')} queries={len(qg_payload.get('queries') or [])}",
        qg_stage_ok,
        "quality_gate.json",
    )

    reports = list(index_payload.get("reports") or [])
    poison_hard = [
        r
        for r in reports
        if str(r.get("status")) == "hard_fail" and "poisoned_text_ratio" in str(r.get("switch_reason") or "")
    ]
    poison_soft = [r for r in reports if "poisoned_text_detected" in str(r.get("switch_reason") or "")]
    poison_hard_ok = len(poison_hard) == 0
    add_check(
        "anti_poison_hard",
        "no poisoned_text_ratio hard-fail on final report path",
        f"poison_hard_count={len(poison_hard)}",
        poison_hard_ok,
        "index_sber_fast.json",
    )
    if poison_soft:
        warnings.append(f"poison_soft_detected_on_reports={len(poison_soft)}")

    ui_payload = _try_load_json(run_dir / "ui_screenshot_result.json") or {}
    ui_ok = bool(ui_payload.get("ok") is True)
    ui_hint_ok = bool((ui_payload.get("ok") is False) and ui_payload.get("action_hint"))
    if not ui_ok and ui_hint_ok:
        warnings.append("ui_capture_non_blocking_playwright_missing")
    add_check(
        "capture_ui",
        "ok=true or non-blocking action_hint",
        f"ok={ui_payload.get('ok')} reason={ui_payload.get('reason')}",
        ui_ok or ui_hint_ok,
        "ui_screenshot_result.json",
    )

    mandatory = {
        "selfcheck",
        "prewarm",
        "index_full_corpus",
        "ask_in_corpus",
        "ask_out_of_corpus",
        "benchmark_rerank",
        "pdf_regression",
        "quality_gate_thresholds",
        "quality_gate_stage",
        "anti_poison_hard",
    }
    mandatory_failed = [c for c in checks if c["check"] in mandatory and not c["ok"]]

    if mandatory_failed:
        verdict = "Not Ready"
    elif warnings:
        verdict = "Conditionally Ready"
    else:
        verdict = "Ready"

    payload = {
        "verdict": verdict,
        "warnings": warnings,
        "mandatory_failed": [c["check"] for c in mandatory_failed],
        "checks": checks,
    }

    lines = [
        f"# Demo Readiness Verdict: {verdict}",
        "",
        "## Summary",
        f"- Expected corpus size: {expected_pdf_count} PDF",
        f"- Goldset size: {goldset_lines} queries",
        f"- Mandatory failed checks: {len(mandatory_failed)}",
        f"- Warnings: {len(warnings)}",
        "",
        "## Checks",
        "| Check | Expected | Actual | Pass | Evidence |",
        "|---|---|---|---|---|",
    ]
    for c in checks:
        lines.append(
            f"| {c['check']} | {c['expected']} | {c['actual']} | {'yes' if c['ok'] else 'no'} | {c['evidence']} |"
        )
    if warnings:
        lines.extend(["", "## Warnings"])
        for w in warnings:
            lines.append(f"- {w}")
    return payload, "\n".join(lines) + "\n"


def main() -> None:
    """Entrypoint: run full flow and emit final verdict artifacts."""
    parser = argparse.ArgumentParser(description="Run full demo readiness flow with auto-verdict")
    parser.add_argument("--env", type=str, default=".env")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--stall-sec", type=int, default=300)
    parser.add_argument("--preflight-ttl-sec", type=int, default=300)
    parser.add_argument("--run-dir", type=str, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(args.run_dir).expanduser().resolve()
        if args.run_dir
        else (project_root / "artifacts" / f"final_ready_run_{timestamp}")
    )
    cmd_dir = run_dir / "cmd"
    cmd_dir.mkdir(parents=True, exist_ok=True)

    python_bin = sys.executable
    steps: list[StepSpec] = [
        StepSpec(
            name="selfcheck",
            cmd=[python_bin, "-m", "rag_system.cli", "--env", args.env, "selfcheck"],
            stdout_copy_to="selfcheck.json",
        ),
        StepSpec(
            name="prewarm",
            cmd=[python_bin, "-m", "rag_system.cli", "--env", args.env, "prewarm"],
            stdout_copy_to="prewarm.json",
        ),
        StepSpec(
            name="index",
            cmd=[
                python_bin,
                "-m",
                "rag_system.cli",
                "--env",
                args.env,
                "index",
                "--data-dir",
                args.data_dir,
                "--extractor",
                "pymupdf4llm",
                "--profile",
                "demo-fast",
                "--reset-index",
                "--fast",
                "--output",
                str(run_dir / "index_sber_fast.json"),
            ],
        ),
        StepSpec(
            name="ask_in",
            cmd=[
                python_bin,
                "-m",
                "rag_system.cli",
                "--env",
                args.env,
                "ask",
                "Какие ключевые темы в отчете Сбера за 2024 год?",
                "--reranker",
                "amberoad",
                "--preflight-ttl-sec",
                str(args.preflight_ttl_sec),
            ],
            stdout_copy_to="ask_full.txt",
        ),
        StepSpec(
            name="ask_out",
            cmd=[
                python_bin,
                "-m",
                "rag_system.cli",
                "--env",
                args.env,
                "ask",
                "Расскажи про ядерный реактор на Луне в отчете Сбера 2015.",
                "--reranker",
                "amberoad",
                "--preflight-ttl-sec",
                str(args.preflight_ttl_sec),
            ],
            stdout_copy_to="ask_out_of_corpus.txt",
        ),
        StepSpec(
            name="benchmark_rerank",
            cmd=[
                python_bin,
                "-m",
                "rag_system.cli",
                "--env",
                args.env,
                "benchmark-rerank",
                "--retrieve-top-k",
                "50",
                "--rerank-top-n",
                "10",
            ],
            stdout_copy_to="benchmark_rerank.json",
        ),
        StepSpec(
            name="pdf_regression",
            cmd=[
                python_bin,
                "-m",
                "rag_system.cli",
                "--env",
                args.env,
                "pdf-regression",
                "--data-dir",
                args.data_dir,
                "--extractor",
                "docling",
                "--profile",
                "full-quality",
                "--timeout-sec",
                "45",
                "--output",
                str(run_dir / "pdf_regression_docling.json"),
                "--log-file",
                str(run_dir / "pdf_regression_docling.log"),
            ],
        ),
        StepSpec(
            name="quality_gate",
            cmd=[
                python_bin,
                "-m",
                "rag_system.cli",
                "--env",
                args.env,
                "quality-gate",
                "--goldset",
                str(project_root / "artifacts" / "goldset_sber_qa.jsonl"),
                "--reranker",
                "amberoad",
                "--metric-k",
                "10",
                "--min-recall",
                "0.55",
                "--min-mrr",
                "0.45",
                "--min-ndcg",
                "0.50",
                "--output",
                str(run_dir / "quality_gate.json"),
            ],
        ),
        StepSpec(
            name="capture_ui",
            cmd=[
                python_bin,
                str(project_root / "scripts" / "capture_ui.py"),
                "--app",
                "streamlit_app.py",
                "--artifacts-dir",
                str(run_dir),
            ],
        ),
    ]

    tsv_path = run_dir / "commands.tsv"
    tsv_path.write_text("step\texit_code\tduration_sec\toutcome\tcommand\n", encoding="utf-8")

    results: list[StepResult] = []
    for step in steps:
        result = _run_step(step, cwd=project_root, cmd_dir=cmd_dir, stall_sec=max(30, int(args.stall_sec)))
        results.append(result)
        with tsv_path.open("a", encoding="utf-8") as tsv:
            tsv.write(
                f"{result.name}\t{result.exit_code}\t{result.duration_sec}\t{result.outcome}\t{result.command}\n"
            )
        if step.stdout_copy_to:
            copy_target = run_dir / step.stdout_copy_to
            copy_target.write_text(result.out_path.read_text(encoding="utf-8"), encoding="utf-8")

    by_name = {r.name: r for r in results}
    verdict_payload, verdict_markdown = _build_verdict(
        run_dir=run_dir,
        results_by_name=by_name,
        data_dir=Path(args.data_dir).expanduser().resolve(),
        goldset_path=project_root / "artifacts" / "goldset_sber_qa.jsonl",
    )
    (run_dir / "verdict.json").write_text(
        json.dumps(verdict_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "verdict.md").write_text(verdict_markdown, encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "verdict": verdict_payload["verdict"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
