"""Resilient full-judge runner for per-query quality-gate comparisons.

This script orchestrates single-query quality-gate calls for multiple judge providers,
adds watchdog timeouts with retry logic, persists checkpoint files, and can resume.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


RETRYABLE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"LLMDidNotFinish", re.IGNORECASE), "llm_did_not_finish"),
    (re.compile(r"rate\s*limit|too\s+many\s+requests|\b429\b", re.IGNORECASE), "rate_limit"),
    (
        re.compile(
            r"service\s+unavailable|internal\s+server\s+error|bad\s+gateway|gateway\s+timeout|\b5\d\d\b",
            re.IGNORECASE,
        ),
        "server_5xx",
    ),
    (
        re.compile(
            r"connection\s+(?:error|reset|aborted)|read\s+timeout|connect\s+timeout|temporarily\s+unavailable|network",
            re.IGNORECASE,
        ),
        "network",
    ),
]

NON_RETRYABLE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"No relevant sources matched goldset patterns", re.IGNORECASE), "goldset_validation"),
    (re.compile(r"Unsupported judge provider", re.IGNORECASE), "judge_validation"),
    (re.compile(r"Invalid RAGAS score", re.IGNORECASE), "ragas_validation"),
]


@dataclass(slots=True)
class QueryCase:
    """Single goldset query prepared for standalone quality-gate run."""

    query_id: str
    query_index: int
    query: str
    relevant_patterns: list[str]
    goldset_path: Path


def _now_ts() -> str:
    """Return compact local timestamp for run directory names."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _slug(text: str) -> str:
    """Convert arbitrary text into a filesystem-safe slug."""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_") or "item"


def _parse_providers(raw: str) -> list[str]:
    """Parse, validate, and deduplicate provider list while preserving order."""
    providers = [p.strip().lower() for p in raw.split(",") if p.strip()]
    allowed = {"anthropic", "gigachat"}
    if not providers:
        raise ValueError("providers list is empty")
    invalid = [p for p in providers if p not in allowed]
    if invalid:
        raise ValueError(f"unsupported providers: {invalid}")
    deduped: list[str] = []
    seen: set[str] = set()
    for p in providers:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


def _load_goldset_queries(goldset_path: Path) -> list[dict[str, Any]]:
    """Load goldset JSONL rows and validate required query fields."""
    if not goldset_path.exists():
        raise FileNotFoundError(f"goldset not found: {goldset_path}")

    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(goldset_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        obj = json.loads(line)
        query = str(obj.get("query", "")).strip()
        if not query:
            raise RuntimeError(f"goldset row {line_no} has empty query")
        relevant_patterns = [str(x) for x in obj.get("relevant_patterns", [])]
        rows.append(
            {
                "query": query,
                "relevant_patterns": relevant_patterns,
                "raw": obj,
            }
        )

    if not rows:
        raise RuntimeError("goldset has no valid query rows")
    return rows


def _prepare_query_cases(rows: list[dict[str, Any]], split_dir: Path) -> list[QueryCase]:
    """Write one-query JSONL files and return executable query cases."""
    split_dir.mkdir(parents=True, exist_ok=True)
    cases: list[QueryCase] = []
    for idx, row in enumerate(rows, start=1):
        query_id = f"q{idx:02d}"
        payload = {
            "query": row["query"],
            "relevant_patterns": row["relevant_patterns"],
        }
        one_query_path = split_dir / f"{query_id}.jsonl"
        one_query_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
        cases.append(
            QueryCase(
                query_id=query_id,
                query_index=idx,
                query=row["query"],
                relevant_patterns=list(row["relevant_patterns"]),
                goldset_path=one_query_path,
            )
        )
    return cases


def _append_commands_log(commands_log: Path, line: str) -> None:
    """Append one line to the runner command log."""
    commands_log.parent.mkdir(parents=True, exist_ok=True)
    with commands_log.open("a", encoding="utf-8") as fh:
        fh.write(line.rstrip() + "\n")


def _run_with_watchdog(
    *,
    cmd: list[str],
    cwd: Path,
    timeout_sec: int,
    out_path: Path,
    err_path: Path,
) -> dict[str, Any]:
    """Run subprocess with timeout watchdog and return execution metadata."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    err_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timed_out = False
    status = "ok"
    exit_code: int | None = None

    with out_path.open("w", encoding="utf-8") as out_f, err_path.open("w", encoding="utf-8") as err_f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=out_f,
            stderr=err_f,
            text=True,
        )
        try:
            exit_code = int(proc.wait(timeout=max(1, int(timeout_sec))))
            status = "ok" if exit_code == 0 else "failed"
        except subprocess.TimeoutExpired:
            timed_out = True
            status = "timeout"
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except Exception:
                    pass
            exit_code = int(proc.returncode) if proc.returncode is not None else -15

    return {
        "status": status,
        "exit_code": exit_code,
        "duration_sec": round(time.monotonic() - started, 3),
        "timed_out": bool(timed_out),
        "timeout_sec": int(timeout_sec),
        "command": cmd,
    }


def _read_text(path: Path) -> str:
    """Read UTF-8 text file, returning empty string when file is missing."""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def classify_failure(
    *,
    status: str,
    stdout_text: str,
    stderr_text: str,
) -> tuple[str, bool]:
    """Classify failure reason and retryability."""
    if status == "timeout":
        return "timeout", True

    haystack = f"{stderr_text}\n{stdout_text}"

    for pattern, code in NON_RETRYABLE_PATTERNS:
        if pattern.search(haystack):
            return code, False

    for pattern, code in RETRYABLE_PATTERNS:
        if pattern.search(haystack):
            return code, True

    return "failed", False


def parse_single_quality_gate_output(
    payload_path: Path,
    *,
    expected_provider: str,
) -> dict[str, Any]:
    """Validate and normalize single-query quality-gate JSON output."""
    if not payload_path.exists():
        raise RuntimeError(f"quality-gate output does not exist: {payload_path}")

    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"quality-gate output is not valid JSON: {payload_path}") from exc

    retrieval = payload.get("retrieval")
    ragas = payload.get("ragas")
    ragas_judge = payload.get("ragas_judge")
    queries = payload.get("queries")

    if not isinstance(retrieval, dict):
        raise RuntimeError("quality-gate output missing retrieval block")
    if not isinstance(ragas, dict):
        raise RuntimeError("quality-gate output missing ragas block (use --with-ragas)")
    if not isinstance(ragas_judge, dict):
        raise RuntimeError("quality-gate output missing ragas_judge block")
    if not isinstance(queries, list) or len(queries) != 1:
        raise RuntimeError("single-query run must contain exactly one item in queries[]")

    provider = str(ragas_judge.get("provider", "")).strip().lower()
    model = str(ragas_judge.get("model", "")).strip()
    if provider != expected_provider:
        raise RuntimeError(
            f"ragas_judge.provider mismatch: got={provider!r} expected={expected_provider!r}"
        )
    if not model:
        raise RuntimeError("ragas_judge.model is empty")

    ragas_norm: dict[str, float] = {}
    for metric in ("faithfulness", "answer_relevance", "context_relevance"):
        value = ragas.get(metric)
        if value is None:
            raise RuntimeError(f"ragas metric missing: {metric}")
        val = float(value)
        if not math.isfinite(val) or val < 0.0 or val > 1.0:
            raise RuntimeError(f"ragas metric out of range [0,1] for {metric}: {val}")
        ragas_norm[metric] = val

    retrieval_norm: dict[str, float] = {}
    for metric in ("mean_recall_at_k", "mean_mrr", "mean_ndcg_at_k"):
        value = retrieval.get(metric)
        if value is None:
            raise RuntimeError(f"retrieval metric missing: {metric}")
        retrieval_norm[metric] = float(value)

    return {
        "ragas": ragas_norm,
        "ragas_judge": {
            "provider": provider,
            "model": model,
        },
        "retrieval": retrieval_norm,
        "query": queries[0].get("query"),
        "ok": bool(payload.get("ok")),
    }


def _checkpoint_load(path: Path) -> list[dict[str, Any]]:
    """Load checkpoint JSONL rows with best-effort parsing."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _checkpoint_write_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    """Persist checkpoint rows atomically using tmp-file replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    body = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if body:
        body += "\n"
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)


def _checkpoint_latest_by_query(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return latest checkpoint record for each query id."""
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        query_id = str(row.get("query_id", ""))
        if not query_id:
            continue
        latest[query_id] = row
    return latest


def _mean(values: list[float]) -> float | None:
    """Compute arithmetic mean for non-empty value list."""
    if not values:
        return None
    return float(sum(values) / len(values))


def summarize_provider_records(
    *,
    provider: str,
    latest_records: dict[str, dict[str, Any]],
    total_queries: int,
) -> dict[str, Any]:
    """Summarize provider completion and aggregate metrics from checkpoint records."""
    completed: list[str] = []
    failed: list[str] = []
    timed_out: list[str] = []

    ragas_f: list[float] = []
    ragas_a: list[float] = []
    ragas_c: list[float] = []
    r_recall: list[float] = []
    r_mrr: list[float] = []
    r_ndcg: list[float] = []

    for query_id, rec in latest_records.items():
        status = str(rec.get("status", ""))
        if status == "success":
            completed.append(query_id)
            ragas = rec.get("ragas") or {}
            retrieval = rec.get("retrieval") or {}
            if "faithfulness" in ragas:
                ragas_f.append(float(ragas["faithfulness"]))
            if "answer_relevance" in ragas:
                ragas_a.append(float(ragas["answer_relevance"]))
            if "context_relevance" in ragas:
                ragas_c.append(float(ragas["context_relevance"]))
            if "mean_recall_at_k" in retrieval:
                r_recall.append(float(retrieval["mean_recall_at_k"]))
            if "mean_mrr" in retrieval:
                r_mrr.append(float(retrieval["mean_mrr"]))
            if "mean_ndcg_at_k" in retrieval:
                r_ndcg.append(float(retrieval["mean_ndcg_at_k"]))
        else:
            failed.append(query_id)
            if status == "timeout":
                timed_out.append(query_id)

    completed = sorted(completed)
    failed = sorted(failed)
    timed_out = sorted(timed_out)

    return {
        "provider": provider,
        "provider_status": "complete" if len(completed) == total_queries else "incomplete",
        "completed_queries": len(completed),
        "failed_queries": failed,
        "timed_out_queries": timed_out,
        "aggregate_metrics": {
            "faithfulness": _mean(ragas_f),
            "answer_relevance": _mean(ragas_a),
            "context_relevance": _mean(ragas_c),
            "mean_recall_at_k": _mean(r_recall),
            "mean_mrr": _mean(r_mrr),
            "mean_ndcg_at_k": _mean(r_ndcg),
        },
    }


def _provider_model(provider: str, args: argparse.Namespace) -> str | None:
    """Resolve judge model override for the selected provider."""
    if provider == "anthropic":
        return args.anthropic_model
    if provider == "gigachat":
        return args.gigachat_model
    return None


def _build_quality_gate_cmd(
    *,
    args: argparse.Namespace,
    provider: str,
    model: str | None,
    one_query_goldset: Path,
    output_path: Path,
) -> list[str]:
    """Build one-shot quality-gate command for a single-query goldset file."""
    cmd = [
        sys.executable,
        "-m",
        "rag_system.cli",
        "--env",
        args.env,
        "quality-gate",
        "--goldset",
        str(one_query_goldset),
        "--reranker",
        args.reranker,
        "--metric-k",
        str(args.metric_k),
        "--min-recall",
        str(args.min_recall),
        "--min-mrr",
        str(args.min_mrr),
        "--min-ndcg",
        str(args.min_ndcg),
        "--with-ragas",
        "--judge-provider",
        provider,
        "--output",
        str(output_path),
    ]
    if model:
        cmd.extend(["--judge-model", str(model)])
    return cmd


def run_provider(
    *,
    provider: str,
    args: argparse.Namespace,
    cwd: Path,
    run_dir: Path,
    cases: list[QueryCase],
    commands_log: Path,
    run_cmd_fn: Callable[..., dict[str, Any]] = _run_with_watchdog,
    parse_output_fn: Callable[[Path], dict[str, Any]] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run per-query quality-gate jobs for one provider with retries/checkpoints."""
    provider_dir = run_dir / provider
    provider_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / f"checkpoint_{provider}.jsonl"

    checkpoint_rows = _checkpoint_load(checkpoint_path)
    latest = _checkpoint_latest_by_query(checkpoint_rows)
    parser_fn = parse_output_fn or (lambda payload_path: parse_single_quality_gate_output(payload_path, expected_provider=provider))

    for case in cases:
        output_path = provider_dir / f"{case.query_id}.json"

        if args.resume:
            prev = latest.get(case.query_id)
            if prev and str(prev.get("status")) == "success" and output_path.exists():
                _append_commands_log(
                    commands_log,
                    f"[{provider}] skip {case.query_id} due to checkpoint success",
                )
                continue

        attempt_no = 0
        final_status = "failed"
        final_error = "failed"
        final_meta: dict[str, Any] | None = None
        parsed_payload: dict[str, Any] | None = None
        cumulative_duration = 0.0

        model = _provider_model(provider, args)

        while attempt_no <= int(args.max_retries):
            attempt_no += 1
            step_name = f"{provider}_{case.query_id}_a{attempt_no}"
            out_path = run_dir / "cmd" / f"{step_name}.out"
            err_path = run_dir / "cmd" / f"{step_name}.err"
            meta_path = run_dir / "cmd" / f"{step_name}.meta.json"

            cmd = _build_quality_gate_cmd(
                args=args,
                provider=provider,
                model=model,
                one_query_goldset=case.goldset_path,
                output_path=output_path,
            )
            _append_commands_log(commands_log, f"[{provider}] {case.query_id} attempt {attempt_no}: {' '.join(cmd)}")

            meta = run_cmd_fn(
                cmd=cmd,
                cwd=cwd,
                timeout_sec=int(args.per_query_timeout_sec),
                out_path=out_path,
                err_path=err_path,
            )
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            final_meta = {
                **meta,
                "meta_path": str(meta_path),
                "stdout_path": str(out_path),
                "stderr_path": str(err_path),
            }
            cumulative_duration += float(meta.get("duration_sec") or 0.0)

            if str(meta.get("status")) == "ok":
                try:
                    parsed_payload = parser_fn(output_path)
                    final_status = "success"
                    final_error = "none"
                    break
                except Exception as exc:
                    final_status = "failed"
                    final_error = f"output_parse_error:{type(exc).__name__}"
                    _append_commands_log(
                        commands_log,
                        f"[{provider}] {case.query_id} parse failed on attempt {attempt_no}: {exc}",
                    )
                    break

            stdout_text = _read_text(out_path)
            stderr_text = _read_text(err_path)
            error_class, retryable = classify_failure(
                status=str(meta.get("status", "failed")),
                stdout_text=stdout_text,
                stderr_text=stderr_text,
            )
            final_error = error_class
            final_status = "timeout" if str(meta.get("status")) == "timeout" else "failed"

            if retryable and attempt_no <= int(args.max_retries):
                _append_commands_log(
                    commands_log,
                    f"[{provider}] {case.query_id} retryable={error_class} attempt {attempt_no}; backoff={args.retry_backoff_sec}s",
                )
                sleep_fn(float(args.retry_backoff_sec))
                continue

            break

        record: dict[str, Any] = {
            "query_id": case.query_id,
            "query_index": case.query_index,
            "query": case.query,
            "provider": provider,
            "status": final_status,
            "attempt_no": attempt_no,
            "duration_sec": round(cumulative_duration, 3),
            "error_class": final_error,
            "output_path": str(output_path),
            "meta_path": (final_meta or {}).get("meta_path"),
            "stdout_path": (final_meta or {}).get("stdout_path"),
            "stderr_path": (final_meta or {}).get("stderr_path"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "ragas": (parsed_payload or {}).get("ragas"),
            "retrieval": (parsed_payload or {}).get("retrieval"),
            "ragas_judge": (parsed_payload or {}).get("ragas_judge"),
        }
        checkpoint_rows.append(record)
        _checkpoint_write_atomic(checkpoint_path, checkpoint_rows)
        latest[case.query_id] = record

    provider_summary = summarize_provider_records(
        provider=provider,
        latest_records=latest,
        total_queries=len(cases),
    )
    provider_summary["checkpoint_path"] = str(checkpoint_path)
    provider_summary["records_by_query"] = latest
    return provider_summary


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    """Round float values while preserving None."""
    if value is None:
        return None
    return round(float(value), digits)


def _build_provider_view(summary: dict[str, Any]) -> dict[str, Any]:
    """Normalize provider summary into report-friendly payload."""
    metrics = summary.get("aggregate_metrics") or {}
    return {
        "provider_status": summary.get("provider_status"),
        "completed_queries": int(summary.get("completed_queries") or 0),
        "failed_queries": list(summary.get("failed_queries") or []),
        "timed_out_queries": list(summary.get("timed_out_queries") or []),
        "aggregate_metrics": {
            "faithfulness": _round_or_none(metrics.get("faithfulness")),
            "answer_relevance": _round_or_none(metrics.get("answer_relevance")),
            "context_relevance": _round_or_none(metrics.get("context_relevance")),
            "mean_recall_at_k": _round_or_none(metrics.get("mean_recall_at_k")),
            "mean_mrr": _round_or_none(metrics.get("mean_mrr")),
            "mean_ndcg_at_k": _round_or_none(metrics.get("mean_ndcg_at_k")),
        },
    }


def _compute_delta_anthropic_minus_gigachat(provider_views: dict[str, dict[str, Any]]) -> dict[str, float | None] | None:
    """Compute aggregate metric delta between Anthropic and GigaChat judges."""
    anth = provider_views.get("anthropic")
    giga = provider_views.get("gigachat")
    if not anth or not giga:
        return None

    anth_metrics = anth.get("aggregate_metrics") or {}
    giga_metrics = giga.get("aggregate_metrics") or {}

    def delta(key: str) -> float | None:
        """Calculate rounded metric difference for a single metric key."""
        left = anth_metrics.get(key)
        right = giga_metrics.get(key)
        if left is None or right is None:
            return None
        return round(float(left) - float(right), 4)

    return {
        "faithfulness": delta("faithfulness"),
        "answer_relevance": delta("answer_relevance"),
        "context_relevance": delta("context_relevance"),
        "mean_recall_at_k": delta("mean_recall_at_k"),
        "mean_mrr": delta("mean_mrr"),
        "mean_ndcg_at_k": delta("mean_ndcg_at_k"),
    }


def _compute_judge_quality_warning(provider_views: dict[str, dict[str, Any]]) -> str | None:
    """Return warning code when both judges produce all-zero RAGAS aggregates."""
    needed = ("anthropic", "gigachat")
    for provider in needed:
        if provider not in provider_views:
            return None

    keys = ("faithfulness", "answer_relevance", "context_relevance")
    for provider in needed:
        metrics = (provider_views[provider].get("aggregate_metrics") or {})
        values = [metrics.get(k) for k in keys]
        if any(v is None for v in values):
            return None
        if any(abs(float(v)) > 1e-12 for v in values):
            return None
    return "all_ragas_zero_both_judges"


def _top_disagreements(
    provider_summaries: dict[str, dict[str, Any]],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Return top-N queries with largest absolute RAGAS judge disagreement."""
    anth = provider_summaries.get("anthropic", {}).get("records_by_query", {})
    giga = provider_summaries.get("gigachat", {}).get("records_by_query", {})

    rows: list[dict[str, Any]] = []
    for query_id, anth_rec in anth.items():
        giga_rec = giga.get(query_id)
        if not giga_rec:
            continue
        if anth_rec.get("status") != "success" or giga_rec.get("status") != "success":
            continue

        anth_r = anth_rec.get("ragas") or {}
        giga_r = giga_rec.get("ragas") or {}

        keys = ("faithfulness", "answer_relevance", "context_relevance")
        if not all(key in anth_r and key in giga_r for key in keys):
            continue

        delta = sum(abs(float(anth_r[key]) - float(giga_r[key])) for key in keys)
        rows.append(
            {
                "query_id": query_id,
                "query_index": anth_rec.get("query_index"),
                "query": anth_rec.get("query"),
                "abs_delta_sum": round(delta, 4),
                "anthropic": {key: round(float(anth_r[key]), 4) for key in keys},
                "gigachat": {key: round(float(giga_r[key]), 4) for key in keys},
            }
        )

    rows.sort(key=lambda x: (x["abs_delta_sum"], -(x.get("query_index") or 0)), reverse=True)
    return rows[:top_n]


def _write_markdown_report(comparison: dict[str, Any], target: Path) -> None:
    """Render human-readable markdown report from judge comparison payload."""
    lines: list[str] = [
        "# Full Judge Comparison",
        "",
        f"- Run dir: `{comparison['run_dir']}`",
        f"- Goldset: `{comparison['goldset']}`",
        f"- Total queries: `{comparison['total_queries']}`",
        f"- Overall status: `{comparison['overall_status']}`",
        "",
        "## Providers",
    ]

    providers = comparison.get("providers") or {}
    for name in sorted(providers):
        p = providers[name]
        lines.extend(
            [
                f"### {name}",
                f"- provider_status: `{p.get('provider_status')}`",
                f"- completed_queries: `{p.get('completed_queries')}`",
                f"- failed_queries: `{len(p.get('failed_queries') or [])}`",
                f"- timed_out_queries: `{len(p.get('timed_out_queries') or [])}`",
                f"- aggregate_metrics: `{json.dumps(p.get('aggregate_metrics'), ensure_ascii=False)}`",
                "",
            ]
        )

    if comparison.get("delta_anthropic_minus_gigachat") is not None:
        lines.append("## Delta (Anthropic - GigaChat)")
        lines.append("")
        lines.append(f"`{json.dumps(comparison['delta_anthropic_minus_gigachat'], ensure_ascii=False)}`")
        lines.append("")

    if comparison.get("judge_quality_warning"):
        lines.append("## Judge Quality Warning")
        lines.append("")
        lines.append(f"- warning: `{comparison['judge_quality_warning']}`")
        lines.append("")

    lines.append("## Top-5 RAGAS Disagreements")
    lines.append("")
    disagreements = comparison.get("top5_disagreements") or []
    if not disagreements:
        lines.append("- No overlap or no successful paired queries.")
    else:
        for row in disagreements:
            lines.append(
                f"- {row['query_id']}: abs_delta_sum={row['abs_delta_sum']} | query={row.get('query')!r}"
            )

    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_full_judge(args: argparse.Namespace) -> dict[str, Any]:
    """Execute full multi-provider judge run and write comparison artifacts."""
    cwd = Path.cwd()
    run_dir = Path(args.run_dir) if args.run_dir else (Path("artifacts") / f"judge_full_runner_{_now_ts()}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cmd").mkdir(parents=True, exist_ok=True)

    commands_log = run_dir / "commands.log"
    providers = _parse_providers(args.providers)
    goldset_path = Path(args.goldset)
    gold_rows = _load_goldset_queries(goldset_path)
    cases = _prepare_query_cases(gold_rows, run_dir / "goldset_split")

    run_meta = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "cwd": str(cwd),
        "run_dir": str(run_dir),
        "providers": providers,
        "goldset": str(goldset_path),
        "total_queries": len(cases),
        "args": {
            "env": args.env,
            "reranker": args.reranker,
            "metric_k": args.metric_k,
            "min_recall": args.min_recall,
            "min_mrr": args.min_mrr,
            "min_ndcg": args.min_ndcg,
            "anthropic_model": args.anthropic_model,
            "gigachat_model": args.gigachat_model,
            "per_query_timeout_sec": args.per_query_timeout_sec,
            "max_retries": args.max_retries,
            "retry_backoff_sec": args.retry_backoff_sec,
            "resume": args.resume,
        },
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    provider_summaries: dict[str, dict[str, Any]] = {}
    provider_views: dict[str, dict[str, Any]] = {}

    for provider in providers:
        _append_commands_log(commands_log, f"=== provider={provider} started_at={datetime.now().isoformat(timespec='seconds')} ===")
        summary = run_provider(
            provider=provider,
            args=args,
            cwd=cwd,
            run_dir=run_dir,
            cases=cases,
            commands_log=commands_log,
        )
        provider_summaries[provider] = summary
        provider_views[provider] = _build_provider_view(summary)
        _append_commands_log(
            commands_log,
            f"=== provider={provider} finished status={provider_views[provider]['provider_status']} completed={provider_views[provider]['completed_queries']}/{len(cases)} ===",
        )

    delta = _compute_delta_anthropic_minus_gigachat(provider_views)
    judge_quality_warning = _compute_judge_quality_warning(provider_views)
    disagreements = _top_disagreements(provider_summaries, top_n=5)
    overall_status = "complete" if all(v["provider_status"] == "complete" for v in provider_views.values()) else "incomplete"

    comparison = {
        "run_dir": str(run_dir),
        "goldset": str(goldset_path),
        "total_queries": len(cases),
        "overall_status": overall_status,
        "providers": provider_views,
        "delta_anthropic_minus_gigachat": delta,
        "judge_quality_warning": judge_quality_warning,
        "top5_disagreements": disagreements,
    }

    (run_dir / "judge_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown_report(comparison, run_dir / "judge_comparison.md")
    print(json.dumps({"overall_status": overall_status, "run_dir": str(run_dir)}, ensure_ascii=False))
    return comparison


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for the full-judge runner."""
    parser = argparse.ArgumentParser(description="Resilient full-judge runner (per-query quality-gate)")
    parser.add_argument("--env", type=str, default=".env")
    parser.add_argument("--goldset", type=str, default="artifacts/goldset_sber_qa.jsonl")
    parser.add_argument("--reranker", type=str, default="amberoad")
    parser.add_argument("--metric-k", type=int, default=10)
    parser.add_argument("--min-recall", type=float, default=0.55)
    parser.add_argument("--min-mrr", type=float, default=0.45)
    parser.add_argument("--min-ndcg", type=float, default=0.50)
    parser.add_argument("--providers", type=str, default="anthropic,gigachat")
    parser.add_argument("--anthropic-model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--gigachat-model", type=str, default=None)
    parser.add_argument("--per-query-timeout-sec", type=int, default=900)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-backoff-sec", type=int, default=5)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-dir", type=str, default=None)
    return parser


def main() -> int:
    """CLI entrypoint returning process exit code."""
    parser = build_parser()
    args = parser.parse_args()
    run_full_judge(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
