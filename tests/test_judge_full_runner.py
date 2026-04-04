"""Unit tests for resilient judge full runner orchestration."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def runner_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "judge_full_runner.py"
    name = "judge_full_runner_test_module"
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load judge_full_runner.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _mk_args(*, resume: bool, max_retries: int) -> SimpleNamespace:
    return SimpleNamespace(
        env=".env",
        reranker="amberoad",
        metric_k=10,
        min_recall=0.55,
        min_mrr=0.45,
        min_ndcg=0.50,
        anthropic_model="claude-sonnet-4-20250514",
        gigachat_model=None,
        per_query_timeout_sec=1,
        max_retries=max_retries,
        retry_backoff_sec=0,
        resume=resume,
    )


def _mk_case(runner_module, tmp_path: Path, *, query_id: str = "q01"):
    gold = tmp_path / f"{query_id}.jsonl"
    gold.write_text(
        json.dumps({"query": "Q", "relevant_patterns": ["2020"]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return [
        runner_module.QueryCase(
            query_id=query_id,
            query_index=1,
            query="Q",
            relevant_patterns=["2020"],
            goldset_path=gold,
        )
    ]


def _ok_parsed(provider: str) -> dict:
    return {
        "ragas": {
            "faithfulness": 0.8,
            "answer_relevance": 0.7,
            "context_relevance": 0.9,
        },
        "ragas_judge": {"provider": provider, "model": "m"},
        "retrieval": {
            "mean_recall_at_k": 1.0,
            "mean_mrr": 0.5,
            "mean_ndcg_at_k": 0.6,
        },
    }


def test_run_provider_timeout_retries_then_marks_timeout(tmp_path: Path, runner_module) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "cmd").mkdir(parents=True)
    args = _mk_args(resume=False, max_retries=2)
    cases = _mk_case(runner_module, tmp_path)
    commands_log = run_dir / "commands.log"

    calls = {"n": 0}

    def fake_run_cmd_fn(*, cmd, cwd, timeout_sec, out_path, err_path):  # noqa: ARG001
        calls["n"] += 1
        out_path.write_text("", encoding="utf-8")
        err_path.write_text("Timed out", encoding="utf-8")
        return {
            "status": "timeout",
            "exit_code": -15,
            "duration_sec": 1.0,
            "timed_out": True,
            "timeout_sec": timeout_sec,
            "command": cmd,
        }

    summary = runner_module.run_provider(
        provider="anthropic",
        args=args,
        cwd=tmp_path,
        run_dir=run_dir,
        cases=cases,
        commands_log=commands_log,
        run_cmd_fn=fake_run_cmd_fn,
        parse_output_fn=lambda _: _ok_parsed("anthropic"),
        sleep_fn=lambda _: None,
    )

    rec = summary["records_by_query"]["q01"]
    assert calls["n"] == 3
    assert rec["status"] == "timeout"
    assert rec["attempt_no"] == 3
    assert rec["error_class"] == "timeout"


def test_run_provider_retries_llm_error_then_succeeds(tmp_path: Path, runner_module) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "cmd").mkdir(parents=True)
    args = _mk_args(resume=False, max_retries=2)
    cases = _mk_case(runner_module, tmp_path)
    commands_log = run_dir / "commands.log"

    calls = {"n": 0}

    def fake_run_cmd_fn(*, cmd, cwd, timeout_sec, out_path, err_path):  # noqa: ARG001
        calls["n"] += 1
        if calls["n"] == 1:
            out_path.write_text("", encoding="utf-8")
            err_path.write_text("LLMDidNotFinish", encoding="utf-8")
            return {
                "status": "failed",
                "exit_code": 1,
                "duration_sec": 0.2,
                "timed_out": False,
                "timeout_sec": timeout_sec,
                "command": cmd,
            }
        out_path.write_text("ok", encoding="utf-8")
        err_path.write_text("", encoding="utf-8")
        return {
            "status": "ok",
            "exit_code": 0,
            "duration_sec": 0.1,
            "timed_out": False,
            "timeout_sec": timeout_sec,
            "command": cmd,
        }

    summary = runner_module.run_provider(
        provider="anthropic",
        args=args,
        cwd=tmp_path,
        run_dir=run_dir,
        cases=cases,
        commands_log=commands_log,
        run_cmd_fn=fake_run_cmd_fn,
        parse_output_fn=lambda _: _ok_parsed("anthropic"),
        sleep_fn=lambda _: None,
    )

    rec = summary["records_by_query"]["q01"]
    assert calls["n"] == 2
    assert rec["status"] == "success"
    assert rec["attempt_no"] == 2


def test_run_provider_does_not_retry_non_retryable_failure(tmp_path: Path, runner_module) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "cmd").mkdir(parents=True)
    args = _mk_args(resume=False, max_retries=3)
    cases = _mk_case(runner_module, tmp_path)
    commands_log = run_dir / "commands.log"

    calls = {"n": 0}

    def fake_run_cmd_fn(*, cmd, cwd, timeout_sec, out_path, err_path):  # noqa: ARG001
        calls["n"] += 1
        out_path.write_text("", encoding="utf-8")
        err_path.write_text("No relevant sources matched goldset patterns", encoding="utf-8")
        return {
            "status": "failed",
            "exit_code": 1,
            "duration_sec": 0.2,
            "timed_out": False,
            "timeout_sec": timeout_sec,
            "command": cmd,
        }

    summary = runner_module.run_provider(
        provider="anthropic",
        args=args,
        cwd=tmp_path,
        run_dir=run_dir,
        cases=cases,
        commands_log=commands_log,
        run_cmd_fn=fake_run_cmd_fn,
        parse_output_fn=lambda _: _ok_parsed("anthropic"),
        sleep_fn=lambda _: None,
    )

    rec = summary["records_by_query"]["q01"]
    assert calls["n"] == 1
    assert rec["status"] == "failed"
    assert rec["error_class"] == "goldset_validation"


def test_run_provider_resume_skips_successful_query(tmp_path: Path, runner_module) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "cmd").mkdir(parents=True)
    (run_dir / "anthropic").mkdir(parents=True)
    args = _mk_args(resume=True, max_retries=2)
    cases = _mk_case(runner_module, tmp_path)
    commands_log = run_dir / "commands.log"

    existing_out = run_dir / "anthropic" / "q01.json"
    existing_out.write_text("{}", encoding="utf-8")
    checkpoint = run_dir / "checkpoint_anthropic.jsonl"
    checkpoint.write_text(
        json.dumps(
            {
                "query_id": "q01",
                "query_index": 1,
                "query": "Q",
                "provider": "anthropic",
                "status": "success",
                "attempt_no": 1,
                "duration_sec": 1.0,
                "error_class": "none",
                "output_path": str(existing_out),
                "ragas": {
                    "faithfulness": 0.9,
                    "answer_relevance": 0.8,
                    "context_relevance": 0.85,
                },
                "retrieval": {
                    "mean_recall_at_k": 1.0,
                    "mean_mrr": 1.0,
                    "mean_ndcg_at_k": 1.0,
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    calls = {"n": 0}

    def fake_run_cmd_fn(*, cmd, cwd, timeout_sec, out_path, err_path):  # noqa: ARG001
        calls["n"] += 1
        raise AssertionError("run_cmd_fn should not be called for resumed successful query")

    summary = runner_module.run_provider(
        provider="anthropic",
        args=args,
        cwd=tmp_path,
        run_dir=run_dir,
        cases=cases,
        commands_log=commands_log,
        run_cmd_fn=fake_run_cmd_fn,
        parse_output_fn=lambda _: _ok_parsed("anthropic"),
        sleep_fn=lambda _: None,
    )

    assert calls["n"] == 0
    assert summary["provider_status"] == "complete"
    assert summary["completed_queries"] == 1


def test_parse_single_output_validates_ragas_judge_and_provider(tmp_path: Path, runner_module) -> None:
    good = tmp_path / "good.json"
    good.write_text(
        json.dumps(
            {
                "ok": True,
                "retrieval": {
                    "mean_recall_at_k": 1.0,
                    "mean_mrr": 0.5,
                    "mean_ndcg_at_k": 0.6,
                },
                "ragas": {
                    "faithfulness": 0.8,
                    "answer_relevance": 0.7,
                    "context_relevance": 0.9,
                },
                "ragas_judge": {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
                "queries": [{"query": "Q"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    parsed = runner_module.parse_single_quality_gate_output(good, expected_provider="anthropic")
    assert parsed["ragas_judge"]["provider"] == "anthropic"

    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "ok": True,
                "retrieval": {
                    "mean_recall_at_k": 1.0,
                    "mean_mrr": 0.5,
                    "mean_ndcg_at_k": 0.6,
                },
                "ragas": {
                    "faithfulness": 0.8,
                    "answer_relevance": 0.7,
                    "context_relevance": 0.9,
                },
                "ragas_judge": {"provider": "gigachat", "model": ""},
                "queries": [{"query": "Q"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="provider mismatch"):
        runner_module.parse_single_quality_gate_output(bad, expected_provider="anthropic")


def test_summarize_provider_records_computes_complete_and_incomplete(runner_module) -> None:
    latest = {
        "q01": {
            "status": "success",
            "ragas": {"faithfulness": 1.0, "answer_relevance": 0.8, "context_relevance": 0.6},
            "retrieval": {"mean_recall_at_k": 0.9, "mean_mrr": 0.7, "mean_ndcg_at_k": 0.8},
        },
        "q02": {
            "status": "failed",
        },
    }
    incomplete = runner_module.summarize_provider_records(
        provider="anthropic",
        latest_records=latest,
        total_queries=2,
    )
    assert incomplete["provider_status"] == "incomplete"
    assert incomplete["completed_queries"] == 1

    complete = runner_module.summarize_provider_records(
        provider="anthropic",
        latest_records={"q01": latest["q01"]},
        total_queries=1,
    )
    assert complete["provider_status"] == "complete"
    assert complete["aggregate_metrics"]["faithfulness"] == pytest.approx(1.0)


def test_compute_judge_quality_warning_detects_all_zero_across_both_judges(runner_module) -> None:
    views = {
        "anthropic": {
            "aggregate_metrics": {
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "context_relevance": 0.0,
            }
        },
        "gigachat": {
            "aggregate_metrics": {
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "context_relevance": 0.0,
            }
        },
    }
    warning = runner_module._compute_judge_quality_warning(views)
    assert warning == "all_ragas_zero_both_judges"

    views["gigachat"]["aggregate_metrics"]["faithfulness"] = 0.01
    warning2 = runner_module._compute_judge_quality_warning(views)
    assert warning2 is None
