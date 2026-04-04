"""Unit tests for CLI contracts and parser defaults."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest

import rag_system.cli as cli_module
from rag_system.cli import _relevant_chunk_ids, build_parser
from rag_system.types import RetrievedChunk


def _index_extractor_action():
    parser = build_parser()
    subparsers = parser._subparsers._group_actions[0].choices  # noqa: SLF001
    index_parser = subparsers["index"]
    return next(action for action in index_parser._actions if action.dest == "extractor")  # noqa: SLF001


def test_index_extractor_default_comes_from_settings() -> None:
    action = _index_extractor_action()
    assert action.default is None


def test_index_extractor_has_strict_choices() -> None:
    action = _index_extractor_action()
    assert set(action.choices or []) == {"docling", "pymupdf4llm", "unstructured"}


def test_index_rejects_unknown_extractor() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["index", "--data-dir", "/tmp", "--extractor", "unknown_backend"])


def test_relevant_chunk_ids_expand_source_matches() -> None:
    index = SimpleNamespace(
        path_to_chunk_ids={
            "/tmp/Копия Сбер 2015.pdf": ["c1", "c2"],
            "/tmp/Копия Сбер 2016.pdf": ["c3"],
        }
    )
    assert _relevant_chunk_ids(index, ["2015"]) == {"c1", "c2"}


def _candidate(chunk_id: str, source_path: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=f"text-{chunk_id}",
        source_path=source_path,
        page=1,
        element_type="text",
        bm25_score=1.0,
        dense_score=1.0,
        fusion_score=1.0,
        metadata={},
    )


def _quality_gate_args(
    goldset: str,
    output: str,
    final_top_k: int,
    *,
    with_ragas: bool = False,
    judge_provider: str | None = None,
    judge_model: str | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        env=None,
        goldset=goldset,
        reranker="amberoad",
        retrieve_top_k=10,
        rerank_top_n=5,
        final_top_k=final_top_k,
        skip_preflight=True,
        preflight_ttl_sec=0,
        metric_k=10,
        min_recall=0.0,
        min_mrr=0.0,
        min_ndcg=0.0,
        with_ragas=with_ragas,
        judge_provider=judge_provider,
        judge_model=judge_model,
        output=output,
    )


def test_quality_gate_uses_retriever_stage_independent_of_final_top_k(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    goldset = tmp_path / "goldset.jsonl"
    goldset.write_text(
        json.dumps({"query": "q1", "relevant_patterns": ["docA"]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    def _collect_once(pipeline, query: str, retrieve_k: int):  # noqa: ARG001
        return [query], [
            _candidate("a1", "/tmp/docA.pdf"),
            _candidate("b1", "/tmp/docB.pdf"),
        ]

    class _FakePipeline:
        def __init__(self, settings):  # noqa: ARG002
            self.index = SimpleNamespace(
                indexed_chunks=[object()],
                path_to_chunk_ids={
                    "/tmp/docA.pdf": ["a1"],
                    "/tmp/docB.pdf": ["b1"],
                },
            )

        def ask(self, **kwargs):
            final_top_k = int(kwargs.get("final_top_k") or 0)
            selected = [_candidate("a1", "/tmp/docA.pdf"), _candidate("b1", "/tmp/docB.pdf")][:final_top_k]
            return SimpleNamespace(
                answer="ok",
                context_chunks=selected,
                trace=SimpleNamespace(grounded_refusal=False, grounded_reason=None),
            )

    captured_rows: list[list[dict]] = []

    def _fake_eval(rows: list[dict], k: int):  # noqa: ARG001
        captured_rows.append(rows)
        return SimpleNamespace(mean_recall_at_k=1.0, mean_mrr=1.0, mean_ndcg_at_k=1.0)

    monkeypatch.setattr(cli_module, "_collect_candidates_once", _collect_once)
    monkeypatch.setattr("rag_system.config.load_settings", lambda dotenv_path=None: SimpleNamespace())
    monkeypatch.setattr("rag_system.pipeline.RAGPipeline", _FakePipeline)
    monkeypatch.setattr("rag_system.eval.evaluate_retrieval", _fake_eval)

    out_1 = tmp_path / "qg_ftk1.json"
    out_8 = tmp_path / "qg_ftk8.json"
    cli_module._cmd_quality_gate(_quality_gate_args(str(goldset), str(out_1), final_top_k=1))
    cli_module._cmd_quality_gate(_quality_gate_args(str(goldset), str(out_8), final_top_k=8))

    assert len(captured_rows) == 2
    assert captured_rows[0][0]["retrieved_ids"] == ["/tmp/docA.pdf", "/tmp/docB.pdf"]
    assert captured_rows[1][0]["retrieved_ids"] == ["/tmp/docA.pdf", "/tmp/docB.pdf"]
    assert captured_rows[0][0]["retrieved_ids"] == captured_rows[1][0]["retrieved_ids"]

    payload_1 = json.loads(out_1.read_text(encoding="utf-8"))
    payload_8 = json.loads(out_8.read_text(encoding="utf-8"))
    assert payload_1["metric_stage"] == "retriever_candidates"
    assert payload_8["metric_stage"] == "retriever_candidates"
    assert payload_1["metric_entity"] == "source_path"
    assert payload_8["metric_entity"] == "source_path"
    assert payload_1["queries"][0]["retrieved_ids_stage"] == ["a1", "b1"]
    assert payload_8["queries"][0]["retrieved_ids_stage"] == ["a1", "b1"]
    assert payload_1["queries"][0]["retrieved_sources_stage"] == ["/tmp/docA.pdf", "/tmp/docB.pdf"]
    assert payload_8["queries"][0]["retrieved_sources_stage"] == ["/tmp/docA.pdf", "/tmp/docB.pdf"]
    assert payload_1["queries"][0]["relevant_sources_stage"] == ["/tmp/docA.pdf"]
    assert payload_8["queries"][0]["relevant_sources_stage"] == ["/tmp/docA.pdf"]


def test_quality_gate_fails_fast_when_goldset_patterns_match_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    goldset = tmp_path / "goldset.jsonl"
    goldset.write_text(
        json.dumps({"query": "q1", "relevant_patterns": ["missing-file.pdf"]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    def _collect_once(pipeline, query: str, retrieve_k: int):  # noqa: ARG001
        return [query], [_candidate("a1", "/tmp/docA.pdf")]

    class _FakePipeline:
        def __init__(self, settings):  # noqa: ARG002
            self.index = SimpleNamespace(
                indexed_chunks=[object()],
                path_to_chunk_ids={"/tmp/docA.pdf": ["a1"]},
            )

        def ask(self, **kwargs):  # noqa: ARG002
            raise AssertionError("ask should not be called when goldset patterns do not match sources")

    monkeypatch.setattr(cli_module, "_collect_candidates_once", _collect_once)
    monkeypatch.setattr("rag_system.config.load_settings", lambda dotenv_path=None: SimpleNamespace())
    monkeypatch.setattr("rag_system.pipeline.RAGPipeline", _FakePipeline)

    output_path = tmp_path / "qg_failfast.json"
    with pytest.raises(RuntimeError, match="No relevant sources matched goldset patterns"):
        cli_module._cmd_quality_gate(_quality_gate_args(str(goldset), str(output_path), final_top_k=5))


def test_quality_gate_with_ragas_includes_judge_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    goldset = tmp_path / "goldset.jsonl"
    goldset.write_text(
        json.dumps({"query": "q1", "relevant_patterns": ["docA"]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    def _collect_once(pipeline, query: str, retrieve_k: int):  # noqa: ARG001
        return [query], [
            _candidate("a1", "/tmp/docA.pdf"),
            _candidate("b1", "/tmp/docB.pdf"),
        ]

    class _FakePipeline:
        def __init__(self, settings):  # noqa: ARG002
            self.index = SimpleNamespace(
                indexed_chunks=[object()],
                path_to_chunk_ids={
                    "/tmp/docA.pdf": ["a1"],
                    "/tmp/docB.pdf": ["b1"],
                },
            )

        def ask(self, **kwargs):  # noqa: ARG002
            selected = [_candidate("a1", "/tmp/docA.pdf"), _candidate("b1", "/tmp/docB.pdf")]
            return SimpleNamespace(
                answer="ok",
                context_chunks=selected,
                trace=SimpleNamespace(grounded_refusal=False, grounded_reason=None),
            )

    captured: dict[str, str | None] = {}

    def _fake_build_adapters(settings, *, judge_provider=None, judge_model=None):  # noqa: ARG001
        captured["provider"] = judge_provider
        captured["model"] = judge_model
        return object(), object(), {"provider": "anthropic", "model": "claude-sonnet-4-20250514"}

    def _fake_eval_retrieval(rows: list[dict], k: int):  # noqa: ARG001
        return SimpleNamespace(mean_recall_at_k=1.0, mean_mrr=1.0, mean_ndcg_at_k=1.0)

    def _fake_eval_ragas(samples: list[dict], *, llm, embeddings):  # noqa: ARG001
        return {
            "faithfulness": 0.9,
            "answer_relevance": 0.8,
            "context_relevance": 0.85,
        }

    monkeypatch.setattr(cli_module, "_collect_candidates_once", _collect_once)
    monkeypatch.setattr("rag_system.config.load_settings", lambda dotenv_path=None: SimpleNamespace())
    monkeypatch.setattr("rag_system.pipeline.RAGPipeline", _FakePipeline)
    monkeypatch.setattr("rag_system.eval.evaluate_retrieval", _fake_eval_retrieval)
    monkeypatch.setattr("rag_system.eval.build_ragas_adapters", _fake_build_adapters)
    monkeypatch.setattr("rag_system.eval.evaluate_ragas", _fake_eval_ragas)

    output_path = tmp_path / "qg_judge.json"
    cli_module._cmd_quality_gate(
        _quality_gate_args(
            str(goldset),
            str(output_path),
            final_top_k=8,
            with_ragas=True,
            judge_provider="anthropic",
            judge_model="claude-sonnet-4-20250514",
        )
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert captured["provider"] == "anthropic"
    assert captured["model"] == "claude-sonnet-4-20250514"
    assert payload["ragas"] == {
        "faithfulness": 0.9,
        "answer_relevance": 0.8,
        "context_relevance": 0.85,
    }
    assert payload["ragas_judge"] == {
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
    }


def test_quality_gate_with_ragas_rejects_out_of_range_scores(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    goldset = tmp_path / "goldset.jsonl"
    goldset.write_text(
        json.dumps({"query": "q1", "relevant_patterns": ["docA"]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    def _collect_once(pipeline, query: str, retrieve_k: int):  # noqa: ARG001
        return [query], [_candidate("a1", "/tmp/docA.pdf")]

    class _FakePipeline:
        def __init__(self, settings):  # noqa: ARG002
            self.index = SimpleNamespace(
                indexed_chunks=[object()],
                path_to_chunk_ids={"/tmp/docA.pdf": ["a1"]},
            )

        def ask(self, **kwargs):  # noqa: ARG002
            return SimpleNamespace(
                answer="ok",
                context_chunks=[_candidate("a1", "/tmp/docA.pdf")],
                trace=SimpleNamespace(grounded_refusal=False, grounded_reason=None),
            )

    monkeypatch.setattr(cli_module, "_collect_candidates_once", _collect_once)
    monkeypatch.setattr("rag_system.config.load_settings", lambda dotenv_path=None: SimpleNamespace())
    monkeypatch.setattr("rag_system.pipeline.RAGPipeline", _FakePipeline)
    monkeypatch.setattr(
        "rag_system.eval.evaluate_retrieval",
        lambda rows, k: SimpleNamespace(mean_recall_at_k=1.0, mean_mrr=1.0, mean_ndcg_at_k=1.0),
    )
    monkeypatch.setattr(
        "rag_system.eval.build_ragas_adapters",
        lambda settings, **kwargs: (object(), object(), {"provider": "anthropic", "model": "claude-sonnet-4-20250514"}),
    )
    monkeypatch.setattr(
        "rag_system.eval.evaluate_ragas",
        lambda samples, **kwargs: {"faithfulness": 1.2, "answer_relevance": 0.5, "context_relevance": 0.5},
    )

    output_path = tmp_path / "qg_invalid_score.json"
    with pytest.raises(RuntimeError, match="Invalid RAGAS score"):
        cli_module._cmd_quality_gate(
            _quality_gate_args(
                str(goldset),
                str(output_path),
                final_top_k=8,
                with_ragas=True,
                judge_provider="anthropic",
                judge_model="claude-sonnet-4-20250514",
            )
        )


def test_index_output_contains_poison_fields(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    report = SimpleNamespace(
        source_path="/tmp/doc.pdf",
        extractor_name="pymupdf4llm",
        total_chunks=12,
        chars_per_page=1111.0,
        empty_page_ratio=0.0,
        short_chunk_ratio=0.1,
        has_table_elements=False,
        status="pass",
        switch_reason=None,
        page_coverage=1.0,
        fallback_path=["pymupdf4llm"],
        low_quality_pages=[],
        poisoned_pages=[2],
        poisoned_page_ratio=0.2,
        poison_signals={"max_escaped_seq_per_1k": 25.0},
        attempts=[{"extractor": "pymupdf4llm"}],
        ocr_backend_effective=None,
        ocr_fallback_path=[],
    )
    stats = SimpleNamespace(
        indexed_files=1,
        indexed_chunks=12,
        deduplicated_chunks=0,
        duplicate_files=0,
        failed_files=0,
        extraction_reports=[report],
    )

    class _FakePipeline:
        def __init__(self, settings):  # noqa: ARG002
            pass

        def index_documents(self, **kwargs):  # noqa: ARG002
            return stats

    monkeypatch.setattr("rag_system.config.load_settings", lambda dotenv_path=None: SimpleNamespace())
    monkeypatch.setattr("rag_system.pipeline.RAGPipeline", _FakePipeline)

    output_path = tmp_path / "index.json"
    args = argparse.Namespace(
        env=None,
        data_dir="/tmp",
        extractor="pymupdf4llm",
        fast=True,
        reset_index=True,
        profile="demo-fast",
        output=str(output_path),
    )
    cli_module._cmd_index(args)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    row = payload["reports"][0]
    assert row["poisoned_pages"] == [2]
    assert row["poisoned_page_ratio"] == 0.2
    assert row["poison_signals"]["max_escaped_seq_per_1k"] == 25.0


def test_pdf_regression_output_contains_poison_fields(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    class _FakeOrchestrator:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

        def extract_with_policy(self, **kwargs):  # noqa: ARG002
            stats = SimpleNamespace(
                total_chunks=10,
                chars_per_page=900.0,
                empty_page_ratio=0.0,
                short_chunk_ratio=0.1,
                has_table_elements=False,
                page_coverage=1.0,
                ocr_backend_effective=None,
                ocr_fallback_path=[],
                poisoned_pages=[1],
                poisoned_page_ratio=1.0,
                poison_signals={"max_backslash_per_1k": 55.0},
            )
            return SimpleNamespace(
                status="soft_fail",
                extractor_used="pymupdf_window_fallback",
                fallback_path=["pymupdf4llm", "pymupdf_window_fallback"],
                switch_reason="poisoned_text_detected:1/1",
                low_quality_pages=[1],
                attempts=[
                    {
                        "extractor": "pymupdf4llm",
                        "switch_reason": "poisoned_text_detected:1/1",
                        "remaining_budget_sec_before_attempt": 120.0,
                        "effective_timeout_sec": 90,
                    }
                ],
                notes=["poisoned"],
                stats=stats,
            )

    settings = SimpleNamespace(
        extract_timeout_sec=45,
        extract_min_chars_per_page=450.0,
        extract_max_empty_page_ratio=0.25,
        extract_max_short_chunk_ratio=0.55,
        extract_max_escaped_seq_per_1k=18.0,
        extract_max_backslash_per_1k=30.0,
        extract_max_control_char_ratio=0.003,
        extract_poisoned_page_ratio_hard=0.35,
        chunk_mode="token",
        chunk_tokens_prose=360,
        chunk_tokens_table=220,
        chunk_overlap_prose=60,
        chunk_overlap_table=30,
        chunk_chars_prose=1600,
        chunk_chars_table=1500,
        chunk_overlap_chars_prose=160,
        chunk_overlap_chars_table=120,
        extract_timeout_base_sec=45,
        extract_timeout_per_100_pages_sec=30,
        extract_timeout_per_10mb_sec=20,
        extract_timeout_max_sec=600,
        page_window_size=40,
        extract_min_page_coverage=0.85,
        unstructured_targeted_only=True,
        docling_ocr_backend="easyocr",
        docling_ocr_fallbacks=("easyocr", "tesseract", "rapidocr", "none"),
        docling_ocr_langs_easyocr=("ru", "en"),
        docling_ocr_langs_tesseract=("rus", "eng"),
    )
    monkeypatch.setattr("rag_system.config.load_settings", lambda dotenv_path=None: settings)
    monkeypatch.setattr("rag_system.extractors.factory.ExtractorOrchestrator", _FakeOrchestrator)
    monkeypatch.setattr(cli_module, "_discover_pdf_files", lambda data_dir: ["/tmp/doc.pdf"])

    output_path = tmp_path / "pdf_regression.json"
    log_path = tmp_path / "pdf_regression.log"
    args = argparse.Namespace(
        env=None,
        data_dir="/tmp",
        extractor="docling",
        fast=False,
        timeout_sec=45,
        wall_clock_cap_sec=None,
        profile="full-quality",
        output=str(output_path),
        log_file=str(log_path),
    )
    cli_module._cmd_pdf_regression(args)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    row = payload["rows"][0]
    assert row["ok"] is True
    assert row["poisoned_pages"] == [1]
    assert row["poisoned_page_ratio"] == 1.0
    assert row["poison_signals"]["max_backslash_per_1k"] == 55.0
    assert row["attempts"][0]["remaining_budget_sec_before_attempt"] == 120.0
    assert row["attempts"][0]["effective_timeout_sec"] == 90
    assert payload["ok_files"] == 1
    assert payload["failed_files"] == 0


def test_pdf_regression_counts_only_hard_fail_as_failed(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    class _FakeOrchestrator:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

        def extract_with_policy(self, **kwargs):  # noqa: ARG002
            stats = SimpleNamespace(
                total_chunks=0,
                chars_per_page=0.0,
                empty_page_ratio=1.0,
                short_chunk_ratio=1.0,
                has_table_elements=False,
                page_coverage=0.0,
                ocr_backend_effective=None,
                ocr_fallback_path=[],
                poisoned_pages=[],
                poisoned_page_ratio=0.0,
                poison_signals={},
            )
            return SimpleNamespace(
                status="hard_fail",
                extractor_used="n/a",
                fallback_path=[],
                switch_reason="timeout:demo",
                low_quality_pages=[],
                attempts=[],
                notes=[],
                stats=stats,
            )

    settings = SimpleNamespace(
        extract_timeout_sec=45,
        extract_min_chars_per_page=450.0,
        extract_max_empty_page_ratio=0.25,
        extract_max_short_chunk_ratio=0.55,
        extract_max_escaped_seq_per_1k=18.0,
        extract_max_backslash_per_1k=30.0,
        extract_max_control_char_ratio=0.003,
        extract_poisoned_page_ratio_hard=0.35,
        chunk_mode="token",
        chunk_tokens_prose=360,
        chunk_tokens_table=220,
        chunk_overlap_prose=60,
        chunk_overlap_table=30,
        chunk_chars_prose=1600,
        chunk_chars_table=1500,
        chunk_overlap_chars_prose=160,
        chunk_overlap_chars_table=120,
        extract_timeout_base_sec=45,
        extract_timeout_per_100_pages_sec=30,
        extract_timeout_per_10mb_sec=20,
        extract_timeout_max_sec=600,
        page_window_size=40,
        extract_min_page_coverage=0.85,
        unstructured_targeted_only=True,
        docling_ocr_backend="easyocr",
        docling_ocr_fallbacks=("easyocr", "tesseract", "rapidocr", "none"),
        docling_ocr_langs_easyocr=("ru", "en"),
        docling_ocr_langs_tesseract=("rus", "eng"),
    )
    monkeypatch.setattr("rag_system.config.load_settings", lambda dotenv_path=None: settings)
    monkeypatch.setattr("rag_system.extractors.factory.ExtractorOrchestrator", _FakeOrchestrator)
    monkeypatch.setattr(cli_module, "_discover_pdf_files", lambda data_dir: ["/tmp/doc.pdf"])

    output_path = tmp_path / "pdf_regression_fail.json"
    args = argparse.Namespace(
        env=None,
        data_dir="/tmp",
        extractor="docling",
        fast=False,
        timeout_sec=45,
        wall_clock_cap_sec=None,
        profile="full-quality",
        output=str(output_path),
        log_file=None,
    )
    cli_module._cmd_pdf_regression(args)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    row = payload["rows"][0]
    assert row["ok"] is False
    assert payload["ok_files"] == 0
    assert payload["failed_files"] == 1


def test_pdf_regression_wall_clock_cap_returns_partial_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class _FakeOrchestrator:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

        def extract_with_policy(self, **kwargs):  # noqa: ARG002
            stats = SimpleNamespace(
                total_chunks=5,
                chars_per_page=600.0,
                empty_page_ratio=0.0,
                short_chunk_ratio=0.1,
                has_table_elements=False,
                page_coverage=1.0,
                ocr_backend_effective=None,
                ocr_fallback_path=[],
                poisoned_pages=[],
                poisoned_page_ratio=0.0,
                poison_signals={},
            )
            return SimpleNamespace(
                status="pass",
                extractor_used="pymupdf4llm",
                fallback_path=["pymupdf4llm"],
                switch_reason=None,
                low_quality_pages=[],
                attempts=[],
                notes=[],
                stats=stats,
            )

    settings = SimpleNamespace(
        extract_timeout_sec=45,
        extract_min_chars_per_page=450.0,
        extract_max_empty_page_ratio=0.25,
        extract_max_short_chunk_ratio=0.55,
        extract_max_escaped_seq_per_1k=18.0,
        extract_max_backslash_per_1k=30.0,
        extract_max_control_char_ratio=0.003,
        extract_poisoned_page_ratio_hard=0.35,
        chunk_mode="token",
        chunk_tokens_prose=360,
        chunk_tokens_table=220,
        chunk_overlap_prose=60,
        chunk_overlap_table=30,
        chunk_chars_prose=1600,
        chunk_chars_table=1500,
        chunk_overlap_chars_prose=160,
        chunk_overlap_chars_table=120,
        extract_timeout_base_sec=45,
        extract_timeout_per_100_pages_sec=30,
        extract_timeout_per_10mb_sec=20,
        extract_timeout_max_sec=600,
        page_window_size=40,
        extract_min_page_coverage=0.85,
        unstructured_targeted_only=True,
        docling_ocr_backend="easyocr",
        docling_ocr_fallbacks=("easyocr", "tesseract", "rapidocr", "none"),
        docling_ocr_langs_easyocr=("ru", "en"),
        docling_ocr_langs_tesseract=("rus", "eng"),
        pdf_regression_wallclock_cap_sec=3,
    )
    monkeypatch.setattr("rag_system.config.load_settings", lambda dotenv_path=None: settings)
    monkeypatch.setattr("rag_system.extractors.factory.ExtractorOrchestrator", _FakeOrchestrator)
    monkeypatch.setattr(
        cli_module,
        "_discover_pdf_files",
        lambda data_dir: ["/tmp/doc1.pdf", "/tmp/doc2.pdf", "/tmp/doc3.pdf"],
    )

    ticks = {"t": 0.0}

    def _fake_perf_counter() -> float:
        ticks["t"] += 1.0
        return ticks["t"]

    monkeypatch.setattr(cli_module.time, "perf_counter", _fake_perf_counter)

    output_path = tmp_path / "pdf_regression_partial.json"
    args = argparse.Namespace(
        env=None,
        data_dir="/tmp",
        extractor="docling",
        fast=False,
        timeout_sec=45,
        wall_clock_cap_sec=None,
        profile="full-quality",
        output=str(output_path),
        log_file=None,
    )
    cli_module._cmd_pdf_regression(args)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["aborted_after_wall_clock"] is True
    assert payload["total_files"] == 3
    assert payload["processed_files"] == 1
    assert payload["remaining_files"] == 2
    assert payload["ok_files"] == 1
    assert payload["failed_files"] == 0


def test_pdf_regression_in_file_wall_clock_timeout_writes_hard_fail_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class _FakeOrchestrator:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

        def extract_with_policy(self, **kwargs):  # noqa: ARG002
            raise TimeoutError("wall_clock_deadline_exceeded_during_file")

    settings = SimpleNamespace(
        extract_timeout_sec=45,
        extract_min_chars_per_page=450.0,
        extract_max_empty_page_ratio=0.25,
        extract_max_short_chunk_ratio=0.55,
        extract_max_escaped_seq_per_1k=18.0,
        extract_max_backslash_per_1k=30.0,
        extract_max_control_char_ratio=0.003,
        extract_poisoned_page_ratio_hard=0.35,
        chunk_mode="token",
        chunk_tokens_prose=360,
        chunk_tokens_table=220,
        chunk_overlap_prose=60,
        chunk_overlap_table=30,
        chunk_chars_prose=1600,
        chunk_chars_table=1500,
        chunk_overlap_chars_prose=160,
        chunk_overlap_chars_table=120,
        extract_timeout_base_sec=45,
        extract_timeout_per_100_pages_sec=30,
        extract_timeout_per_10mb_sec=20,
        extract_timeout_max_sec=600,
        page_window_size=40,
        extract_min_page_coverage=0.85,
        unstructured_targeted_only=True,
        docling_ocr_backend="easyocr",
        docling_ocr_fallbacks=("easyocr", "tesseract", "rapidocr", "none"),
        docling_ocr_langs_easyocr=("ru", "en"),
        docling_ocr_langs_tesseract=("rus", "eng"),
        pdf_regression_wallclock_cap_sec=900,
    )
    monkeypatch.setattr("rag_system.config.load_settings", lambda dotenv_path=None: settings)
    monkeypatch.setattr("rag_system.extractors.factory.ExtractorOrchestrator", _FakeOrchestrator)
    monkeypatch.setattr(cli_module, "_discover_pdf_files", lambda data_dir: ["/tmp/doc_timeout.pdf"])

    output_path = tmp_path / "pdf_regression_timeout.json"
    args = argparse.Namespace(
        env=None,
        data_dir="/tmp",
        extractor="docling",
        fast=False,
        timeout_sec=45,
        wall_clock_cap_sec=1,
        profile="full-quality",
        output=str(output_path),
        log_file=None,
    )
    cli_module._cmd_pdf_regression(args)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["aborted_after_wall_clock"] is True
    assert payload["processed_files"] == 1
    assert payload["remaining_files"] == 0
    assert payload["ok_files"] == 0
    assert payload["failed_files"] == 1
    row = payload["rows"][0]
    assert row["ok"] is False
    assert row["status"] == "hard_fail"
    assert str(row["switch_reason"]).startswith("timeout:")
    assert "wall_clock_deadline" in str(row["switch_reason"])
