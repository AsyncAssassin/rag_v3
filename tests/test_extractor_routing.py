"""Tests for extractor routing contract and chunking policy helpers."""

from __future__ import annotations

import time

import pytest

from rag_system.extractors.base import ExtractionQualityThresholds, compute_extraction_stats
from rag_system.extractors.factory import ExtractorOrchestrator
from rag_system.types import DocumentChunk


def _chunk(text: str, page: int, element_type: str = "text") -> DocumentChunk:
    return DocumentChunk(
        text=text,
        source_path="/tmp/doc.pdf",
        page=page,
        element_type=element_type,
        metadata={},
    )


def _attempt_payload(
    *,
    extractor: str,
    status: str,
    switch_reason: str | None,
    chunks: list[DocumentChunk],
    page_coverage: float,
    chars_per_page: float = 0.0,
    short_chunk_ratio: float = 0.0,
) -> dict:
    return {
        "extractor": extractor,
        "status": status,
        "switch_reason": switch_reason,
        "duration_ms": 10.0,
        "error": None,
        "total_chunks": len(chunks),
        "chars_per_page": chars_per_page,
        "empty_page_ratio": 0.0 if chunks else 1.0,
        "short_chunk_ratio": short_chunk_ratio,
        "page_coverage": page_coverage,
        "low_quality_pages": [],
        "poisoned_pages": [],
        "poisoned_page_ratio": 0.0,
        "poison_signals": {},
        "ocr_backend_effective": None,
        "ocr_fallback_path": [],
        "remaining_budget_sec_before_attempt": 900.0,
        "effective_timeout_sec": 120,
        "chunks": chunks,
        "stats": None,
    }


def test_docling_backend_order_prefers_configured_backend() -> None:
    orchestrator = ExtractorOrchestrator(
        docling_ocr_backend="rapidocr",
        docling_ocr_fallbacks=("easyocr", "none"),
    )
    assert orchestrator._docling_backend_order()[:3] == ["rapidocr", "easyocr", "none"]


def test_routing_hard_fail_for_low_page_coverage() -> None:
    orchestrator = ExtractorOrchestrator(min_page_coverage=0.85)
    chunks = [_chunk("очень длинный текст " * 120, page=1)]
    stats = compute_extraction_stats(chunks, "/tmp/doc.pdf", "pymupdf4llm", total_pages=3)

    status, reason, low_pages, page_coverage, *_ = orchestrator._evaluate_routing(
        stats,
        chunks,
        page_scope=[1, 2, 3],
    )

    assert status == "hard_fail"
    assert reason is not None and "page_coverage" in reason
    assert page_coverage < 0.85
    assert set(low_pages).issuperset({2, 3})


def test_routing_soft_fail_for_low_chars_without_hard_fail() -> None:
    orchestrator = ExtractorOrchestrator(min_page_coverage=0.5)
    chunks = [_chunk("коротко", page=1), _chunk("тоже коротко", page=2)]
    stats = compute_extraction_stats(chunks, "/tmp/doc.pdf", "docling_easyocr", total_pages=2)

    status, reason, _, page_coverage, *_ = orchestrator._evaluate_routing(
        stats,
        chunks,
        page_scope=[1, 2],
    )

    assert page_coverage == 1.0
    assert status == "soft_fail"
    assert reason is not None and "chars_per_page" in reason


def test_effective_timeout_uses_hard_cap_when_adaptive_is_higher() -> None:
    orchestrator = ExtractorOrchestrator(
        extract_timeout_sec=45,
        extract_timeout_base_sec=45,
    )
    assert orchestrator._resolve_timeout_sec(120) == 45
    assert orchestrator._resolve_timeout_sec(30) == 30


class _SleepExtractor:
    name = "sleep_extractor"

    def __init__(self, sleep_sec: float) -> None:
        self.sleep_sec = sleep_sec

    def extract(self, file_path: str):  # noqa: ARG002
        time.sleep(self.sleep_sec)
        return []


class _LargePayloadExtractor:
    name = "large_payload_extractor"

    def __init__(self, count: int = 2000, text_size: int = 256) -> None:
        self.count = int(count)
        self.text = "x" * max(64, int(text_size))

    def extract(self, file_path: str):  # noqa: ARG002
        return [
            DocumentChunk(
                text=self.text,
                source_path="/tmp/doc.pdf",
                page=(idx % 5) + 1,
                element_type="text",
                metadata={},
            )
            for idx in range(self.count)
        ]


def test_extract_single_times_out_and_stops_hanging_extractor() -> None:
    orchestrator = ExtractorOrchestrator(
        extract_timeout_sec=1,
        extract_timeout_base_sec=120,
    )
    started = time.perf_counter()
    with pytest.raises(TimeoutError):
        orchestrator._extract_single(
            extractor=_SleepExtractor(4.0),
            file_path="/tmp/doc.pdf",
            timeout_override_sec=30,
        )
    elapsed = time.perf_counter() - started
    assert elapsed < 3.5


def test_extract_single_large_payload_does_not_timeout_with_file_transport() -> None:
    orchestrator = ExtractorOrchestrator(
        extract_timeout_sec=3,
        extract_timeout_base_sec=3,
    )
    chunks = orchestrator._extract_single(
        extractor=_LargePayloadExtractor(count=2000, text_size=256),
        file_path="/tmp/doc.pdf",
        timeout_override_sec=3,
    )
    assert len(chunks) == 2000


def test_extract_single_timeout_invokes_subprocess_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = ExtractorOrchestrator(
        extract_timeout_sec=1,
        extract_timeout_base_sec=120,
    )
    called = {"cleanup": 0}
    original = orchestrator._stop_subprocess_tree

    def _spy_stop(proc) -> None:  # noqa: ANN001
        called["cleanup"] += 1
        original(proc)

    monkeypatch.setattr(orchestrator, "_stop_subprocess_tree", _spy_stop)

    with pytest.raises(TimeoutError):
        orchestrator._extract_single(
            extractor=_SleepExtractor(4.0),
            file_path="/tmp/doc.pdf",
            timeout_override_sec=30,
        )
    assert called["cleanup"] == 1


def test_attempt_extractor_marks_timeout_reason() -> None:
    orchestrator = ExtractorOrchestrator(
        extract_timeout_sec=1,
        extract_timeout_base_sec=120,
    )
    attempt = orchestrator._attempt_extractor(
        extractor=_SleepExtractor(4.0),
        file_path="/tmp/doc.pdf",
        total_pages=1,
        page_scope=[1],
        timeout_override_sec=30,
        extractor_label="sleep_extractor",
    )
    assert attempt["status"] == "hard_fail"
    assert str(attempt["switch_reason"]).startswith("timeout:")


def test_attempt_extractor_deadline_expired_before_start() -> None:
    orchestrator = ExtractorOrchestrator(
        extract_timeout_sec=45,
        extract_timeout_base_sec=120,
    )
    attempt = orchestrator._attempt_extractor(
        extractor=_SleepExtractor(0.1),
        file_path="/tmp/doc.pdf",
        total_pages=1,
        page_scope=[1],
        timeout_override_sec=30,
        extractor_label="sleep_extractor",
        deadline_monotonic=time.perf_counter() - 0.01,
    )
    assert attempt["status"] == "hard_fail"
    assert str(attempt["switch_reason"]).startswith("timeout:")
    assert "before_attempt" in str(attempt["switch_reason"])


def test_window_fallback_marks_timeout_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = ExtractorOrchestrator(
        extract_timeout_sec=1,
        extract_timeout_base_sec=120,
    )

    def _raise_timeout(self, **kwargs):  # noqa: ARG001
        raise TimeoutError("window deadline")

    monkeypatch.setattr(ExtractorOrchestrator, "_extract_single", _raise_timeout)

    attempt = orchestrator._attempt_window_fallback(
        file_path="/tmp/doc.pdf",
        total_pages=1,
        page_scope=[1],
        extractor_label="pymupdf_window_fallback",
        timeout_override_sec=30,
    )
    assert attempt["status"] == "hard_fail"
    assert str(attempt["switch_reason"]).startswith("timeout:")


def test_routing_soft_fail_for_poisoned_text() -> None:
    thresholds = ExtractionQualityThresholds(
        min_chars_per_page=1.0,
        max_short_chunk_ratio=1.0,
        max_empty_page_ratio=1.0,
        max_escaped_seq_per_1k=5.0,
        max_backslash_per_1k=15.0,
        max_control_char_ratio=0.5,
        poisoned_page_ratio_hard=0.9,
    )
    orchestrator = ExtractorOrchestrator(min_page_coverage=0.5, quality_thresholds=thresholds)
    chunks = [
        _chunk(("normal text " * 30).strip(), page=1),
        _chunk("\\n\\r\\x0A\\u000A " * 80, page=2),
    ]
    stats = compute_extraction_stats(chunks, "/tmp/doc.pdf", "pymupdf4llm", total_pages=2)
    status, reason, low_pages, _, poisoned_pages, poisoned_ratio, _ = orchestrator._evaluate_routing(
        stats,
        chunks,
        page_scope=[1, 2],
    )
    assert status == "soft_fail"
    assert reason is not None and "poisoned_text_detected" in reason
    assert set(poisoned_pages) == {2}
    assert set(low_pages).issuperset({2})
    assert 0.0 < poisoned_ratio < 0.9


def test_routing_hard_fail_for_high_poison_ratio() -> None:
    thresholds = ExtractionQualityThresholds(
        min_chars_per_page=1.0,
        max_short_chunk_ratio=1.0,
        max_empty_page_ratio=1.0,
        max_escaped_seq_per_1k=5.0,
        max_backslash_per_1k=15.0,
        max_control_char_ratio=0.5,
        poisoned_page_ratio_hard=0.35,
    )
    orchestrator = ExtractorOrchestrator(min_page_coverage=0.5, quality_thresholds=thresholds)
    chunks = [
        _chunk("\\n\\r\\x0A\\u000A " * 80, page=1),
        _chunk("\\n\\r\\x0A\\u000A " * 90, page=2),
        _chunk("normal text " * 20, page=3),
    ]
    stats = compute_extraction_stats(chunks, "/tmp/doc.pdf", "pymupdf4llm", total_pages=3)
    status, reason, _, _, poisoned_pages, poisoned_ratio, _ = orchestrator._evaluate_routing(
        stats,
        chunks,
        page_scope=[1, 2, 3],
    )
    assert status == "hard_fail"
    assert reason is not None and "poisoned_text_ratio" in reason
    assert set(poisoned_pages) == {1, 2}
    assert poisoned_ratio >= 0.35


def test_full_quality_uses_soft_window_fallback_result(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = ExtractorOrchestrator(unstructured_targeted_only=True)
    failed_attempts = iter(
        [
            {"extractor": "docling_easyocr", "status": "hard_fail", "switch_reason": "timeout:docling", "chunks": []},
            {"extractor": "docling_rapidocr", "status": "hard_fail", "switch_reason": "timeout:docling", "chunks": []},
            {"extractor": "pymupdf4llm", "status": "hard_fail", "switch_reason": "timeout:pymupdf", "chunks": []},
        ]
    )

    def _fake_attempt(self, **kwargs):  # noqa: ARG001
        return next(failed_attempts)

    def _fake_window(self, **kwargs):  # noqa: ARG001
        chunks = [_chunk("достаточно длинный текст " * 40, page=1)]
        return {
            "extractor": "pymupdf_window_fallback",
            "status": "soft_fail",
            "switch_reason": "poisoned_text_detected:1/1",
            "duration_ms": 15.0,
            "error": None,
            "total_chunks": 1,
            "chars_per_page": 1000.0,
            "empty_page_ratio": 0.0,
            "short_chunk_ratio": 0.0,
            "page_coverage": 1.0,
            "low_quality_pages": [1],
            "poisoned_pages": [1],
            "poisoned_page_ratio": 1.0,
            "poison_signals": {"max_escaped_seq_per_1k": 50.0},
            "ocr_backend_effective": None,
            "ocr_fallback_path": [],
            "chunks": chunks,
            "stats": None,
        }

    monkeypatch.setattr(ExtractorOrchestrator, "_attempt_extractor", _fake_attempt)
    monkeypatch.setattr(ExtractorOrchestrator, "_attempt_window_fallback", _fake_window)

    out = orchestrator._run_full_quality(
        file_path="/tmp/doc.pdf",
        total_pages=1,
        page_scope=[1],
        notes=[],
        profile_name="full-quality",
    )
    assert out.status == "soft_fail"
    assert out.extractor_used == "pymupdf_window_fallback"
    assert out.chunks
    assert out.switch_reason is not None and "poisoned_text_detected" in out.switch_reason


def test_full_quality_prefers_best_usable_over_window_when_coverage_gap_high(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = ExtractorOrchestrator(
        unstructured_targeted_only=True,
        extract_prefer_best_usable=True,
        extract_best_usable_min_coverage_gap=0.15,
    )
    first_docling_chunks = [_chunk("качественный docling текст " * 45, page=1)]
    attempts = iter(
        [
            _attempt_payload(
                extractor="docling_easyocr",
                status="soft_fail",
                switch_reason="chars_per_page<450.0",
                chunks=first_docling_chunks,
                page_coverage=0.92,
                chars_per_page=920.0,
                short_chunk_ratio=0.08,
            ),
            _attempt_payload(
                extractor="docling_rapidocr",
                status="hard_fail",
                switch_reason="timeout:docling",
                chunks=[],
                page_coverage=0.0,
                chars_per_page=0.0,
                short_chunk_ratio=1.0,
            ),
            _attempt_payload(
                extractor="pymupdf4llm",
                status="hard_fail",
                switch_reason="timeout:pymupdf",
                chunks=[],
                page_coverage=0.0,
                chars_per_page=0.0,
                short_chunk_ratio=1.0,
            ),
        ]
    )

    def _fake_attempt(self, **kwargs):  # noqa: ARG001
        return next(attempts)

    def _fake_window(self, **kwargs):  # noqa: ARG001
        window_chunks = [_chunk("fallback window text " * 35, page=1)]
        return _attempt_payload(
            extractor="pymupdf_window_fallback",
            status="soft_fail",
            switch_reason="chars_per_page<450.0",
            chunks=window_chunks,
            page_coverage=0.62,
            chars_per_page=600.0,
            short_chunk_ratio=0.04,
        )

    monkeypatch.setattr(ExtractorOrchestrator, "_attempt_extractor", _fake_attempt)
    monkeypatch.setattr(ExtractorOrchestrator, "_attempt_window_fallback", _fake_window)

    out = orchestrator._run_full_quality(
        file_path="/tmp/doc.pdf",
        total_pages=2,
        page_scope=[1, 2],
        notes=[],
        profile_name="full-quality",
    )
    assert out.status == "soft_fail"
    assert out.extractor_used == "docling_easyocr"
    assert "chars_per_page" in str(out.switch_reason)
    assert any("best usable extraction result" in str(note) for note in out.notes)


def test_normalize_unstructured_targeted_merges_short_fragments() -> None:
    orchestrator = ExtractorOrchestrator(unstructured_min_merged_chunk_chars=220)
    chunks = [
        _chunk("краткий фрагмент " * 4, page=2),
        _chunk("еще кусок " * 5, page=2),
        _chunk("добавочный блок " * 5, page=2),
        _chunk("краткий фрагмент " * 4, page=2),  # duplicate
        _chunk("длинный самостоятельный текст " * 25, page=2),
    ]

    normalized = orchestrator._normalize_unstructured_targeted_chunks(chunks)
    assert len(normalized) == 2
    assert all(chunk.page == 2 for chunk in normalized)
    merged = normalized[0]
    assert len(merged.text) >= 180
    assert "краткий фрагмент" in merged.text
    assert "добавочный блок" in merged.text


def test_full_quality_downgrades_window_low_coverage_hard_fail_to_soft_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = ExtractorOrchestrator(unstructured_targeted_only=True)
    failed_attempts = iter(
        [
            {"extractor": "docling_easyocr", "status": "hard_fail", "switch_reason": "timeout:docling", "chunks": []},
            {"extractor": "docling_rapidocr", "status": "hard_fail", "switch_reason": "timeout:docling", "chunks": []},
            {"extractor": "pymupdf4llm", "status": "hard_fail", "switch_reason": "timeout:pymupdf", "chunks": []},
        ]
    )

    def _fake_attempt(self, **kwargs):  # noqa: ARG001
        return next(failed_attempts)

    def _fake_window(self, **kwargs):  # noqa: ARG001
        chunks = [_chunk("достаточно длинный текст " * 40, page=1)]
        return {
            "extractor": "pymupdf_window_fallback",
            "status": "hard_fail",
            "switch_reason": "page_coverage<0.2136<0.85",
            "duration_ms": 15.0,
            "error": None,
            "total_chunks": 1,
            "chars_per_page": 1000.0,
            "empty_page_ratio": 0.0,
            "short_chunk_ratio": 0.0,
            "page_coverage": 0.2136,
            "low_quality_pages": [2, 3],
            "poisoned_pages": [],
            "poisoned_page_ratio": 0.0,
            "poison_signals": {},
            "ocr_backend_effective": None,
            "ocr_fallback_path": [],
            "chunks": chunks,
            "stats": None,
        }

    monkeypatch.setattr(ExtractorOrchestrator, "_attempt_extractor", _fake_attempt)
    monkeypatch.setattr(ExtractorOrchestrator, "_attempt_window_fallback", _fake_window)

    out = orchestrator._run_full_quality(
        file_path="/tmp/doc.pdf",
        total_pages=3,
        page_scope=[1, 2, 3],
        notes=[],
        profile_name="full-quality",
    )
    assert out.status == "soft_fail"
    assert out.extractor_used == "pymupdf_window_fallback"
    assert out.attempts[-1]["downgraded_to_soft_fail_due_to_low_coverage"] is True
    assert out.attempts[-1]["ipc_transport"] == "file"
    assert any("pragmatic soft-fail fallback result" in str(note) for note in out.notes)


def test_full_quality_keeps_hard_fail_when_low_coverage_includes_poison_hard_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = ExtractorOrchestrator(unstructured_targeted_only=True)
    failed_attempts = iter(
        [
            {"extractor": "docling_easyocr", "status": "hard_fail", "switch_reason": "timeout:docling", "chunks": []},
            {"extractor": "docling_rapidocr", "status": "hard_fail", "switch_reason": "timeout:docling", "chunks": []},
            {"extractor": "pymupdf4llm", "status": "hard_fail", "switch_reason": "timeout:pymupdf", "chunks": []},
        ]
    )

    def _fake_attempt(self, **kwargs):  # noqa: ARG001
        return next(failed_attempts)

    def _fake_window(self, **kwargs):  # noqa: ARG001
        chunks = [_chunk("достаточно длинный текст " * 40, page=1)]
        return {
            "extractor": "pymupdf_window_fallback",
            "status": "hard_fail",
            "switch_reason": "page_coverage<0.2136<0.85; poisoned_text_ratio:0.4>=thr:0.35",
            "duration_ms": 15.0,
            "error": None,
            "total_chunks": 1,
            "chars_per_page": 1000.0,
            "empty_page_ratio": 0.0,
            "short_chunk_ratio": 0.0,
            "page_coverage": 0.2136,
            "low_quality_pages": [2, 3],
            "poisoned_pages": [2],
            "poisoned_page_ratio": 0.4,
            "poison_signals": {"max_escaped_seq_per_1k": 50.0},
            "ocr_backend_effective": None,
            "ocr_fallback_path": [],
            "chunks": chunks,
            "stats": None,
        }

    monkeypatch.setattr(ExtractorOrchestrator, "_attempt_extractor", _fake_attempt)
    monkeypatch.setattr(ExtractorOrchestrator, "_attempt_window_fallback", _fake_window)

    out = orchestrator._run_full_quality(
        file_path="/tmp/doc.pdf",
        total_pages=3,
        page_scope=[1, 2, 3],
        notes=[],
        profile_name="full-quality",
    )
    assert out.status == "hard_fail"
    assert out.extractor_used == "pymupdf_window_fallback"
    assert out.attempts[-1]["downgraded_to_soft_fail_due_to_low_coverage"] is False


def test_full_quality_skips_secondary_docling_after_timeout_without_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = ExtractorOrchestrator(
        extract_full_quality_reserve_window_fallback_sec=180,
    )
    calls: list[str] = []

    def _fake_attempt(self, **kwargs):  # noqa: ARG001
        label = kwargs["extractor_label"]
        calls.append(label)
        if label == "docling_easyocr":
            return {
                "extractor": label,
                "status": "hard_fail",
                "switch_reason": "timeout:Extractor 'docling' timed out after 300s",
                "duration_ms": 300000.0,
                "error": "timeout",
                "total_chunks": 0,
                "chars_per_page": 0.0,
                "empty_page_ratio": 1.0,
                "short_chunk_ratio": 1.0,
                "page_coverage": 0.0,
                "low_quality_pages": [1],
                "poisoned_pages": [],
                "poisoned_page_ratio": 0.0,
                "poison_signals": {},
                "ocr_backend_effective": None,
                "ocr_fallback_path": [],
                "remaining_budget_sec_before_attempt": 900.0,
                "effective_timeout_sec": 300,
                "chunks": [],
                "stats": None,
            }
        if label == "docling_rapidocr":
            pytest.fail("secondary docling should be skipped after primary timeout+0chunks")
        if label == "pymupdf4llm":
            return {
                "extractor": label,
                "status": "hard_fail",
                "switch_reason": "timeout:Extractor 'pymupdf4llm' timed out after 180s",
                "duration_ms": 180000.0,
                "error": "timeout",
                "total_chunks": 0,
                "chars_per_page": 0.0,
                "empty_page_ratio": 1.0,
                "short_chunk_ratio": 1.0,
                "page_coverage": 0.0,
                "low_quality_pages": [1],
                "poisoned_pages": [],
                "poisoned_page_ratio": 0.0,
                "poison_signals": {},
                "ocr_backend_effective": None,
                "ocr_fallback_path": [],
                "remaining_budget_sec_before_attempt": 600.0,
                "effective_timeout_sec": 180,
                "chunks": [],
                "stats": None,
            }
        pytest.fail(f"Unexpected extractor label: {label}")

    def _fake_window(self, **kwargs):  # noqa: ARG001
        chunks = [_chunk("достаточно длинный текст " * 40, page=1)]
        return {
            "extractor": "pymupdf_window_fallback",
            "status": "soft_fail",
            "switch_reason": "chars_per_page<450.0",
            "duration_ms": 20.0,
            "error": None,
            "total_chunks": 1,
            "chars_per_page": 1200.0,
            "empty_page_ratio": 0.0,
            "short_chunk_ratio": 0.0,
            "page_coverage": 1.0,
            "low_quality_pages": [1],
            "poisoned_pages": [],
            "poisoned_page_ratio": 0.0,
            "poison_signals": {},
            "ocr_backend_effective": None,
            "ocr_fallback_path": [],
            "remaining_budget_sec_before_attempt": 180.0,
            "effective_timeout_sec": 180,
            "chunks": chunks,
            "stats": None,
        }

    monkeypatch.setattr(ExtractorOrchestrator, "_attempt_extractor", _fake_attempt)
    monkeypatch.setattr(ExtractorOrchestrator, "_attempt_window_fallback", _fake_window)

    out = orchestrator._run_full_quality(
        file_path="/tmp/doc.pdf",
        total_pages=1,
        page_scope=[1],
        notes=[],
        profile_name="full-quality",
    )
    assert "docling_easyocr" in calls
    assert "docling_rapidocr" not in calls
    assert "pymupdf4llm" in calls
    assert out.status == "soft_fail"


def test_full_quality_passes_reserve_budget_and_starts_window_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = ExtractorOrchestrator(
        extract_full_quality_reserve_window_fallback_sec=180,
    )
    extractor_reserves: list[int] = []
    window_called = {"value": False}

    def _fake_attempt(self, **kwargs):  # noqa: ARG001
        extractor_reserves.append(int(kwargs.get("reserved_tail_budget_sec") or 0))
        return {
            "extractor": kwargs["extractor_label"],
            "status": "hard_fail",
            "switch_reason": f"timeout:wall_clock_deadline_exceeded_before_attempt:{kwargs['extractor_label']}",
            "duration_ms": 0.0,
            "error": "budget exhausted",
            "total_chunks": 0,
            "chars_per_page": 0.0,
            "empty_page_ratio": 1.0,
            "short_chunk_ratio": 1.0,
            "page_coverage": 0.0,
            "low_quality_pages": [1],
            "poisoned_pages": [],
            "poisoned_page_ratio": 0.0,
            "poison_signals": {},
            "ocr_backend_effective": None,
            "ocr_fallback_path": [],
            "remaining_budget_sec_before_attempt": 10.0,
            "effective_timeout_sec": 0,
            "chunks": [],
            "stats": None,
        }

    def _fake_window(self, **kwargs):  # noqa: ARG001
        window_called["value"] = True
        assert int(kwargs.get("reserved_tail_budget_sec") or 0) == 0
        chunks = [_chunk("текст " * 120, page=1)]
        return {
            "extractor": "pymupdf_window_fallback",
            "status": "soft_fail",
            "switch_reason": "chars_per_page<450.0",
            "duration_ms": 15.0,
            "error": None,
            "total_chunks": 1,
            "chars_per_page": 500.0,
            "empty_page_ratio": 0.0,
            "short_chunk_ratio": 0.0,
            "page_coverage": 1.0,
            "low_quality_pages": [],
            "poisoned_pages": [],
            "poisoned_page_ratio": 0.0,
            "poison_signals": {},
            "ocr_backend_effective": None,
            "ocr_fallback_path": [],
            "remaining_budget_sec_before_attempt": 10.0,
            "effective_timeout_sec": 10,
            "chunks": chunks,
            "stats": None,
        }

    monkeypatch.setattr(ExtractorOrchestrator, "_attempt_extractor", _fake_attempt)
    monkeypatch.setattr(ExtractorOrchestrator, "_attempt_window_fallback", _fake_window)

    out = orchestrator._run_full_quality(
        file_path="/tmp/doc.pdf",
        total_pages=1,
        page_scope=[1],
        notes=[],
        profile_name="full-quality",
    )
    assert window_called["value"] is True
    assert extractor_reserves and all(value == 180 for value in extractor_reserves)
    assert out.status == "soft_fail"


def test_full_quality_returns_best_soft_when_deadline_blocks_following_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = ExtractorOrchestrator(
        extract_full_quality_reserve_window_fallback_sec=180,
    )
    attempts = iter(
        [
            {
                "extractor": "docling_easyocr",
                "status": "soft_fail",
                "switch_reason": "chars_per_page<450.0",
                "duration_ms": 80.0,
                "error": None,
                "total_chunks": 1,
                "chars_per_page": 300.0,
                "empty_page_ratio": 0.0,
                "short_chunk_ratio": 0.0,
                "page_coverage": 1.0,
                "low_quality_pages": [],
                "poisoned_pages": [],
                "poisoned_page_ratio": 0.0,
                "poison_signals": {},
                "ocr_backend_effective": None,
                "ocr_fallback_path": [],
                "remaining_budget_sec_before_attempt": 200.0,
                "effective_timeout_sec": 80,
                "chunks": [_chunk("длинный текст " * 50, page=1)],
                "stats": None,
            },
            {
                "extractor": "docling_rapidocr",
                "status": "hard_fail",
                "switch_reason": "timeout:wall_clock_deadline_exceeded_before_attempt:docling_rapidocr",
                "duration_ms": 0.0,
                "error": "deadline",
                "total_chunks": 0,
                "chars_per_page": 0.0,
                "empty_page_ratio": 1.0,
                "short_chunk_ratio": 1.0,
                "page_coverage": 0.0,
                "low_quality_pages": [1],
                "poisoned_pages": [],
                "poisoned_page_ratio": 0.0,
                "poison_signals": {},
                "ocr_backend_effective": None,
                "ocr_fallback_path": [],
                "remaining_budget_sec_before_attempt": 10.0,
                "effective_timeout_sec": 0,
                "chunks": [],
                "stats": None,
            },
            {
                "extractor": "pymupdf4llm",
                "status": "hard_fail",
                "switch_reason": "timeout:wall_clock_deadline_exceeded_before_attempt:pymupdf4llm",
                "duration_ms": 0.0,
                "error": "deadline",
                "total_chunks": 0,
                "chars_per_page": 0.0,
                "empty_page_ratio": 1.0,
                "short_chunk_ratio": 1.0,
                "page_coverage": 0.0,
                "low_quality_pages": [1],
                "poisoned_pages": [],
                "poisoned_page_ratio": 0.0,
                "poison_signals": {},
                "ocr_backend_effective": None,
                "ocr_fallback_path": [],
                "remaining_budget_sec_before_attempt": 5.0,
                "effective_timeout_sec": 0,
                "chunks": [],
                "stats": None,
            },
        ]
    )

    def _fake_attempt(self, **kwargs):  # noqa: ARG001
        return next(attempts)

    def _fake_window(self, **kwargs):  # noqa: ARG001
        return {
            "extractor": "pymupdf_window_fallback",
            "status": "hard_fail",
            "switch_reason": "timeout:wall_clock_deadline_exceeded_before_attempt:pymupdf_window_fallback",
            "duration_ms": 0.0,
            "error": "deadline",
            "total_chunks": 0,
            "chars_per_page": 0.0,
            "empty_page_ratio": 1.0,
            "short_chunk_ratio": 1.0,
            "page_coverage": 0.0,
            "low_quality_pages": [1],
            "poisoned_pages": [],
            "poisoned_page_ratio": 0.0,
            "poison_signals": {},
            "ocr_backend_effective": None,
            "ocr_fallback_path": [],
            "remaining_budget_sec_before_attempt": 2.0,
            "effective_timeout_sec": 0,
            "chunks": [],
            "stats": None,
        }

    monkeypatch.setattr(ExtractorOrchestrator, "_attempt_extractor", _fake_attempt)
    monkeypatch.setattr(ExtractorOrchestrator, "_attempt_window_fallback", _fake_window)

    out = orchestrator._run_full_quality(
        file_path="/tmp/doc.pdf",
        total_pages=1,
        page_scope=[1],
        notes=[],
        profile_name="full-quality",
    )
    assert out.status == "soft_fail"
    assert out.extractor_used == "docling_easyocr"
    assert out.chunks


def test_full_quality_low_coverage_recovery_batched_improves_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = ExtractorOrchestrator(
        min_page_coverage=0.85,
        extract_low_coverage_recovery_enabled=True,
        extract_low_coverage_recovery_trigger_coverage=0.35,
        extract_low_coverage_recovery_batch_pages=120,
        extract_low_coverage_recovery_max_pages=360,
        extract_low_coverage_recovery_softfail_min_coverage=0.55,
        unstructured_targeted_only=True,
    )
    page_scope = list(range(1, 201))
    base_chunks = [_chunk("базовый текст " * 40, page=page) for page in range(1, 21)]
    batch1_chunks = [_chunk("recovery batch1 " * 40, page=page) for page in range(21, 141)]
    batch2_chunks = [_chunk("recovery batch2 " * 40, page=page) for page in range(141, 201)]
    calls: list[str] = []

    def _fake_attempt(self, **kwargs):  # noqa: ARG001
        label = kwargs["extractor_label"]
        calls.append(label)
        if label == "docling_easyocr":
            return _attempt_payload(
                extractor=label,
                status="hard_fail",
                switch_reason="exception:ocr_backend_failed",
                chunks=[],
                page_coverage=0.0,
            )
        if label == "docling_rapidocr":
            return _attempt_payload(
                extractor=label,
                status="hard_fail",
                switch_reason="exception:ocr_backend_failed",
                chunks=[],
                page_coverage=0.0,
            )
        if label == "pymupdf4llm":
            payload = _attempt_payload(
                extractor=label,
                status="hard_fail",
                switch_reason="page_coverage<0.1<0.85",
                chunks=base_chunks,
                page_coverage=0.1,
                chars_per_page=900.0,
                short_chunk_ratio=0.05,
            )
            payload["low_quality_pages"] = list(range(21, 201))
            return payload
        if label == "unstructured_hi_res_targeted":
            return _attempt_payload(
                extractor=label,
                status="hard_fail",
                switch_reason="exception:targeted_batch_failed",
                chunks=[],
                page_coverage=0.0,
            )
        if label == "unstructured_hi_res_recovery_batch_1":
            return _attempt_payload(
                extractor=label,
                status="pass",
                switch_reason=None,
                chunks=batch1_chunks,
                page_coverage=1.0,
                chars_per_page=950.0,
                short_chunk_ratio=0.0,
            )
        if label == "unstructured_hi_res_recovery_batch_2":
            return _attempt_payload(
                extractor=label,
                status="pass",
                switch_reason=None,
                chunks=batch2_chunks,
                page_coverage=1.0,
                chars_per_page=940.0,
                short_chunk_ratio=0.0,
            )
        pytest.fail(f"Unexpected extractor call: {label}")

    def _fake_window(self, **kwargs):  # noqa: ARG001
        pytest.fail("window fallback should not run after successful recovery")

    monkeypatch.setattr(ExtractorOrchestrator, "_attempt_extractor", _fake_attempt)
    monkeypatch.setattr(ExtractorOrchestrator, "_attempt_window_fallback", _fake_window)

    out = orchestrator._run_full_quality(
        file_path="/tmp/doc.pdf",
        total_pages=200,
        page_scope=page_scope,
        notes=[],
        profile_name="full-quality",
    )

    assert out.status == "pass"
    assert out.extractor_used == "pymupdf4llm+unstructured_hi_res_recovery"
    assert out.page_coverage >= 0.85
    assert any(label.startswith("unstructured_hi_res_recovery_batch_") for label in calls)
    recovery_attempts = [a for a in out.attempts if a.get("recovery_mode") == "low_coverage"]
    assert recovery_attempts
    assert recovery_attempts[-1].get("recovery_batches_run", 0) >= 1
    assert recovery_attempts[-1].get("coverage_after_batch")
    assert recovery_attempts[-1].get("targeted_pages_processed")


def test_recovery_downgrades_low_coverage_hard_fail_to_soft_fail_when_usable() -> None:
    orchestrator = ExtractorOrchestrator(
        min_page_coverage=0.85,
        extract_low_coverage_recovery_softfail_min_coverage=0.55,
    )
    attempt = _attempt_payload(
        extractor="pymupdf4llm+unstructured_hi_res_recovery",
        status="hard_fail",
        switch_reason="page_coverage<0.62<0.85",
        chunks=[_chunk("usable " * 50, page=1)],
        page_coverage=0.62,
        chars_per_page=900.0,
        short_chunk_ratio=0.0,
    )
    notes: list[str] = []

    orchestrator._apply_recovery_soft_fail_downgrade(attempt, notes)

    assert attempt["status"] == "soft_fail"
    assert attempt["downgraded_to_soft_fail_due_to_recovery_low_coverage"] is True
    assert notes


@pytest.mark.parametrize(
    ("switch_reason", "chunks", "coverage"),
    [
        ("timeout:Extractor timed out after 300s", [_chunk("usable " * 50, page=1)], 0.7),
        ("page_coverage<0.7<0.85; poisoned_text_ratio:0.4>=thr:0.35", [_chunk("usable " * 50, page=1)], 0.7),
        ("chunks==0", [], 0.0),
    ],
)
def test_recovery_keeps_hard_fail_for_timeout_poison_or_empty(
    switch_reason: str,
    chunks: list[DocumentChunk],
    coverage: float,
) -> None:
    orchestrator = ExtractorOrchestrator(
        min_page_coverage=0.85,
        extract_low_coverage_recovery_softfail_min_coverage=0.55,
    )
    attempt = _attempt_payload(
        extractor="pymupdf4llm+unstructured_hi_res_recovery",
        status="hard_fail",
        switch_reason=switch_reason,
        chunks=chunks,
        page_coverage=coverage,
        chars_per_page=900.0 if chunks else 0.0,
        short_chunk_ratio=0.0 if chunks else 1.0,
    )
    notes: list[str] = []

    orchestrator._apply_recovery_soft_fail_downgrade(attempt, notes)

    assert attempt["status"] == "hard_fail"
    assert attempt.get("downgraded_to_soft_fail_due_to_recovery_low_coverage") is False


def test_full_quality_recovery_disabled_keeps_baseline_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = ExtractorOrchestrator(
        min_page_coverage=0.85,
        extract_low_coverage_recovery_enabled=False,
        unstructured_targeted_only=True,
    )
    calls: list[str] = []
    base_chunks = [_chunk("базовый текст " * 40, page=1)]

    def _fake_attempt(self, **kwargs):  # noqa: ARG001
        label = kwargs["extractor_label"]
        calls.append(label)
        if label == "docling_easyocr":
            return _attempt_payload(
                extractor=label,
                status="hard_fail",
                switch_reason="exception:docling",
                chunks=[],
                page_coverage=0.0,
            )
        if label == "docling_rapidocr":
            return _attempt_payload(
                extractor=label,
                status="hard_fail",
                switch_reason="exception:docling",
                chunks=[],
                page_coverage=0.0,
            )
        if label == "pymupdf4llm":
            payload = _attempt_payload(
                extractor=label,
                status="hard_fail",
                switch_reason="page_coverage<0.2<0.85",
                chunks=base_chunks,
                page_coverage=0.2,
                chars_per_page=800.0,
                short_chunk_ratio=0.1,
            )
            payload["low_quality_pages"] = [2, 3]
            return payload
        if label == "unstructured_hi_res_targeted":
            return _attempt_payload(
                extractor=label,
                status="hard_fail",
                switch_reason="exception:unstructured",
                chunks=[],
                page_coverage=0.0,
            )
        if label.startswith("unstructured_hi_res_recovery_batch_"):
            pytest.fail("recovery batches must be skipped when disabled")
        pytest.fail(f"Unexpected label {label}")

    def _fake_window(self, **kwargs):  # noqa: ARG001
        return _attempt_payload(
            extractor="pymupdf_window_fallback",
            status="soft_fail",
            switch_reason="page_coverage<0.2<0.85",
            chunks=[_chunk("fallback text " * 40, page=1)],
            page_coverage=0.2,
            chars_per_page=850.0,
            short_chunk_ratio=0.0,
        )

    monkeypatch.setattr(ExtractorOrchestrator, "_attempt_extractor", _fake_attempt)
    monkeypatch.setattr(ExtractorOrchestrator, "_attempt_window_fallback", _fake_window)

    out = orchestrator._run_full_quality(
        file_path="/tmp/doc.pdf",
        total_pages=3,
        page_scope=[1, 2, 3],
        notes=[],
        profile_name="full-quality",
    )

    assert out.status == "soft_fail"
    assert out.extractor_used == "pymupdf_window_fallback"
    assert not any(label.startswith("unstructured_hi_res_recovery_batch_") for label in calls)
