"""Unit tests for extraction quality checks."""

from __future__ import annotations

from rag_system.extractors.base import (
    ExtractionQualityThresholds,
    compute_extraction_stats,
    compute_poison_signals,
    detect_poisoned_pages,
    is_quality_poor,
)
from rag_system.types import DocumentChunk



def test_quality_good_for_dense_text() -> None:
    chunks = [
        DocumentChunk(text="x" * 1000, source_path="/tmp/a.pdf", page=1, element_type="text"),
        DocumentChunk(text="x" * 800, source_path="/tmp/a.pdf", page=2, element_type="text"),
    ]
    stats = compute_extraction_stats(chunks, "/tmp/a.pdf", "docling")
    assert is_quality_poor(stats) is False



def test_quality_poor_for_too_short_chunks() -> None:
    chunks = [
        DocumentChunk(text="short", source_path="/tmp/b.pdf", page=1, element_type="text"),
        DocumentChunk(text="tiny", source_path="/tmp/b.pdf", page=2, element_type="text"),
    ]
    stats = compute_extraction_stats(chunks, "/tmp/b.pdf", "docling")
    assert is_quality_poor(stats, ExtractionQualityThresholds(min_chars_per_page=100, max_short_chunk_ratio=0.4)) is True


def test_compute_poison_signals_low_for_normal_text() -> None:
    sig = compute_poison_signals("Обычный текст без скрытых escape-последовательностей.")
    assert sig["escaped_seq_per_1k"] < 1.0
    assert sig["backslash_per_1k"] < 2.0
    assert sig["control_char_ratio"] == 0.0


def test_detect_poisoned_pages_flags_escape_heavy_text() -> None:
    thresholds = ExtractionQualityThresholds(
        max_escaped_seq_per_1k=10.0,
        max_backslash_per_1k=20.0,
        max_control_char_ratio=0.01,
    )
    text_by_page = {
        1: ["нормальный текст " * 20],
        2: ["\\n\\r\\x0A\\u000A " * 120],
    }
    poisoned_pages, poisoned_ratio, signals = detect_poisoned_pages(text_by_page, [1, 2], thresholds)
    assert poisoned_pages == [2]
    assert 0.0 < poisoned_ratio < 1.0
    assert signals["max_escaped_seq_per_1k"] > thresholds.max_escaped_seq_per_1k
