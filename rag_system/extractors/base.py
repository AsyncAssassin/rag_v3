"""Base extraction abstractions and shared quality checks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

from ..types import DocumentChunk, ExtractionStats, normalize_source_path
from ..utils import (
    chunk_table_by_tokens,
    chunk_text,
    chunk_text_by_tokens_sections,
    normalize_whitespace,
)


@dataclass(slots=True)
class ExtractionQualityThresholds:
    """Thresholds used to decide whether extraction quality is poor."""

    min_chars_per_page: float = 450.0
    max_empty_page_ratio: float = 0.25
    max_short_chunk_ratio: float = 0.55
    max_escaped_seq_per_1k: float = 18.0
    max_backslash_per_1k: float = 30.0
    max_control_char_ratio: float = 0.003
    poisoned_page_ratio_hard: float = 0.35


@dataclass(slots=True)
class ChunkingOptions:
    """Chunking configuration shared across extractors."""

    mode: str = "token"
    token_prose: int = 360
    token_table: int = 220
    overlap_token_prose: int = 60
    overlap_token_table: int = 30
    char_prose: int = 1600
    char_table: int = 1500
    overlap_char_prose: int = 160
    overlap_char_table: int = 120


class DocumentExtractor(ABC):
    """Abstract interface for all document extractors."""

    name: str = "base"

    @abstractmethod
    def extract(self, file_path: str) -> list[DocumentChunk]:
        """Extract chunks from a source file."""


class UnsupportedFileTypeError(RuntimeError):
    """Raised when extractor does not support provided file type."""



ESCAPED_SEQUENCE_RE = re.compile(r"(\\[nrt])|(\\x[0-9a-fA-F]{2})|(\\u[0-9a-fA-F]{4})")


def compute_poison_signals(text: str) -> dict[str, float]:
    """Compute anti-poison anomaly metrics for raw text."""
    raw = text or ""
    text_len = max(1, len(raw))
    escaped_seq_count = len(ESCAPED_SEQUENCE_RE.findall(raw))
    backslash_count = raw.count("\\")
    control_chars = sum(
        1 for ch in raw if ((ord(ch) < 32 or ord(ch) == 127) and ch not in {"\n", "\r", "\t"})
    )
    return {
        "escaped_seq_per_1k": (escaped_seq_count * 1000.0) / text_len,
        "backslash_per_1k": (backslash_count * 1000.0) / text_len,
        "control_char_ratio": control_chars / text_len,
    }


def detect_poisoned_pages(
    text_by_page: dict[int, list[str]],
    page_scope: list[int],
    thresholds: ExtractionQualityThresholds,
) -> tuple[list[int], float, dict[str, float]]:
    """Detect poisoned pages and return ratio + aggregated signal diagnostics."""
    scope = sorted({int(p) for p in page_scope if int(p) > 0})
    if not scope:
        scope = [1]

    poisoned_pages: list[int] = []
    max_escaped = 0.0
    max_backslash = 0.0
    max_control = 0.0

    for page in scope:
        page_text = "\n".join(text_by_page.get(page, []))
        sig = compute_poison_signals(page_text)
        max_escaped = max(max_escaped, sig["escaped_seq_per_1k"])
        max_backslash = max(max_backslash, sig["backslash_per_1k"])
        max_control = max(max_control, sig["control_char_ratio"])

        is_poisoned = (
            sig["escaped_seq_per_1k"] > thresholds.max_escaped_seq_per_1k
            or sig["backslash_per_1k"] > thresholds.max_backslash_per_1k
            or sig["control_char_ratio"] > thresholds.max_control_char_ratio
        )
        if is_poisoned:
            poisoned_pages.append(page)

    poisoned_ratio = float(len(poisoned_pages) / max(1, len(scope)))
    signals = {
        "max_escaped_seq_per_1k": round(max_escaped, 4),
        "max_backslash_per_1k": round(max_backslash, 4),
        "max_control_char_ratio": round(max_control, 6),
        "threshold_escaped_seq_per_1k": float(thresholds.max_escaped_seq_per_1k),
        "threshold_backslash_per_1k": float(thresholds.max_backslash_per_1k),
        "threshold_control_char_ratio": float(thresholds.max_control_char_ratio),
        "poisoned_pages_count": float(len(poisoned_pages)),
        "scope_pages_count": float(len(scope)),
    }
    return poisoned_pages, poisoned_ratio, signals


def compute_extraction_stats(
    chunks: Iterable[DocumentChunk],
    source_path: str,
    extractor_name: str,
    short_chunk_chars: int = 120,
    total_pages: int | None = None,
) -> ExtractionStats:
    """Compute extraction quality stats for one file."""
    items = list(chunks)
    total_chars = sum(len(ch.text or "") for ch in items)

    page_ids = [ch.page for ch in items if ch.page is not None]
    pages_seen = len(set(page_ids)) if page_ids else (1 if total_chars > 0 else 0)
    chars_per_page = float(total_chars / max(pages_seen, 1))
    effective_total_pages = max(1, int(total_pages if total_pages is not None else max(1, pages_seen)))
    page_coverage = float(min(1.0, pages_seen / effective_total_pages)) if pages_seen > 0 else 0.0

    if page_ids:
        page_to_has_text: dict[int, bool] = {}
        for ch in items:
            if ch.page is None:
                continue
            page_to_has_text[ch.page] = page_to_has_text.get(ch.page, False) or bool((ch.text or "").strip())
        empty_pages = sum(1 for has_text in page_to_has_text.values() if not has_text)
        empty_ratio = float(empty_pages / max(len(page_to_has_text), 1))
    else:
        empty_ratio = 0.0 if total_chars > 0 else 1.0

    if items:
        short_chunks = sum(1 for ch in items if len((ch.text or "").strip()) < short_chunk_chars)
        short_ratio = float(short_chunks / len(items))
    else:
        short_ratio = 1.0

    has_table_elements = any((ch.element_type or "").lower() == "table" or bool(ch.table_html) for ch in items)

    return ExtractionStats(
        source_path=normalize_source_path(source_path),
        total_chunks=len(items),
        total_chars=total_chars,
        pages_seen=pages_seen,
        chars_per_page=chars_per_page,
        empty_page_ratio=empty_ratio,
        short_chunk_ratio=short_ratio,
        has_table_elements=has_table_elements,
        extractor_name=extractor_name,
        total_pages=effective_total_pages,
        page_coverage=page_coverage,
    )



def is_quality_poor(
    stats: ExtractionStats,
    thresholds: ExtractionQualityThresholds | None = None,
) -> bool:
    """Return True when extraction quality is below accepted thresholds."""
    th = thresholds or ExtractionQualityThresholds()
    return (
        stats.chars_per_page < th.min_chars_per_page
        or stats.empty_page_ratio > th.max_empty_page_ratio
        or stats.short_chunk_ratio > th.max_short_chunk_ratio
    )



def ensure_pdf(path: str) -> None:
    """Validate that the provided file path has PDF extension."""
    suffix = Path(path).suffix.lower()
    if suffix != ".pdf":
        raise UnsupportedFileTypeError(f"Extractor supports only PDF, got: {suffix}")



def chunk_long_text(
    text: str,
    source_path: str,
    page: int | None,
    element_type: str | None,
    metadata: dict,
    chunking: ChunkingOptions | None = None,
) -> list[DocumentChunk]:
    """Split long text into manageable document chunks."""
    cfg = chunking or ChunkingOptions()
    kind = (element_type or "").lower()
    raw = (text or "").strip()
    if not raw:
        return []

    pieces: list[str]
    if cfg.mode == "token":
        if kind == "table":
            max_tokens = max(64, int(cfg.token_table))
            overlap_tokens = max(0, min(int(cfg.overlap_token_table), max_tokens - 1))
            pieces = chunk_table_by_tokens(
                raw,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )
        else:
            max_tokens = max(64, int(cfg.token_prose))
            overlap_tokens = max(0, min(int(cfg.overlap_token_prose), max_tokens - 1))
            pieces = chunk_text_by_tokens_sections(
                raw,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )
    else:
        normalized = normalize_whitespace(raw) if kind != "table" else raw
        chunk_size = int(cfg.char_table if kind == "table" else cfg.char_prose)
        overlap = int(cfg.overlap_char_table if kind == "table" else cfg.overlap_char_prose)
        if overlap >= chunk_size:
            overlap = max(0, chunk_size // 5)
        pieces = chunk_text(
            normalized,
            chunk_size=max(200, chunk_size),
            overlap=max(0, overlap),
        )

    out: list[DocumentChunk] = []
    for piece in pieces:
        out.append(
            DocumentChunk(
                text=piece,
                source_path=normalize_source_path(source_path),
                page=page,
                element_type=element_type,
                metadata=dict(metadata or {}),
            )
        )
    return out
