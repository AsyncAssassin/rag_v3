"""Shared type definitions for the RAG v3 system."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DocumentChunk:
    """Atomic chunk extracted from a source document."""

    text: str
    source_path: str
    page: int | None = None
    element_type: str | None = None
    bbox: dict[str, float] | None = None
    table_html: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize chunk to a plain dictionary."""
        return asdict(self)


@dataclass(slots=True)
class ExtractionStats:
    """Quality stats for extraction pass over one file."""

    source_path: str
    total_chunks: int
    total_chars: int
    pages_seen: int
    chars_per_page: float
    empty_page_ratio: float
    short_chunk_ratio: float
    has_table_elements: bool
    extractor_name: str
    total_pages: int = 1
    page_coverage: float = 1.0
    status: str = "pass"
    switch_reason: str | None = None
    low_quality_pages: list[int] = field(default_factory=list)
    fallback_path: list[str] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    ocr_backend_effective: str | None = None
    ocr_fallback_path: list[str] = field(default_factory=list)
    poisoned_pages: list[int] = field(default_factory=list)
    poisoned_page_ratio: float = 0.0
    poison_signals: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IndexedChunk:
    """Chunk enriched with indexing artifacts."""

    chunk_id: str
    chunk: DocumentChunk
    tokenized: list[str]
    dense_vector: list[float]
    content_hash: str
    file_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize indexed chunk for persistence."""
        data = {
            "chunk_id": self.chunk_id,
            "chunk": self.chunk.to_dict(),
            "tokenized": self.tokenized,
            "dense_vector": self.dense_vector,
            "content_hash": self.content_hash,
            "file_hash": self.file_hash,
        }
        return data


@dataclass(slots=True)
class RetrievedChunk:
    """Retrieved candidate with score breakdown."""

    chunk_id: str
    text: str
    source_path: str
    page: int | None
    element_type: str | None
    bm25_score: float = 0.0
    dense_score: float = 0.0
    fusion_score: float = 0.0
    rerank_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QueryTrace:
    """Debug trace for one query run."""

    original_query: str
    rewritten_queries: list[str]
    extractor_used: str | None
    reranker_used: str
    reranker_cached: bool
    reranker_load_ms: float
    retrieve_top_k: int
    rerank_top_n: int
    final_top_k: int
    timings_ms: dict[str, float]
    grounded_refusal: bool = False
    grounded_reason: str | None = None
    extractor_attempts: list[dict[str, Any]] = field(default_factory=list)
    switch_reason: str | None = None
    page_coverage: float | None = None
    low_quality_pages: list[int] = field(default_factory=list)
    final_extractor_used: str | None = None
    dense_disabled: bool = False
    dense_disable_reason: str | None = None


@dataclass(slots=True)
class AnswerResult:
    """Final answer payload returned by pipeline/UI."""

    answer: str
    citations: list[dict[str, Any]]
    context_chunks: list[RetrievedChunk]
    trace: QueryTrace


@dataclass(slots=True)
class IndexStats:
    """Summary from indexing run."""

    indexed_files: int
    indexed_chunks: int
    deduplicated_chunks: int
    duplicate_files: int
    failed_files: int
    extraction_reports: list[ExtractionStats]


@dataclass(slots=True)
class PDFRegressionRecord:
    """Result for one file in PDF ingestion regression."""

    file_path: str
    extractor: str
    status: str
    total_chunks: int
    chars_per_page: float
    empty_page_ratio: float
    short_chunk_ratio: float
    has_table_elements: bool
    note: str = ""


def normalize_source_path(path: str | Path) -> str:
    """Return a canonical absolute source path string."""
    return str(Path(path).expanduser().resolve())
