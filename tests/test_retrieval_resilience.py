"""Unit tests for retrieval resilience and dense fallback behavior."""

from __future__ import annotations

from rag_system.indexing import HybridIndex
from rag_system.retrieval import HybridRetriever, apply_source_diversity, apply_year_aware_source_boost
from rag_system.types import DocumentChunk, IndexedChunk, RetrievedChunk
from rag_system.utils import sha256_text, tokenize


class _EmbedMismatch:
    def embed_query(self, text: str) -> list[float]:  # noqa: ARG002
        return [0.1, 0.2]


def test_retrieval_falls_back_to_bm25_on_dense_dim_mismatch() -> None:
    chunk = DocumentChunk(
        text="alpha beta gamma",
        source_path="/tmp/a.pdf",
        page=1,
        element_type="text",
        metadata={},
    )
    content_hash = sha256_text(chunk.text)
    indexed = IndexedChunk(
        chunk_id="c1",
        chunk=chunk,
        tokenized=tokenize(chunk.text),
        dense_vector=[0.1, 0.2, 0.3],
        content_hash=content_hash,
        file_hash="f1",
    )
    index = HybridIndex()
    index.indexed_chunks = [indexed]
    index.rebuild_runtime()

    retriever = HybridRetriever(index, _EmbedMismatch())
    results, debug = retriever.retrieve("alpha", top_k=5)

    assert len(results) == 1
    assert debug.dense_ranked_ids == []
    assert debug.dense_disabled is True
    assert debug.dense_disable_reason is not None
    assert "dense_dim_mismatch" in debug.dense_disable_reason


def _rc(chunk_id: str, source_path: str, fusion_score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=f"text-{chunk_id}",
        source_path=source_path,
        page=1,
        element_type="text",
        bm25_score=1.0,
        dense_score=1.0,
        fusion_score=fusion_score,
        metadata={},
    )


def test_source_diversity_limits_single_source_dominance_with_backfill() -> None:
    ordered = [
        _rc("a1", "/tmp/a.pdf", 10.0),
        _rc("a2", "/tmp/a.pdf", 9.0),
        _rc("a3", "/tmp/a.pdf", 8.0),
        _rc("b1", "/tmp/b.pdf", 7.0),
    ]
    diversified = apply_source_diversity(
        ordered,
        top_k=4,
        enabled=True,
        max_chunks_per_source=2,
    )
    assert [x.chunk_id for x in diversified] == ["a1", "b1", "a2", "a3"]


def test_source_diversity_disabled_keeps_original_order() -> None:
    ordered = [
        _rc("a1", "/tmp/a.pdf", 10.0),
        _rc("a2", "/tmp/a.pdf", 9.0),
        _rc("b1", "/tmp/b.pdf", 8.0),
    ]
    same = apply_source_diversity(
        ordered,
        top_k=3,
        enabled=False,
        max_chunks_per_source=2,
    )
    assert [x.chunk_id for x in same] == ["a1", "a2", "b1"]


def test_year_aware_boost_promotes_matching_source_year() -> None:
    ordered = [
        _rc("c2019", "/tmp/report_2019.pdf", 10.0),
        _rc("c2020", "/tmp/report_2020.pdf", 9.95),
        _rc("c2021", "/tmp/report_2021.pdf", 9.8),
    ]
    boosted = apply_year_aware_source_boost(
        ordered,
        query="Какие эффекты пандемии отражены в отчете Сбера 2020?",
        enabled=True,
        boost=0.12,
    )
    assert [x.chunk_id for x in boosted] == ["c2020", "c2019", "c2021"]


def test_year_aware_boost_disabled_keeps_original_order() -> None:
    ordered = [
        _rc("c2019", "/tmp/report_2019.pdf", 10.0),
        _rc("c2020", "/tmp/report_2020.pdf", 9.95),
    ]
    same = apply_year_aware_source_boost(
        ordered,
        query="Что в отчете 2020?",
        enabled=False,
        boost=0.12,
    )
    assert [x.chunk_id for x in same] == ["c2019", "c2020"]


def test_year_aware_boost_without_year_in_query_keeps_original_order() -> None:
    ordered = [
        _rc("c2020", "/tmp/report_2020.pdf", 10.0),
        _rc("c2021", "/tmp/report_2021.pdf", 9.9),
    ]
    same = apply_year_aware_source_boost(
        ordered,
        query="Какие ключевые темы в отчете?",
        enabled=True,
        boost=0.12,
    )
    assert [x.chunk_id for x in same] == ["c2020", "c2021"]
