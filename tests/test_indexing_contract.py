"""Unit tests for indexing contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from rag_system.indexing import GigaEmbeddingClient, HybridIndex, IndexBuilder
from rag_system.types import DocumentChunk, IndexedChunk
from rag_system.utils import load_json, save_json, sha256_file, sha256_text, tokenize


def test_discover_files_fails_for_missing_data_dir() -> None:
    missing_dir = Path("/tmp") / f"rag_v3_missing_{uuid4().hex}"
    builder = object.__new__(IndexBuilder)
    builder.settings = SimpleNamespace(index_dir=".rag_index")
    with pytest.raises(FileNotFoundError, match="Data directory not found"):
        builder.discover_files(str(missing_dir))


def _mk_stats(path: str, *, chunks: int, coverage: float, status: str, reason: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        source_path=path,
        total_chunks=chunks,
        chars_per_page=1000.0,
        empty_page_ratio=0.0,
        short_chunk_ratio=0.0,
        has_table_elements=False,
        page_coverage=coverage,
        status=status,
        switch_reason=reason,
        low_quality_pages=[],
        poisoned_pages=[],
        poisoned_page_ratio=0.0,
        poison_signals={},
        ocr_backend_effective=None,
        ocr_fallback_path=[],
    )


def _mk_outcome(path: str, *, text: str, status: str = "pass", coverage: float = 1.0) -> SimpleNamespace:
    chunk = DocumentChunk(
        text=text,
        source_path=path,
        page=1,
        element_type="text",
        metadata={},
    )
    return SimpleNamespace(
        chunks=[chunk],
        stats=_mk_stats(path, chunks=1, coverage=coverage, status=status, reason=None),
        extractor_used="pymupdf4llm",
        notes=[],
        fallback_path=["pymupdf4llm"],
        status=status,
        switch_reason=None,
        page_coverage=coverage,
        low_quality_pages=[],
        attempts=[],
    )


def _mk_indexed_chunk(path: str, *, text: str, file_hash: str, chunk_id: str) -> IndexedChunk:
    chunk = DocumentChunk(
        text=text,
        source_path=path,
        page=1,
        element_type="text",
        metadata={},
    )
    return IndexedChunk(
        chunk_id=chunk_id,
        chunk=chunk,
        tokenized=tokenize(text),
        dense_vector=[0.1, 0.2, 0.3],
        content_hash=sha256_text(text),
        file_hash=file_hash,
    )


def _mk_builder(tmp_path: Path) -> IndexBuilder:
    builder = object.__new__(IndexBuilder)
    builder.settings = SimpleNamespace(
        index_dir=str(tmp_path / ".rag_index"),
        extract_min_page_coverage=0.85,
        ingest_profile="demo-fast",
    )
    builder.embed_client = SimpleNamespace(embed_documents=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])
    return builder


def test_build_or_update_dedup_is_scoped_per_file(tmp_path: Path) -> None:
    first = tmp_path / "a.pdf"
    second = tmp_path / "b.pdf"
    first.write_text("same", encoding="utf-8")
    second.write_text("same", encoding="utf-8")

    builder = _mk_builder(tmp_path)
    builder.discover_files = lambda data_dir: [str(first), str(second)]  # type: ignore[method-assign]
    builder.extractor_orchestrator = SimpleNamespace(
        extract_with_policy=lambda file_path, **kwargs: _mk_outcome(file_path, text="identical paragraph")
    )

    index, stats = builder.build_or_update(str(tmp_path), reset_index=True)

    assert len(index.indexed_chunks) == 2
    assert stats.deduplicated_chunks == 0
    assert len(index.path_to_chunk_ids[str(first.resolve())]) == 1
    assert len(index.path_to_chunk_ids[str(second.resolve())]) == 1


def test_quality_aware_invalidation_reextracts_low_quality_same_hash(tmp_path: Path) -> None:
    target = tmp_path / "doc.pdf"
    target.write_text("content", encoding="utf-8")
    source_path = str(target.resolve())
    file_hash = sha256_file(source_path)

    existing = HybridIndex()
    existing_chunk = _mk_indexed_chunk(source_path, text="old", file_hash=file_hash, chunk_id="old1")
    existing.indexed_chunks = [existing_chunk]
    existing.path_to_chunk_ids = {source_path: ["old1"]}
    existing.file_hash_by_path = {source_path: file_hash}
    existing.file_quality_by_path = {
        source_path: {
            "status": "hard_fail",
            "page_coverage": 0.2,
            "total_chunks": 1,
            "switch_reason": "page_coverage<0.2<0.85",
            "poisoned_page_ratio": 0.0,
            "extractor_used": "pymupdf4llm",
        }
    }
    existing.rebuild_runtime()
    existing.persist(str(tmp_path / ".rag_index"))

    builder = _mk_builder(tmp_path)
    builder.discover_files = lambda data_dir: [source_path]  # type: ignore[method-assign]
    calls = {"n": 0}

    def _extract(file_path, **kwargs):  # noqa: ANN001, ARG001
        calls["n"] += 1
        return _mk_outcome(file_path, text="new", status="pass", coverage=1.0)

    builder.extractor_orchestrator = SimpleNamespace(extract_with_policy=_extract)

    index, _ = builder.build_or_update(str(tmp_path), reset_index=False)
    assert calls["n"] == 1
    assert [c.chunk.text for c in index.indexed_chunks] == ["new"]
    assert index.file_quality_by_path[source_path]["status"] == "pass"


def test_quality_aware_invalidation_reuses_high_quality_same_hash(tmp_path: Path) -> None:
    target = tmp_path / "doc.pdf"
    target.write_text("content", encoding="utf-8")
    source_path = str(target.resolve())
    file_hash = sha256_file(source_path)

    existing = HybridIndex()
    existing_chunk = _mk_indexed_chunk(source_path, text="old", file_hash=file_hash, chunk_id="old1")
    existing.indexed_chunks = [existing_chunk]
    existing.path_to_chunk_ids = {source_path: ["old1"]}
    existing.file_hash_by_path = {source_path: file_hash}
    existing.file_quality_by_path = {
        source_path: {
            "status": "pass",
            "page_coverage": 1.0,
            "total_chunks": 1,
            "switch_reason": None,
            "poisoned_page_ratio": 0.0,
            "extractor_used": "pymupdf4llm",
        }
    }
    existing.rebuild_runtime()
    existing.persist(str(tmp_path / ".rag_index"))

    builder = _mk_builder(tmp_path)
    builder.discover_files = lambda data_dir: [source_path]  # type: ignore[method-assign]
    builder.extractor_orchestrator = SimpleNamespace(
        extract_with_policy=lambda *args, **kwargs: pytest.fail("extractor should not be called for reusable quality")
    )

    index, _ = builder.build_or_update(str(tmp_path), reset_index=False)
    assert [c.chunk.text for c in index.indexed_chunks] == ["old"]
    assert index.file_quality_by_path[source_path]["status"] == "pass"


def test_missing_quality_meta_triggers_reextract(tmp_path: Path) -> None:
    target = tmp_path / "doc.pdf"
    target.write_text("content", encoding="utf-8")
    source_path = str(target.resolve())
    file_hash = sha256_file(source_path)

    existing = HybridIndex()
    existing_chunk = _mk_indexed_chunk(source_path, text="old", file_hash=file_hash, chunk_id="old1")
    existing.indexed_chunks = [existing_chunk]
    existing.path_to_chunk_ids = {source_path: ["old1"]}
    existing.file_hash_by_path = {source_path: file_hash}
    existing.rebuild_runtime()
    existing.persist(str(tmp_path / ".rag_index"))

    meta_path = tmp_path / ".rag_index" / "meta.json"
    raw_meta = load_json(meta_path)
    raw_meta.pop("file_quality_by_path", None)
    raw_meta.pop("meta_schema_version", None)
    save_json(meta_path, raw_meta)

    builder = _mk_builder(tmp_path)
    builder.discover_files = lambda data_dir: [source_path]  # type: ignore[method-assign]
    calls = {"n": 0}

    def _extract(file_path, **kwargs):  # noqa: ANN001, ARG001
        calls["n"] += 1
        return _mk_outcome(file_path, text="reindexed", status="pass", coverage=1.0)

    builder.extractor_orchestrator = SimpleNamespace(extract_with_policy=_extract)

    index, _ = builder.build_or_update(str(tmp_path), reset_index=False)

    assert calls["n"] == 1
    assert [c.chunk.text for c in index.indexed_chunks] == ["reindexed"]
    assert source_path in index.file_quality_by_path

    persisted_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert persisted_meta["meta_schema_version"] == 2
    assert source_path in persisted_meta["file_quality_by_path"]


def test_embedding_client_splits_oversized_items_without_breaking_batch_order() -> None:
    class _FakeEmbedder:
        def embeddings(self, *, texts, model):  # noqa: ANN001, ARG002
            for idx, text in enumerate(texts):
                if len(text) > 20:
                    raise RuntimeError(f"Tokens limit exceeded for index {idx}: 5000 (max 4096)")
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[float(len(text)), 1.0]) for text in texts]
            )

    client = object.__new__(GigaEmbeddingClient)
    client.settings = SimpleNamespace(
        giga_embedding_model="mock-embed",
        api_max_retries=0,
        api_retry_backoff_sec=0.01,
        embedding_batch_size=4,
    )
    client._embedder = _FakeEmbedder()

    texts = ["alpha", "beta", "x" * 80, "gamma"]
    vectors = client.embed_documents(texts)

    assert len(vectors) == len(texts)
    assert vectors[0][0] == 5.0
    assert vectors[1][0] == 4.0
    assert vectors[3][0] == 5.0
    assert vectors[2][0] > 0.0
