"""Index building, persistence, and loading for hybrid retrieval."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .config import Settings
from .extractors.base import ChunkingOptions, ExtractionQualityThresholds
from .extractors.factory import ExtractorOrchestrator
from .logging_utils import get_logger
from .types import DocumentChunk, ExtractionStats, IndexStats, IndexedChunk, normalize_source_path
from .utils import load_json, retry_call, save_json, sha256_file, sha256_text, tokenize


LOGGER = get_logger()
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv", ".md"}
TOKEN_LIMIT_INDEX_RE = re.compile(r"index\s+(\d+)\s*:\s*\d+\s*\(max\s*\d+\)", re.IGNORECASE)


class GigaEmbeddingClient:
    """Dense embedding client backed by GigaEmbeddings."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._embedder = self._build_embedder()

    def _build_embedder(self):
        """Create low-level GigaChat client with explicit timeout semantics."""
        try:
            from gigachat import GigaChat
        except Exception as exc:
            raise RuntimeError("gigachat client import failed") from exc

        return GigaChat(
            credentials=self.settings.giga_api_key,
            scope=self.settings.giga_scope,
            verify_ssl_certs=False,
            timeout=float(self.settings.giga_http_timeout_sec),
        )

    @staticmethod
    def _is_token_limit_error(exc: Exception) -> bool:
        """Return true when backend rejects input due to token limit."""
        msg = f"{type(exc).__name__}: {exc}".lower()
        return "tokens limit exceeded" in msg or "requestentitytoolargeerror" in msg

    @staticmethod
    def _oversize_index_from_error(exc: Exception, *, batch_len: int) -> int | None:
        """Parse offending item index from backend 413 text."""
        m = TOKEN_LIMIT_INDEX_RE.search(str(exc))
        if not m:
            return None
        try:
            idx = int(m.group(1))
        except Exception:
            return None
        if 0 <= idx < batch_len:
            return idx
        return None

    @staticmethod
    def _split_text_for_embeddings(text: str) -> tuple[str, str]:
        """Split long text into two semantically stable halves."""
        payload = (text or "").strip()
        if len(payload) <= 2:
            return payload, ""
        mid = len(payload) // 2
        window = min(400, max(40, len(payload) // 10))
        lo = max(1, mid - window)
        hi = min(len(payload) - 1, mid + window)
        split_at = None
        for pos in range(mid, lo - 1, -1):
            if payload[pos].isspace():
                split_at = pos
                break
        if split_at is None:
            for pos in range(mid + 1, hi + 1):
                if payload[pos].isspace():
                    split_at = pos
                    break
        if split_at is None:
            split_at = mid
        left = payload[:split_at].strip()
        right = payload[split_at:].strip()
        return left, right

    def _embed_batch_once(self, batch: list[str]) -> list[list[float]]:
        """Embed batch in one API call with retries."""
        response = retry_call(
            lambda: self._embedder.embeddings(
                texts=batch,
                model=self.settings.giga_embedding_model,
            ),
            max_retries=self.settings.api_max_retries,
            base_backoff_sec=self.settings.api_retry_backoff_sec,
        )
        vectors = [list(map(float, item.embedding)) for item in (response.data or [])]
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"embeddings batch size mismatch: requested={len(batch)} received={len(vectors)}"
            )
        return vectors

    def _embed_text_safe(self, text: str, *, depth: int = 0) -> list[float]:
        """Embed single text, recursively splitting when token limit is exceeded."""
        try:
            return self._embed_batch_once([text])[0]
        except Exception as exc:
            if not self._is_token_limit_error(exc) or depth >= 8:
                raise
            left, right = self._split_text_for_embeddings(text)
            if not left or not right:
                raise
            if left == text or right == text:
                raise
            v_left = self._embed_text_safe(left, depth=depth + 1)
            v_right = self._embed_text_safe(right, depth=depth + 1)
            merged = np.mean(np.asarray([v_left, v_right], dtype=np.float32), axis=0)
            return list(map(float, merged.tolist()))

    def _embed_batch_resilient(self, batch: list[str]) -> list[list[float]]:
        """Embed batch, isolating only oversize items instead of failing whole run."""
        if not batch:
            return []
        try:
            return self._embed_batch_once(batch)
        except Exception as exc:
            if not self._is_token_limit_error(exc):
                raise
            if len(batch) == 1:
                return [self._embed_text_safe(batch[0])]
            oversize_idx = self._oversize_index_from_error(exc, batch_len=len(batch))
            if oversize_idx is None:
                return [self._embed_text_safe(text) for text in batch]
            left = self._embed_batch_resilient(batch[:oversize_idx])
            mid = [self._embed_text_safe(batch[oversize_idx])]
            right = self._embed_batch_resilient(batch[oversize_idx + 1 :])
            return left + mid + right

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed list of texts into dense vectors."""
        if not texts:
            return []
        batch_size = max(1, int(self.settings.embedding_batch_size))
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors = self._embed_batch_resilient(batch)
            all_vectors.extend(vectors)
        return all_vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed single query text."""
        return self._embed_text_safe(text)


class HybridIndex:
    """In-memory hybrid index with persistence hooks."""

    def __init__(self) -> None:
        self.indexed_chunks: list[IndexedChunk] = []
        self.chunk_ids: list[str] = []
        self.chunk_map: dict[str, IndexedChunk] = {}
        self.path_to_chunk_ids: dict[str, list[str]] = {}
        self.file_hash_by_path: dict[str, str] = {}
        self.file_quality_by_path: dict[str, dict[str, object]] = {}
        self.dense_matrix: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.bm25: BM25Okapi | None = None

    def rebuild_runtime(self) -> None:
        """Recompute BM25 and dense matrix from indexed chunks."""
        self.chunk_ids = [c.chunk_id for c in self.indexed_chunks]
        self.chunk_map = {c.chunk_id: c for c in self.indexed_chunks}

        tokenized_corpus = [c.tokenized for c in self.indexed_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

        if self.indexed_chunks:
            self.dense_matrix = np.asarray([c.dense_vector for c in self.indexed_chunks], dtype=np.float32)
        else:
            self.dense_matrix = np.empty((0, 0), dtype=np.float32)

    def persist(self, index_dir: str) -> None:
        """Persist index artifacts to disk."""
        root = Path(index_dir)
        root.mkdir(parents=True, exist_ok=True)

        chunks_payload = [c.to_dict() for c in self.indexed_chunks]
        save_json(root / "chunks.json", {"items": chunks_payload})
        save_json(
            root / "meta.json",
            {
                "meta_schema_version": 2,
                "path_to_chunk_ids": self.path_to_chunk_ids,
                "file_hash_by_path": self.file_hash_by_path,
                "file_quality_by_path": self.file_quality_by_path,
            },
        )

    @classmethod
    def load(cls, index_dir: str) -> "HybridIndex":
        """Load previously persisted index from disk."""
        root = Path(index_dir)
        idx = cls()
        chunks_path = root / "chunks.json"
        meta_path = root / "meta.json"
        if not chunks_path.exists() or not meta_path.exists():
            return idx

        chunks_data = load_json(chunks_path).get("items", [])
        for item in chunks_data:
            chunk_data = item["chunk"]
            chunk = DocumentChunk(
                text=chunk_data["text"],
                source_path=chunk_data["source_path"],
                page=chunk_data.get("page"),
                element_type=chunk_data.get("element_type"),
                bbox=chunk_data.get("bbox"),
                table_html=chunk_data.get("table_html"),
                metadata=chunk_data.get("metadata", {}),
            )
            idx.indexed_chunks.append(
                IndexedChunk(
                    chunk_id=item["chunk_id"],
                    chunk=chunk,
                    tokenized=item["tokenized"],
                    dense_vector=item["dense_vector"],
                    content_hash=item["content_hash"],
                    file_hash=item["file_hash"],
                )
            )

        meta = load_json(meta_path)
        idx.path_to_chunk_ids = meta.get("path_to_chunk_ids", {})
        idx.file_hash_by_path = meta.get("file_hash_by_path", {})
        idx.file_quality_by_path = meta.get("file_quality_by_path", {})
        idx.rebuild_runtime()
        return idx


class IndexBuilder:
    """Builds or updates the hybrid index from filesystem documents."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embed_client = GigaEmbeddingClient(settings)
        quality_thresholds = ExtractionQualityThresholds(
            min_chars_per_page=float(settings.extract_min_chars_per_page),
            max_empty_page_ratio=float(settings.extract_max_empty_page_ratio),
            max_short_chunk_ratio=float(settings.extract_max_short_chunk_ratio),
            max_escaped_seq_per_1k=float(settings.extract_max_escaped_seq_per_1k),
            max_backslash_per_1k=float(settings.extract_max_backslash_per_1k),
            max_control_char_ratio=float(settings.extract_max_control_char_ratio),
            poisoned_page_ratio_hard=float(settings.extract_poisoned_page_ratio_hard),
        )
        chunking = ChunkingOptions(
            mode=settings.chunk_mode,
            token_prose=settings.chunk_tokens_prose,
            token_table=settings.chunk_tokens_table,
            overlap_token_prose=settings.chunk_overlap_prose,
            overlap_token_table=settings.chunk_overlap_table,
            char_prose=settings.chunk_chars_prose,
            char_table=settings.chunk_chars_table,
            overlap_char_prose=settings.chunk_overlap_chars_prose,
            overlap_char_table=settings.chunk_overlap_chars_table,
        )
        self.extractor_orchestrator = ExtractorOrchestrator(
            languages=("rus", "eng"),
            quality_thresholds=quality_thresholds,
            extract_timeout_sec=settings.extract_timeout_sec,
            extract_timeout_base_sec=settings.extract_timeout_base_sec,
            extract_timeout_per_100_pages_sec=settings.extract_timeout_per_100_pages_sec,
            extract_timeout_per_10mb_sec=settings.extract_timeout_per_10mb_sec,
            extract_timeout_max_sec=settings.extract_timeout_max_sec,
            extract_full_quality_docling_primary_max_sec=getattr(
                settings,
                "extract_full_quality_docling_primary_max_sec",
                300,
            ),
            extract_full_quality_docling_secondary_max_sec=getattr(
                settings,
                "extract_full_quality_docling_secondary_max_sec",
                240,
            ),
            extract_full_quality_pymupdf_max_sec=getattr(
                settings,
                "extract_full_quality_pymupdf_max_sec",
                180,
            ),
            extract_full_quality_reserve_window_fallback_sec=getattr(
                settings,
                "extract_full_quality_reserve_window_fallback_sec",
                180,
            ),
            extract_full_quality_min_stage_start_sec=getattr(
                settings,
                "extract_full_quality_min_stage_start_sec",
                15,
            ),
            extract_full_quality_unstructured_min_remaining_sec=getattr(
                settings,
                "extract_full_quality_unstructured_min_remaining_sec",
                90,
            ),
            extract_prefer_best_usable=getattr(
                settings,
                "extract_prefer_best_usable",
                True,
            ),
            extract_best_usable_min_coverage_gap=getattr(
                settings,
                "extract_best_usable_min_coverage_gap",
                0.15,
            ),
            unstructured_min_merged_chunk_chars=getattr(
                settings,
                "unstructured_min_merged_chunk_chars",
                220,
            ),
            extract_low_coverage_recovery_enabled=getattr(
                settings,
                "extract_low_coverage_recovery_enabled",
                True,
            ),
            extract_low_coverage_recovery_trigger_coverage=getattr(
                settings,
                "extract_low_coverage_recovery_trigger_coverage",
                0.35,
            ),
            extract_low_coverage_recovery_batch_pages=getattr(
                settings,
                "extract_low_coverage_recovery_batch_pages",
                120,
            ),
            extract_low_coverage_recovery_max_pages=getattr(
                settings,
                "extract_low_coverage_recovery_max_pages",
                360,
            ),
            extract_low_coverage_recovery_softfail_min_coverage=getattr(
                settings,
                "extract_low_coverage_recovery_softfail_min_coverage",
                0.55,
            ),
            page_window_size=settings.page_window_size,
            min_page_coverage=settings.extract_min_page_coverage,
            unstructured_targeted_only=settings.unstructured_targeted_only,
            docling_ocr_backend=settings.docling_ocr_backend,
            docling_ocr_fallbacks=settings.docling_ocr_fallbacks,
            docling_ocr_langs_easyocr=settings.docling_ocr_langs_easyocr,
            docling_ocr_langs_tesseract=settings.docling_ocr_langs_tesseract,
            chunking=chunking,
        )

    def discover_files(self, data_dir: str) -> list[str]:
        """Find supported document files recursively."""
        root = Path(data_dir).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(
                f"Data directory not found: {root}. Pass --data-dir or set DATA_DIR to an existing folder."
            )
        index_root = Path(self.settings.index_dir).expanduser().resolve()

        files: list[str] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            # Avoid indexing persisted index artifacts.
            if index_root in path.parents:
                continue
            files.append(str(path))
        return sorted(files)

    @staticmethod
    def _quality_is_reusable(
        quality: dict[str, object] | None,
        *,
        min_page_coverage: float,
    ) -> bool:
        """Return true when prior extraction quality allows safe incremental reuse."""
        if not isinstance(quality, dict):
            return False

        status = str(quality.get("status") or "").strip().lower()
        if status not in {"pass", "soft_fail"}:
            return False

        try:
            total_chunks = int(quality.get("total_chunks") or 0)
        except (TypeError, ValueError):
            total_chunks = 0
        if total_chunks <= 0:
            return False

        try:
            page_coverage = float(quality.get("page_coverage") or 0.0)
        except (TypeError, ValueError):
            page_coverage = 0.0
        return page_coverage >= float(min_page_coverage)

    @staticmethod
    def _quality_payload(
        *,
        status: str,
        page_coverage: float,
        total_chunks: int,
        switch_reason: str | None,
        poisoned_page_ratio: float,
        extractor_used: str,
    ) -> dict[str, object]:
        """Build compact per-file extraction quality payload for incremental reuse."""
        return {
            "status": str(status),
            "page_coverage": float(page_coverage),
            "total_chunks": int(total_chunks),
            "switch_reason": switch_reason,
            "poisoned_page_ratio": float(poisoned_page_ratio),
            "extractor_used": str(extractor_used),
        }

    def build_or_update(
        self,
        data_dir: str,
        preferred_extractor: str = "docling",
        fast_mode: bool = False,
        reset_index: bool = False,
        profile: str | None = None,
    ) -> tuple[HybridIndex, IndexStats]:
        """Build or incrementally update hybrid index from files."""
        existing = HybridIndex() if reset_index else HybridIndex.load(self.settings.index_dir)
        files = self.discover_files(data_dir)

        extraction_reports: list[ExtractionStats] = []

        reused_chunks: list[IndexedChunk] = []
        new_chunks: list[IndexedChunk] = []
        new_path_to_chunk_ids: dict[str, list[str]] = {}
        new_file_hash_by_path: dict[str, str] = {}
        new_file_quality_by_path: dict[str, dict[str, object]] = {}

        old_by_id = existing.chunk_map
        old_path_to_ids = existing.path_to_chunk_ids
        old_file_quality_by_path = existing.file_quality_by_path

        dedup_skipped = 0
        duplicate_files = 0
        failed_files = 0

        seen_file_hashes: set[str] = set()

        for file_path in files:
            abs_path = normalize_source_path(file_path)
            file_hash = sha256_file(abs_path)
            new_file_hash_by_path[abs_path] = file_hash

            if file_hash in seen_file_hashes:
                duplicate_files += 1
            seen_file_hashes.add(file_hash)

            if (
                abs_path in existing.file_hash_by_path
                and existing.file_hash_by_path.get(abs_path) == file_hash
                and abs_path in old_path_to_ids
                and self._quality_is_reusable(
                    old_file_quality_by_path.get(abs_path),
                    min_page_coverage=self.settings.extract_min_page_coverage,
                )
            ):
                ids = old_path_to_ids[abs_path]
                keep = [old_by_id[i] for i in ids if i in old_by_id]
                reused_chunks.extend(keep)
                new_path_to_chunk_ids[abs_path] = [c.chunk_id for c in keep]
                if abs_path in old_file_quality_by_path:
                    new_file_quality_by_path[abs_path] = dict(old_file_quality_by_path[abs_path])
                else:
                    new_file_quality_by_path[abs_path] = self._quality_payload(
                        status="pass" if keep else "hard_fail",
                        page_coverage=1.0 if keep else 0.0,
                        total_chunks=len(keep),
                        switch_reason=None if keep else "missing_quality_meta",
                        poisoned_page_ratio=0.0,
                        extractor_used="reused",
                    )
                continue

            if (
                abs_path in existing.file_hash_by_path
                and existing.file_hash_by_path.get(abs_path) == file_hash
                and abs_path in old_path_to_ids
            ):
                LOGGER.info(
                    "Re-extracting %s despite unchanged hash due to low/unknown prior quality",
                    abs_path,
                )

            try:
                outcome = self.extractor_orchestrator.extract_with_policy(
                    abs_path,
                    preferred=preferred_extractor,
                    fast_mode=fast_mode,
                    profile=profile or self.settings.ingest_profile,
                )
                LOGGER.info(
                    "Extraction done for %s: extractor=%s fallback_path=%s chunks=%s status=%s reason=%s",
                    abs_path,
                    outcome.extractor_used,
                    "->".join(outcome.fallback_path) if outcome.fallback_path else outcome.extractor_used,
                    len(outcome.chunks),
                    outcome.status,
                    outcome.switch_reason,
                )
                extraction_reports.append(outcome.stats)
                new_file_quality_by_path[abs_path] = self._quality_payload(
                    status=outcome.status,
                    page_coverage=float(outcome.stats.page_coverage),
                    total_chunks=int(outcome.stats.total_chunks),
                    switch_reason=outcome.switch_reason,
                    poisoned_page_ratio=float(outcome.stats.poisoned_page_ratio),
                    extractor_used=outcome.extractor_used,
                )
                if outcome.status == "hard_fail":
                    LOGGER.warning(
                        "Extraction hard-failed for %s: reason=%s fallback_path=%s",
                        abs_path,
                        outcome.switch_reason,
                        "->".join(outcome.fallback_path),
                    )
                    failed_files += 1
                    continue
                chunks = outcome.chunks
            except Exception as exc:
                LOGGER.exception("Failed to extract %s: %s", abs_path, exc)
                failed_files += 1
                new_file_quality_by_path[abs_path] = self._quality_payload(
                    status="hard_fail",
                    page_coverage=0.0,
                    total_chunks=0,
                    switch_reason=f"exception:{exc}",
                    poisoned_page_ratio=0.0,
                    extractor_used="n/a",
                )
                continue

            if not chunks:
                failed_files += 1
                new_file_quality_by_path[abs_path] = self._quality_payload(
                    status="hard_fail",
                    page_coverage=0.0,
                    total_chunks=0,
                    switch_reason="chunks==0",
                    poisoned_page_ratio=0.0,
                    extractor_used=outcome.extractor_used,
                )
                continue

            texts = [c.text for c in chunks]
            vectors = self.embed_client.embed_documents(texts)

            file_chunk_ids: list[str] = []
            dedup_content_hashes: set[str] = set()
            for chunk, vec in zip(chunks, vectors):
                content_hash = sha256_text(chunk.text)
                if content_hash in dedup_content_hashes:
                    dedup_skipped += 1
                    continue
                dedup_content_hashes.add(content_hash)

                chunk_id = sha256_text(
                    f"{abs_path}|{file_hash}|{chunk.page}|{chunk.element_type}|{content_hash}"
                )[:24]
                indexed = IndexedChunk(
                    chunk_id=chunk_id,
                    chunk=chunk,
                    tokenized=tokenize(chunk.text),
                    dense_vector=vec,
                    content_hash=content_hash,
                    file_hash=file_hash,
                )
                new_chunks.append(indexed)
                file_chunk_ids.append(chunk_id)

            new_path_to_chunk_ids[abs_path] = file_chunk_ids

        all_chunks = reused_chunks + new_chunks

        index = HybridIndex()
        index.indexed_chunks = all_chunks
        index.path_to_chunk_ids = new_path_to_chunk_ids
        index.file_hash_by_path = new_file_hash_by_path
        index.file_quality_by_path = new_file_quality_by_path
        index.rebuild_runtime()
        index.persist(self.settings.index_dir)

        stats = IndexStats(
            indexed_files=len(files) - failed_files,
            indexed_chunks=len(all_chunks),
            deduplicated_chunks=dedup_skipped,
            duplicate_files=duplicate_files,
            failed_files=failed_files,
            extraction_reports=extraction_reports,
        )
        return index, stats
