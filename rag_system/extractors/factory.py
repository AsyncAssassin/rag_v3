"""Extractor orchestration and fallback policies."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
import io
import math
from multiprocessing import get_context
import os
import pickle
from pathlib import Path
import signal
import tempfile
import time

from ..logging_utils import get_logger
from ..types import DocumentChunk, ExtractionStats
from .base import (
    ChunkingOptions,
    ExtractionQualityThresholds,
    UnsupportedFileTypeError,
    chunk_long_text,
    compute_extraction_stats,
    detect_poisoned_pages,
)
from .docling_extractor import DoclingExtractor, PlainTextExtractor
from .pymupdf4llm_extractor import PyMuPDF4LLMExtractor
from .unstructured_extractor import UnstructuredExtractor


LOGGER = get_logger()


def _looks_like_table_text(text: str) -> bool:
    """Lightweight heuristic: detect table-like page text."""
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < 5:
        return False
    delim_hits = 0
    for ln in lines[:60]:
        if "|" in ln or "\t" in ln:
            delim_hits += 1
            continue
        multi_space = len([p for p in ln.split("  ") if p.strip()])
        if multi_space >= 3:
            delim_hits += 1
    return delim_hits >= 4


def _extract_by_page_windows_impl(file_path: str, window_size: int, chunking: ChunkingOptions) -> list[DocumentChunk]:
    """Fallback extractor implementation: process pages in windows using raw PyMuPDF text."""
    try:
        import fitz  # type: ignore
    except Exception as exc:  # pragma: no cover - import depends on runtime env
        raise RuntimeError("PyMuPDF fallback import failed") from exc

    start_ms = time.perf_counter()
    output: list[DocumentChunk] = []
    doc = fitz.open(file_path)
    try:
        total_pages = int(doc.page_count)
        for window_start in range(0, total_pages, window_size):
            window_end = min(total_pages, window_start + window_size)
            window_tag = f"{window_start + 1}-{window_end}"
            for page_idx in range(window_start, window_end):
                page = doc.load_page(page_idx)
                text = (page.get_text("text") or "").strip()
                if not text:
                    continue
                element_type = "table" if _looks_like_table_text(text) else "text"
                output.extend(
                    chunk_long_text(
                        text=text,
                        source_path=file_path,
                        page=page_idx + 1,
                        element_type=element_type,
                        metadata={
                            "extractor": "pymupdf_window_fallback",
                            "window": window_tag,
                            "window_size": int(window_size),
                            "elapsed_sec": round(time.perf_counter() - start_ms, 3),
                        },
                        chunking=chunking,
                    )
                )
    finally:
        doc.close()

    if not output:
        raise RuntimeError("pymupdf window fallback returned no chunks")
    return output


class _PageWindowExtractor:
    """Pickle-friendly extractor wrapper for page-window fallback."""

    def __init__(self, *, window_size: int, chunking: ChunkingOptions, name: str = "pymupdf_window_fallback") -> None:
        self.window_size = int(window_size)
        self.chunking = chunking
        self.name = name

    def extract(self, file_path: str) -> list[DocumentChunk]:
        """Extract chunks by page windows."""
        return _extract_by_page_windows_impl(file_path=file_path, window_size=self.window_size, chunking=self.chunking)


def _load_subprocess_payload(payload_path: str) -> tuple | None:
    """Read extraction payload produced by subprocess from temp file."""
    if not payload_path or not os.path.exists(payload_path):
        return None
    with contextlib.suppress(OSError):
        if os.path.getsize(payload_path) <= 0:
            return None
    try:
        with open(payload_path, "rb") as fh:
            payload = pickle.load(fh)
        return payload if isinstance(payload, tuple) and payload else None
    except Exception:
        return None


def _run_extractor_in_subprocess(extractor, file_path: str, sender, payload_path: str) -> None:
    """Run extractor inside subprocess and persist payload to file-backed transport."""
    if os.name == "posix":
        with contextlib.suppress(Exception):  # pragma: no cover - platform/process-specific
            os.setsid()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            chunks = extractor.extract(file_path)
            with open(payload_path, "wb") as fh:
                pickle.dump(("ok", chunks), fh, protocol=pickle.HIGHEST_PROTOCOL)
            with contextlib.suppress(BrokenPipeError, OSError):  # parent may stop waiting on timeout
                sender.send(("ok_file", payload_path))
        except Exception as exc:  # pragma: no cover - child process path
            with contextlib.suppress(Exception):
                with open(payload_path, "wb") as fh:
                    pickle.dump(("err", type(exc).__name__, str(exc)), fh, protocol=pickle.HIGHEST_PROTOCOL)
            with contextlib.suppress(BrokenPipeError, OSError):
                sender.send(("err", type(exc).__name__, str(exc)))
        finally:
            with contextlib.suppress(Exception):
                sender.close()


@dataclass(slots=True)
class ExtractionOutcome:
    """Extraction result with stats and selected backend name."""

    chunks: list[DocumentChunk]
    stats: ExtractionStats
    extractor_used: str
    notes: list[str]
    fallback_path: list[str]
    status: str = "pass"
    switch_reason: str | None = None
    page_coverage: float = 1.0
    low_quality_pages: list[int] = field(default_factory=list)
    attempts: list[dict[str, object]] = field(default_factory=list)


class ExtractorOrchestrator:
    """Run extraction policy with deterministic quality-aware routing."""

    def __init__(
        self,
        languages: tuple[str, ...] = ("rus", "eng"),
        quality_thresholds: ExtractionQualityThresholds | None = None,
        extract_timeout_sec: int | None = 180,
        extract_timeout_base_sec: int | None = None,
        extract_timeout_per_100_pages_sec: int = 30,
        extract_timeout_per_10mb_sec: int = 20,
        extract_timeout_max_sec: int = 600,
        extract_full_quality_docling_primary_max_sec: int = 300,
        extract_full_quality_docling_secondary_max_sec: int = 240,
        extract_full_quality_pymupdf_max_sec: int = 180,
        extract_full_quality_reserve_window_fallback_sec: int = 180,
        extract_full_quality_min_stage_start_sec: int = 15,
        extract_full_quality_unstructured_min_remaining_sec: int = 90,
        extract_prefer_best_usable: bool = True,
        extract_best_usable_min_coverage_gap: float = 0.15,
        unstructured_min_merged_chunk_chars: int = 220,
        extract_low_coverage_recovery_enabled: bool = True,
        extract_low_coverage_recovery_trigger_coverage: float = 0.35,
        extract_low_coverage_recovery_batch_pages: int = 120,
        extract_low_coverage_recovery_max_pages: int = 360,
        extract_low_coverage_recovery_softfail_min_coverage: float = 0.55,
        page_window_size: int = 40,
        min_page_coverage: float = 0.85,
        unstructured_targeted_only: bool = True,
        docling_ocr_backend: str = "easyocr",
        docling_ocr_fallbacks: tuple[str, ...] = ("easyocr", "tesseract", "rapidocr", "none"),
        docling_ocr_langs_easyocr: tuple[str, ...] = ("ru", "en"),
        docling_ocr_langs_tesseract: tuple[str, ...] = ("rus", "eng"),
        chunking: ChunkingOptions | None = None,
    ) -> None:
        self.languages = languages
        self.quality_thresholds = quality_thresholds or ExtractionQualityThresholds()
        self.extract_timeout_sec = extract_timeout_sec if (extract_timeout_sec and extract_timeout_sec > 0) else None
        self.extract_timeout_base_sec = (
            extract_timeout_base_sec
            if (extract_timeout_base_sec and extract_timeout_base_sec > 0)
            else self.extract_timeout_sec
        )
        self.extract_timeout_per_100_pages_sec = max(0, int(extract_timeout_per_100_pages_sec))
        self.extract_timeout_per_10mb_sec = max(0, int(extract_timeout_per_10mb_sec))
        self.extract_timeout_max_sec = max(30, int(extract_timeout_max_sec))
        self.extract_full_quality_docling_primary_max_sec = max(
            0,
            int(extract_full_quality_docling_primary_max_sec),
        )
        self.extract_full_quality_docling_secondary_max_sec = max(
            0,
            int(extract_full_quality_docling_secondary_max_sec),
        )
        self.extract_full_quality_pymupdf_max_sec = max(
            0,
            int(extract_full_quality_pymupdf_max_sec),
        )
        self.extract_full_quality_reserve_window_fallback_sec = max(
            0,
            int(extract_full_quality_reserve_window_fallback_sec),
        )
        self.extract_full_quality_min_stage_start_sec = max(
            1,
            int(extract_full_quality_min_stage_start_sec),
        )
        self.extract_full_quality_unstructured_min_remaining_sec = max(
            0,
            int(extract_full_quality_unstructured_min_remaining_sec),
        )
        self.extract_prefer_best_usable = bool(extract_prefer_best_usable)
        self.extract_best_usable_min_coverage_gap = max(0.0, float(extract_best_usable_min_coverage_gap))
        self.unstructured_min_merged_chunk_chars = max(80, int(unstructured_min_merged_chunk_chars))
        self.extract_low_coverage_recovery_enabled = bool(extract_low_coverage_recovery_enabled)
        self.extract_low_coverage_recovery_trigger_coverage = min(
            1.0,
            max(0.0, float(extract_low_coverage_recovery_trigger_coverage)),
        )
        self.extract_low_coverage_recovery_batch_pages = max(1, int(extract_low_coverage_recovery_batch_pages))
        self.extract_low_coverage_recovery_max_pages = max(
            self.extract_low_coverage_recovery_batch_pages,
            int(extract_low_coverage_recovery_max_pages),
        )
        self.page_window_size = max(5, int(page_window_size))
        self.min_page_coverage = float(min_page_coverage)
        self.extract_low_coverage_recovery_softfail_min_coverage = min(
            self.min_page_coverage,
            max(0.0, float(extract_low_coverage_recovery_softfail_min_coverage)),
        )
        self.unstructured_targeted_only = bool(unstructured_targeted_only)
        self.docling_ocr_backend = (docling_ocr_backend or "easyocr").strip().lower()
        self.docling_ocr_fallbacks = tuple(
            (item or "").strip().lower() for item in docling_ocr_fallbacks if (item or "").strip()
        )
        self.docling_ocr_langs_easyocr = tuple(x.strip().lower() for x in docling_ocr_langs_easyocr if x.strip())
        self.docling_ocr_langs_tesseract = tuple(x.strip().lower() for x in docling_ocr_langs_tesseract if x.strip())
        self.chunking = chunking or ChunkingOptions()
        self.max_targeted_pages = max(20, self.page_window_size * 3)

    def extract_with_policy(
        self,
        file_path: str,
        preferred: str = "docling",
        fast_mode: bool = False,
        profile: str | None = None,
        deadline_monotonic: float | None = None,
    ) -> ExtractionOutcome:
        """Extract one file according to configured profile and fallback policy."""
        suffix = Path(file_path).suffix.lower()
        notes: list[str] = []
        profile_name = self._normalize_profile(profile, fast_mode=fast_mode, preferred=preferred)

        if suffix in {".txt", ".csv", ".md"}:
            extractor = PlainTextExtractor(chunking=self.chunking)
            chunks = extractor.extract(file_path)
            stats = compute_extraction_stats(chunks, file_path, extractor.name, total_pages=1)
            stats.status = "pass"
            return ExtractionOutcome(
                chunks=chunks,
                stats=stats,
                extractor_used=extractor.name,
                notes=notes,
                fallback_path=[extractor.name],
                status="pass",
                switch_reason=None,
                page_coverage=float(stats.page_coverage),
                low_quality_pages=[],
                attempts=[
                    {
                        "extractor": extractor.name,
                        "status": "pass",
                        "switch_reason": None,
                        "duration_ms": 0.0,
                        "error": None,
                        "total_chunks": len(chunks),
                        "chars_per_page": round(stats.chars_per_page, 2),
                        "empty_page_ratio": round(stats.empty_page_ratio, 4),
                        "short_chunk_ratio": round(stats.short_chunk_ratio, 4),
                        "page_coverage": round(stats.page_coverage, 4),
                        "low_quality_pages": [],
                        "poisoned_pages": [],
                        "poisoned_page_ratio": 0.0,
                        "poison_signals": {},
                        "ocr_backend_effective": None,
                        "ocr_fallback_path": [],
                    }
                ],
            )

        if suffix != ".pdf":
            raise UnsupportedFileTypeError(f"Unsupported file extension: {suffix}")

        total_pages = self._pdf_page_count(file_path)
        full_scope = list(range(1, max(1, total_pages) + 1))

        if profile_name == "demo-fast":
            return self._run_demo_fast(
                file_path=file_path,
                total_pages=total_pages,
                page_scope=full_scope,
                notes=notes,
                deadline_monotonic=deadline_monotonic,
            )

        if preferred == "unstructured":
            attempt = self._attempt_extractor(
                extractor=UnstructuredExtractor(languages=self.languages, strategy="hi_res", chunking=self.chunking),
                file_path=file_path,
                total_pages=total_pages,
                page_scope=full_scope,
                timeout_override_sec=self._adaptive_timeout_sec(file_path, profile_name),
                extractor_label="unstructured_hi_res",
                deadline_monotonic=deadline_monotonic,
            )
            attempts_public = [self._public_attempt(attempt)]
            fallback_path = [attempt["extractor"]]
            return self._finalize_from_attempt(
                attempt=attempt,
                notes=notes,
                fallback_path=fallback_path,
                attempts_public=attempts_public,
                fallback_status=attempt["status"],
            )

        return self._run_full_quality(
            file_path=file_path,
            total_pages=total_pages,
            page_scope=full_scope,
            notes=notes,
            profile_name=profile_name,
            deadline_monotonic=deadline_monotonic,
        )

    def _run_demo_fast(
        self,
        *,
        file_path: str,
        total_pages: int,
        page_scope: list[int],
        notes: list[str],
        deadline_monotonic: float | None = None,
    ) -> ExtractionOutcome:
        """demo-fast profile: pymupdf4llm -> page-window fallback."""
        attempts_public: list[dict[str, object]] = []
        fallback_path: list[str] = []

        first = self._attempt_extractor(
            extractor=PyMuPDF4LLMExtractor(chunking=self.chunking),
            file_path=file_path,
            total_pages=total_pages,
            page_scope=page_scope,
            timeout_override_sec=self._adaptive_timeout_sec(file_path, "demo-fast"),
            extractor_label="pymupdf4llm",
            deadline_monotonic=deadline_monotonic,
        )
        attempts_public.append(self._public_attempt(first))
        fallback_path.append("pymupdf4llm")

        if first["status"] == "pass":
            return self._finalize_from_attempt(
                attempt=first,
                notes=notes,
                fallback_path=fallback_path,
                attempts_public=attempts_public,
                fallback_status="pass",
            )

        notes.append(f"pymupdf4llm: {first['switch_reason'] or 'quality threshold not met'}")

        second = self._attempt_window_fallback(
            file_path=file_path,
            total_pages=total_pages,
            page_scope=page_scope,
            extractor_label="pymupdf_window_fallback",
            timeout_override_sec=self._adaptive_timeout_sec(file_path, "demo-fast"),
            deadline_monotonic=deadline_monotonic,
        )
        attempts_public.append(self._public_attempt(second))
        fallback_path.append("pymupdf_window_fallback")

        fallback_status = "pass" if second["status"] == "pass" else "soft_fail"
        return self._finalize_from_attempt(
            attempt=second,
            notes=notes,
            fallback_path=fallback_path,
            attempts_public=attempts_public,
            fallback_status=fallback_status,
        )

    def _run_full_quality(
        self,
        *,
        file_path: str,
        total_pages: int,
        page_scope: list[int],
        notes: list[str],
        profile_name: str,
        deadline_monotonic: float | None = None,
    ) -> ExtractionOutcome:
        """full-quality profile routing.

        Order:
        docling(easyocr) -> docling(rapidocr) -> pymupdf4llm -> unstructured_hi_res(targeted) -> page-window fallback
        """
        attempts_public: list[dict[str, object]] = []
        fallback_path: list[str] = []
        best_soft: dict | None = None
        usable_attempts: list[dict] = []
        first_targeted_pages: list[int] = []
        merge_attempt: dict | None = None
        adaptive_timeout = self._adaptive_timeout_sec(file_path, profile_name)
        reserve_window_sec = self.extract_full_quality_reserve_window_fallback_sec

        # 1-2) Docling passes with selected OCR backends.
        docling_backends = self._docling_backend_order()[:2]
        for idx, backend in enumerate(docling_backends):
            label = f"docling_{backend}"
            docling_attempt = self._attempt_extractor(
                extractor=DoclingExtractor(
                    languages=self.languages,
                    full_page_ocr=False,
                    ocr_backend=backend,
                    ocr_fallbacks=(backend, "none"),
                    easyocr_langs=self.docling_ocr_langs_easyocr,
                    tesseract_langs=self.docling_ocr_langs_tesseract,
                    chunking=self.chunking,
                ),
                file_path=file_path,
                total_pages=total_pages,
                page_scope=page_scope,
                timeout_override_sec=adaptive_timeout,
                extractor_label=label,
                stage_name=label,
                stage_timeout_cap_sec=(
                    self.extract_full_quality_docling_primary_max_sec
                    if idx == 0
                    else self.extract_full_quality_docling_secondary_max_sec
                ),
                reserved_tail_budget_sec=reserve_window_sec,
                deadline_monotonic=deadline_monotonic,
            )
            attempts_public.append(self._public_attempt(docling_attempt))
            fallback_path.append(label)
            if docling_attempt["status"] == "pass":
                return self._finalize_from_attempt(
                    attempt=docling_attempt,
                    notes=notes,
                    fallback_path=fallback_path,
                    attempts_public=attempts_public,
                    fallback_status="pass",
                )
            notes.append(f"{label}: {docling_attempt['switch_reason'] or 'failed'}")
            if docling_attempt["chunks"]:
                best_soft = docling_attempt
            if self._is_usable_attempt(docling_attempt):
                usable_attempts.append(docling_attempt)
            if idx == 0 and self._is_timeout_without_chunks(docling_attempt):
                notes.append("docling_secondary skipped: primary timed out with empty output")
                break

        # 3) Operational fallback to pymupdf4llm.
        pymupdf = self._attempt_extractor(
            extractor=PyMuPDF4LLMExtractor(chunking=self.chunking),
            file_path=file_path,
            total_pages=total_pages,
            page_scope=page_scope,
            timeout_override_sec=adaptive_timeout,
            extractor_label="pymupdf4llm",
            stage_name="pymupdf4llm",
            stage_timeout_cap_sec=self.extract_full_quality_pymupdf_max_sec,
            reserved_tail_budget_sec=reserve_window_sec,
            deadline_monotonic=deadline_monotonic,
        )
        attempts_public.append(self._public_attempt(pymupdf))
        fallback_path.append("pymupdf4llm")
        if pymupdf["status"] == "pass":
            return self._finalize_from_attempt(
                attempt=pymupdf,
                notes=notes,
                fallback_path=fallback_path,
                attempts_public=attempts_public,
                fallback_status="pass",
            )
        notes.append(f"pymupdf4llm: {pymupdf['switch_reason'] or 'quality threshold not met'}")
        if pymupdf["chunks"]:
            best_soft = pymupdf
        if self._is_usable_attempt(pymupdf):
            usable_attempts.append(pymupdf)

        # 4) Targeted hi_res fallback only for low-quality pages.
        target_pages: list[int] = []
        if best_soft and best_soft["low_quality_pages"]:
            target_pages = list(best_soft["low_quality_pages"])[: self.max_targeted_pages]
            first_targeted_pages = list(target_pages)

        remaining_before_unstructured = self._remaining_budget_sec(deadline_monotonic)
        run_unstructured = bool(target_pages) and (
            remaining_before_unstructured is None
            or remaining_before_unstructured >= float(self.extract_full_quality_unstructured_min_remaining_sec)
        )
        if run_unstructured:
            page_scope_for_unstructured = target_pages if target_pages else page_scope
            unstructured = self._attempt_extractor(
                extractor=UnstructuredExtractor(
                    languages=self.languages,
                    strategy="hi_res",
                    chunking=self.chunking,
                    target_pages=target_pages if target_pages else None,
                ),
                file_path=file_path,
                total_pages=total_pages,
                page_scope=page_scope_for_unstructured,
                timeout_override_sec=adaptive_timeout,
                extractor_label=("unstructured_hi_res_targeted" if target_pages else "unstructured_hi_res"),
                stage_name=("unstructured_hi_res_targeted" if target_pages else "unstructured_hi_res"),
                reserved_tail_budget_sec=reserve_window_sec,
                deadline_monotonic=deadline_monotonic,
            )
            attempts_public.append(self._public_attempt(unstructured))
            fallback_path.append("unstructured_hi_res_targeted" if target_pages else "unstructured_hi_res")

            if unstructured["status"] == "pass" and target_pages and best_soft and best_soft["chunks"]:
                merged_chunks = self._merge_targeted_chunks(best_soft["chunks"], unstructured["chunks"], target_pages)
                merged_stats = compute_extraction_stats(
                    merged_chunks,
                    file_path,
                    "pymupdf4llm+unstructured_hi_res_targeted",
                    total_pages=total_pages,
                )
                (
                    merged_status,
                    merged_reason,
                    merged_low_pages,
                    merged_cov,
                    merged_poisoned_pages,
                    merged_poisoned_ratio,
                    merged_poison_signals,
                ) = self._evaluate_routing(
                    merged_stats,
                    merged_chunks,
                    page_scope=page_scope,
                )
                merged_stats.poisoned_pages = list(merged_poisoned_pages)
                merged_stats.poisoned_page_ratio = float(merged_poisoned_ratio)
                merged_stats.poison_signals = dict(merged_poison_signals)
                merge_attempt = {
                    "extractor": "pymupdf4llm+unstructured_hi_res_targeted",
                    "status": merged_status,
                    "switch_reason": merged_reason,
                    "duration_ms": 0.0,
                    "error": None,
                    "total_chunks": int(merged_stats.total_chunks),
                    "chars_per_page": float(merged_stats.chars_per_page),
                    "empty_page_ratio": float(merged_stats.empty_page_ratio),
                    "short_chunk_ratio": float(merged_stats.short_chunk_ratio),
                    "page_coverage": float(merged_cov),
                    "low_quality_pages": list(merged_low_pages),
                    "poisoned_pages": list(merged_poisoned_pages),
                    "poisoned_page_ratio": float(merged_poisoned_ratio),
                    "poison_signals": dict(merged_poison_signals),
                    "ocr_backend_effective": None,
                    "ocr_fallback_path": [],
                    "remaining_budget_sec_before_attempt": None,
                    "effective_timeout_sec": None,
                    "chunks": merged_chunks,
                    "stats": merged_stats,
                    "recovery_mode": None,
                    "recovery_batches_run": 0,
                    "coverage_after_batch": [],
                    "targeted_pages_processed": [],
                }
                attempts_public.append(self._public_attempt(merge_attempt))
                fallback_path.append("targeted_merge")
                if merge_attempt["status"] == "pass":
                    return self._finalize_from_attempt(
                        attempt=merge_attempt,
                        notes=notes,
                        fallback_path=fallback_path,
                        attempts_public=attempts_public,
                        fallback_status="pass",
                    )
                notes.append(f"targeted_merge: {merge_attempt['switch_reason'] or 'quality threshold not met'}")
                best_soft = merge_attempt
                if self._is_usable_attempt(merge_attempt):
                    usable_attempts.append(merge_attempt)
            elif unstructured["status"] == "pass":
                return self._finalize_from_attempt(
                    attempt=unstructured,
                    notes=notes,
                    fallback_path=fallback_path,
                    attempts_public=attempts_public,
                    fallback_status="pass",
                )
            else:
                notes.append(f"{unstructured['extractor']}: {unstructured['switch_reason'] or 'failed'}")
                if unstructured["chunks"] and not best_soft:
                    best_soft = unstructured
                if self._is_usable_attempt(unstructured):
                    usable_attempts.append(unstructured)
        else:
            if target_pages:
                notes.append("Skipping unstructured_hi_res_targeted: insufficient remaining budget")
            else:
                notes.append("Skipping unstructured_hi_res_targeted: no low_quality_pages for targeted fallback")

        # 4.5) Recovery mode for low-coverage hard-fail (2020-like PDFs).
        recovery_seed: dict | None = None
        if self.extract_low_coverage_recovery_enabled:
            candidates = [merge_attempt, best_soft, pymupdf]
            for candidate in candidates:
                if candidate is None:
                    continue
                if self._is_low_coverage_recovery_trigger(candidate):
                    recovery_seed = candidate
                    break
        if recovery_seed is not None:
            targeted_pages_for_recovery = first_targeted_pages if recovery_seed is merge_attempt else []
            recovered_attempt = self._run_low_coverage_recovery(
                file_path=file_path,
                total_pages=total_pages,
                page_scope=page_scope,
                adaptive_timeout=adaptive_timeout,
                reserve_window_sec=reserve_window_sec,
                deadline_monotonic=deadline_monotonic,
                base_attempt=recovery_seed,
                targeted_pages_already=targeted_pages_for_recovery,
                notes=notes,
                attempts_public=attempts_public,
                fallback_path=fallback_path,
            )
            if recovered_attempt is not None:
                best_soft = recovered_attempt
                if self._is_usable_attempt(recovered_attempt):
                    usable_attempts.append(recovered_attempt)
                if recovered_attempt["status"] == "pass":
                    return self._finalize_from_attempt(
                        attempt=recovered_attempt,
                        notes=notes,
                        fallback_path=fallback_path,
                        attempts_public=attempts_public,
                        fallback_status="pass",
                    )

        # 5) Final operational fallback by page windows.
        window = self._attempt_window_fallback(
            file_path=file_path,
            total_pages=total_pages,
            page_scope=page_scope,
            extractor_label="pymupdf_window_fallback",
            timeout_override_sec=adaptive_timeout,
            stage_name="pymupdf_window_fallback",
            deadline_monotonic=deadline_monotonic,
        )
        if self._is_pragmatic_low_coverage_hard_fail(window):
            window["status"] = "soft_fail"
            window["downgraded_to_soft_fail_due_to_low_coverage"] = True
            notes.append("Using pragmatic soft-fail fallback result: low coverage with usable chunks")
        if self._is_usable_attempt(window):
            usable_attempts.append(window)
        attempts_public.append(self._public_attempt(window))
        fallback_path.append("pymupdf_window_fallback")

        if window["status"] == "pass":
            return self._finalize_from_attempt(
                attempt=window,
                notes=notes,
                fallback_path=fallback_path,
                attempts_public=attempts_public,
                fallback_status="pass",
            )

        if self.extract_prefer_best_usable:
            best_usable = self._select_best_usable_attempt(
                usable_attempts,
                min_coverage_gap=float(self.extract_best_usable_min_coverage_gap),
            )
        else:
            best_usable = None
        if best_usable is None and window["status"] == "soft_fail" and window["chunks"]:
            best_usable = window
        if best_usable is None and best_soft and best_soft["chunks"]:
            best_usable = best_soft

        if best_usable and best_usable["chunks"]:
            chosen_name = str(best_usable.get("extractor") or "")
            if chosen_name != "pymupdf_window_fallback":
                notes.append(f"Using best usable extraction result: {chosen_name}")
            elif window["status"] == "soft_fail":
                notes.append("Using soft-fail page-window fallback result")
            fallback_status = "pass" if best_usable.get("status") == "pass" else "soft_fail"
            return self._finalize_from_attempt(
                attempt=best_usable,
                notes=notes,
                fallback_path=fallback_path,
                attempts_public=attempts_public,
                fallback_status=fallback_status,
            )

        return self._finalize_from_attempt(
            attempt=window,
            notes=notes,
            fallback_path=fallback_path,
            attempts_public=attempts_public,
            fallback_status="hard_fail",
        )

    def _attempt_extractor(
        self,
        *,
        extractor,
        file_path: str,
        total_pages: int,
        page_scope: list[int],
        timeout_override_sec: int | None,
        extractor_label: str,
        stage_name: str | None = None,
        stage_timeout_cap_sec: int | None = None,
        reserved_tail_budget_sec: int = 0,
        deadline_monotonic: float | None = None,
    ) -> dict:
        """Run one extraction attempt and evaluate routing criteria."""
        started = time.perf_counter()
        stage = (stage_name or extractor_label or "unknown_stage").strip() or "unknown_stage"
        remaining_sec = self._remaining_budget_sec(deadline_monotonic)
        remaining_before_attempt = (
            round(max(0.0, float(remaining_sec)), 2)
            if remaining_sec is not None
            else None
        )
        budget_sec = remaining_sec
        if budget_sec is not None:
            budget_sec = budget_sec - float(max(0, int(reserved_tail_budget_sec)))

        effective_override = timeout_override_sec
        if stage_timeout_cap_sec and int(stage_timeout_cap_sec) > 0:
            stage_cap = int(stage_timeout_cap_sec)
            effective_override = stage_cap if effective_override is None else min(int(effective_override), stage_cap)

        timeout_sec = self._resolve_timeout_sec(
            effective_override,
            remaining_sec=budget_sec if budget_sec is not None else remaining_sec,
        )
        min_start_sec = int(self.extract_full_quality_min_stage_start_sec)
        if timeout_sec is None:
            if budget_sec is not None and budget_sec < float(min_start_sec):
                timeout_sec = 0
        elif timeout_sec < min_start_sec:
            timeout_sec = 0

        if timeout_sec is not None and timeout_sec <= 0:
            return self._before_attempt_timeout(
                extractor_label=extractor_label,
                stage_name=stage,
                started=started,
                page_scope=page_scope,
                error=(
                    "wall-clock deadline exceeded before extractor attempt"
                    if budget_sec is None or budget_sec <= 0
                    else (
                        "insufficient stage budget before extractor attempt "
                        f"(available={round(max(0.0, budget_sec), 2)}s reserve={int(max(0, reserved_tail_budget_sec))}s)"
                    )
                ),
                remaining_budget_sec_before_attempt=remaining_before_attempt,
                effective_timeout_sec=timeout_sec,
            )
        try:
            chunks = self._extract_single(
                extractor=extractor,
                file_path=file_path,
                timeout_override_sec=timeout_sec,
                deadline_monotonic=deadline_monotonic,
            )
            if extractor_label == "unstructured_hi_res_targeted" or extractor_label.startswith(
                "unstructured_hi_res_recovery"
            ):
                chunks = self._normalize_unstructured_targeted_chunks(chunks)
            scope_size = max(1, len(page_scope))
            stats = compute_extraction_stats(
                chunks,
                file_path,
                extractor_label,
                total_pages=scope_size,
            )
            (
                status,
                switch_reason,
                low_quality_pages,
                page_coverage,
                poisoned_pages,
                poisoned_page_ratio,
                poison_signals,
            ) = self._evaluate_routing(
                stats,
                chunks,
                page_scope=page_scope,
            )
            stats.status = status
            stats.switch_reason = switch_reason
            stats.low_quality_pages = list(low_quality_pages)
            stats.page_coverage = float(page_coverage)
            stats.poisoned_pages = list(poisoned_pages)
            stats.poisoned_page_ratio = float(poisoned_page_ratio)
            stats.poison_signals = dict(poison_signals)

            ocr_backend_effective = self._extract_metadata_scalar(chunks, "ocr_backend_effective")
            ocr_fallback_path = self._extract_metadata_list(chunks, "ocr_fallback_path")

            return {
                "extractor": extractor_label,
                "status": status,
                "switch_reason": switch_reason,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "error": None,
                "total_chunks": int(stats.total_chunks),
                "chars_per_page": float(stats.chars_per_page),
                "empty_page_ratio": float(stats.empty_page_ratio),
                "short_chunk_ratio": float(stats.short_chunk_ratio),
                "page_coverage": float(page_coverage),
                "low_quality_pages": list(low_quality_pages),
                "poisoned_pages": list(poisoned_pages),
                "poisoned_page_ratio": float(poisoned_page_ratio),
                "poison_signals": dict(poison_signals),
                "ocr_backend_effective": ocr_backend_effective,
                "ocr_fallback_path": ocr_fallback_path,
                "remaining_budget_sec_before_attempt": remaining_before_attempt,
                "effective_timeout_sec": timeout_sec,
                "ipc_transport": "file",
                "chunks": chunks,
                "stats": stats,
            }
        except Exception as exc:
            reason = "timeout" if isinstance(exc, TimeoutError) else "exception"
            return {
                "extractor": extractor_label,
                "status": "hard_fail",
                "switch_reason": f"{reason}:{exc}",
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "error": str(exc),
                "total_chunks": 0,
                "chars_per_page": 0.0,
                "empty_page_ratio": 1.0,
                "short_chunk_ratio": 1.0,
                "page_coverage": 0.0,
                "low_quality_pages": list(page_scope),
                "poisoned_pages": [],
                "poisoned_page_ratio": 0.0,
                "poison_signals": {},
                "ocr_backend_effective": None,
                "ocr_fallback_path": [],
                "remaining_budget_sec_before_attempt": remaining_before_attempt,
                "effective_timeout_sec": timeout_sec,
                "ipc_transport": "file",
                "chunks": [],
                "stats": None,
            }

    def _attempt_window_fallback(
        self,
        *,
        file_path: str,
        total_pages: int,
        page_scope: list[int],
        extractor_label: str,
        timeout_override_sec: int | None,
        stage_name: str | None = None,
        stage_timeout_cap_sec: int | None = None,
        reserved_tail_budget_sec: int = 0,
        deadline_monotonic: float | None = None,
    ) -> dict:
        """Run page-window fallback extractor and evaluate routing criteria."""
        started = time.perf_counter()
        stage = (stage_name or extractor_label or "unknown_stage").strip() or "unknown_stage"
        remaining_sec = self._remaining_budget_sec(deadline_monotonic)
        remaining_before_attempt = (
            round(max(0.0, float(remaining_sec)), 2)
            if remaining_sec is not None
            else None
        )
        budget_sec = remaining_sec
        if budget_sec is not None:
            budget_sec = budget_sec - float(max(0, int(reserved_tail_budget_sec)))

        effective_override = timeout_override_sec
        if stage_timeout_cap_sec and int(stage_timeout_cap_sec) > 0:
            stage_cap = int(stage_timeout_cap_sec)
            effective_override = stage_cap if effective_override is None else min(int(effective_override), stage_cap)
        timeout_sec = self._resolve_timeout_sec(
            effective_override,
            remaining_sec=budget_sec if budget_sec is not None else remaining_sec,
        )
        min_start_sec = int(self.extract_full_quality_min_stage_start_sec)
        if timeout_sec is None:
            if budget_sec is not None and budget_sec < float(min_start_sec):
                timeout_sec = 0
        elif timeout_sec < min_start_sec:
            timeout_sec = 0

        if timeout_sec is not None and timeout_sec <= 0:
            return self._before_attempt_timeout(
                extractor_label=extractor_label,
                stage_name=stage,
                started=started,
                page_scope=page_scope,
                error=(
                    "wall-clock deadline exceeded before extractor attempt"
                    if budget_sec is None or budget_sec <= 0
                    else (
                        "insufficient stage budget before extractor attempt "
                        f"(available={round(max(0.0, budget_sec), 2)}s reserve={int(max(0, reserved_tail_budget_sec))}s)"
                    )
                ),
                remaining_budget_sec_before_attempt=remaining_before_attempt,
                effective_timeout_sec=timeout_sec,
            )
        try:
            chunks = self._extract_single(
                extractor=_PageWindowExtractor(
                    window_size=self.page_window_size,
                    chunking=self.chunking,
                    name=extractor_label,
                ),
                file_path=file_path,
                timeout_override_sec=timeout_sec,
                deadline_monotonic=deadline_monotonic,
            )
            stats = compute_extraction_stats(
                chunks,
                file_path,
                extractor_label,
                total_pages=max(1, total_pages),
            )
            (
                status,
                switch_reason,
                low_quality_pages,
                page_coverage,
                poisoned_pages,
                poisoned_page_ratio,
                poison_signals,
            ) = self._evaluate_routing(
                stats,
                chunks,
                page_scope=page_scope,
            )
            stats.status = status
            stats.switch_reason = switch_reason
            stats.low_quality_pages = list(low_quality_pages)
            stats.page_coverage = float(page_coverage)
            stats.poisoned_pages = list(poisoned_pages)
            stats.poisoned_page_ratio = float(poisoned_page_ratio)
            stats.poison_signals = dict(poison_signals)
            return {
                "extractor": extractor_label,
                "status": status,
                "switch_reason": switch_reason,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "error": None,
                "total_chunks": int(stats.total_chunks),
                "chars_per_page": float(stats.chars_per_page),
                "empty_page_ratio": float(stats.empty_page_ratio),
                "short_chunk_ratio": float(stats.short_chunk_ratio),
                "page_coverage": float(page_coverage),
                "low_quality_pages": list(low_quality_pages),
                "poisoned_pages": list(poisoned_pages),
                "poisoned_page_ratio": float(poisoned_page_ratio),
                "poison_signals": dict(poison_signals),
                "ocr_backend_effective": None,
                "ocr_fallback_path": [],
                "remaining_budget_sec_before_attempt": remaining_before_attempt,
                "effective_timeout_sec": timeout_sec,
                "ipc_transport": "file",
                "chunks": chunks,
                "stats": stats,
            }
        except Exception as exc:
            reason = "timeout" if isinstance(exc, TimeoutError) else "exception"
            return {
                "extractor": extractor_label,
                "status": "hard_fail",
                "switch_reason": f"{reason}:{exc}",
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "error": str(exc),
                "total_chunks": 0,
                "chars_per_page": 0.0,
                "empty_page_ratio": 1.0,
                "short_chunk_ratio": 1.0,
                "page_coverage": 0.0,
                "low_quality_pages": list(page_scope),
                "poisoned_pages": [],
                "poisoned_page_ratio": 0.0,
                "poison_signals": {},
                "ocr_backend_effective": None,
                "ocr_fallback_path": [],
                "remaining_budget_sec_before_attempt": remaining_before_attempt,
                "effective_timeout_sec": timeout_sec,
                "ipc_transport": "file",
                "chunks": [],
                "stats": None,
            }

    @staticmethod
    def _before_attempt_timeout(
        *,
        extractor_label: str,
        stage_name: str,
        started: float,
        page_scope: list[int],
        error: str,
        remaining_budget_sec_before_attempt: float | None,
        effective_timeout_sec: int | None,
    ) -> dict:
        """Build standardized payload for stage skipped due exhausted budget."""
        return {
            "extractor": extractor_label,
            "status": "hard_fail",
            "switch_reason": f"timeout:wall_clock_deadline_exceeded_before_attempt:{stage_name}",
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "error": error,
            "total_chunks": 0,
            "chars_per_page": 0.0,
            "empty_page_ratio": 1.0,
            "short_chunk_ratio": 1.0,
            "page_coverage": 0.0,
            "low_quality_pages": list(page_scope),
            "poisoned_pages": [],
            "poisoned_page_ratio": 0.0,
            "poison_signals": {},
            "ocr_backend_effective": None,
            "ocr_fallback_path": [],
            "remaining_budget_sec_before_attempt": remaining_budget_sec_before_attempt,
            "effective_timeout_sec": effective_timeout_sec,
            "ipc_transport": "file",
            "chunks": [],
            "stats": None,
        }

    @staticmethod
    def _is_deadline_timeout(attempt: dict) -> bool:
        """Return true when attempt failed due to deadline/timeout."""
        reason = str(attempt.get("switch_reason") or "").lower()
        return (
            attempt.get("status") == "hard_fail"
            and reason.startswith("timeout:")
            and ("wall_clock_deadline" in reason or "deadline reached" in reason)
        )

    @staticmethod
    def _is_timeout_without_chunks(attempt: dict) -> bool:
        """Return true when attempt timed out and produced no chunks."""
        reason = str(attempt.get("switch_reason") or "").lower()
        chunks = attempt.get("chunks") or []
        total_chunks = attempt.get("total_chunks")
        empty_total = False
        try:
            empty_total = int(total_chunks or 0) <= 0
        except (TypeError, ValueError):
            empty_total = True
        return (
            str(attempt.get("status")) == "hard_fail"
            and reason.startswith("timeout:")
            and not bool(chunks)
            and empty_total
        )

    @staticmethod
    def _is_pragmatic_low_coverage_hard_fail(attempt: dict) -> bool:
        """Return true when hard-fail is caused only by low page coverage with usable chunks."""
        if str(attempt.get("status")) != "hard_fail":
            return False
        chunks = attempt.get("chunks") or []
        try:
            total_chunks = int(attempt.get("total_chunks") or len(chunks))
        except (TypeError, ValueError):
            total_chunks = len(chunks)
        if total_chunks <= 0:
            return False

        reason = str(attempt.get("switch_reason") or "")
        if not reason:
            return False
        lower_reason = reason.lower()
        if lower_reason.startswith("timeout:") or lower_reason.startswith("exception:"):
            return False
        if "poisoned_text_ratio" in lower_reason or "chunks==0" in lower_reason:
            return False

        reason_parts = [part.strip() for part in reason.split(";") if part.strip()]
        if not reason_parts:
            return False
        return all(part.startswith("page_coverage<") for part in reason_parts)

    def _is_low_coverage_recovery_trigger(self, attempt: dict | None) -> bool:
        """Return true when attempt should trigger low-coverage recovery batches."""
        if not isinstance(attempt, dict):
            return False
        if not self._is_pragmatic_low_coverage_hard_fail(attempt):
            return False
        try:
            coverage = float(attempt.get("page_coverage") or 0.0)
        except (TypeError, ValueError):
            coverage = 0.0
        return coverage < float(self.extract_low_coverage_recovery_trigger_coverage)

    def _apply_recovery_soft_fail_downgrade(self, attempt: dict, notes: list[str]) -> None:
        """Downgrade low-coverage hard-fail to soft-fail for recovery output when coverage is usable."""
        if not self._is_pragmatic_low_coverage_hard_fail(attempt):
            attempt["downgraded_to_soft_fail_due_to_recovery_low_coverage"] = False
            return
        try:
            coverage = float(attempt.get("page_coverage") or 0.0)
        except (TypeError, ValueError):
            coverage = 0.0
        if coverage < float(self.extract_low_coverage_recovery_softfail_min_coverage):
            attempt["downgraded_to_soft_fail_due_to_recovery_low_coverage"] = False
            return
        attempt["status"] = "soft_fail"
        attempt["downgraded_to_soft_fail_due_to_recovery_low_coverage"] = True
        notes.append(
            "Recovery downgraded low-coverage hard-fail to soft-fail "
            f"(coverage={round(coverage, 4)} >= {round(float(self.extract_low_coverage_recovery_softfail_min_coverage), 4)})"
        )

    @staticmethod
    def _is_usable_attempt(attempt: dict) -> bool:
        """Return true when attempt produced non-empty usable output."""
        if str(attempt.get("status")) not in {"pass", "soft_fail"}:
            return False
        chunks = attempt.get("chunks") or []
        if not chunks:
            return False
        try:
            return int(attempt.get("total_chunks") or len(chunks)) > 0
        except (TypeError, ValueError):
            return bool(chunks)

    @staticmethod
    def _attempt_quality_key(attempt: dict) -> tuple[float, float, float, int]:
        """Build deterministic quality-order key for usable attempts."""
        page_coverage = float(attempt.get("page_coverage") or 0.0)
        short_chunk_ratio = float(attempt.get("short_chunk_ratio") or 1.0)
        chars_per_page = float(attempt.get("chars_per_page") or 0.0)
        try:
            total_chunks = int(attempt.get("total_chunks") or 0)
        except (TypeError, ValueError):
            total_chunks = 0
        return (page_coverage, -short_chunk_ratio, chars_per_page, total_chunks)

    def _select_best_usable_attempt(
        self,
        attempts: list[dict],
        *,
        min_coverage_gap: float,
    ) -> dict | None:
        """Select best usable attempt by quality-order with window fallback guard."""
        usable = [attempt for attempt in attempts if self._is_usable_attempt(attempt)]
        if not usable:
            return None

        best = max(usable, key=self._attempt_quality_key)
        best_name = str(best.get("extractor") or "")
        if "window_fallback" not in best_name:
            return best

        window_coverage = float(best.get("page_coverage") or 0.0)
        non_window = [a for a in usable if "window_fallback" not in str(a.get("extractor") or "")]
        if not non_window:
            return best

        competitor = max(non_window, key=self._attempt_quality_key)
        competitor_coverage = float(competitor.get("page_coverage") or 0.0)
        if competitor_coverage >= window_coverage + float(min_coverage_gap):
            return competitor
        return best

    def _finalize_from_attempt(
        self,
        *,
        attempt: dict,
        notes: list[str],
        fallback_path: list[str],
        attempts_public: list[dict[str, object]],
        fallback_status: str,
    ) -> ExtractionOutcome:
        """Build ExtractionOutcome from attempt payload."""
        stats = attempt.get("stats")
        if stats is None:
            stats = ExtractionStats(
                source_path="",
                total_chunks=0,
                total_chars=0,
                pages_seen=0,
                chars_per_page=0.0,
                empty_page_ratio=1.0,
                short_chunk_ratio=1.0,
                has_table_elements=False,
                extractor_name=str(attempt.get("extractor", "n/a")),
                total_pages=1,
                page_coverage=0.0,
                status=fallback_status,
                switch_reason=str(attempt.get("switch_reason")),
                low_quality_pages=list(attempt.get("low_quality_pages") or []),
                fallback_path=list(fallback_path),
                attempts=list(attempts_public),
                poisoned_pages=list(attempt.get("poisoned_pages") or []),
                poisoned_page_ratio=float(attempt.get("poisoned_page_ratio") or 0.0),
                poison_signals=dict(attempt.get("poison_signals") or {}),
            )
        else:
            stats.status = fallback_status
            stats.switch_reason = str(attempt.get("switch_reason")) if attempt.get("switch_reason") else None
            stats.low_quality_pages = list(attempt.get("low_quality_pages") or [])
            stats.fallback_path = list(fallback_path)
            stats.attempts = list(attempts_public)
            stats.page_coverage = float(attempt.get("page_coverage") or 0.0)
            stats.ocr_backend_effective = attempt.get("ocr_backend_effective")
            stats.ocr_fallback_path = list(attempt.get("ocr_fallback_path") or [])
            stats.poisoned_pages = list(attempt.get("poisoned_pages") or [])
            stats.poisoned_page_ratio = float(attempt.get("poisoned_page_ratio") or 0.0)
            stats.poison_signals = dict(attempt.get("poison_signals") or {})

        return ExtractionOutcome(
            chunks=list(attempt.get("chunks") or []),
            stats=stats,
            extractor_used=str(attempt.get("extractor") or "n/a"),
            notes=list(notes),
            fallback_path=list(fallback_path),
            status=fallback_status,
            switch_reason=stats.switch_reason,
            page_coverage=float(stats.page_coverage),
            low_quality_pages=list(stats.low_quality_pages),
            attempts=list(attempts_public),
        )

    @staticmethod
    def _public_attempt(attempt: dict) -> dict[str, object]:
        """Strip internal payload fields from attempt for trace/logging."""
        return {
            "extractor": attempt.get("extractor"),
            "status": attempt.get("status"),
            "switch_reason": attempt.get("switch_reason"),
            "duration_ms": attempt.get("duration_ms"),
            "error": attempt.get("error"),
            "total_chunks": attempt.get("total_chunks"),
            "chars_per_page": round(float(attempt.get("chars_per_page") or 0.0), 2),
            "empty_page_ratio": round(float(attempt.get("empty_page_ratio") or 0.0), 4),
            "short_chunk_ratio": round(float(attempt.get("short_chunk_ratio") or 0.0), 4),
            "page_coverage": round(float(attempt.get("page_coverage") or 0.0), 4),
            "low_quality_pages": list(attempt.get("low_quality_pages") or []),
            "poisoned_pages": list(attempt.get("poisoned_pages") or []),
            "poisoned_page_ratio": round(float(attempt.get("poisoned_page_ratio") or 0.0), 4),
            "poison_signals": dict(attempt.get("poison_signals") or {}),
            "ocr_backend_effective": attempt.get("ocr_backend_effective"),
            "ocr_fallback_path": list(attempt.get("ocr_fallback_path") or []),
            "remaining_budget_sec_before_attempt": (
                round(float(attempt.get("remaining_budget_sec_before_attempt")), 2)
                if attempt.get("remaining_budget_sec_before_attempt") is not None
                else None
            ),
            "effective_timeout_sec": attempt.get("effective_timeout_sec"),
            "ipc_transport": attempt.get("ipc_transport") or "file",
            "downgraded_to_soft_fail_due_to_low_coverage": bool(
                attempt.get("downgraded_to_soft_fail_due_to_low_coverage")
            ),
            "downgraded_to_soft_fail_due_to_recovery_low_coverage": bool(
                attempt.get("downgraded_to_soft_fail_due_to_recovery_low_coverage")
            ),
            "recovery_mode": attempt.get("recovery_mode"),
            "recovery_batches_run": int(attempt.get("recovery_batches_run") or 0),
            "coverage_after_batch": [
                round(float(x), 4) for x in (attempt.get("coverage_after_batch") or []) if x is not None
            ],
            "targeted_pages_processed": [
                int(x) for x in (attempt.get("targeted_pages_processed") or []) if str(x).strip()
            ],
        }

    def _extract_single(
        self,
        *,
        extractor,
        file_path: str,
        timeout_override_sec: int | None,
        deadline_monotonic: float | None = None,
    ) -> list[DocumentChunk]:
        """Execute extraction with one backend and optional hard timeout."""
        LOGGER.info("Extracting %s with %s", file_path, extractor.name)
        remaining_sec = self._remaining_budget_sec(deadline_monotonic)
        timeout_sec = self._resolve_timeout_sec(timeout_override_sec, remaining_sec=remaining_sec)
        if timeout_sec is None:
            return extractor.extract(file_path)
        if timeout_sec <= 0:
            raise TimeoutError(f"Extractor '{extractor.name}' timed out before start (deadline reached)")

        fd, payload_path = tempfile.mkstemp(prefix="rag_extract_payload_", suffix=".pkl")
        os.close(fd)
        ctx = get_context("spawn")
        receiver, sender = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_run_extractor_in_subprocess,
            args=(extractor, file_path, sender, payload_path),
            daemon=True,
        )
        proc.start()
        sender.close()

        payload = None
        ipc_message = None
        try:
            proc.join(timeout=float(timeout_sec))
            if proc.is_alive():
                self._stop_subprocess_tree(proc)
                raise TimeoutError(f"Extractor '{extractor.name}' timed out after {timeout_sec}s")

            with contextlib.suppress(EOFError, OSError):
                if receiver.poll(1.0):
                    ipc_message = receiver.recv()
            payload = _load_subprocess_payload(payload_path)
            if (
                payload is None
                and isinstance(ipc_message, tuple)
                and ipc_message
                and str(ipc_message[0]) == "err"
            ):
                payload = ipc_message
        finally:
            with contextlib.suppress(Exception):
                receiver.close()
            with contextlib.suppress(Exception):
                proc.close()
            with contextlib.suppress(Exception):
                os.remove(payload_path)

        if not isinstance(payload, tuple) or not payload:
            exit_code = proc.exitcode
            raise RuntimeError(f"Extractor '{extractor.name}' finished without payload (exit={exit_code})")

        status = payload[0]
        if status == "ok":
            chunks = payload[1]
            if not isinstance(chunks, list):
                raise RuntimeError(f"Extractor '{extractor.name}' returned invalid payload type")
            return chunks
        if status == "err":
            err_type = str(payload[1]) if len(payload) > 1 else "RuntimeError"
            err_text = str(payload[2]) if len(payload) > 2 else "unknown child error"
            raise RuntimeError(f"{err_type}: {err_text}")
        raise RuntimeError(f"Extractor '{extractor.name}' returned unknown payload status: {status}")

    def _resolve_timeout_sec(self, timeout_override_sec: int | None, *, remaining_sec: float | None = None) -> int | None:
        """Resolve effective timeout using hard cap and adaptive estimate."""
        hard_cap = self.extract_timeout_sec if self.extract_timeout_sec and self.extract_timeout_sec > 0 else None
        adaptive = timeout_override_sec if timeout_override_sec and timeout_override_sec > 0 else None

        if adaptive is None:
            timeout = int(hard_cap) if hard_cap is not None else None
        elif hard_cap is None:
            timeout = int(adaptive)
        else:
            timeout = max(1, int(min(adaptive, hard_cap)))

        if remaining_sec is None:
            return timeout
        if remaining_sec <= 0:
            return 0

        remaining_limit = max(1, int(math.ceil(remaining_sec)))
        if timeout is None:
            return remaining_limit
        return max(1, min(timeout, remaining_limit))

    @staticmethod
    def _remaining_budget_sec(deadline_monotonic: float | None) -> float | None:
        """Return remaining wall-clock budget in seconds."""
        if deadline_monotonic is None:
            return None
        return deadline_monotonic - time.perf_counter()

    @staticmethod
    def _stop_subprocess_tree(proc) -> None:
        """Best-effort terminate/kill subprocess and its process group."""
        with contextlib.suppress(Exception):
            proc.terminate()
        proc.join(timeout=5)
        if not proc.is_alive():
            return

        if os.name == "posix" and proc.pid:
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.killpg(proc.pid, signal.SIGTERM)
            proc.join(timeout=2)
            if proc.is_alive():
                with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                    os.killpg(proc.pid, signal.SIGKILL)

        if proc.is_alive():  # pragma: no cover - best effort hard stop
            with contextlib.suppress(Exception):
                proc.kill()
        proc.join(timeout=2)

    def _evaluate_routing(
        self,
        stats: ExtractionStats,
        chunks: list[DocumentChunk],
        *,
        page_scope: list[int],
    ) -> tuple[str, str | None, list[int], float, list[int], float, dict[str, float]]:
        """Apply deterministic pass/soft_fail/hard_fail routing contract."""
        scope = sorted({int(p) for p in page_scope if int(p) > 0})
        if not scope:
            scope = [1]

        (
            low_pages,
            empty_ratio_scope,
            table_like_doc,
            poisoned_pages,
            poisoned_page_ratio,
            poison_signals,
        ) = self._detect_low_quality_pages(chunks, page_scope=scope)
        seen_pages = {int(ch.page) for ch in chunks if ch.page is not None and int(ch.page) in scope}
        page_coverage = float(len(seen_pages) / max(1, len(scope)))

        hard_reasons: list[str] = []
        if stats.total_chunks <= 0:
            hard_reasons.append("chunks==0")
        if page_coverage < self.min_page_coverage:
            hard_reasons.append(
                f"page_coverage<{round(page_coverage, 4)}<{round(self.min_page_coverage, 4)}"
            )
        if poisoned_page_ratio >= self.quality_thresholds.poisoned_page_ratio_hard:
            hard_reasons.append(
                f"poisoned_text_ratio:{round(poisoned_page_ratio, 4)}>=thr:{round(self.quality_thresholds.poisoned_page_ratio_hard, 4)}"
            )
        if hard_reasons:
            return (
                "hard_fail",
                "; ".join(hard_reasons),
                sorted(low_pages),
                page_coverage,
                sorted(poisoned_pages),
                poisoned_page_ratio,
                poison_signals,
            )

        soft_reasons: list[str] = []
        if stats.chars_per_page < self.quality_thresholds.min_chars_per_page:
            soft_reasons.append(
                f"chars_per_page<{round(stats.chars_per_page, 2)}<{round(self.quality_thresholds.min_chars_per_page, 2)}"
            )
        if empty_ratio_scope > self.quality_thresholds.max_empty_page_ratio:
            soft_reasons.append(
                f"empty_page_ratio>{round(empty_ratio_scope, 4)}>{round(self.quality_thresholds.max_empty_page_ratio, 4)}"
            )
        if stats.short_chunk_ratio > self.quality_thresholds.max_short_chunk_ratio:
            soft_reasons.append(
                f"short_chunk_ratio>{round(stats.short_chunk_ratio, 4)}>{round(self.quality_thresholds.max_short_chunk_ratio, 4)}"
            )
        if table_like_doc and not stats.has_table_elements:
            soft_reasons.append("table_loss")
        if poisoned_pages:
            soft_reasons.append(f"poisoned_text_detected:{len(poisoned_pages)}/{len(scope)}")

        if soft_reasons:
            return (
                "soft_fail",
                "; ".join(soft_reasons),
                sorted(low_pages),
                page_coverage,
                sorted(poisoned_pages),
                poisoned_page_ratio,
                poison_signals,
            )
        return (
            "pass",
            None,
            sorted(low_pages),
            page_coverage,
            sorted(poisoned_pages),
            poisoned_page_ratio,
            poison_signals,
        )

    def _detect_low_quality_pages(
        self,
        chunks: list[DocumentChunk],
        *,
        page_scope: list[int],
    ) -> tuple[list[int], float, bool, list[int], float, dict[str, float]]:
        """Identify low-quality pages for targeted fallback decisions."""
        scope = sorted({int(p) for p in page_scope if int(p) > 0})
        if not scope:
            return [1], 1.0, False, [], 0.0, {
                "max_escaped_seq_per_1k": 0.0,
                "max_backslash_per_1k": 0.0,
                "max_control_char_ratio": 0.0,
                "threshold_escaped_seq_per_1k": float(self.quality_thresholds.max_escaped_seq_per_1k),
                "threshold_backslash_per_1k": float(self.quality_thresholds.max_backslash_per_1k),
                "threshold_control_char_ratio": float(self.quality_thresholds.max_control_char_ratio),
                "poisoned_pages_count": 0.0,
                "scope_pages_count": 1.0,
            }

        chars_by_page = {p: 0 for p in scope}
        chunk_count_by_page = {p: 0 for p in scope}
        short_count_by_page = {p: 0 for p in scope}
        table_by_page = {p: False for p in scope}
        text_by_page: dict[int, list[str]] = {p: [] for p in scope}

        for ch in chunks:
            if ch.page is None:
                continue
            page = int(ch.page)
            if page not in chars_by_page:
                continue
            txt = (ch.text or "").strip()
            chars_by_page[page] += len(txt)
            chunk_count_by_page[page] += 1
            if len(txt) < 120:
                short_count_by_page[page] += 1
            if txt:
                text_by_page[page].append(txt[:2000])
            if (ch.element_type or "").lower() == "table" or bool(ch.table_html):
                table_by_page[page] = True

        low_pages: list[int] = []
        table_like_pages = 0
        empty_pages = 0

        for page in scope:
            chars = chars_by_page[page]
            if chars == 0:
                empty_pages += 1
            total_chunks = chunk_count_by_page[page]
            short_ratio = float(short_count_by_page[page] / total_chunks) if total_chunks else 1.0
            sample_text = "\n".join(text_by_page[page][:8])
            page_table_like = self._looks_like_table(sample_text)
            if page_table_like:
                table_like_pages += 1

            if chars < self.quality_thresholds.min_chars_per_page:
                low_pages.append(page)
            if short_ratio > self.quality_thresholds.max_short_chunk_ratio:
                low_pages.append(page)
            if page_table_like and not table_by_page[page]:
                low_pages.append(page)

        poisoned_pages, poisoned_ratio, poison_signals = detect_poisoned_pages(
            text_by_page,
            scope,
            self.quality_thresholds,
        )
        if poisoned_pages:
            low_pages.extend(poisoned_pages)

        empty_ratio_scope = float(empty_pages / max(1, len(scope)))
        table_like_doc = table_like_pages >= max(2, math.ceil(len(scope) * 0.12))
        return (
            sorted(set(low_pages)),
            empty_ratio_scope,
            table_like_doc,
            sorted(set(poisoned_pages)),
            poisoned_ratio,
            poison_signals,
        )

    @staticmethod
    def _extract_metadata_scalar(chunks: list[DocumentChunk], key: str) -> str | None:
        """Read first non-empty scalar metadata value from chunks."""
        for ch in chunks:
            value = ch.metadata.get(key) if ch.metadata else None
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _extract_metadata_list(chunks: list[DocumentChunk], key: str) -> list[str]:
        """Read first non-empty list metadata value from chunks."""
        for ch in chunks:
            value = ch.metadata.get(key) if ch.metadata else None
            if isinstance(value, list) and value:
                return [str(x) for x in value]
        return []

    def _normalize_unstructured_targeted_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Merge fragmented short chunks per page and remove exact duplicate snippets."""
        if not chunks:
            return []
        min_chars = max(80, int(self.unstructured_min_merged_chunk_chars))
        buckets: dict[tuple[str, int], list[DocumentChunk]] = {}
        order: list[tuple[str, int]] = []
        passthrough: list[DocumentChunk] = []

        for chunk in chunks:
            if chunk.page is None:
                passthrough.append(chunk)
                continue
            key = (str(chunk.source_path or ""), int(chunk.page))
            if key not in buckets:
                buckets[key] = []
                order.append(key)
            buckets[key].append(chunk)

        normalized: list[DocumentChunk] = []
        for key in order:
            page_chunks = buckets[key]
            if not page_chunks:
                continue
            seen_page_texts: set[str] = set()
            buffer_texts: list[str] = []
            buffer_seed: DocumentChunk | None = None

            def flush_buffer() -> None:
                """Emit merged short-fragment buffer as one normalized chunk."""
                nonlocal buffer_texts, buffer_seed
                if not buffer_seed or not buffer_texts:
                    buffer_texts = []
                    buffer_seed = None
                    return
                merged_text = " ".join(buffer_texts).strip()
                if not merged_text:
                    buffer_texts = []
                    buffer_seed = None
                    return
                normalized.append(self._clone_chunk_with_text(buffer_seed, merged_text))
                buffer_texts = []
                buffer_seed = None

            for chunk in page_chunks:
                text = " ".join(str(chunk.text or "").split())
                if not text:
                    continue
                dedupe_key = self._normalized_text_key(text)
                if dedupe_key in seen_page_texts:
                    continue
                seen_page_texts.add(dedupe_key)

                if len(text) < min_chars:
                    if buffer_seed is None:
                        buffer_seed = chunk
                    buffer_texts.append(text)
                    if sum(len(part) for part in buffer_texts) >= min_chars:
                        flush_buffer()
                    continue

                flush_buffer()
                normalized.append(self._clone_chunk_with_text(chunk, text))

            flush_buffer()

        normalized.extend(passthrough)
        return normalized

    @staticmethod
    def _normalized_text_key(text: str) -> str:
        """Build normalized dedupe key for chunk text snippets."""
        return " ".join(str(text or "").lower().split())

    @staticmethod
    def _clone_chunk_with_text(chunk: DocumentChunk, text: str) -> DocumentChunk:
        """Copy chunk while replacing text and preserving metadata payload."""
        return DocumentChunk(
            text=text,
            source_path=chunk.source_path,
            page=chunk.page,
            element_type=chunk.element_type,
            bbox=dict(chunk.bbox) if isinstance(chunk.bbox, dict) else chunk.bbox,
            table_html=chunk.table_html,
            metadata=dict(chunk.metadata or {}),
        )

    @staticmethod
    def _merge_targeted_chunks(
        base_chunks: list[DocumentChunk],
        targeted_chunks: list[DocumentChunk],
        target_pages: list[int],
    ) -> list[DocumentChunk]:
        """Replace low-quality pages in base chunks with targeted fallback chunks."""
        page_set = {int(p) for p in target_pages}
        kept = [ch for ch in base_chunks if ch.page is None or int(ch.page) not in page_set]
        merged = kept + list(targeted_chunks)
        merged.sort(key=lambda ch: (ch.source_path, ch.page if ch.page is not None else 0, len(ch.text or "")))
        return merged

    @staticmethod
    def _chunk_pages(pages: list[int], batch_size: int) -> list[list[int]]:
        """Split sorted pages into deterministic fixed-size batches."""
        clean = sorted({int(p) for p in pages if int(p) > 0})
        if not clean:
            return []
        size = max(1, int(batch_size))
        return [clean[idx : idx + size] for idx in range(0, len(clean), size)]

    def _run_low_coverage_recovery(
        self,
        *,
        file_path: str,
        total_pages: int,
        page_scope: list[int],
        adaptive_timeout: int | None,
        reserve_window_sec: int,
        deadline_monotonic: float | None,
        base_attempt: dict,
        targeted_pages_already: list[int],
        notes: list[str],
        attempts_public: list[dict[str, object]],
        fallback_path: list[str],
    ) -> dict | None:
        """Run additional targeted unstructured batches for low-coverage hard-fail attempts."""
        if not self._is_low_coverage_recovery_trigger(base_attempt):
            return None

        base_low_pages = [int(p) for p in (base_attempt.get("low_quality_pages") or []) if int(p) > 0]
        if not base_low_pages:
            return None

        processed_pages: list[int] = []
        already_done = {int(p) for p in targeted_pages_already if int(p) > 0}
        candidate_pages = [p for p in sorted(set(base_low_pages)) if p not in already_done]
        if not candidate_pages:
            candidate_pages = sorted(set(base_low_pages))
        if not candidate_pages:
            return None

        candidate_pages = candidate_pages[: int(self.extract_low_coverage_recovery_max_pages)]
        batches = self._chunk_pages(candidate_pages, int(self.extract_low_coverage_recovery_batch_pages))
        if not batches:
            return None

        notes.append(
            "Starting low-coverage recovery mode "
            f"(pages={len(candidate_pages)}, batch={int(self.extract_low_coverage_recovery_batch_pages)})"
        )

        working_attempt = dict(base_attempt)
        working_chunks = list(base_attempt.get("chunks") or [])
        coverage_after_batch: list[float] = []
        batches_run = 0

        for idx, page_batch in enumerate(batches, start=1):
            remaining_before_batch = self._remaining_budget_sec(deadline_monotonic)
            min_required = float(
                int(self.extract_full_quality_min_stage_start_sec) + int(max(0, reserve_window_sec))
            )
            if remaining_before_batch is not None and remaining_before_batch < min_required:
                notes.append(
                    "Stopping low-coverage recovery: insufficient remaining budget "
                    f"({round(max(0.0, float(remaining_before_batch)), 2)}s)"
                )
                break

            stage_label = f"unstructured_hi_res_recovery_batch_{idx}"
            batch_attempt = self._attempt_extractor(
                extractor=UnstructuredExtractor(
                    languages=self.languages,
                    strategy="hi_res",
                    chunking=self.chunking,
                    target_pages=page_batch,
                ),
                file_path=file_path,
                total_pages=total_pages,
                page_scope=page_batch,
                timeout_override_sec=adaptive_timeout,
                extractor_label=stage_label,
                stage_name=stage_label,
                reserved_tail_budget_sec=reserve_window_sec,
                deadline_monotonic=deadline_monotonic,
            )
            batch_attempt["recovery_mode"] = "low_coverage"
            batch_attempt["recovery_batches_run"] = int(idx)
            batch_attempt["coverage_after_batch"] = list(coverage_after_batch)
            batch_attempt["targeted_pages_processed"] = list(processed_pages)
            attempts_public.append(self._public_attempt(batch_attempt))
            fallback_path.append(stage_label)
            batches_run = idx

            if not batch_attempt["chunks"]:
                notes.append(f"{stage_label}: {batch_attempt['switch_reason'] or 'failed'}")
                if self._is_deadline_timeout(batch_attempt):
                    notes.append("Stopping low-coverage recovery due to deadline timeout")
                    break
                continue

            processed_pages.extend(page_batch)
            merged_chunks = self._merge_targeted_chunks(working_chunks, batch_attempt["chunks"], page_batch)
            merged_stats = compute_extraction_stats(
                merged_chunks,
                file_path,
                "pymupdf4llm+unstructured_hi_res_recovery",
                total_pages=total_pages,
            )
            (
                merged_status,
                merged_reason,
                merged_low_pages,
                merged_cov,
                merged_poisoned_pages,
                merged_poisoned_ratio,
                merged_poison_signals,
            ) = self._evaluate_routing(
                merged_stats,
                merged_chunks,
                page_scope=page_scope,
            )
            merged_stats.poisoned_pages = list(merged_poisoned_pages)
            merged_stats.poisoned_page_ratio = float(merged_poisoned_ratio)
            merged_stats.poison_signals = dict(merged_poison_signals)
            coverage_after_batch.append(float(merged_cov))
            merge_attempt = {
                "extractor": "pymupdf4llm+unstructured_hi_res_recovery",
                "status": merged_status,
                "switch_reason": merged_reason,
                "duration_ms": 0.0,
                "error": None,
                "total_chunks": int(merged_stats.total_chunks),
                "chars_per_page": float(merged_stats.chars_per_page),
                "empty_page_ratio": float(merged_stats.empty_page_ratio),
                "short_chunk_ratio": float(merged_stats.short_chunk_ratio),
                "page_coverage": float(merged_cov),
                "low_quality_pages": list(merged_low_pages),
                "poisoned_pages": list(merged_poisoned_pages),
                "poisoned_page_ratio": float(merged_poisoned_ratio),
                "poison_signals": dict(merged_poison_signals),
                "ocr_backend_effective": None,
                "ocr_fallback_path": [],
                "remaining_budget_sec_before_attempt": batch_attempt.get("remaining_budget_sec_before_attempt"),
                "effective_timeout_sec": batch_attempt.get("effective_timeout_sec"),
                "chunks": merged_chunks,
                "stats": merged_stats,
                "recovery_mode": "low_coverage",
                "recovery_batches_run": int(idx),
                "coverage_after_batch": list(coverage_after_batch),
                "targeted_pages_processed": sorted(set(processed_pages)),
            }
            self._apply_recovery_soft_fail_downgrade(merge_attempt, notes)
            attempts_public.append(self._public_attempt(merge_attempt))
            fallback_path.append(f"recovery_merge_{idx}")
            working_attempt = merge_attempt
            working_chunks = merged_chunks

            if merge_attempt["status"] == "pass":
                notes.append(f"Low-coverage recovery reached pass on batch {idx}")
                break
            if float(merged_cov) >= float(self.min_page_coverage):
                notes.append(f"Low-coverage recovery reached min coverage on batch {idx}")
                break

        if batches_run <= 0:
            return None

        working_attempt["recovery_mode"] = "low_coverage"
        working_attempt["recovery_batches_run"] = int(batches_run)
        working_attempt["coverage_after_batch"] = list(coverage_after_batch)
        working_attempt["targeted_pages_processed"] = sorted(set(processed_pages))
        return working_attempt

    def _docling_backend_order(self) -> list[str]:
        """Resolve deterministic Docling OCR backend order."""
        allowed = {"easyocr", "rapidocr", "tesseract", "none"}
        first = (self.docling_ocr_backend or "easyocr").strip().lower()
        if first not in allowed:
            first = "easyocr"

        preferred_second = "rapidocr" if first != "rapidocr" else "easyocr"

        ordered: list[str] = [first]
        for backend in [preferred_second, *self.docling_ocr_fallbacks, "none"]:
            b = (backend or "").strip().lower()
            if b in allowed and b not in ordered:
                ordered.append(b)
        if len(ordered) == 1:
            ordered.append("none")
        return ordered

    def _normalize_profile(self, profile: str | None, *, fast_mode: bool, preferred: str) -> str:
        """Normalize ingestion profile name."""
        if profile:
            norm = profile.strip().lower()
            if norm in {"demo-fast", "demo_fast", "fast"}:
                return "demo-fast"
            if norm in {"full-quality", "full_quality", "quality"}:
                return "full-quality"
        if fast_mode or preferred == "pymupdf4llm":
            return "demo-fast"
        return "full-quality"

    def _adaptive_timeout_sec(self, file_path: str, profile_name: str) -> int | None:
        """Estimate timeout from PDF size/pages to avoid hard fixed cutoff."""
        if self.extract_timeout_base_sec is None:
            return self.extract_timeout_sec
        try:
            size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        except Exception:
            size_mb = 0.0
        pages = self._pdf_page_count(file_path)
        base = int(self.extract_timeout_base_sec)
        timeout = (
            base
            + math.ceil(max(0, pages) / 100) * self.extract_timeout_per_100_pages_sec
            + math.ceil(max(0.0, size_mb) / 10.0) * self.extract_timeout_per_10mb_sec
        )
        if profile_name == "full-quality":
            timeout = int(timeout * 1.8)
        timeout = min(timeout, self.extract_timeout_max_sec)
        return max(15, timeout)

    @staticmethod
    def _pdf_page_count(file_path: str) -> int:
        """Read page count with best-effort fallback."""
        try:
            import fitz  # type: ignore

            doc = fitz.open(file_path)
            pages = int(doc.page_count)
            doc.close()
            return max(1, pages)
        except Exception:
            return 1

    @staticmethod
    def _looks_like_table(text: str) -> bool:
        """Lightweight heuristic: detect table-like page text."""
        return _looks_like_table_text(text)

    def _extract_by_page_windows(self, file_path: str, window_size: int) -> list[DocumentChunk]:
        """Fallback extractor: process pages in windows using raw PyMuPDF text."""
        return _extract_by_page_windows_impl(
            file_path=file_path,
            window_size=int(window_size),
            chunking=self.chunking,
        )
