"""Configuration management for RAG v3."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(slots=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    giga_api_key: str
    anthropic_api_key: str = ""
    giga_scope: str = "GIGACHAT_API_B2B"
    giga_chat_model: str = "GigaChat-2-Max"
    giga_embedding_model: str = "GigaEmbeddings-3B-2025-09"
    giga_http_timeout_sec: float = 60.0
    data_dir: str = "data"
    index_dir: str = ".rag_index"
    ingest_profile: str = "demo-fast"
    default_extractor: str = "pymupdf4llm"
    default_reranker: str = "amberoad"
    retrieve_top_k: int = 50
    rerank_top_n: int = 10
    final_top_k: int = 8
    retrieval_source_diversity_enabled: bool = True
    retrieval_source_max_chunks_per_source: int = 2
    retrieval_year_boost_enabled: bool = True
    retrieval_year_boost: float = 0.12
    rewrite_n: int = 3
    rrf_k: int = 60
    extract_timeout_sec: int = 180
    extract_timeout_base_sec: int = 45
    extract_timeout_per_100_pages_sec: int = 30
    extract_timeout_per_10mb_sec: int = 20
    extract_timeout_max_sec: int = 600
    extract_full_quality_docling_primary_max_sec: int = 300
    extract_full_quality_docling_secondary_max_sec: int = 240
    extract_full_quality_pymupdf_max_sec: int = 180
    extract_full_quality_reserve_window_fallback_sec: int = 180
    extract_full_quality_min_stage_start_sec: int = 15
    extract_full_quality_unstructured_min_remaining_sec: int = 90
    extract_prefer_best_usable: bool = True
    extract_best_usable_min_coverage_gap: float = 0.15
    unstructured_min_merged_chunk_chars: int = 220
    extract_low_coverage_recovery_enabled: bool = True
    extract_low_coverage_recovery_trigger_coverage: float = 0.35
    extract_low_coverage_recovery_batch_pages: int = 120
    extract_low_coverage_recovery_max_pages: int = 360
    extract_low_coverage_recovery_softfail_min_coverage: float = 0.55
    page_window_size: int = 40
    extract_min_page_coverage: float = 0.85
    extract_min_chars_per_page: float = 450.0
    extract_max_empty_page_ratio: float = 0.25
    extract_max_short_chunk_ratio: float = 0.55
    extract_max_escaped_seq_per_1k: float = 18.0
    extract_max_backslash_per_1k: float = 30.0
    extract_max_control_char_ratio: float = 0.003
    extract_poisoned_page_ratio_hard: float = 0.35
    unstructured_targeted_only: bool = True
    docling_ocr_backend: str = "easyocr"
    docling_ocr_fallbacks: tuple[str, ...] = ("easyocr", "tesseract", "rapidocr", "none")
    docling_ocr_langs_easyocr: tuple[str, ...] = ("ru", "en")
    docling_ocr_langs_tesseract: tuple[str, ...] = ("rus", "eng")
    chunk_mode: str = "token"
    chunk_tokens_prose: int = 360
    chunk_tokens_table: int = 220
    chunk_overlap_prose: int = 60
    chunk_overlap_table: int = 30
    chunk_chars_prose: int = 1600
    chunk_chars_table: int = 1500
    chunk_overlap_chars_prose: int = 160
    chunk_overlap_chars_table: int = 120
    embedding_batch_size: int = 32
    grounded_min_top_rerank_score: float = 4.5
    grounded_min_top_rerank_score_amberoad: float = 1.8
    grounded_min_top_rerank_score_bge_m3: float = 0.85
    grounded_min_top_rerank_score_jina_multilingual: float = 0.75
    rerank_year_retention_enabled: bool = True
    rerank_year_retention_max_score_gap: float = 0.35
    grounded_min_total_context_chars: int = 800
    grounded_min_chunks: int = 3
    preflight_ttl_sec: int = 300
    api_max_retries: int = 3
    api_retry_backoff_sec: float = 1.25
    prewarm_rerankers: bool = False
    reranker_cache_dir: str = ".rag_cache/rerankers"
    pdf_regression_wallclock_cap_sec: int = 900
    ragas_judge_provider: str = "gigachat"
    ragas_judge_model: str = "claude-sonnet-4-20250514"

    def ensure_dirs(self) -> None:
        """Create index directory if needed."""
        Path(self.index_dir).mkdir(parents=True, exist_ok=True)
        Path(self.reranker_cache_dir).mkdir(parents=True, exist_ok=True)



def _read_int(name: str, default: int) -> int:
    """Read integer env var with fallback."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _read_bool(name: str, default: bool) -> bool:
    """Read boolean env var with common string/int forms."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _read_float(name: str, default: float) -> float:
    """Read float env var with fallback."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _read_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Read comma-separated env list with normalized lowercase values."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    parts = [part.strip() for part in value.split(",")]
    normalized = tuple(part.lower() for part in parts if part)
    return normalized or default



def load_settings(dotenv_path: str | None = None) -> Settings:
    """Load settings from environment and optional .env file."""
    load_dotenv(dotenv_path=dotenv_path)

    giga_api_key = os.getenv("GIGA_API_KEY") or os.getenv("GIGACHAT_AUTH_KEY") or ""
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY") or ""
    giga_scope = os.getenv("GIGA_SCOPE") or os.getenv("GIGACHAT_SCOPE") or "GIGACHAT_API_B2B"

    settings = Settings(
        giga_api_key=giga_api_key,
        anthropic_api_key=anthropic_api_key,
        giga_scope=giga_scope,
        giga_chat_model=os.getenv("GIGA_CHAT_MODEL", "GigaChat-2-Max"),
        giga_embedding_model=os.getenv("GIGA_EMBEDDING_MODEL", "GigaEmbeddings-3B-2025-09"),
        giga_http_timeout_sec=max(5.0, _read_float("GIGA_HTTP_TIMEOUT_SEC", 60.0)),
        data_dir=os.getenv("DATA_DIR", "data"),
        index_dir=os.getenv("INDEX_DIR", ".rag_index"),
        ingest_profile=os.getenv("INGEST_PROFILE", "demo-fast"),
        default_extractor=os.getenv("DEFAULT_EXTRACTOR", "pymupdf4llm"),
        default_reranker=os.getenv("DEFAULT_RERANKER", "amberoad"),
        retrieve_top_k=_read_int("RETRIEVE_TOP_K", 50),
        rerank_top_n=_read_int("RERANK_TOP_N", 10),
        final_top_k=_read_int("FINAL_TOP_K", 8),
        retrieval_source_diversity_enabled=_read_bool("RETRIEVAL_SOURCE_DIVERSITY_ENABLED", True),
        retrieval_source_max_chunks_per_source=max(1, _read_int("RETRIEVAL_SOURCE_MAX_CHUNKS_PER_SOURCE", 2)),
        retrieval_year_boost_enabled=_read_bool("RETRIEVAL_YEAR_BOOST_ENABLED", True),
        retrieval_year_boost=max(0.0, _read_float("RETRIEVAL_YEAR_BOOST", 0.12)),
        rewrite_n=_read_int("REWRITE_N", 3),
        rrf_k=_read_int("RRF_K", 60),
        extract_timeout_sec=_read_int("EXTRACT_TIMEOUT_SEC", 180),
        extract_timeout_base_sec=_read_int("EXTRACT_TIMEOUT_BASE_SEC", 45),
        extract_timeout_per_100_pages_sec=_read_int("EXTRACT_TIMEOUT_PER_100_PAGES_SEC", 30),
        extract_timeout_per_10mb_sec=_read_int("EXTRACT_TIMEOUT_PER_10MB_SEC", 20),
        extract_timeout_max_sec=_read_int("EXTRACT_TIMEOUT_MAX_SEC", 600),
        extract_full_quality_docling_primary_max_sec=_read_int(
            "EXTRACT_FULL_QUALITY_DOCLING_PRIMARY_MAX_SEC",
            300,
        ),
        extract_full_quality_docling_secondary_max_sec=_read_int(
            "EXTRACT_FULL_QUALITY_DOCLING_SECONDARY_MAX_SEC",
            240,
        ),
        extract_full_quality_pymupdf_max_sec=_read_int(
            "EXTRACT_FULL_QUALITY_PYMUPDF_MAX_SEC",
            180,
        ),
        extract_full_quality_reserve_window_fallback_sec=_read_int(
            "EXTRACT_FULL_QUALITY_RESERVE_WINDOW_FALLBACK_SEC",
            180,
        ),
        extract_full_quality_min_stage_start_sec=_read_int(
            "EXTRACT_FULL_QUALITY_MIN_STAGE_START_SEC",
            15,
        ),
        extract_full_quality_unstructured_min_remaining_sec=_read_int(
            "EXTRACT_FULL_QUALITY_UNSTRUCTURED_MIN_REMAINING_SEC",
            90,
        ),
        extract_prefer_best_usable=_read_bool("EXTRACT_PREFER_BEST_USABLE", True),
        extract_best_usable_min_coverage_gap=max(
            0.0,
            _read_float("EXTRACT_BEST_USABLE_MIN_COVERAGE_GAP", 0.15),
        ),
        unstructured_min_merged_chunk_chars=max(
            80,
            _read_int("UNSTRUCTURED_MIN_MERGED_CHUNK_CHARS", 220),
        ),
        extract_low_coverage_recovery_enabled=_read_bool("EXTRACT_LOW_COVERAGE_RECOVERY_ENABLED", True),
        extract_low_coverage_recovery_trigger_coverage=min(
            1.0,
            max(0.0, _read_float("EXTRACT_LOW_COVERAGE_RECOVERY_TRIGGER_COVERAGE", 0.35)),
        ),
        extract_low_coverage_recovery_batch_pages=max(
            1,
            _read_int("EXTRACT_LOW_COVERAGE_RECOVERY_BATCH_PAGES", 120),
        ),
        extract_low_coverage_recovery_max_pages=max(
            1,
            _read_int("EXTRACT_LOW_COVERAGE_RECOVERY_MAX_PAGES", 360),
        ),
        extract_low_coverage_recovery_softfail_min_coverage=min(
            1.0,
            max(0.0, _read_float("EXTRACT_LOW_COVERAGE_RECOVERY_SOFTFAIL_MIN_COVERAGE", 0.55)),
        ),
        page_window_size=_read_int("PAGE_WINDOW_SIZE", 40),
        extract_min_page_coverage=_read_float("EXTRACT_MIN_PAGE_COVERAGE", 0.85),
        extract_min_chars_per_page=_read_float("EXTRACT_MIN_CHARS_PER_PAGE", 450.0),
        extract_max_empty_page_ratio=_read_float("EXTRACT_MAX_EMPTY_PAGE_RATIO", 0.25),
        extract_max_short_chunk_ratio=_read_float("EXTRACT_MAX_SHORT_CHUNK_RATIO", 0.55),
        extract_max_escaped_seq_per_1k=_read_float("EXTRACT_MAX_ESCAPED_SEQ_PER_1K", 18.0),
        extract_max_backslash_per_1k=_read_float("EXTRACT_MAX_BACKSLASH_PER_1K", 30.0),
        extract_max_control_char_ratio=_read_float("EXTRACT_MAX_CONTROL_CHAR_RATIO", 0.003),
        extract_poisoned_page_ratio_hard=_read_float("EXTRACT_POISONED_PAGE_RATIO_HARD", 0.35),
        unstructured_targeted_only=_read_bool("UNSTRUCTURED_TARGETED_ONLY", True),
        docling_ocr_backend=os.getenv("DOCLING_OCR_BACKEND", "easyocr").strip().lower(),
        docling_ocr_fallbacks=_read_list(
            "DOCLING_OCR_FALLBACKS",
            ("easyocr", "tesseract", "rapidocr", "none"),
        ),
        docling_ocr_langs_easyocr=_read_list("DOCLING_OCR_LANGS_EASYOCR", ("ru", "en")),
        docling_ocr_langs_tesseract=_read_list("DOCLING_OCR_LANGS_TESSERACT", ("rus", "eng")),
        chunk_mode=os.getenv("CHUNK_MODE", "token").strip().lower(),
        chunk_tokens_prose=max(64, _read_int("CHUNK_TOKENS_PROSE", 360)),
        chunk_tokens_table=max(64, _read_int("CHUNK_TOKENS_TABLE", 220)),
        chunk_overlap_prose=max(0, _read_int("CHUNK_OVERLAP_PROSE", 60)),
        chunk_overlap_table=max(0, _read_int("CHUNK_OVERLAP_TABLE", 30)),
        chunk_chars_prose=max(600, _read_int("CHUNK_CHARS_PROSE", 1600)),
        chunk_chars_table=max(600, _read_int("CHUNK_CHARS_TABLE", 1500)),
        chunk_overlap_chars_prose=max(0, _read_int("CHUNK_OVERLAP_CHARS_PROSE", 160)),
        chunk_overlap_chars_table=max(0, _read_int("CHUNK_OVERLAP_CHARS_TABLE", 120)),
        embedding_batch_size=max(1, _read_int("EMBEDDING_BATCH_SIZE", 32)),
        grounded_min_top_rerank_score=_read_float("GROUNDED_MIN_TOP_RERANK_SCORE", 4.5),
        grounded_min_top_rerank_score_amberoad=_read_float("GROUNDED_MIN_TOP_RERANK_SCORE_AMBEROAD", 1.8),
        grounded_min_top_rerank_score_bge_m3=_read_float("GROUNDED_MIN_TOP_RERANK_SCORE_BGE_M3", 0.85),
        grounded_min_top_rerank_score_jina_multilingual=_read_float(
            "GROUNDED_MIN_TOP_RERANK_SCORE_JINA_MULTILINGUAL",
            0.75,
        ),
        rerank_year_retention_enabled=_read_bool("RERANK_YEAR_RETENTION_ENABLED", True),
        rerank_year_retention_max_score_gap=max(
            0.0,
            _read_float("RERANK_YEAR_RETENTION_MAX_SCORE_GAP", 0.35),
        ),
        grounded_min_total_context_chars=_read_int("GROUNDED_MIN_TOTAL_CONTEXT_CHARS", 800),
        grounded_min_chunks=max(1, _read_int("GROUNDED_MIN_CHUNKS", 3)),
        preflight_ttl_sec=max(0, _read_int("PREFLIGHT_TTL_SEC", 300)),
        api_max_retries=max(0, _read_int("API_MAX_RETRIES", 3)),
        api_retry_backoff_sec=max(0.1, _read_float("API_RETRY_BACKOFF_SEC", 1.25)),
        prewarm_rerankers=_read_bool("PREWARM_RERANKERS", False),
        reranker_cache_dir=os.getenv("RERANKER_CACHE_DIR", ".rag_cache/rerankers"),
        pdf_regression_wallclock_cap_sec=max(0, _read_int("PDF_REGRESSION_WALLCLOCK_CAP_SEC", 900)),
        ragas_judge_provider=os.getenv("RAGAS_JUDGE_PROVIDER", "gigachat").strip().lower(),
        ragas_judge_model=os.getenv("RAGAS_JUDGE_MODEL", "claude-sonnet-4-20250514").strip(),
    )
    if settings.chunk_mode not in {"token", "char"}:
        settings.chunk_mode = "token"
    if settings.docling_ocr_backend not in {"easyocr", "rapidocr", "tesseract", "none"}:
        settings.docling_ocr_backend = "easyocr"
    settings.extract_max_escaped_seq_per_1k = max(0.0, float(settings.extract_max_escaped_seq_per_1k))
    settings.extract_max_backslash_per_1k = max(0.0, float(settings.extract_max_backslash_per_1k))
    settings.extract_max_control_char_ratio = min(
        1.0,
        max(0.0, float(settings.extract_max_control_char_ratio)),
    )
    settings.extract_poisoned_page_ratio_hard = min(
        1.0,
        max(0.0, float(settings.extract_poisoned_page_ratio_hard)),
    )
    settings.retrieval_year_boost = max(0.0, float(settings.retrieval_year_boost))
    settings.extract_full_quality_docling_primary_max_sec = max(
        0,
        int(settings.extract_full_quality_docling_primary_max_sec),
    )
    settings.extract_full_quality_docling_secondary_max_sec = max(
        0,
        int(settings.extract_full_quality_docling_secondary_max_sec),
    )
    settings.extract_full_quality_pymupdf_max_sec = max(
        0,
        int(settings.extract_full_quality_pymupdf_max_sec),
    )
    settings.extract_full_quality_reserve_window_fallback_sec = max(
        0,
        int(settings.extract_full_quality_reserve_window_fallback_sec),
    )
    settings.extract_full_quality_min_stage_start_sec = max(
        1,
        int(settings.extract_full_quality_min_stage_start_sec),
    )
    settings.extract_full_quality_unstructured_min_remaining_sec = max(
        0,
        int(settings.extract_full_quality_unstructured_min_remaining_sec),
    )
    settings.unstructured_min_merged_chunk_chars = max(
        80,
        int(settings.unstructured_min_merged_chunk_chars),
    )
    settings.extract_low_coverage_recovery_trigger_coverage = min(
        1.0,
        max(0.0, float(settings.extract_low_coverage_recovery_trigger_coverage)),
    )
    settings.extract_low_coverage_recovery_batch_pages = max(
        1,
        int(settings.extract_low_coverage_recovery_batch_pages),
    )
    settings.extract_low_coverage_recovery_max_pages = max(
        int(settings.extract_low_coverage_recovery_batch_pages),
        int(settings.extract_low_coverage_recovery_max_pages),
    )
    settings.extract_low_coverage_recovery_softfail_min_coverage = min(
        float(settings.extract_min_page_coverage),
        max(0.0, float(settings.extract_low_coverage_recovery_softfail_min_coverage)),
    )
    settings.extract_best_usable_min_coverage_gap = max(
        0.0,
        float(settings.extract_best_usable_min_coverage_gap),
    )
    settings.rerank_year_retention_max_score_gap = max(
        0.0,
        float(settings.rerank_year_retention_max_score_gap),
    )
    if settings.ragas_judge_provider not in {"gigachat", "anthropic"}:
        settings.ragas_judge_provider = "gigachat"
    if not settings.ragas_judge_model:
        settings.ragas_judge_model = "claude-sonnet-4-20250514"
    settings.ensure_dirs()
    return settings
