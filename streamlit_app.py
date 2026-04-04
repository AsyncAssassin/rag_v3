"""Streamlit frontend for RAG v3."""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import re
import shutil
from typing import Any
import uuid

import pandas as pd
import streamlit as st

from rag_system.config import load_settings
from rag_system.pipeline import RAGPipeline
from rag_system.rerankers.factory import available_rerankers
from rag_system.types import IndexStats


MAX_FILES_PER_SESSION = 20
MAX_FILE_SIZE_MB = 100
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".csv", ".md"}

SOURCE_MODES = ("demo", "uploads", "demo+uploads")


st.set_page_config(page_title="RAG v3", layout="wide")
st.title("RAG v3: Hybrid + Multi-Reranker + Docling-first")


@st.cache_resource
def _load_pipeline(env_path: str | None) -> RAGPipeline:
    """Create cached pipeline object."""
    settings = load_settings(env_path)
    return RAGPipeline(settings)


def _ensure_session_defaults() -> None:
    """Initialize session state fields used by UI."""
    if "history" not in st.session_state:
        st.session_state.history = []
    if "selfcheck_result" not in st.session_state:
        st.session_state.selfcheck_result = None
    if "prewarm_result" not in st.session_state:
        st.session_state.prewarm_result = None
    if "last_index_stats" not in st.session_state:
        st.session_state.last_index_stats = None
    if "upload_feedback" not in st.session_state:
        st.session_state.upload_feedback = None
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""
    if "corpus_overview" not in st.session_state:
        st.session_state.corpus_overview = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex[:12]


def _session_dirs() -> tuple[Path, Path, Path]:
    """Return per-session directories for uploaded and union source sets."""
    root = Path(".session_uploads") / st.session_state.session_id
    uploads_dir = root / "uploads"
    union_dir = root / "union"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    return root, uploads_dir, union_dir


def _sanitize_filename(filename: str) -> str:
    """Return safe filename for temporary upload storage."""
    base = Path(filename or "uploaded.bin").name
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._-")
    return safe or "uploaded.bin"


def _is_supported_file(path_or_name: str) -> bool:
    """Return True when extension is supported by the RAG indexer."""
    return Path(path_or_name).suffix.lower() in ALLOWED_EXTENSIONS


def _list_supported_files(root: Path) -> list[Path]:
    """List supported files recursively in sorted order."""
    if not root.exists() or not root.is_dir():
        return []
    files = [p for p in root.rglob("*") if p.is_file() and _is_supported_file(p.name)]
    return sorted(files, key=lambda p: str(p))


def _save_uploaded_files(uploaded_files: list[Any] | None, uploads_dir: Path) -> dict[str, Any]:
    """Save uploaded files with per-session limits and sha256 de-dup."""
    uploaded_files = uploaded_files or []
    existing_files = _list_supported_files(uploads_dir)
    existing_count = len(existing_files)
    existing_hashes: set[str] = set()
    for path in existing_files:
        name = path.name
        if "__" in name:
            existing_hashes.add(name.split("__", 1)[0])

    feedback = {
        "saved": [],
        "duplicates": 0,
        "too_large": [],
        "unsupported": [],
        "skipped_limit": 0,
        "existing_count": existing_count,
        "total_selected": len(uploaded_files),
    }
    if not uploaded_files:
        return feedback

    capacity = max(0, MAX_FILES_PER_SESSION - existing_count)
    saved_count = 0
    for item in uploaded_files:
        if saved_count >= capacity:
            feedback["skipped_limit"] += 1
            continue

        file_name = getattr(item, "name", "uploaded.bin")
        if not _is_supported_file(file_name):
            feedback["unsupported"].append(file_name)
            continue

        payload = item.getvalue()
        size_bytes = len(payload)
        if size_bytes > MAX_FILE_SIZE_BYTES:
            feedback["too_large"].append((file_name, size_bytes))
            continue

        digest = hashlib.sha256(payload).hexdigest()
        if digest in existing_hashes:
            feedback["duplicates"] += 1
            continue

        safe_name = _sanitize_filename(file_name)
        target = uploads_dir / f"{digest}__{safe_name}"
        target.write_bytes(payload)
        existing_hashes.add(digest)
        saved_count += 1
        feedback["saved"].append({"name": safe_name, "size_bytes": size_bytes, "path": str(target)})

    return feedback


def _prepare_source_dir(source_mode: str, demo_dir: Path, uploads_dir: Path, union_dir: Path) -> tuple[Path | None, str | None]:
    """Resolve one source directory for indexer from selected mode."""
    source_mode = source_mode.strip().lower()
    demo_files = _list_supported_files(demo_dir)
    upload_files = _list_supported_files(uploads_dir)

    if source_mode == "demo":
        if not demo_dir.exists() or not demo_dir.is_dir():
            return None, f"Demo directory not found: {demo_dir}"
        if not demo_files:
            return None, f"Demo directory has no supported files: {demo_dir}"
        return demo_dir, None

    if source_mode == "uploads":
        if not upload_files:
            return None, "Uploads source is empty. Add files in uploader first."
        return uploads_dir, None

    if source_mode == "demo+uploads":
        if not demo_files and not upload_files:
            return None, "No files found in demo or uploads sources."

        if union_dir.exists():
            shutil.rmtree(union_dir, ignore_errors=True)
        union_demo = union_dir / "demo"
        union_uploads = union_dir / "uploads"
        union_demo.mkdir(parents=True, exist_ok=True)
        union_uploads.mkdir(parents=True, exist_ok=True)

        for src in demo_files:
            shutil.copy2(src, union_demo / src.name)
        for src in upload_files:
            shutil.copy2(src, union_uploads / src.name)
        return union_dir, None

    return None, f"Unknown source mode: {source_mode}"


def _to_float(value: Any) -> float | None:
    """Safely cast numeric-like value to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_document_stats_df(pipeline: RAGPipeline, last_index_stats: IndexStats | None) -> pd.DataFrame:
    """Build per-document stats table from index + extraction reports."""
    source_chunk_counts: Counter[str] = Counter()
    source_pages: dict[str, set[int]] = {}
    source_token_counts: Counter[str] = Counter()

    for item in pipeline.index.indexed_chunks:
        source = item.chunk.source_path
        source_chunk_counts[source] += 1
        source_token_counts[source] += len(item.tokenized or [])
        if item.chunk.page is not None:
            source_pages.setdefault(source, set()).add(int(item.chunk.page))

    report_by_source: dict[str, Any] = {}
    if last_index_stats is not None:
        for report in last_index_stats.extraction_reports:
            report_by_source[str(report.source_path)] = report

    quality_by_source = pipeline.index.file_quality_by_path or {}
    source_keys = set(source_chunk_counts) | set(quality_by_source) | set(report_by_source)
    if not source_keys:
        return pd.DataFrame(
            columns=[
                "source",
                "doc_name",
                "ext",
                "chunks",
                "pages",
                "extractor",
                "status",
                "page_coverage",
                "chars_per_page",
                "short_chunk_ratio",
                "empty_page_ratio",
                "poisoned_page_ratio",
                "switch_reason",
            ]
        )

    rows: list[dict[str, Any]] = []
    for source in sorted(source_keys):
        report = report_by_source.get(source)
        quality = quality_by_source.get(source, {})

        chunks = int(source_chunk_counts.get(source, int(quality.get("total_chunks") or 0)))
        pages = 0
        if report is not None:
            pages = int(report.total_pages or 0)
        elif source in source_pages:
            pages = len(source_pages[source])

        extractor = str(
            quality.get("extractor_used")
            or (report.extractor_name if report is not None else "unknown")
        )
        status = str(
            quality.get("status")
            or (report.status if report is not None else "pass")
        )

        page_coverage = _to_float(
            quality.get("page_coverage")
            if quality.get("page_coverage") is not None
            else (report.page_coverage if report is not None else None)
        )
        chars_per_page = _to_float(report.chars_per_page if report is not None else None)
        short_chunk_ratio = _to_float(report.short_chunk_ratio if report is not None else None)
        empty_page_ratio = _to_float(report.empty_page_ratio if report is not None else None)
        poisoned_page_ratio = _to_float(
            quality.get("poisoned_page_ratio")
            if quality.get("poisoned_page_ratio") is not None
            else (report.poisoned_page_ratio if report is not None else None)
        )
        switch_reason = quality.get("switch_reason")
        if switch_reason is None and report is not None:
            switch_reason = report.switch_reason

        rows.append(
            {
                "source": source,
                "doc_name": Path(source).name,
                "ext": Path(source).suffix.lower() or "n/a",
                "chunks": chunks,
                "pages": pages,
                "extractor": extractor,
                "status": status,
                "page_coverage": page_coverage,
                "chars_per_page": chars_per_page,
                "short_chunk_ratio": short_chunk_ratio,
                "empty_page_ratio": empty_page_ratio,
                "poisoned_page_ratio": poisoned_page_ratio,
                "switch_reason": switch_reason,
                "token_count": int(source_token_counts.get(source, 0)),
            }
        )

    return pd.DataFrame(rows)


def _build_corpus_overview(pipeline: RAGPipeline) -> dict[str, Any]:
    """Return deterministic corpus overview without LLM calls."""
    chunks = pipeline.index.indexed_chunks
    if not chunks:
        return {"ready": False, "message": "Индекс пуст. Сначала запустите индексацию."}

    source_counts: Counter[str] = Counter()
    token_counts: Counter[str] = Counter()
    for item in chunks:
        source_counts[item.chunk.source_path] += 1
        for token in item.tokenized:
            norm = (token or "").strip().lower()
            if len(norm) < 4:
                continue
            if not norm.isalpha():
                continue
            token_counts[norm] += 1

    top_sources = [
        {"source": src, "doc_name": Path(src).name, "chunks": count}
        for src, count in source_counts.most_common(5)
    ]
    top_terms = [{"token": t, "count": c} for t, c in token_counts.most_common(12)]

    summary_parts = [
        f"Документов в индексе: {len(source_counts)}.",
        f"Всего чанков: {len(chunks)}.",
    ]
    if top_sources:
        summary_parts.append(
            "Топ документов по объему: "
            + ", ".join(f"{row['doc_name']} ({row['chunks']})" for row in top_sources[:3])
            + "."
        )
    if top_terms:
        summary_parts.append(
            "Частые темы: " + ", ".join(term["token"] for term in top_terms[:8]) + "."
        )

    return {
        "ready": True,
        "summary": " ".join(summary_parts),
        "top_sources": top_sources,
        "top_terms": top_terms,
    }


def _render_stats_section(pipeline: RAGPipeline, last_index_stats: IndexStats | None) -> None:
    """Render detailed document statistics with filters and CSV export."""
    st.subheader("Подробная статистика документов")
    df = _build_document_stats_df(pipeline, last_index_stats)
    if df.empty:
        st.info("Статистика появится после первой индексации.")
        return

    docs_total = int(len(df))
    chunks_total = int(df["chunks"].sum())
    failed_docs = int((df["status"] == "hard_fail").sum())
    avg_page_coverage = float(df["page_coverage"].dropna().mean() or 0.0)
    avg_chars_per_page = float(df["chars_per_page"].dropna().mean() or 0.0)
    duplicate_files = int(last_index_stats.duplicate_files if last_index_stats else 0)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("docs_total", docs_total)
    k2.metric("chunks_total", chunks_total)
    k3.metric("failed_docs", failed_docs)
    k4.metric("avg_page_coverage", f"{avg_page_coverage:.3f}")
    k5.metric("avg_chars_per_page", f"{avg_chars_per_page:.1f}")
    k6.metric("duplicate_files", duplicate_files)

    st.markdown("Фильтры")
    f1, f2, f3 = st.columns(3)
    statuses = sorted(df["status"].dropna().unique().tolist())
    exts = sorted(df["ext"].dropna().unique().tolist())
    extractors = sorted(df["extractor"].dropna().unique().tolist())

    status_filter = f1.multiselect("status", statuses, default=statuses)
    ext_filter = f2.multiselect("ext", exts, default=exts)
    extractor_filter = f3.multiselect("extractor", extractors, default=extractors)

    filtered = df[
        df["status"].isin(status_filter)
        & df["ext"].isin(ext_filter)
        & df["extractor"].isin(extractor_filter)
    ].copy()

    sort_columns = [
        "doc_name",
        "chunks",
        "pages",
        "status",
        "extractor",
        "page_coverage",
        "chars_per_page",
        "short_chunk_ratio",
        "empty_page_ratio",
        "poisoned_page_ratio",
    ]
    s1, s2 = st.columns(2)
    sort_by = s1.selectbox("Сортировка", sort_columns, index=1)
    sort_asc = s2.checkbox("По возрастанию", value=False)
    filtered = filtered.sort_values(by=sort_by, ascending=sort_asc, na_position="last")

    view_columns = [
        "doc_name",
        "source",
        "ext",
        "chunks",
        "pages",
        "extractor",
        "status",
        "page_coverage",
        "chars_per_page",
        "short_chunk_ratio",
        "empty_page_ratio",
        "poisoned_page_ratio",
        "switch_reason",
    ]
    st.dataframe(filtered[view_columns], use_container_width=True, hide_index=True)

    csv_bytes = filtered[view_columns].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Скачать CSV статистику",
        data=csv_bytes,
        file_name="document_stats.csv",
        mime="text/csv",
        use_container_width=True,
    )


_ensure_session_defaults()
session_root, uploads_dir, union_dir = _session_dirs()


with st.sidebar:
    st.header("Настройки")
    env_path = st.text_input(".env path", value=".env")
    pipeline = _load_pipeline(env_path if env_path else None)

    st.caption(
        f"Scope: `{pipeline.settings.giga_scope}`\n\n"
        f"Chat model: `{pipeline.settings.giga_chat_model}`\n\n"
        f"Embedding model: `{pipeline.settings.giga_embedding_model}`"
    )

    demo_data_dir = st.text_input("Папка demo-документов", value=pipeline.settings.data_dir)
    source_mode = st.selectbox("Источник индексации", SOURCE_MODES, index=0)

    st.markdown("### Загрузка документов")
    st.caption(
        f"Лимиты: до {MAX_FILES_PER_SESSION} файлов за сессию, "
        f"до {MAX_FILE_SIZE_MB} MB на файл."
    )
    uploaded_items = st.file_uploader(
        "Добавьте документы (pdf/txt/csv/md)",
        type=["pdf", "txt", "csv", "md"],
        accept_multiple_files=True,
    )

    if st.button("Сохранить upload-файлы", use_container_width=True):
        st.session_state.upload_feedback = _save_uploaded_files(uploaded_items, uploads_dir)

    feedback = st.session_state.upload_feedback
    if isinstance(feedback, dict):
        if feedback.get("saved"):
            st.success(f"Сохранено: {len(feedback['saved'])} file(s)")
        if feedback.get("duplicates"):
            st.info(f"Дубликаты по sha256: {feedback['duplicates']}")
        if feedback.get("skipped_limit"):
            st.warning(
                f"Пропущено из-за лимита {MAX_FILES_PER_SESSION} файлов: {feedback['skipped_limit']}"
            )
        if feedback.get("unsupported"):
            st.warning(f"Неподдерживаемые типы: {len(feedback['unsupported'])}")
        if feedback.get("too_large"):
            st.warning(f"Слишком большие файлы (> {MAX_FILE_SIZE_MB} MB): {len(feedback['too_large'])}")

    uploaded_files = _list_supported_files(uploads_dir)
    st.caption(f"Uploaded files in session: {len(uploaded_files)}")
    with st.expander("Uploaded files list"):
        if uploaded_files:
            rows = [
                {
                    "file": p.name,
                    "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
                    "path": str(p),
                }
                for p in uploaded_files
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.write("Пока пусто.")

    extractor_options = ["pymupdf4llm", "docling", "unstructured"]
    default_extractor = (
        pipeline.settings.default_extractor
        if pipeline.settings.default_extractor in extractor_options
        else "pymupdf4llm"
    )
    extractor = st.selectbox("Extractor", extractor_options, index=extractor_options.index(default_extractor))

    profile_options = ["demo-fast", "full-quality"]
    default_profile = (
        pipeline.settings.ingest_profile
        if pipeline.settings.ingest_profile in profile_options
        else "demo-fast"
    )
    profile = st.selectbox("Ingestion profile", profile_options, index=profile_options.index(default_profile))
    fast_mode = st.checkbox("Fast mode", value=(extractor == "pymupdf4llm"))

    reranker_keys = list(available_rerankers().keys())
    default_reranker = (
        pipeline.settings.default_reranker
        if pipeline.settings.default_reranker in reranker_keys
        else reranker_keys[0]
    )
    reranker = st.selectbox("Reranker", reranker_keys, index=reranker_keys.index(default_reranker))

    retrieve_top_k = st.slider(
        "Retrieve top-k",
        min_value=10,
        max_value=200,
        value=pipeline.settings.retrieve_top_k,
        step=5,
    )
    rerank_top_n = st.slider(
        "Rerank top-n",
        min_value=3,
        max_value=50,
        value=pipeline.settings.rerank_top_n,
        step=1,
    )
    final_top_k = st.slider(
        "Final context top-k",
        min_value=1,
        max_value=20,
        value=pipeline.settings.final_top_k,
        step=1,
    )

    if st.button("Selfcheck API", use_container_width=True):
        with st.spinner("Проверяю доступ к models/chat/embeddings..."):
            st.session_state.selfcheck_result = pipeline.selfcheck(
                check_chat=True,
                check_embeddings=True,
                force=True,
            )

    if st.session_state.selfcheck_result:
        if st.session_state.selfcheck_result.get("ok"):
            st.success("Selfcheck: OK")
        else:
            st.error("Selfcheck: FAIL")
        with st.expander("Selfcheck details"):
            st.json(st.session_state.selfcheck_result)

    if st.button("Preload rerankers", use_container_width=True):
        with st.spinner(f"Загружаю выбранный reranker: {reranker} ..."):
            st.session_state.prewarm_result = pipeline.prewarm_rerankers([reranker])

    warmed = pipeline.warmed_rerankers()
    selected_warm = pipeline.is_reranker_warm(reranker)
    st.caption(f"Selected reranker warm: {'yes' if selected_warm else 'no'}")
    st.caption(f"Warmed rerankers: {', '.join(warmed) if warmed else 'none'}")
    if st.session_state.prewarm_result:
        with st.expander("Prewarm details"):
            st.json(st.session_state.prewarm_result)

    if st.button("Индексировать документы", type="primary", use_container_width=True):
        selected_source_dir, source_error = _prepare_source_dir(
            source_mode=source_mode,
            demo_dir=Path(demo_data_dir).expanduser().resolve(),
            uploads_dir=uploads_dir,
            union_dir=union_dir,
        )
        if source_error:
            st.error(source_error)
        else:
            with st.spinner("Идет индексация..."):
                stats = pipeline.index_documents(
                    data_dir=str(selected_source_dir),
                    preferred_extractor=extractor,
                    fast_mode=fast_mode,
                    profile=profile,
                )
            st.session_state.last_index_stats = stats
            st.success(
                f"Готово: files={stats.indexed_files}, chunks={stats.indexed_chunks}, "
                f"duplicates={stats.duplicate_files}, failed={stats.failed_files}"
            )
            with st.expander("Extraction reports"):
                st.json(
                    [
                        {
                            "source": r.source_path,
                            "extractor": r.extractor_name,
                            "chunks": r.total_chunks,
                            "chars_per_page": round(r.chars_per_page, 2),
                            "empty_page_ratio": round(r.empty_page_ratio, 4),
                            "short_chunk_ratio": round(r.short_chunk_ratio, 4),
                            "has_table_elements": r.has_table_elements,
                            "status": r.status,
                            "switch_reason": r.switch_reason,
                            "page_coverage": round(r.page_coverage, 4),
                            "fallback_path": r.fallback_path,
                            "low_quality_pages": r.low_quality_pages,
                            "attempts": r.attempts,
                            "ocr_backend_effective": r.ocr_backend_effective,
                            "ocr_fallback_path": r.ocr_fallback_path,
                        }
                        for r in stats.extraction_reports
                    ]
                )


index_chunks = len(pipeline.index.indexed_chunks)
index_state = "ready" if index_chunks > 0 else "empty"

selfcheck_payload = st.session_state.selfcheck_result
if not selfcheck_payload:
    selfcheck_state = "not_run"
elif bool(selfcheck_payload.get("ok")):
    selfcheck_state = "ok"
else:
    selfcheck_state = "fail"

reranker_warm = "yes" if pipeline.is_reranker_warm(reranker) else "no"

st.subheader("Готовность к запросу")
status_cols = st.columns(3)
status_cols[0].metric("Index state", index_state, delta=f"chunks={index_chunks}")
status_cols[1].metric("Selfcheck", selfcheck_state)
status_cols[2].metric("Reranker warm", reranker_warm, delta=reranker)

_render_stats_section(pipeline, st.session_state.last_index_stats)

st.subheader("Подсказки для запросов")
st.caption(
    "Guardrail строгий: лучше задавать in-domain вопросы по содержимому корпуса."
)
example_cols = st.columns(3)
if example_cols[0].button("Пример: темы 2024", use_container_width=True):
    st.session_state.query_input = "Какие ключевые темы в отчете Сбера за 2024 год?"
if example_cols[1].button("Пример: сравнение 2015/2024", use_container_width=True):
    st.session_state.query_input = "Сравни ключевые темы документов 2015 и 2024 годов."
if example_cols[2].button("Пример: OOD", use_container_width=True):
    st.session_state.query_input = "Кто полетел на Луну первым?"

if st.button("Показать обзор корпуса", use_container_width=True):
    st.session_state.corpus_overview = _build_corpus_overview(pipeline)

overview = st.session_state.corpus_overview
if isinstance(overview, dict):
    if not overview.get("ready"):
        st.info(str(overview.get("message") or "Индекс пуст."))
    else:
        st.success(str(overview.get("summary")))
        with st.expander("Top sources / terms"):
            st.json(
                {
                    "top_sources": overview.get("top_sources", []),
                    "top_terms": overview.get("top_terms", []),
                }
            )

query = st.text_area(
    "Ваш вопрос",
    key="query_input",
    height=120,
    placeholder="Например: о чем отчет Сбера за 2015 год?",
)

if not pipeline.is_reranker_warm(reranker):
    st.info("Реранкер не прогрет: первый запрос может выполняться дольше из-за загрузки модели.")

if st.button("Спросить", use_container_width=True, disabled=not query.strip()):
    with st.spinner("Подготовка preflight и выполнение retrieval + rerank + generation..."):
        result = pipeline.ask(
            query=query.strip(),
            reranker_name=reranker,
            retrieve_top_k=retrieve_top_k,
            rerank_top_n=rerank_top_n,
            final_top_k=final_top_k,
        )
    st.session_state.history.append(
        {
            "query": query.strip(),
            "answer": result.answer,
            "citations": result.citations,
            "trace": result.trace,
            "context": result.context_chunks,
        }
    )

for item in reversed(st.session_state.history):
    st.markdown("---")
    st.markdown(f"### Вопрос\n{item['query']}")
    st.markdown(f"### Ответ\n{item['answer']}")
    st.caption(f"Reranker used: {item['trace'].reranker_used}")

    with st.expander("Источники"):
        for idx, c in enumerate(item["citations"], start=1):
            st.write(
                f"[{idx}] {c['source_path']} | page={c.get('page')} | "
                f"rerank={c.get('rerank_score'):.4f} | fusion={c.get('fusion_score'):.4f}"
            )

    with st.expander("Debug trace"):
        st.json(
            {
                "query": item["trace"].original_query,
                "rewrites": item["trace"].rewritten_queries,
                "extractor_used": item["trace"].extractor_used,
                "reranker_used": item["trace"].reranker_used,
                "reranker_cached": item["trace"].reranker_cached,
                "reranker_load_ms": round(item["trace"].reranker_load_ms, 2),
                "timings_ms": item["trace"].timings_ms,
                "grounded_refusal": item["trace"].grounded_refusal,
                "grounded_reason": item["trace"].grounded_reason,
                "final_extractor_used": item["trace"].final_extractor_used,
                "switch_reason": item["trace"].switch_reason,
                "page_coverage": item["trace"].page_coverage,
                "low_quality_pages": item["trace"].low_quality_pages,
                "dense_disabled": item["trace"].dense_disabled,
                "dense_disable_reason": item["trace"].dense_disable_reason,
            }
        )

    with st.expander("Top context chunks"):
        for c in item["context"]:
            st.code(
                f"source={c.source_path} page={c.page} "
                f"bm25={c.bm25_score:.4f} dense={c.dense_score:.4f} "
                f"fusion={c.fusion_score:.4f} rerank={c.rerank_score:.4f}\n\n{c.text[:1200]}",
                language="text",
            )
