"""Streamlit frontend for RAG v3."""

from __future__ import annotations

import streamlit as st

from rag_system.config import load_settings
from rag_system.pipeline import RAGPipeline
from rag_system.rerankers.factory import available_rerankers


st.set_page_config(page_title="RAG v3", layout="wide")
st.title("RAG v3: Hybrid + Multi-Reranker + Docling-first")


@st.cache_resource
def _load_pipeline(env_path: str | None) -> RAGPipeline:
    """Create cached pipeline object."""
    settings = load_settings(env_path)
    return RAGPipeline(settings)


if "history" not in st.session_state:
    st.session_state.history = []
if "selfcheck_result" not in st.session_state:
    st.session_state.selfcheck_result = None
if "prewarm_result" not in st.session_state:
    st.session_state.prewarm_result = None


with st.sidebar:
    st.header("Настройки")
    env_path = st.text_input(".env path", value=".env")
    pipeline = _load_pipeline(env_path if env_path else None)

    st.caption(
        f"Scope: `{pipeline.settings.giga_scope}`\n\n"
        f"Chat model: `{pipeline.settings.giga_chat_model}`\n\n"
        f"Embedding model: `{pipeline.settings.giga_embedding_model}`"
    )

    data_dir = st.text_input("Папка документов", value=pipeline.settings.data_dir)
    extractor_options = ["pymupdf4llm", "docling", "unstructured"]
    default_extractor = (
        pipeline.settings.default_extractor
        if pipeline.settings.default_extractor in extractor_options
        else "pymupdf4llm"
    )
    extractor = st.selectbox("Extractor", extractor_options, index=extractor_options.index(default_extractor))
    profile_options = ["demo-fast", "full-quality"]
    default_profile = pipeline.settings.ingest_profile if pipeline.settings.ingest_profile in profile_options else "demo-fast"
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
        with st.spinner("Идет индексация..."):
            stats = pipeline.index_documents(
                data_dir=data_dir,
                preferred_extractor=extractor,
                fast_mode=fast_mode,
                profile=profile,
            )
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


query = st.text_area("Ваш вопрос", height=120, placeholder="Например: о чем отчет Сбера за 2015 год?")

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
