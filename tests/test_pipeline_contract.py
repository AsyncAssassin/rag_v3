"""Unit tests for pipeline behavior contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_system.pipeline import RAGPipeline
from rag_system.types import RetrievedChunk


def _settings_stub() -> SimpleNamespace:
    return SimpleNamespace(
        default_reranker="amberoad",
        retrieve_top_k=50,
        rerank_top_n=10,
        final_top_k=8,
        rewrite_n=1,
        rrf_k=60,
        grounded_min_chunks=3,
        grounded_min_total_context_chars=800,
        grounded_min_top_rerank_score=4.5,
        grounded_min_top_rerank_score_amberoad=2.5,
        grounded_min_top_rerank_score_bge_m3=0.85,
        grounded_min_top_rerank_score_jina_multilingual=0.75,
        rerank_year_retention_enabled=True,
        rerank_year_retention_max_score_gap=0.35,
    )


def test_ask_does_not_call_preflight_when_index_is_empty() -> None:
    pipe = object.__new__(RAGPipeline)
    pipe.settings = _settings_stub()
    pipe.index = SimpleNamespace(indexed_chunks=[])

    def _should_not_run(**kwargs):  # noqa: ARG001
        raise AssertionError("ensure_runtime should not run for empty index")

    pipe.ensure_runtime = _should_not_run
    result = RAGPipeline.ask(pipe, "тестовый вопрос", skip_preflight=False)
    assert "Индекс пуст" in result.answer


def test_grounded_refusal_hides_citations_and_context(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeRetriever:
        def __init__(self, *args, **kwargs):  # noqa: D401, ANN002, ANN003, ARG002
            pass

        def retrieve(self, query: str, top_k: int = 50):  # noqa: ARG002
            candidate = RetrievedChunk(
                chunk_id="c1",
                text="контекст",
                source_path="/tmp/doc.pdf",
                page=1,
                element_type="text",
                bm25_score=1.0,
                dense_score=0.5,
                fusion_score=0.0,
                metadata={"extractor": "pymupdf4llm"},
            )
            debug = SimpleNamespace(
                bm25_ranked_ids=["c1"],
                dense_ranked_ids=["c1"],
                fused_ids=["c1"],
                dense_disabled=False,
                dense_disable_reason=None,
            )
            return [candidate], debug

    class _DummyReranker:
        def rerank(self, query: str, candidates: list[RetrievedChunk], top_n: int):  # noqa: ARG002
            for chunk in candidates:
                chunk.rerank_score = 3.0
            return SimpleNamespace(
                chunks=candidates[:top_n],
                backend="amberoad",
                latency_ms=1.0,
            )

    monkeypatch.setattr("rag_system.pipeline.HybridRetriever", _FakeRetriever)

    pipe = object.__new__(RAGPipeline)
    pipe.settings = _settings_stub()
    pipe.index = SimpleNamespace(indexed_chunks=[object()], chunk_map={})
    pipe.embed_client = SimpleNamespace()
    pipe.llm_client = SimpleNamespace(
        rewrite_query=lambda query, n=1: [query],  # noqa: ARG005
        generate_answer=lambda query, chunks: (_ for _ in ()).throw(AssertionError("LLM should not run on refusal")),
    )
    pipe.ensure_runtime = lambda **kwargs: None
    pipe.get_or_create_reranker = lambda name, warmup=True: (_DummyReranker(), False, 0.0)  # noqa: ARG005
    pipe._should_refuse_due_to_grounding = lambda chunks, reranker_backend=None: (True, "forced_refusal")  # noqa: ARG005

    result = RAGPipeline.ask(pipe, "тест", skip_preflight=True)
    assert result.trace.grounded_refusal is True
    assert result.citations == []
    assert result.context_chunks == []


def _retrieved(chunk_id: str, score: float, text_len: int = 320) -> RetrievedChunk:
    chunk = RetrievedChunk(
        chunk_id=chunk_id,
        text="x" * text_len,
        source_path="/tmp/doc.pdf",
        page=1,
        element_type="text",
        bm25_score=0.0,
        dense_score=0.0,
        fusion_score=0.0,
        metadata={},
    )
    chunk.rerank_score = score
    return chunk


def _retrieved_source(chunk_id: str, source_path: str, score: float, text_len: int = 320) -> RetrievedChunk:
    chunk = RetrievedChunk(
        chunk_id=chunk_id,
        text="x" * text_len,
        source_path=source_path,
        page=1,
        element_type="text",
        bm25_score=0.0,
        dense_score=0.0,
        fusion_score=0.0,
        metadata={},
    )
    chunk.rerank_score = score
    return chunk


def test_grounding_guardrail_accepts_in_corpus_like_context_for_amberoad() -> None:
    pipe = object.__new__(RAGPipeline)
    pipe.settings = _settings_stub()
    chunks = [
        _retrieved("c1", score=3.2),
        _retrieved("c2", score=2.9),
        _retrieved("c3", score=2.7),
    ]
    refuse, reason = RAGPipeline._should_refuse_due_to_grounding(
        pipe,
        chunks,
        reranker_backend="amberoad",
    )
    assert refuse is False
    assert reason is None


def test_grounding_guardrail_refuses_out_of_corpus_like_context_for_amberoad() -> None:
    pipe = object.__new__(RAGPipeline)
    pipe.settings = _settings_stub()
    chunks = [
        _retrieved("c1", score=-1.1),
        _retrieved("c2", score=-1.5),
        _retrieved("c3", score=-1.9),
    ]
    refuse, reason = RAGPipeline._should_refuse_due_to_grounding(
        pipe,
        chunks,
        reranker_backend="amberoad",
    )
    assert refuse is True
    assert reason is not None and reason.startswith("low_rerank_score:")


def test_year_retention_keeps_matching_year_when_gap_small() -> None:
    pipe = object.__new__(RAGPipeline)
    pipe.settings = _settings_stub()
    query = "Какие ключевые темы в отчете Сбера за 2020 год?"

    retriever_candidates = [
        _retrieved_source("c1", "/tmp/Копия Сбер 2021.pdf", score=0.0),
        _retrieved_source("c2", "/tmp/Копия Сбер 2020.pdf", score=0.0),
        _retrieved_source("c3", "/tmp/Копия Сбер 2019.pdf", score=0.0),
    ]
    reranked = [
        _retrieved_source("c1", "/tmp/Копия Сбер 2021.pdf", score=0.92),
        _retrieved_source("c3", "/tmp/Копия Сбер 2019.pdf", score=0.81),
        _retrieved_source("c2", "/tmp/Копия Сбер 2020.pdf", score=0.66),
    ]
    final = reranked[:2]

    kept = RAGPipeline._apply_year_retention_safeguard(
        pipe,
        query=query,
        retriever_candidates=retriever_candidates,
        reranked_chunks=reranked,
        final_chunks=final,
        final_k=2,
    )
    kept_sources = [chunk.source_path for chunk in kept]
    assert any("2020" in source for source in kept_sources)


def test_year_retention_does_not_override_when_score_gap_large() -> None:
    pipe = object.__new__(RAGPipeline)
    settings = _settings_stub()
    settings.rerank_year_retention_max_score_gap = 0.2
    pipe.settings = settings
    query = "Что в отчете Сбера 2020 года?"

    retriever_candidates = [
        _retrieved_source("c1", "/tmp/Копия Сбер 2024.pdf", score=0.0),
        _retrieved_source("c2", "/tmp/Копия Сбер 2020.pdf", score=0.0),
    ]
    reranked = [
        _retrieved_source("c1", "/tmp/Копия Сбер 2024.pdf", score=0.93),
        _retrieved_source("c3", "/tmp/Копия Сбер 2023.pdf", score=0.88),
        _retrieved_source("c2", "/tmp/Копия Сбер 2020.pdf", score=0.40),
    ]
    final = reranked[:2]

    kept = RAGPipeline._apply_year_retention_safeguard(
        pipe,
        query=query,
        retriever_candidates=retriever_candidates,
        reranked_chunks=reranked,
        final_chunks=final,
        final_k=2,
    )
    kept_ids = [chunk.chunk_id for chunk in kept]
    assert kept_ids == ["c1", "c3"]
