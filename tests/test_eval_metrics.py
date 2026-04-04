"""Unit tests for retrieval metrics."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from rag_system.eval import (
    _normalize_ragas_score,
    build_ragas_adapters,
    evaluate_retrieval,
    ndcg_from_relevance_list,
    recall_at_k,
)



def test_evaluate_retrieval_returns_expected_means() -> None:
    summary = evaluate_retrieval(
        [
            {
                "query": "q1",
                "retrieved_ids": ["a", "b", "c"],
                "relevant_ids": ["a"],
            },
            {
                "query": "q2",
                "retrieved_ids": ["x", "y", "z"],
                "relevant_ids": ["y"],
            },
        ],
        k=2,
    )
    assert summary.mean_recall_at_k > 0
    assert summary.mean_mrr > 0
    assert 0 <= summary.mean_ndcg_at_k <= 1


def test_recall_at_k_ignores_duplicate_hits() -> None:
    assert recall_at_k(["a", "a"], {"a"}, k=2) == 1.0


def test_ndcg_partial_retrieval_penalizes_missing_relevant_docs() -> None:
    score = ndcg_from_relevance_list(["a", "x", "y"], {"a", "b"}, k=3)
    expected = 1.0 / (1.0 + 1.0 / math.log2(3))
    assert score == pytest.approx(expected, rel=1e-6)


def test_build_ragas_adapters_rejects_unknown_provider() -> None:
    settings = SimpleNamespace(
        ragas_judge_provider="gigachat",
        ragas_judge_model="claude-sonnet-4-20250514",
        giga_api_key="x",
        giga_scope="GIGACHAT_API_B2B",
        giga_chat_model="GigaChat-2-Max",
        anthropic_api_key="y",
    )
    with pytest.raises(RuntimeError, match="Unsupported judge provider"):
        build_ragas_adapters(settings, judge_provider="unknown")


def test_build_ragas_adapters_requires_anthropic_key_for_anthropic_provider() -> None:
    settings = SimpleNamespace(
        ragas_judge_provider="anthropic",
        ragas_judge_model="claude-sonnet-4-20250514",
        giga_api_key="x",
        giga_scope="GIGACHAT_API_B2B",
        giga_chat_model="GigaChat-2-Max",
        anthropic_api_key="",
    )
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        build_ragas_adapters(settings, judge_provider="anthropic")


def test_build_ragas_adapters_requires_giga_key_for_embeddings() -> None:
    settings = SimpleNamespace(
        ragas_judge_provider="anthropic",
        ragas_judge_model="claude-sonnet-4-20250514",
        giga_api_key="",
        giga_scope="GIGACHAT_API_B2B",
        giga_chat_model="GigaChat-2-Max",
        anthropic_api_key="sk-ant-test",
    )
    with pytest.raises(RuntimeError, match="GIGA_API_KEY"):
        build_ragas_adapters(settings, judge_provider="anthropic")


def test_normalize_ragas_score_sanitizes_nan_inf_and_range() -> None:
    assert _normalize_ragas_score(float("nan")) == 0.0
    assert _normalize_ragas_score(float("inf")) == 0.0
    assert _normalize_ragas_score(-0.5) == 0.0
    assert _normalize_ragas_score(1.7) == 1.0
    assert _normalize_ragas_score(0.42) == pytest.approx(0.42)
