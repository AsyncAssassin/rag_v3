"""Evaluation utilities for retrieval and answer quality."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .utils import mean


def _normalize_ragas_score(value: float) -> float:
    """Normalize raw metric score to stable [0, 1] range."""
    numeric = float(value)
    if not math.isfinite(numeric):
        return 0.0
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


@dataclass(slots=True)
class RetrievalEvalRow:
    """Per-query retrieval metric row."""

    query: str
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float


@dataclass(slots=True)
class RetrievalEvalSummary:
    """Aggregated retrieval metrics."""

    mean_recall_at_k: float
    mean_mrr: float
    mean_ndcg_at_k: float
    per_query: list[RetrievalEvalRow]


def build_ragas_adapters(
    settings,
    *,
    judge_provider: str | None = None,
    judge_model: str | None = None,
) -> tuple[Any, Any, dict[str, str]]:
    """Build RAGAS-compatible adapters for selected judge provider."""
    provider = str(judge_provider or getattr(settings, "ragas_judge_provider", "gigachat")).strip().lower()
    if provider not in {"gigachat", "anthropic"}:
        raise RuntimeError(f"Unsupported judge provider: {provider}")

    resolved_model: str
    if provider == "anthropic":
        resolved_model = str(
            judge_model
            or getattr(settings, "ragas_judge_model", "")
            or "claude-sonnet-4-20250514"
        ).strip()
        if not resolved_model:
            resolved_model = "claude-sonnet-4-20250514"
    else:
        resolved_model = str(judge_model or getattr(settings, "giga_chat_model", "GigaChat-2-Max")).strip()
        if not resolved_model:
            resolved_model = "GigaChat-2-Max"

    if not settings.giga_api_key:
        raise RuntimeError("GIGA_API_KEY is required for RAGAS AnswerRelevance embeddings")

    anthropic_key = str(getattr(settings, "anthropic_api_key", "") or "").strip()
    if provider == "anthropic" and not anthropic_key:
        raise RuntimeError("ANTHROPIC_API_KEY (or CLAUDE_API_KEY) is not configured for anthropic judge provider")

    try:
        from ragas.llms import LangchainLLMWrapper
        from langchain_gigachat.embeddings import GigaChatEmbeddings
    except Exception as exc:
        raise RuntimeError("Failed to import RAGAS/GigaChat embedding adapters") from exc

    embeddings = GigaChatEmbeddings(
        credentials=settings.giga_api_key,
        scope=settings.giga_scope,
        verify_ssl_certs=False,
    )

    if provider == "gigachat":
        if not settings.giga_api_key:
            raise RuntimeError("GIGA_API_KEY is not configured for gigachat judge provider")
        try:
            from langchain_gigachat.chat_models import GigaChat
        except Exception as exc:
            raise RuntimeError("Failed to import GigaChat judge adapter") from exc

        llm = LangchainLLMWrapper(
            GigaChat(
                credentials=settings.giga_api_key,
                verify_ssl_certs=False,
                scope=settings.giga_scope,
                model=resolved_model,
                temperature=0.0,
            )
        )
    else:
        try:
            from langchain_anthropic import ChatAnthropic
        except Exception as exc:
            raise RuntimeError("Failed to import Anthropic judge adapter (install langchain-anthropic)") from exc

        llm = LangchainLLMWrapper(
            ChatAnthropic(
                anthropic_api_key=anthropic_key,
                model=resolved_model,
                temperature=0.0,
            )
        )

    meta = {
        "provider": provider,
        "model": resolved_model,
    }
    return llm, embeddings, meta


def build_ragas_gigachat_adapters(settings) -> tuple[Any, Any]:
    """Backward-compatible GigaChat judge adapter builder."""
    llm, embeddings, _ = build_ragas_adapters(settings, judge_provider="gigachat")
    return llm, embeddings



def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute recall@k."""
    if not relevant_ids:
        return 0.0
    if k <= 0:
        return 0.0
    top = retrieved_ids[:k]
    hit = len({doc_id for doc_id in top if doc_id in relevant_ids})
    return float(min(1.0, hit / len(relevant_ids)))



def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Compute reciprocal rank for first relevant doc."""
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return float(1.0 / i)
    return 0.0



def ndcg_from_relevance_list(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute NDCG@k from binary relevance labels."""
    if k <= 0:
        return 0.0
    if not relevant_ids:
        return 0.0

    top = retrieved_ids[:k]
    if not top:
        return 0.0

    dcg = 0.0
    for i, doc_id in enumerate(top):
        rel = 1.0 if doc_id in relevant_ids else 0.0
        if rel > 0:
            dcg += rel / math.log2(i + 2)

    ideal_hits = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)



def evaluate_retrieval(
    query_rows: list[dict[str, Any]],
    *,
    k: int = 10,
) -> RetrievalEvalSummary:
    """Evaluate retrieval metrics from query rows with gold relevance."""
    rows: list[RetrievalEvalRow] = []
    for row in query_rows:
        query = str(row["query"])
        retrieved = [str(x) for x in row.get("retrieved_ids", [])]
        relevant = {str(x) for x in row.get("relevant_ids", [])}

        rr = reciprocal_rank(retrieved, relevant)
        rec = recall_at_k(retrieved, relevant, k=k)
        nd = ndcg_from_relevance_list(retrieved, relevant, k=k)

        rows.append(
            RetrievalEvalRow(
                query=query,
                recall_at_k=rec,
                reciprocal_rank=rr,
                ndcg_at_k=nd,
            )
        )

    return RetrievalEvalSummary(
        mean_recall_at_k=mean([r.recall_at_k for r in rows]),
        mean_mrr=mean([r.reciprocal_rank for r in rows]),
        mean_ndcg_at_k=mean([r.ndcg_at_k for r in rows]),
        per_query=rows,
    )



def evaluate_ragas(
    samples: list[dict[str, Any]],
    *,
    llm,
    embeddings,
) -> dict[str, float]:
    """Compute optional RAGAS metrics for prepared samples.

    Expected sample keys:
    - user_input
    - response
    - retrieved_contexts (list[str])
    """
    try:
        import asyncio
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import AnswerRelevancy, ContextRelevance, Faithfulness
    except Exception as exc:
        raise RuntimeError("ragas dependencies are not available") from exc

    faith_metric = Faithfulness(llm=llm)
    answer_metric = AnswerRelevancy(llm=llm, embeddings=embeddings)
    context_metric = ContextRelevance(llm=llm)

    faith_scores: list[float] = []
    answer_scores: list[float] = []
    context_scores: list[float] = []

    for sample_row in samples:
        sample = SingleTurnSample(
            user_input=sample_row["user_input"],
            response=sample_row["response"],
            retrieved_contexts=list(sample_row.get("retrieved_contexts", [])),
        )
        faith_scores.append(float(asyncio.run(faith_metric.single_turn_ascore(sample))))
        answer_scores.append(float(asyncio.run(answer_metric.single_turn_ascore(sample))))
        context_scores.append(float(asyncio.run(context_metric.single_turn_ascore(sample))))

    return {
        "faithfulness": _normalize_ragas_score(mean(faith_scores)),
        "answer_relevance": _normalize_ragas_score(mean(answer_scores)),
        "context_relevance": _normalize_ragas_score(mean(context_scores)),
    }
