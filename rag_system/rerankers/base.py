"""Base abstractions for reranking backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..types import RetrievedChunk
from ..utils import now_ms


@dataclass(slots=True)
class RerankResult:
    """Rerank output payload."""

    chunks: list[RetrievedChunk]
    latency_ms: float
    backend: str
    model_name: str


class BaseReranker(ABC):
    """Abstract reranker interface."""

    backend: str = "base"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._warmed = False

    @abstractmethod
    def score(self, query: str, passages: list[str]) -> list[float]:
        """Return relevance scores for (query, passage) pairs."""

    def warmup(self) -> None:
        """Warmup hook for expensive model backends."""
        self._warmed = True

    def is_warmed(self) -> bool:
        """Return whether backend runtime is already warmed."""
        return bool(self._warmed)

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_n: int) -> RerankResult:
        """Score candidates and return top-n by rerank score."""
        start = now_ms()
        passages = [c.text for c in candidates]
        scores = self.score(query, passages)
        self._warmed = True
        if len(scores) != len(candidates):
            raise RuntimeError(
                "Reranker score count mismatch "
                f"(backend={self.backend}, model={self.model_name}, "
                f"scores={len(scores)}, candidates={len(candidates)})"
            )

        for chunk, score in zip(candidates, scores):
            chunk.rerank_score = float(score)

        reranked = sorted(candidates, key=lambda x: x.rerank_score, reverse=True)[:top_n]
        latency = now_ms() - start
        return RerankResult(
            chunks=reranked,
            latency_ms=latency,
            backend=self.backend,
            model_name=self.model_name,
        )
