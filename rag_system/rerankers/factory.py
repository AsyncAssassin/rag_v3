"""Factory for creating rerankers by configured backend key."""

from __future__ import annotations

from .base import BaseReranker
from .hf_cross_encoder import AmberoadReranker, BgeReranker, JinaReranker


def available_rerankers() -> dict[str, str]:
    """Return supported reranker key -> human label mapping."""
    return {
        "amberoad": "amberoad/bert-multilingual-passage-reranking-msmarco",
        "bge_m3": "BAAI/bge-reranker-v2-m3",
        "jina_multilingual": "jinaai/jina-reranker-v2-base-multilingual",
    }



def create_reranker(name: str, **kwargs) -> BaseReranker:
    """Create reranker instance by backend name."""
    normalized = (name or "").strip().lower()
    if normalized in {"amberoad", "default"}:
        return AmberoadReranker(**kwargs)
    if normalized in {"bge", "bge_m3", "baai"}:
        return BgeReranker(**kwargs)
    if normalized in {"jina", "jina_multilingual"}:
        return JinaReranker(**kwargs)
    raise ValueError(f"Unknown reranker backend: {name}")
