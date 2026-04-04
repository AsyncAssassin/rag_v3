"""Unit tests for reranker score-shape and score-count contracts."""

from __future__ import annotations

import numpy as np
import pytest

from rag_system.rerankers.base import BaseReranker
from rag_system.rerankers.hf_cross_encoder import HuggingFaceCrossEncoderReranker, _ModelRuntime
from rag_system.types import RetrievedChunk


class _DummyCrossEncoder:
    def __init__(self, output):
        self.output = output

    def predict(self, pairs, batch_size=16):  # noqa: ARG002
        return self.output


class _ShortScoresReranker(BaseReranker):
    backend = "test_short"

    def score(self, query: str, passages: list[str]) -> list[float]:  # noqa: ARG002
        return [0.9]


def _candidate(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source_path="/tmp/doc.pdf",
        page=1,
        element_type="text",
    )


def test_cross_encoder_2d_outputs_use_last_column() -> None:
    reranker = HuggingFaceCrossEncoderReranker(model_name="dummy")
    reranker._runtime = _ModelRuntime(  # type: ignore[attr-defined]
        mode="st_cross_encoder",
        model=_DummyCrossEncoder(np.asarray([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]], dtype=np.float32)),
    )
    scores = reranker.score("q", ["p1", "p2", "p3"])
    assert scores == pytest.approx([0.9, 0.8, 0.7], rel=1e-6)


def test_base_rerank_raises_on_score_count_mismatch() -> None:
    reranker = _ShortScoresReranker(model_name="dummy")
    candidates = [_candidate("1", "a"), _candidate("2", "b")]
    with pytest.raises(RuntimeError, match="score count mismatch"):
        reranker.rerank("query", candidates, top_n=2)
