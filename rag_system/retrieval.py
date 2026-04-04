"""Hybrid retrieval: BM25 + dense + RRF fusion."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

from .indexing import GigaEmbeddingClient, HybridIndex
from .logging_utils import get_logger
from .types import RetrievedChunk
from .utils import cosine_similarity, rrf_fusion, tokenize


LOGGER = get_logger()
YEAR_PATTERN = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")


@dataclass(slots=True)
class RetrievalDebug:
    """Debug payload from retrieval stage."""

    bm25_ranked_ids: list[str]
    dense_ranked_ids: list[str]
    fused_ids: list[str]
    dense_disabled: bool = False
    dense_disable_reason: str | None = None


def apply_source_diversity(
    candidates: list[RetrievedChunk],
    *,
    top_k: int,
    enabled: bool,
    max_chunks_per_source: int,
) -> list[RetrievedChunk]:
    """Apply source-aware interleaving with per-source cap and deterministic backfill."""
    if top_k <= 0:
        return []
    pool = list(candidates)
    if not enabled or max_chunks_per_source <= 0 or len(pool) <= 1:
        return pool[:top_k]

    buckets: dict[str, list[RetrievedChunk]] = {}
    source_order: list[str] = []
    for chunk in pool:
        source = str(chunk.source_path or "")
        if source not in buckets:
            buckets[source] = []
            source_order.append(source)
        buckets[source].append(chunk)

    selected: list[RetrievedChunk] = []
    consumed_by_source: dict[str, int] = {src: 0 for src in source_order}

    while len(selected) < top_k:
        progressed = False
        for source in source_order:
            used = consumed_by_source[source]
            if used >= max_chunks_per_source:
                continue
            bucket = buckets[source]
            if used >= len(bucket):
                continue
            selected.append(bucket[used])
            consumed_by_source[source] = used + 1
            progressed = True
            if len(selected) >= top_k:
                break
        if not progressed:
            break

    if len(selected) >= top_k:
        return selected[:top_k]

    selected_ids = {item.chunk_id for item in selected}
    for chunk in pool:
        if len(selected) >= top_k:
            break
        if chunk.chunk_id in selected_ids:
            continue
        selected.append(chunk)
        selected_ids.add(chunk.chunk_id)

    return selected[:top_k]


def _extract_years(text: str) -> set[int]:
    """Extract 4-digit years from free-form text."""
    years: set[int] = set()
    for match in YEAR_PATTERN.findall(str(text or "")):
        try:
            years.add(int(match))
        except ValueError:
            continue
    return years


def apply_year_aware_source_boost(
    candidates: list[RetrievedChunk],
    *,
    query: str,
    enabled: bool,
    boost: float,
) -> list[RetrievedChunk]:
    """Promote candidates whose source year matches query year."""
    pool = list(candidates)
    if not enabled or boost <= 0.0 or len(pool) <= 1:
        return pool

    query_years = _extract_years(query)
    if not query_years:
        return pool

    ranked: list[tuple[float, int, RetrievedChunk]] = []
    any_boosted = False
    for idx, chunk in enumerate(pool):
        source_years = _extract_years(str(chunk.source_path or ""))
        bonus = float(boost) if (query_years & source_years) else 0.0
        if bonus > 0.0:
            any_boosted = True
        adjusted_score = float(chunk.fusion_score) + bonus
        ranked.append((adjusted_score, idx, chunk))

    if not any_boosted:
        return pool

    ranked.sort(key=lambda row: (-row[0], row[1]))
    return [row[2] for row in ranked]


class HybridRetriever:
    """Retrieves candidate chunks using sparse+dense fusion."""

    def __init__(self, index: HybridIndex, embed_client: GigaEmbeddingClient, rrf_k: int = 60) -> None:
        self.index = index
        self.embed_client = embed_client
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 50) -> tuple[list[RetrievedChunk], RetrievalDebug]:
        """Run BM25 and dense search, fuse with RRF, return candidates."""
        if not self.index.indexed_chunks:
            return [], RetrievalDebug([], [], [], dense_disabled=False, dense_disable_reason=None)

        bm25_ids, bm25_scores = self._bm25_search(query, top_k=top_k)
        dense_ids, dense_scores, dense_disable_reason = self._dense_search(query, top_k=top_k)
        dense_disabled = dense_disable_reason is not None

        fused = rrf_fusion([bm25_ids, dense_ids], rrf_k=self.rrf_k)
        fused_ids = sorted(fused.keys(), key=lambda doc_id: fused[doc_id], reverse=True)[:top_k]

        out: list[RetrievedChunk] = []
        for cid in fused_ids:
            item = self.index.chunk_map[cid]
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=item.chunk.text,
                    source_path=item.chunk.source_path,
                    page=item.chunk.page,
                    element_type=item.chunk.element_type,
                    bm25_score=float(bm25_scores.get(cid, 0.0)),
                    dense_score=float(dense_scores.get(cid, 0.0)),
                    fusion_score=float(fused.get(cid, 0.0)),
                    metadata=dict(item.chunk.metadata),
                )
            )

        return out, RetrievalDebug(
            bm25_ranked_ids=bm25_ids,
            dense_ranked_ids=dense_ids,
            fused_ids=fused_ids,
            dense_disabled=dense_disabled,
            dense_disable_reason=dense_disable_reason,
        )

    def _bm25_search(self, query: str, top_k: int) -> tuple[list[str], dict[str, float]]:
        """Run sparse BM25 search."""
        if self.index.bm25 is None:
            return [], {}

        tokens = tokenize(query)
        if not tokens:
            return [], {}

        scores = self.index.bm25.get_scores(tokens)
        scores_arr = np.asarray(scores, dtype=np.float32)
        if scores_arr.size == 0:
            return [], {}

        top_indices = np.argsort(scores_arr)[::-1][:top_k]
        ranked_ids = [self.index.chunk_ids[i] for i in top_indices]
        score_map = {self.index.chunk_ids[i]: float(scores_arr[i]) for i in top_indices}
        return ranked_ids, score_map

    def _dense_search(self, query: str, top_k: int) -> tuple[list[str], dict[str, float], str | None]:
        """Run dense semantic search over vector matrix."""
        if self.index.dense_matrix.size == 0:
            return [], {}, "dense_unavailable"

        query_vec = np.asarray(self.embed_client.embed_query(query), dtype=np.float32)
        if query_vec.ndim != 1:
            query_vec = query_vec.reshape(-1)

        matrix = self.index.dense_matrix
        expected_dim = int(matrix.shape[1]) if matrix.ndim == 2 else 0
        if expected_dim <= 0:
            return [], {}, "dense_unavailable"
        if query_vec.size != expected_dim:
            reason = f"dense_dim_mismatch:query={query_vec.size},index={expected_dim}"
            LOGGER.warning("Dense retrieval disabled for query due to %s", reason)
            return [], {}, reason

        scores = cosine_similarity(query_vec, self.index.dense_matrix)
        if scores.size == 0:
            return [], {}, "dense_empty_scores"

        top_indices = np.argsort(scores)[::-1][:top_k]
        ranked_ids = [self.index.chunk_ids[i] for i in top_indices]
        score_map = {self.index.chunk_ids[i]: float(scores[i]) for i in top_indices}
        return ranked_ids, score_map, None
