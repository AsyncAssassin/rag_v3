"""End-to-end RAG pipeline orchestration."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import time
from typing import Any

from .config import Settings, load_settings
from .indexing import GigaEmbeddingClient, HybridIndex, IndexBuilder
from .llm import GigaLLMClient
from .logging_utils import get_logger
from .rerankers.base import BaseReranker
from .rerankers.factory import available_rerankers, create_reranker
from .retrieval import HybridRetriever, apply_source_diversity, apply_year_aware_source_boost
from .selfcheck import run_preflight
from .types import AnswerResult, IndexStats, QueryTrace, RetrievedChunk
from .utils import now_ms, rrf_fusion


LOGGER = get_logger()
YEAR_PATTERN = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")


class RAGPipeline:
    """Main app-level service combining indexing, retrieval, reranking, and generation."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.index: HybridIndex = HybridIndex.load(self.settings.index_dir)
        self.embed_client = GigaEmbeddingClient(self.settings)
        self.llm_client = GigaLLMClient(self.settings)
        self._reranker_cache: dict[str, BaseReranker] = {}
        self._selfcheck_cache: dict[str, dict[str, Any]] = {}
        self._selfcheck_cache_ts: dict[str, float] = {}

        if self.settings.prewarm_rerankers:
            try:
                self.prewarm_rerankers()
            except Exception as exc:
                LOGGER.warning("Auto-prewarm failed: %s", exc)

    def selfcheck(
        self,
        *,
        check_chat: bool = True,
        check_embeddings: bool = True,
        force: bool = False,
        ttl_sec: int | None = None,
    ) -> dict[str, Any]:
        """Run/cached preflight checks for model accessibility."""
        cache_key = f"chat={int(check_chat)}|emb={int(check_embeddings)}"
        ttl = self.settings.preflight_ttl_sec if ttl_sec is None else max(0, int(ttl_sec))
        ts = self._selfcheck_cache_ts.get(cache_key)
        if (
            not force
            and cache_key in self._selfcheck_cache
            and ts is not None
            and (time.time() - ts) <= ttl
        ):
            return dict(self._selfcheck_cache[cache_key])

        if not force:
            disk_cached = self._load_disk_preflight(cache_key=cache_key, ttl_sec=ttl)
            if disk_cached is not None:
                self._selfcheck_cache[cache_key] = dict(disk_cached)
                self._selfcheck_cache_ts[cache_key] = time.time()
                return dict(disk_cached)

        result = run_preflight(
            self.settings,
            check_chat=check_chat,
            check_embeddings=check_embeddings,
        )
        self._selfcheck_cache[cache_key] = dict(result)
        self._selfcheck_cache_ts[cache_key] = time.time()
        self._save_disk_preflight(cache_key=cache_key, payload=result)
        return result

    def ensure_runtime(
        self,
        *,
        check_chat: bool = True,
        check_embeddings: bool = True,
        preflight_ttl_sec: int | None = None,
    ) -> dict[str, Any]:
        """Ensure required API capabilities are available, otherwise raise with diagnostics."""
        result = self.selfcheck(
            check_chat=check_chat,
            check_embeddings=check_embeddings,
            ttl_sec=preflight_ttl_sec,
        )
        if result.get("ok"):
            return result

        failed = result.get("failed_checks") or []
        details = "; ".join(f"{item['name']}: {item['detail']}" for item in failed) if failed else "unknown failure"
        raise RuntimeError(f"Selfcheck failed ({details})")

    def _normalize_reranker_name(self, name: str | None) -> str:
        """Normalize reranker key with fallback to defaults."""
        raw = (name or self.settings.default_reranker or "amberoad").strip().lower()
        if raw in {"default"}:
            return "amberoad"
        return raw

    def get_or_create_reranker(
        self,
        name: str | None,
        *,
        warmup: bool = False,
    ) -> tuple[BaseReranker, bool, float]:
        """Get cached reranker or create a new one, optionally warming runtime."""
        key = self._normalize_reranker_name(name)
        reranker = self._reranker_cache.get(key)
        was_cached = reranker is not None

        if reranker is None:
            reranker = create_reranker(key, cache_dir=self.settings.reranker_cache_dir)
            self._reranker_cache[key] = reranker

        load_ms = 0.0
        if warmup and not reranker.is_warmed():
            t0 = now_ms()
            reranker.warmup()
            load_ms = now_ms() - t0

        return reranker, was_cached, float(load_ms)

    def is_reranker_warm(self, name: str | None) -> bool:
        """Return whether reranker runtime is already loaded."""
        key = self._normalize_reranker_name(name)
        reranker = self._reranker_cache.get(key)
        return bool(reranker and reranker.is_warmed())

    def warmed_rerankers(self) -> list[str]:
        """List warmed reranker backends currently cached in memory."""
        return sorted([k for k, v in self._reranker_cache.items() if v.is_warmed()])

    def prewarm_rerankers(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        """Warmup one or many rerankers and return latency report."""
        targets = names or list(available_rerankers().keys())
        rows: list[dict[str, Any]] = []
        for name in targets:
            t0 = now_ms()
            try:
                reranker, was_cached, load_ms = self.get_or_create_reranker(name, warmup=True)
                rows.append(
                    {
                        "backend": name,
                        "model": reranker.model_name,
                        "ok": True,
                        "was_cached": was_cached,
                        "load_ms": round(load_ms, 2),
                        "total_ms": round(now_ms() - t0, 2),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "backend": name,
                        "model": None,
                        "ok": False,
                        "error": str(exc),
                        "total_ms": round(now_ms() - t0, 2),
                    }
                )
        return rows

    def index_documents(
        self,
        data_dir: str | None = None,
        preferred_extractor: str | None = None,
        fast_mode: bool = False,
        reset_index: bool = False,
        profile: str | None = None,
    ) -> IndexStats:
        """Build/update index from local documents."""
        self.ensure_runtime(check_chat=False, check_embeddings=True)
        builder = IndexBuilder(self.settings)
        idx, stats = builder.build_or_update(
            data_dir=data_dir or self.settings.data_dir,
            preferred_extractor=preferred_extractor or self.settings.default_extractor,
            fast_mode=fast_mode,
            reset_index=reset_index,
            profile=profile or self.settings.ingest_profile,
        )
        self.index = idx
        return stats

    def ask(
        self,
        query: str,
        reranker_name: str | None = None,
        retrieve_top_k: int | None = None,
        rerank_top_n: int | None = None,
        final_top_k: int | None = None,
        skip_preflight: bool = False,
        preflight_ttl_sec: int | None = None,
    ) -> AnswerResult:
        """Run full RAG flow for a query and return answer with trace."""
        if not self.index.indexed_chunks:
            return self._empty_answer(query, reranker_name or self.settings.default_reranker)

        if not skip_preflight:
            self.ensure_runtime(
                check_chat=True,
                check_embeddings=True,
                preflight_ttl_sec=preflight_ttl_sec,
            )

        retrieve_k = retrieve_top_k or self.settings.retrieve_top_k
        rerank_n = rerank_top_n or self.settings.rerank_top_n
        final_k = final_top_k or self.settings.final_top_k

        timings: dict[str, float] = {}

        t0 = now_ms()
        rewritten = self.llm_client.rewrite_query(query, n=self.settings.rewrite_n)
        timings["rewrite"] = now_ms() - t0

        retriever = HybridRetriever(self.index, self.embed_client, rrf_k=self.settings.rrf_k)

        t1 = now_ms()
        rank_lists: list[list[str]] = []
        merged_candidates: dict[str, RetrievedChunk] = {}
        dense_disabled = False
        dense_disable_reason: str | None = None

        for q in rewritten:
            candidates, debug = retriever.retrieve(q, top_k=retrieve_k)
            rank_lists.append(debug.bm25_ranked_ids)
            rank_lists.append(debug.dense_ranked_ids)
            if debug.dense_disabled:
                dense_disabled = True
                if dense_disable_reason is None:
                    dense_disable_reason = debug.dense_disable_reason

            for c in candidates:
                existing = merged_candidates.get(c.chunk_id)
                if existing is None:
                    merged_candidates[c.chunk_id] = c
                else:
                    existing.bm25_score = max(existing.bm25_score, c.bm25_score)
                    existing.dense_score = max(existing.dense_score, c.dense_score)
                    existing.fusion_score = max(existing.fusion_score, c.fusion_score)

        fused_scores = rrf_fusion(rank_lists, rrf_k=self.settings.rrf_k)
        sorted_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)[:retrieve_k]

        candidates_for_rerank: list[RetrievedChunk] = []
        for cid in sorted_ids:
            chunk = merged_candidates.get(cid)
            if chunk is None:
                continue
            chunk.fusion_score = float(fused_scores[cid])
            candidates_for_rerank.append(chunk)
        candidates_for_rerank = apply_year_aware_source_boost(
            candidates_for_rerank,
            query=query,
            enabled=bool(getattr(self.settings, "retrieval_year_boost_enabled", True)),
            boost=float(getattr(self.settings, "retrieval_year_boost", 0.12)),
        )
        candidates_for_rerank = apply_source_diversity(
            candidates_for_rerank,
            top_k=retrieve_k,
            enabled=bool(getattr(self.settings, "retrieval_source_diversity_enabled", True)),
            max_chunks_per_source=int(getattr(self.settings, "retrieval_source_max_chunks_per_source", 2)),
        )

        timings["retrieve"] = now_ms() - t1

        t2 = now_ms()
        selected_reranker = self._normalize_reranker_name(reranker_name)
        reranker, was_cached, reranker_load_ms = self.get_or_create_reranker(selected_reranker, warmup=True)
        timings["reranker_prepare"] = now_ms() - t2
        timings["reranker_load"] = reranker_load_ms

        t3 = now_ms()
        rerank_result = reranker.rerank(query, candidates_for_rerank, top_n=rerank_n)
        final_chunks = rerank_result.chunks[:final_k]
        final_chunks = self._apply_year_retention_safeguard(
            query=query,
            retriever_candidates=candidates_for_rerank,
            reranked_chunks=rerank_result.chunks,
            final_chunks=final_chunks,
            final_k=final_k,
        )
        timings["rerank"] = now_ms() - t3
        timings["rerank_model"] = rerank_result.latency_ms

        grounded_refusal, grounded_reason = self._should_refuse_due_to_grounding(
            final_chunks,
            reranker_backend=rerank_result.backend,
        )
        t4 = now_ms()
        if grounded_refusal:
            answer_text = "Не нашел подтверждения в документах корпуса."
        else:
            answer_text = self.llm_client.generate_answer(query, final_chunks)
        timings["generate"] = now_ms() - t4

        visible_chunks = [] if grounded_refusal else final_chunks
        citations = [
            {
                "source_path": c.source_path,
                "page": c.page,
                "chunk_id": c.chunk_id,
                "rerank_score": c.rerank_score,
                "fusion_score": c.fusion_score,
            }
            for c in visible_chunks
        ]

        detected_extractor = self._detect_extractor_name(final_chunks)
        trace = QueryTrace(
            original_query=query,
            rewritten_queries=rewritten,
            extractor_used=detected_extractor,
            reranker_used=rerank_result.backend,
            reranker_cached=was_cached,
            reranker_load_ms=reranker_load_ms,
            retrieve_top_k=retrieve_k,
            rerank_top_n=rerank_n,
            final_top_k=final_k,
            timings_ms=timings,
            grounded_refusal=grounded_refusal,
            grounded_reason=grounded_reason,
            final_extractor_used=detected_extractor,
            dense_disabled=dense_disabled,
            dense_disable_reason=dense_disable_reason,
        )

        return AnswerResult(
            answer=answer_text,
            citations=citations,
            context_chunks=visible_chunks,
            trace=trace,
        )

    def _empty_answer(self, query: str, reranker_name: str) -> AnswerResult:
        """Build default answer for empty index state."""
        trace = QueryTrace(
            original_query=query,
            rewritten_queries=[query],
            extractor_used=None,
            reranker_used=reranker_name,
            reranker_cached=False,
            reranker_load_ms=0.0,
            retrieve_top_k=self.settings.retrieve_top_k,
            rerank_top_n=self.settings.rerank_top_n,
            final_top_k=self.settings.final_top_k,
            timings_ms={},
            grounded_refusal=False,
            grounded_reason=None,
            final_extractor_used=None,
            dense_disabled=False,
            dense_disable_reason=None,
        )
        return AnswerResult(
            answer="Индекс пуст. Сначала запустите индексацию документов.",
            citations=[],
            context_chunks=[],
            trace=trace,
        )

    @staticmethod
    def _detect_extractor_name(chunks: list[RetrievedChunk]) -> str | None:
        """Infer extractor name from chunk metadata."""
        names = [str(c.metadata.get("extractor")) for c in chunks if c.metadata.get("extractor")]
        if not names:
            return None
        return Counter(names).most_common(1)[0][0]

    @staticmethod
    def _extract_years(text: str) -> set[int]:
        """Extract 4-digit years from free-form text."""
        years: set[int] = set()
        for match in YEAR_PATTERN.findall(str(text or "")):
            try:
                years.add(int(match))
            except ValueError:
                continue
        return years

    def _apply_year_retention_safeguard(
        self,
        *,
        query: str,
        retriever_candidates: list[RetrievedChunk],
        reranked_chunks: list[RetrievedChunk],
        final_chunks: list[RetrievedChunk],
        final_k: int,
    ) -> list[RetrievedChunk]:
        """Retain at least one matching-year source in final top-k when score gap is acceptable."""
        if final_k <= 0:
            return []
        if not bool(getattr(self.settings, "rerank_year_retention_enabled", True)):
            return list(final_chunks[:final_k])

        query_years = self._extract_years(query)
        if not query_years:
            return list(final_chunks[:final_k])

        retriever_has_matching_year = any(
            bool(query_years & self._extract_years(str(chunk.source_path or "")))
            for chunk in retriever_candidates
        )
        if not retriever_has_matching_year:
            return list(final_chunks[:final_k])

        selected = list(final_chunks[:final_k])
        if any(bool(query_years & self._extract_years(str(chunk.source_path or ""))) for chunk in selected):
            return selected

        candidates_by_id = {chunk.chunk_id for chunk in selected}
        matching_candidate = next(
            (
                chunk
                for chunk in reranked_chunks
                if chunk.chunk_id not in candidates_by_id
                and bool(query_years & self._extract_years(str(chunk.source_path or "")))
            ),
            None,
        )
        if matching_candidate is None:
            return selected

        gap_threshold = float(getattr(self.settings, "rerank_year_retention_max_score_gap", 0.35))
        if selected:
            cutoff_score = float(selected[-1].rerank_score)
            candidate_score = float(matching_candidate.rerank_score)
            score_gap = cutoff_score - candidate_score
            if score_gap > gap_threshold:
                return selected

        if len(selected) >= final_k and selected:
            selected = selected[:-1]
        selected.append(matching_candidate)
        selected.sort(key=lambda chunk: float(chunk.rerank_score), reverse=True)
        return selected[:final_k]

    def _resolve_top_score_threshold(self, reranker_backend: str | None, top_score: float) -> float:
        """Resolve grounded guardrail threshold depending on reranker score scale."""
        backend = (reranker_backend or "").strip().lower()
        if backend == "amberoad":
            return float(self.settings.grounded_min_top_rerank_score_amberoad)
        if backend == "bge_m3":
            return float(self.settings.grounded_min_top_rerank_score_bge_m3)
        if backend == "jina_multilingual":
            return float(self.settings.grounded_min_top_rerank_score_jina_multilingual)

        base = float(self.settings.grounded_min_top_rerank_score)
        # Safety fallback for unknown backends with normalized score scale.
        if abs(float(top_score)) <= 1.5 and base > 1.5:
            return 0.75
        return base

    def _should_refuse_due_to_grounding(
        self,
        chunks: list[RetrievedChunk],
        reranker_backend: str | None = None,
    ) -> tuple[bool, str | None]:
        """Apply lightweight confidence guardrail before answer generation."""
        if not chunks:
            return True, "no_context"
        if len(chunks) < self.settings.grounded_min_chunks:
            return True, f"too_few_chunks:{len(chunks)}"
        top_score = float(chunks[0].rerank_score)
        threshold = self._resolve_top_score_threshold(reranker_backend, top_score)
        if top_score < threshold:
            return True, f"low_rerank_score:{round(top_score, 4)}<thr:{round(threshold, 4)}"
        total_chars = sum(len(c.text or "") for c in chunks)
        if total_chars < int(self.settings.grounded_min_total_context_chars):
            return True, f"low_context_chars:{total_chars}"
        return False, None

    def _preflight_cache_path(self) -> Path:
        """Return path to persisted preflight cache."""
        cache_root = Path(self.settings.reranker_cache_dir).expanduser().resolve().parent
        cache_root.mkdir(parents=True, exist_ok=True)
        return cache_root / "preflight_cache.json"

    def _load_disk_preflight(self, *, cache_key: str, ttl_sec: int) -> dict[str, Any] | None:
        """Load cached preflight result from disk when still fresh."""
        path = self._preflight_cache_path()
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("items", {})
            item = rows.get(cache_key)
            if not isinstance(item, dict):
                return None
            ts = float(item.get("ts", 0.0))
            if (time.time() - ts) > ttl_sec:
                return None
            result = item.get("result")
            if isinstance(result, dict):
                return result
        except Exception:
            return None
        return None

    def _save_disk_preflight(self, *, cache_key: str, payload: dict[str, Any]) -> None:
        """Persist latest preflight result for cross-process cache reuse."""
        path = self._preflight_cache_path()
        data = {"items": {}}
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    data = {"items": {}}
        except Exception:
            data = {"items": {}}

        items = data.setdefault("items", {})
        if not isinstance(items, dict):
            items = {}
            data["items"] = items
        items[cache_key] = {"ts": time.time(), "result": payload}
        try:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            return
