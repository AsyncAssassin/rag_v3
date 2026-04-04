"""Command-line interface for RAG v3."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
from pathlib import Path
import sys
import time


def _print_json(payload: dict) -> None:
    """Pretty-print dictionary as JSON."""
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _write_json(path: str | None, payload: dict) -> None:
    """Persist JSON payload when output path is provided."""
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")



def _cmd_index(args: argparse.Namespace) -> None:
    """Handle index command."""
    from .config import load_settings
    from .pipeline import RAGPipeline

    settings = load_settings(args.env)
    pipeline = RAGPipeline(settings)
    with contextlib.redirect_stdout(sys.stderr):
        stats = pipeline.index_documents(
            data_dir=args.data_dir,
            preferred_extractor=args.extractor,
            fast_mode=args.fast,
            reset_index=args.reset_index,
            profile=args.profile,
        )

    payload = {
        "profile": args.profile,
        "reset_index": bool(args.reset_index),
        "indexed_files": stats.indexed_files,
        "indexed_chunks": stats.indexed_chunks,
        "deduplicated_chunks": stats.deduplicated_chunks,
        "duplicate_files": stats.duplicate_files,
        "failed_files": stats.failed_files,
        "reports": [
            {
                "source_path": r.source_path,
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
                "poisoned_pages": r.poisoned_pages,
                "poisoned_page_ratio": round(r.poisoned_page_ratio, 4),
                "poison_signals": r.poison_signals,
                "attempts": r.attempts,
                "ocr_backend_effective": r.ocr_backend_effective,
                "ocr_fallback_path": r.ocr_fallback_path,
            }
            for r in stats.extraction_reports
        ],
    }
    _write_json(args.output, payload)
    _print_json(payload)



def _cmd_ask(args: argparse.Namespace) -> None:
    """Handle ask command."""
    from .config import load_settings
    from .pipeline import RAGPipeline

    settings = load_settings(args.env)
    pipeline = RAGPipeline(settings)
    selected_reranker = args.reranker or settings.default_reranker

    if not pipeline.is_reranker_warm(selected_reranker):
        print("Подготовка: preflight + загрузка reranker (первый запуск может быть дольше)...")
    else:
        print("Подготовка: preflight + reranker уже прогрет.")

    result = pipeline.ask(
        query=args.query,
        reranker_name=selected_reranker,
        retrieve_top_k=args.retrieve_top_k,
        rerank_top_n=args.rerank_top_n,
        final_top_k=args.final_top_k,
        skip_preflight=args.skip_preflight,
        preflight_ttl_sec=args.preflight_ttl_sec,
    )

    print("\n=== ANSWER ===\n")
    print(result.answer)

    print("\n=== CITATIONS ===\n")
    for i, c in enumerate(result.citations, start=1):
        print(
            f"[{i}] source={c['source_path']} page={c.get('page')} "
            f"rerank={c.get('rerank_score'):.4f} fusion={c.get('fusion_score'):.4f}"
        )

    print("\n=== TRACE ===\n")
    _print_json(
        {
            "query": result.trace.original_query,
                "rewrites": result.trace.rewritten_queries,
                "extractor_used": result.trace.extractor_used,
                "reranker_used": result.trace.reranker_used,
                "reranker_cached": result.trace.reranker_cached,
                "reranker_load_ms": round(result.trace.reranker_load_ms, 2),
                "timings_ms": result.trace.timings_ms,
                "grounded_refusal": result.trace.grounded_refusal,
                "grounded_reason": result.trace.grounded_reason,
                "final_extractor_used": result.trace.final_extractor_used,
                "switch_reason": result.trace.switch_reason,
                "page_coverage": result.trace.page_coverage,
                "low_quality_pages": result.trace.low_quality_pages,
                "dense_disabled": result.trace.dense_disabled,
                "dense_disable_reason": result.trace.dense_disable_reason,
            }
        )



def _discover_pdf_files(data_dir: str) -> list[str]:
    """Discover PDFs recursively under data_dir."""
    return [str(p) for p in sorted(Path(data_dir).rglob("*.pdf")) if p.is_file()]



def _is_success_status(status: str | None) -> bool:
    """Return true when extraction status is considered usable."""
    return str(status) in {"pass", "soft_fail"}


def _is_wall_clock_timeout_reason(reason: str | None) -> bool:
    """Detect timeout reason caused by overall wall-clock deadline."""
    value = str(reason or "").lower()
    return "wall_clock_deadline" in value or "deadline reached" in value


def _cmd_pdf_regression(args: argparse.Namespace) -> None:
    """Run ingestion regression checks over local PDF corpus."""
    from .config import load_settings
    from .extractors.base import ChunkingOptions, ExtractionQualityThresholds
    from .extractors.factory import ExtractorOrchestrator

    settings = load_settings(args.env)
    timeout_sec = settings.extract_timeout_sec if args.timeout_sec is None else int(args.timeout_sec)
    default_wall_clock_cap_sec = int(getattr(settings, "pdf_regression_wallclock_cap_sec", 900))
    wall_clock_cap_sec = (
        default_wall_clock_cap_sec if args.wall_clock_cap_sec is None else max(0, int(args.wall_clock_cap_sec))
    )
    quality_thresholds = ExtractionQualityThresholds(
        min_chars_per_page=float(settings.extract_min_chars_per_page),
        max_empty_page_ratio=float(settings.extract_max_empty_page_ratio),
        max_short_chunk_ratio=float(settings.extract_max_short_chunk_ratio),
        max_escaped_seq_per_1k=float(settings.extract_max_escaped_seq_per_1k),
        max_backslash_per_1k=float(settings.extract_max_backslash_per_1k),
        max_control_char_ratio=float(settings.extract_max_control_char_ratio),
        poisoned_page_ratio_hard=float(settings.extract_poisoned_page_ratio_hard),
    )
    chunking = ChunkingOptions(
        mode=settings.chunk_mode,
        token_prose=settings.chunk_tokens_prose,
        token_table=settings.chunk_tokens_table,
        overlap_token_prose=settings.chunk_overlap_prose,
        overlap_token_table=settings.chunk_overlap_table,
        char_prose=settings.chunk_chars_prose,
        char_table=settings.chunk_chars_table,
        overlap_char_prose=settings.chunk_overlap_chars_prose,
        overlap_char_table=settings.chunk_overlap_chars_table,
    )
    orchestrator = ExtractorOrchestrator(
        languages=("rus", "eng"),
        quality_thresholds=quality_thresholds,
        extract_timeout_sec=timeout_sec,
        extract_timeout_base_sec=settings.extract_timeout_base_sec,
        extract_timeout_per_100_pages_sec=settings.extract_timeout_per_100_pages_sec,
        extract_timeout_per_10mb_sec=settings.extract_timeout_per_10mb_sec,
        extract_timeout_max_sec=settings.extract_timeout_max_sec,
        extract_full_quality_docling_primary_max_sec=getattr(
            settings,
            "extract_full_quality_docling_primary_max_sec",
            300,
        ),
        extract_full_quality_docling_secondary_max_sec=getattr(
            settings,
            "extract_full_quality_docling_secondary_max_sec",
            240,
        ),
        extract_full_quality_pymupdf_max_sec=getattr(
            settings,
            "extract_full_quality_pymupdf_max_sec",
            180,
        ),
        extract_full_quality_reserve_window_fallback_sec=getattr(
            settings,
            "extract_full_quality_reserve_window_fallback_sec",
            180,
        ),
        extract_full_quality_min_stage_start_sec=getattr(
            settings,
            "extract_full_quality_min_stage_start_sec",
            15,
        ),
        extract_full_quality_unstructured_min_remaining_sec=getattr(
            settings,
            "extract_full_quality_unstructured_min_remaining_sec",
            90,
        ),
        extract_prefer_best_usable=getattr(
            settings,
            "extract_prefer_best_usable",
            True,
        ),
        extract_best_usable_min_coverage_gap=getattr(
            settings,
            "extract_best_usable_min_coverage_gap",
            0.15,
        ),
        unstructured_min_merged_chunk_chars=getattr(
            settings,
            "unstructured_min_merged_chunk_chars",
            220,
        ),
        page_window_size=settings.page_window_size,
        min_page_coverage=settings.extract_min_page_coverage,
        unstructured_targeted_only=settings.unstructured_targeted_only,
        docling_ocr_backend=settings.docling_ocr_backend,
        docling_ocr_fallbacks=settings.docling_ocr_fallbacks,
        docling_ocr_langs_easyocr=settings.docling_ocr_langs_easyocr,
        docling_ocr_langs_tesseract=settings.docling_ocr_langs_tesseract,
        chunking=chunking,
    )
    files = _discover_pdf_files(args.data_dir)

    rows: list[dict] = []
    started_total = time.perf_counter()
    deadline_monotonic = (started_total + float(wall_clock_cap_sec)) if wall_clock_cap_sec > 0 else None
    aborted_after_wall_clock = False
    total_files = len(files)
    for idx, fp in enumerate(files, start=1):
        elapsed_sec = time.perf_counter() - started_total
        if wall_clock_cap_sec > 0 and elapsed_sec >= float(wall_clock_cap_sec):
            aborted_after_wall_clock = True
            print(
                (
                    f"[pdf-regression] wall-clock cap reached before file {idx}/{total_files} "
                    f"(elapsed={round(elapsed_sec, 2)}s cap={wall_clock_cap_sec}s)"
                ),
                file=sys.stderr,
                flush=True,
            )
            break

        started = time.perf_counter()
        print(
            f"[pdf-regression] start {idx}/{total_files}: {fp}",
            file=sys.stderr,
            flush=True,
        )
        try:
            with contextlib.redirect_stdout(sys.stderr):
                outcome = orchestrator.extract_with_policy(
                    file_path=fp,
                    preferred=args.extractor,
                    fast_mode=args.fast,
                    profile=args.profile,
                    deadline_monotonic=deadline_monotonic,
                )
            rows.append(
                {
                    "file_path": fp,
                    "ok": _is_success_status(outcome.status),
                    "status": outcome.status,
                    "extractor_used": outcome.extractor_used,
                    "fallback_path": outcome.fallback_path,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "chunks": outcome.stats.total_chunks,
                    "chars_per_page": round(outcome.stats.chars_per_page, 2),
                    "empty_page_ratio": round(outcome.stats.empty_page_ratio, 4),
                    "short_chunk_ratio": round(outcome.stats.short_chunk_ratio, 4),
                    "has_table_elements": outcome.stats.has_table_elements,
                    "page_coverage": round(outcome.stats.page_coverage, 4),
                    "switch_reason": outcome.switch_reason,
                    "low_quality_pages": outcome.low_quality_pages,
                    "poisoned_pages": outcome.stats.poisoned_pages,
                    "poisoned_page_ratio": round(outcome.stats.poisoned_page_ratio, 4),
                    "poison_signals": outcome.stats.poison_signals,
                    "attempts": outcome.attempts,
                    "notes": outcome.notes,
                    "ocr_backend_effective": outcome.stats.ocr_backend_effective,
                    "ocr_fallback_path": outcome.stats.ocr_fallback_path,
                    "error": None,
                }
            )
            if _is_wall_clock_timeout_reason(outcome.switch_reason):
                aborted_after_wall_clock = True
                print(
                    (
                        f"[pdf-regression] wall-clock cap reached during file {idx}/{total_files} "
                        f"(elapsed={round(time.perf_counter() - started_total, 2)}s cap={wall_clock_cap_sec}s)"
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            print(
                (
                    f"[pdf-regression] done  {idx}/{total_files}: status={outcome.status} "
                    f"duration_ms={round((time.perf_counter() - started) * 1000, 2)} "
                    f"reason={outcome.switch_reason or 'none'}"
                ),
                file=sys.stderr,
                flush=True,
            )
            if aborted_after_wall_clock:
                break
        except Exception as exc:
            reason = str(exc)
            if isinstance(exc, TimeoutError) and not reason.lower().startswith("timeout:"):
                reason = f"timeout:{reason}"
            rows.append(
                {
                    "file_path": fp,
                    "ok": False,
                    "status": "hard_fail",
                    "extractor_used": "n/a",
                    "fallback_path": [],
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "chunks": 0,
                    "chars_per_page": 0.0,
                    "empty_page_ratio": 1.0,
                    "short_chunk_ratio": 1.0,
                    "has_table_elements": False,
                    "page_coverage": 0.0,
                    "switch_reason": reason,
                    "low_quality_pages": [],
                    "poisoned_pages": [],
                    "poisoned_page_ratio": 0.0,
                    "poison_signals": {},
                    "attempts": [],
                    "notes": [],
                    "ocr_backend_effective": None,
                    "ocr_fallback_path": [],
                    "error": reason,
                }
            )
            print(
                (
                    f"[pdf-regression] done  {idx}/{total_files}: status=hard_fail "
                    f"duration_ms={round((time.perf_counter() - started) * 1000, 2)} reason={reason}"
                ),
                file=sys.stderr,
                flush=True,
            )
            if _is_wall_clock_timeout_reason(reason):
                aborted_after_wall_clock = True
                break

    duration_sec = round(time.perf_counter() - started_total, 2)
    payload = {
        "total_files": total_files,
        "processed_files": len(rows),
        "remaining_files": max(0, total_files - len(rows)),
        "aborted_after_wall_clock": aborted_after_wall_clock,
        "wall_clock_cap_sec": int(wall_clock_cap_sec),
        "duration_sec": duration_sec,
        "ok_files": sum(1 for r in rows if bool(r.get("ok"))),
        "failed_files": sum(1 for r in rows if r["status"] == "hard_fail"),
        "profile": args.profile,
        "rows": rows,
    }
    _write_json(args.output, payload)
    if args.log_file:
        log_target = Path(args.log_file)
        log_target.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"{row['status']} | {row['file_path']} | extractor={row['extractor_used']} | "
            f"fallback={','.join(row['fallback_path']) if row['fallback_path'] else 'n/a'} | "
            f"duration_ms={row['duration_ms']} | error={row['error'] or 'none'}"
            for row in rows
        ]
        log_target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    _print_json(payload)


def _parse_reranker_list(raw: str | None) -> list[str]:
    """Parse comma-separated reranker list."""
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _cmd_selfcheck(args: argparse.Namespace) -> None:
    """Run API/model preflight checks."""
    from .config import load_settings
    from .pipeline import RAGPipeline

    settings = load_settings(args.env)
    pipeline = RAGPipeline(settings)
    result = pipeline.selfcheck(
        check_chat=not args.skip_chat,
        check_embeddings=not args.skip_embeddings,
        force=True,
    )
    _print_json(result)
    if not result.get("ok"):
        raise RuntimeError("Selfcheck failed. Inspect failed_checks for details.")


def _cmd_prewarm(args: argparse.Namespace) -> None:
    """Warmup reranker runtimes and report latency."""
    from .config import load_settings
    from .pipeline import RAGPipeline
    from .rerankers.factory import available_rerankers

    settings = load_settings(args.env)
    pipeline = RAGPipeline(settings)
    targets = _parse_reranker_list(args.rerankers) or list(available_rerankers().keys())

    selfcheck = None
    if not args.skip_selfcheck:
        selfcheck = pipeline.selfcheck(check_chat=False, check_embeddings=True, force=True)
        if not selfcheck.get("ok"):
            _print_json({"selfcheck": selfcheck})
            raise RuntimeError("Prewarm aborted: embeddings selfcheck failed.")

    rows = pipeline.prewarm_rerankers(targets)
    payload = {
        "selfcheck": selfcheck,
        "rows": rows,
        "warmed_rerankers": pipeline.warmed_rerankers(),
    }
    _print_json(payload)


def _load_queries(queries_file: str | None) -> list[str]:
    """Load benchmark queries from txt/jsonl, or return defaults."""
    if not queries_file:
        return [
            "Что такое регуляризация в глубоком обучении?",
            "Какие цели у AI Act в ЕС?",
            "Как считается Recall@K в RAG?",
            "Сравни embedding и reranker в RAG пайплайне",
        ]

    path = Path(queries_file)
    if not path.exists():
        raise FileNotFoundError(f"queries file not found: {queries_file}")

    if path.suffix.lower() in {".txt", ".md"}:
        return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    queries: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "query" in obj:
            queries.append(str(obj["query"]))
    return queries



def _collect_candidates_once(pipeline, query: str, retrieve_k: int):
    """Collect fixed candidate set for reranker A/B/C benchmarking."""
    from .retrieval import HybridRetriever, apply_source_diversity, apply_year_aware_source_boost
    from .types import RetrievedChunk
    from .utils import rrf_fusion

    rewritten = pipeline.llm_client.rewrite_query(query, n=pipeline.settings.rewrite_n)
    retriever = HybridRetriever(pipeline.index, pipeline.embed_client, rrf_k=pipeline.settings.rrf_k)

    rank_lists: list[list[str]] = []
    merged = {}

    for q in rewritten:
        candidates, dbg = retriever.retrieve(q, top_k=retrieve_k)
        rank_lists.append(dbg.bm25_ranked_ids)
        rank_lists.append(dbg.dense_ranked_ids)
        for c in candidates:
            merged[c.chunk_id] = c

    fused = rrf_fusion(rank_lists, rrf_k=pipeline.settings.rrf_k)
    sorted_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:retrieve_k]

    final_candidates = []
    for cid in sorted_ids:
        c = merged.get(cid)
        if c is None:
            item = pipeline.index.chunk_map.get(cid)
            if item is None:
                continue
            c = RetrievedChunk(
                chunk_id=cid,
                text=item.chunk.text,
                source_path=item.chunk.source_path,
                page=item.chunk.page,
                element_type=item.chunk.element_type,
                bm25_score=0.0,
                dense_score=0.0,
                fusion_score=0.0,
                metadata=dict(item.chunk.metadata),
            )
        c.fusion_score = float(fused[cid])
        final_candidates.append(c)

    runtime_settings = getattr(pipeline, "settings", None)
    boosted = apply_year_aware_source_boost(
        final_candidates,
        query=query,
        enabled=bool(getattr(runtime_settings, "retrieval_year_boost_enabled", True)),
        boost=float(getattr(runtime_settings, "retrieval_year_boost", 0.12)),
    )
    diversified = apply_source_diversity(
        boosted,
        top_k=int(retrieve_k),
        enabled=bool(getattr(runtime_settings, "retrieval_source_diversity_enabled", True)),
        max_chunks_per_source=int(getattr(runtime_settings, "retrieval_source_max_chunks_per_source", 2)),
    )
    return rewritten, diversified



def _relevant_chunk_ids(index, relevant_patterns: list[str]) -> set[str]:
    """Resolve relevant chunk ids from source-path patterns."""
    relevant_ids: set[str] = set()
    patterns = [str(p).strip().lower() for p in relevant_patterns if str(p).strip()]
    if not patterns:
        return relevant_ids
    for src, chunk_ids in index.path_to_chunk_ids.items():
        src_norm = str(src).lower()
        if any(pattern in src_norm for pattern in patterns):
            relevant_ids.update(str(cid) for cid in chunk_ids)
    return relevant_ids



def _relevant_source_paths(index, relevant_patterns: list[str]) -> list[str]:
    """Resolve relevant source paths from source-path patterns."""
    patterns = [str(p).strip().lower() for p in relevant_patterns if str(p).strip()]
    if not patterns:
        return []

    matched: list[str] = []
    for src in sorted(index.path_to_chunk_ids.keys()):
        src_str = str(src)
        src_norm = src_str.lower()
        if any(pattern in src_norm for pattern in patterns):
            matched.append(src_str)
    return matched


def _dedupe_keep_order(items: list[str]) -> list[str]:
    """Dedupe list preserving first occurrence order."""
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _cmd_benchmark_rerank(args: argparse.Namespace) -> None:
    """Benchmark A/B/C rerankers on fixed candidate sets."""
    from .config import load_settings
    from .pipeline import RAGPipeline
    from .rerankers.factory import available_rerankers
    from .utils import now_ms

    settings = load_settings(args.env)
    pipeline = RAGPipeline(settings)

    if not pipeline.index.indexed_chunks:
        raise RuntimeError("Index is empty. Run index command first.")

    queries = _load_queries(args.queries_file)
    backends = [args.only] if args.only else list(available_rerankers().keys())

    if not args.no_prewarm:
        pipeline.prewarm_rerankers(backends)

    rows = []
    for query in queries:
        rewritten, fixed_candidates = _collect_candidates_once(pipeline, query, args.retrieve_top_k)
        for backend in backends:
            reranker, was_cached, load_ms = pipeline.get_or_create_reranker(backend, warmup=True)
            t0 = now_ms()
            result = reranker.rerank(query, copy.deepcopy(fixed_candidates), top_n=args.rerank_top_n)
            elapsed = now_ms() - t0
            rows.append(
                {
                    "query": query,
                    "backend": backend,
                    "model": result.model_name,
                    "reranker_cached": was_cached,
                    "reranker_load_ms": round(load_ms, 2),
                    "latency_ms": round(elapsed, 2),
                    "rerank_latency_ms": round(result.latency_ms, 2),
                    "top1_source": result.chunks[0].source_path if result.chunks else None,
                    "top1_page": result.chunks[0].page if result.chunks else None,
                    "top1_score": round(result.chunks[0].rerank_score, 5) if result.chunks else None,
                    "rewrites": rewritten,
                }
            )

    _print_json({"rows": rows})


def _cmd_quality_gate(args: argparse.Namespace) -> None:
    """Run retrieval quality gate on a mini goldset."""
    from .config import load_settings
    from .eval import build_ragas_adapters, evaluate_ragas, evaluate_retrieval
    from .pipeline import RAGPipeline

    settings = load_settings(args.env)
    pipeline = RAGPipeline(settings)
    if not pipeline.index.indexed_chunks:
        raise RuntimeError("Index is empty. Run index command first.")

    gold_path = Path(args.goldset)
    if not gold_path.exists():
        raise FileNotFoundError(f"goldset not found: {gold_path}")

    rows: list[dict] = []
    ragas_samples: list[dict] = []
    queries_meta: list[dict] = []

    for line in gold_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        query = str(obj["query"])
        relevant_patterns = [str(x) for x in obj.get("relevant_patterns", [])]
        _, fixed_candidates = _collect_candidates_once(pipeline, query, args.retrieve_top_k)
        retrieved_ids_stage = [str(c.chunk_id) for c in fixed_candidates]
        retrieved_sources_stage = _dedupe_keep_order(
            [str(c.source_path) for c in fixed_candidates if c.source_path]
        )
        relevant_sources_stage = _relevant_source_paths(pipeline.index, relevant_patterns)
        relevant_ids = _relevant_chunk_ids(pipeline.index, relevant_patterns)
        if not relevant_sources_stage:
            raise RuntimeError(
                "No relevant sources matched goldset patterns for "
                f"query={query!r}; patterns={relevant_patterns!r}"
            )

        result = pipeline.ask(
            query=query,
            reranker_name=args.reranker,
            retrieve_top_k=args.retrieve_top_k,
            rerank_top_n=args.rerank_top_n,
            final_top_k=args.final_top_k,
            skip_preflight=args.skip_preflight,
            preflight_ttl_sec=args.preflight_ttl_sec,
        )

        rows.append(
            {
                "query": query,
                "retrieved_ids": retrieved_sources_stage,
                "relevant_ids": list(relevant_sources_stage),
            }
        )
        queries_meta.append(
            {
                "query": query,
                "relevant_patterns": relevant_patterns,
                "retrieved_ids_stage": retrieved_ids_stage,
                "retrieved_sources_stage": retrieved_sources_stage,
                "relevant_sources_stage": list(relevant_sources_stage),
                "relevant_chunk_ids": sorted(relevant_ids),
                "top_sources": [c.source_path for c in fixed_candidates[:3]],
                "grounded_refusal": result.trace.grounded_refusal,
                "grounded_reason": result.trace.grounded_reason,
            }
        )
        ragas_samples.append(
            {
                "user_input": query,
                "response": result.answer,
                "retrieved_contexts": [c.text for c in result.context_chunks],
            }
        )

    retrieval = evaluate_retrieval(rows, k=args.metric_k)
    retrieval_payload = {
        "mean_recall_at_k": round(retrieval.mean_recall_at_k, 4),
        "mean_mrr": round(retrieval.mean_mrr, 4),
        "mean_ndcg_at_k": round(retrieval.mean_ndcg_at_k, 4),
        "k": args.metric_k,
    }

    thresholds = {
        "min_recall_at_k": args.min_recall,
        "min_mrr": args.min_mrr,
        "min_ndcg_at_k": args.min_ndcg,
    }
    gate_ok = (
        retrieval.mean_recall_at_k >= args.min_recall
        and retrieval.mean_mrr >= args.min_mrr
        and retrieval.mean_ndcg_at_k >= args.min_ndcg
    )

    ragas_payload = None
    ragas_judge_payload = None
    if args.with_ragas:
        llm, emb, judge_meta = build_ragas_adapters(
            settings,
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
        )
        ragas_scores = evaluate_ragas(ragas_samples, llm=llm, embeddings=emb)
        for metric_name, metric_value in ragas_scores.items():
            value = float(metric_value)
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise RuntimeError(f"Invalid RAGAS score for {metric_name}: {value}")
        ragas_payload = {k: round(v, 4) for k, v in ragas_scores.items()}
        ragas_judge_payload = {
            "provider": judge_meta["provider"],
            "model": judge_meta["model"],
        }

    payload = {
        "ok": bool(gate_ok),
        "reranker": args.reranker,
        "metric_stage": "retriever_candidates",
        "metric_entity": "source_path",
        "retrieval": retrieval_payload,
        "thresholds": thresholds,
        "queries": queries_meta,
        "ragas": ragas_payload,
        "ragas_judge": ragas_judge_payload,
    }
    _write_json(args.output, payload)
    _print_json(payload)



def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser with all subcommands."""
    parser = argparse.ArgumentParser(description="RAG v3 CLI")
    parser.add_argument("--env", type=str, default=None, help="Path to .env file")

    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Build/update hybrid index")
    p_index.add_argument("--data-dir", type=str, required=True)
    p_index.add_argument(
        "--extractor",
        type=str,
        default=None,
        choices=["docling", "pymupdf4llm", "unstructured"],
        help="Extractor backend override (default: from config DEFAULT_EXTRACTOR)",
    )
    p_index.add_argument("--fast", action="store_true", help="Use fast extractor path")
    p_index.add_argument("--profile", type=str, default=None, help="ingestion profile: demo-fast/full-quality")
    p_index.add_argument("--reset-index", action="store_true", help="Ignore existing index and rebuild from scratch")
    p_index.add_argument("--output", type=str, default=None, help="Path to write JSON report")
    p_index.set_defaults(func=_cmd_index)

    p_ask = sub.add_parser("ask", help="Ask question against indexed docs")
    p_ask.add_argument("query", type=str)
    p_ask.add_argument("--reranker", type=str, default=None)
    p_ask.add_argument("--retrieve-top-k", type=int, default=None)
    p_ask.add_argument("--rerank-top-n", type=int, default=None)
    p_ask.add_argument("--final-top-k", type=int, default=None)
    p_ask.add_argument("--skip-preflight", action="store_true", help="Skip runtime preflight checks before ask")
    p_ask.add_argument("--preflight-ttl-sec", type=int, default=None, help="TTL for cached preflight checks")
    p_ask.set_defaults(func=_cmd_ask)

    p_reg = sub.add_parser("pdf-regression", help="Run PDF extraction regression")
    p_reg.add_argument("--data-dir", type=str, required=True)
    p_reg.add_argument("--extractor", type=str, default="docling")
    p_reg.add_argument("--fast", action="store_true")
    p_reg.add_argument("--timeout-sec", type=int, default=None, help="Hard cap per extractor attempt in seconds")
    p_reg.add_argument(
        "--wall-clock-cap-sec",
        type=int,
        default=None,
        help=(
            "Optional wall-clock cap for the whole command; can abort current file with timeout row "
            "(0 disables; default from PDF_REGRESSION_WALLCLOCK_CAP_SEC)"
        ),
    )
    p_reg.add_argument("--profile", type=str, default="full-quality", help="ingestion profile: demo-fast/full-quality")
    p_reg.add_argument("--output", type=str, default=None, help="Path to write JSON report")
    p_reg.add_argument("--log-file", type=str, default=None, help="Path to write line-based debug log")
    p_reg.set_defaults(func=_cmd_pdf_regression)

    p_check = sub.add_parser("selfcheck", help="Run GigaChat/GigaEmbeddings preflight checks")
    p_check.add_argument("--skip-chat", action="store_true", help="Skip chat completion check")
    p_check.add_argument("--skip-embeddings", action="store_true", help="Skip embeddings check")
    p_check.set_defaults(func=_cmd_selfcheck)

    p_warm = sub.add_parser("prewarm", help="Preload reranker runtimes")
    p_warm.add_argument(
        "--rerankers",
        type=str,
        default=None,
        help="Comma-separated rerankers (amberoad,bge_m3,jina_multilingual). Default: all",
    )
    p_warm.add_argument("--skip-selfcheck", action="store_true", help="Skip embeddings selfcheck before warmup")
    p_warm.set_defaults(func=_cmd_prewarm)

    p_bench = sub.add_parser("benchmark-rerank", help="Compare rerankers on fixed candidate set")
    p_bench.add_argument("--queries-file", type=str, default=None, help="txt/jsonl with queries")
    p_bench.add_argument("--only", type=str, default=None, help="Run one backend only")
    p_bench.add_argument("--retrieve-top-k", type=int, default=50)
    p_bench.add_argument("--rerank-top-n", type=int, default=10)
    p_bench.add_argument("--no-prewarm", action="store_true", help="Disable warmup before benchmark")
    p_bench.set_defaults(func=_cmd_benchmark_rerank)

    p_gate = sub.add_parser("quality-gate", help="Run retrieval quality gate on mini goldset")
    p_gate.add_argument("--goldset", type=str, default="artifacts/goldset_sber_qa.jsonl")
    p_gate.add_argument("--reranker", type=str, default="amberoad")
    p_gate.add_argument("--retrieve-top-k", type=int, default=50)
    p_gate.add_argument("--rerank-top-n", type=int, default=10)
    p_gate.add_argument("--final-top-k", type=int, default=8)
    p_gate.add_argument("--metric-k", type=int, default=10)
    p_gate.add_argument("--min-recall", type=float, default=0.55)
    p_gate.add_argument("--min-mrr", type=float, default=0.45)
    p_gate.add_argument("--min-ndcg", type=float, default=0.50)
    p_gate.add_argument("--with-ragas", action="store_true", help="Also compute RAGAS metrics")
    p_gate.add_argument(
        "--judge-provider",
        type=str,
        choices=["gigachat", "anthropic"],
        default=None,
        help="Judge provider for RAGAS metrics (default from env RAGAS_JUDGE_PROVIDER)",
    )
    p_gate.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Judge model override for RAGAS metrics (default from provider/env)",
    )
    p_gate.add_argument("--skip-preflight", action="store_true")
    p_gate.add_argument("--preflight-ttl-sec", type=int, default=None)
    p_gate.add_argument("--output", type=str, default=None)
    p_gate.set_defaults(func=_cmd_quality_gate)

    return parser



def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
