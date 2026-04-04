"""Utility functions for indexing, tokenization, hashing, and scoring."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Callable, Iterable, TypeVar

import numpy as np


TOKEN_RE = re.compile(r"[\w\-]+", flags=re.UNICODE)
MULTISPACE_RE = re.compile(r"\s+")
PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
_T = TypeVar("_T")

RU_STOPWORDS = {
    "и",
    "в",
    "во",
    "на",
    "с",
    "со",
    "к",
    "ко",
    "о",
    "об",
    "по",
    "за",
    "из",
    "для",
    "как",
    "что",
    "это",
    "при",
    "или",
    "а",
    "не",
    "но",
    "от",
    "до",
    "год",
}
EN_STOPWORDS = {
    "the",
    "and",
    "or",
    "for",
    "with",
    "from",
    "this",
    "that",
    "are",
    "is",
    "to",
    "of",
    "in",
    "on",
    "by",
    "as",
}



def sha256_text(text: str) -> str:
    """Hash text with SHA256."""
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()



def sha256_file(path: str | Path) -> str:
    """Hash file bytes with SHA256."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()



def tokenize(text: str) -> list[str]:
    """Simple unicode-aware tokenizer for BM25."""
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(text):
        tok = raw.lower().replace("ё", "е")
        tok = tok.strip("-_")
        if len(tok) < 2:
            continue
        if tok in RU_STOPWORDS or tok in EN_STOPWORDS:
            continue
        digits = sum(ch.isdigit() for ch in tok)
        if digits and digits / len(tok) >= 0.6:
            continue
        if not any(ch.isalpha() for ch in tok):
            continue
        tokens.append(tok)
    return tokens



def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarities query x matrix."""
    if matrix.size == 0:
        return np.array([], dtype=np.float32)
    q_norm = np.linalg.norm(query)
    m_norm = np.linalg.norm(matrix, axis=1)
    denom = np.maximum(q_norm * m_norm, 1e-12)
    return np.dot(matrix, query) / denom



def rrf_fusion(rank_lists: list[list[str]], rrf_k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion over multiple ranked id lists."""
    scores: dict[str, float] = {}
    for ranked_ids in rank_lists:
        for rank, doc_id in enumerate(ranked_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    return scores



def now_ms() -> float:
    """Current timestamp in milliseconds."""
    return time.perf_counter() * 1000.0


def normalize_whitespace(text: str) -> str:
    """Normalize runs of whitespace to single spaces."""
    return MULTISPACE_RE.sub(" ", text or "").strip()


def is_retryable_error(exc: Exception) -> bool:
    """Best-effort check for transient API/network/rate-limit failures."""
    msg = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "429",
        "too many requests",
        "rate limit",
        "timed out",
        "timeout",
        "temporarily",
        "connection",
        "connecterror",
        "disconnected",
        "remoteprotocolerror",
        "remote protocol",
        "server disconnected",
        "unexpected eof",
        "unavailable",
        "reset by peer",
        "service unavailable",
    )
    return any(m in msg for m in markers)


def retry_call(
    fn: Callable[[], _T],
    *,
    max_retries: int = 2,
    base_backoff_sec: float = 1.0,
    max_backoff_sec: float = 10.0,
    retry_if: Callable[[Exception], bool] | None = None,
) -> _T:
    """Retry callable with exponential backoff on transient failures."""
    checker = retry_if or is_retryable_error
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            if attempt >= max_retries or not checker(exc):
                raise
            sleep_sec = min(float(max_backoff_sec), base_backoff_sec * (2**attempt))
            time.sleep(max(0.05, sleep_sec))
            attempt += 1



def save_json(path: str | Path, payload: dict) -> None:
    """Write JSON file with UTF-8 and stable formatting."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)



def load_json(path: str | Path) -> dict:
    """Load JSON file into dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split long text into overlapping chunks by characters."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end].strip())
        if end == n:
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def _token_spans(text: str) -> list[tuple[int, int]]:
    """Return start/end spans for lightweight token approximation."""
    return [(m.start(), m.end()) for m in TOKEN_RE.finditer(text or "")]


def chunk_text_by_tokens(text: str, max_tokens: int = 360, overlap_tokens: int = 60) -> list[str]:
    """Split text by approximate token windows with overlap."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be < max_tokens")

    raw = (text or "").strip()
    if not raw:
        return []

    spans = _token_spans(raw)
    if not spans:
        # OCR/noise blocks can be tokenized poorly by regex and produce huge single chunks.
        # Fall back to conservative char windowing to keep payloads embedding-safe.
        approx_chunk_size = max(600, int(max_tokens) * 6)
        approx_overlap = max(0, min(approx_chunk_size - 1, int(overlap_tokens) * 6))
        fallback = chunk_text(raw, chunk_size=approx_chunk_size, overlap=approx_overlap)
        return [normalize_whitespace(piece) for piece in fallback if piece.strip()]
    if len(spans) <= max_tokens:
        return [normalize_whitespace(raw)]

    step = max_tokens - overlap_tokens
    out: list[str] = []
    i = 0
    n = len(spans)
    while i < n:
        end_idx = min(i + max_tokens, n)
        start_char = spans[i][0]
        end_char = spans[end_idx - 1][1]
        piece = raw[start_char:end_char].strip()
        if piece:
            out.append(normalize_whitespace(piece))
        if end_idx >= n:
            break
        i += step
    return out


def chunk_text_by_tokens_sections(
    text: str,
    max_tokens: int = 360,
    overlap_tokens: int = 60,
) -> list[str]:
    """Token-aware chunking with hard section boundaries on paragraph breaks."""
    raw = (text or "").strip()
    if not raw:
        return []

    sections = [s.strip() for s in PARAGRAPH_SPLIT_RE.split(raw) if s.strip()]
    if not sections:
        return chunk_text_by_tokens(raw, max_tokens=max_tokens, overlap_tokens=overlap_tokens)

    out: list[str] = []
    for section in sections:
        out.extend(chunk_text_by_tokens(section, max_tokens=max_tokens, overlap_tokens=overlap_tokens))
    return out


def chunk_table_by_tokens(
    text: str,
    max_tokens: int = 220,
    overlap_tokens: int = 30,
) -> list[str]:
    """Chunk table-like text by row lines to avoid splitting rows when possible."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be < max_tokens")

    lines = [ln.rstrip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return []

    line_token_counts = [max(1, len(_token_spans(ln))) for ln in lines]
    chunks: list[str] = []
    start_idx = 0

    while start_idx < len(lines):
        token_sum = 0
        end_idx = start_idx
        while end_idx < len(lines):
            next_tokens = line_token_counts[end_idx]
            if token_sum > 0 and token_sum + next_tokens > max_tokens:
                break
            token_sum += next_tokens
            end_idx += 1

        piece = "\n".join(lines[start_idx:end_idx]).strip()
        if piece:
            chunks.append(piece)
        if end_idx >= len(lines):
            break

        overlap_sum = 0
        overlap_start = end_idx
        while overlap_start > start_idx and overlap_sum < overlap_tokens:
            overlap_start -= 1
            overlap_sum += line_token_counts[overlap_start]
        if overlap_start == start_idx:
            start_idx = end_idx
        else:
            start_idx = overlap_start

    return chunks



def mean(values: Iterable[float]) -> float:
    """Compute mean for iterable values."""
    vals = list(values)
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))



def ndcg_at_k(relevances: list[float], k: int) -> float:
    """Compute NDCG@k from ordered relevances."""
    if k <= 0:
        return 0.0
    rel = relevances[:k]
    if not rel:
        return 0.0
    dcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rel))
    ideal = sorted(rel, reverse=True)
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)
