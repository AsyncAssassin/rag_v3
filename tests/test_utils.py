"""Unit tests for utility helpers."""

from __future__ import annotations

import pytest

from rag_system.utils import (
    chunk_table_by_tokens,
    chunk_text_by_tokens,
    chunk_text,
    chunk_text_by_tokens_sections,
    is_retryable_error,
    ndcg_at_k,
    retry_call,
    rrf_fusion,
)


def test_chunk_text_produces_overlap() -> None:
    text = "a" * 2500
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    assert len(chunks) >= 3
    assert len(chunks[0]) == 1000


def test_rrf_fusion_prefers_consistent_top_docs() -> None:
    scores = rrf_fusion(
        [
            ["d1", "d2", "d3"],
            ["d1", "d4", "d2"],
        ]
    )
    assert scores["d1"] > scores["d2"]


def test_chunk_text_by_tokens_sections_respects_chunk_limit() -> None:
    text = ("Раздел 1\n\n" + "слово " * 600 + "\n\nРаздел 2\n\n" + "данные " * 600).strip()
    chunks = chunk_text_by_tokens_sections(text, max_tokens=200, overlap_tokens=20)
    assert len(chunks) >= 4
    assert all(len(ch.strip()) > 0 for ch in chunks)


def test_chunk_table_by_tokens_keeps_rows() -> None:
    rows = [f"row{i} | value{i} | metric{i}" for i in range(30)]
    text = "\n".join(rows)
    chunks = chunk_table_by_tokens(text, max_tokens=30, overlap_tokens=6)
    assert len(chunks) > 1
    assert all("|" in chunk for chunk in chunks)


def test_chunk_text_by_tokens_falls_back_to_char_windows_when_no_token_spans() -> None:
    text = "💥" * 12000
    chunks = chunk_text_by_tokens(text, max_tokens=200, overlap_tokens=20)
    assert len(chunks) > 1
    assert max(len(chunk) for chunk in chunks) <= 1200
    assert all(chunk.strip() for chunk in chunks)


def test_ndcg_at_k_basic_bounds() -> None:
    score = ndcg_at_k([1, 0, 1], k=3)
    assert 0.0 <= score <= 1.0


def test_is_retryable_error_handles_remote_disconnects() -> None:
    exc = RuntimeError("RemoteProtocolError: Server disconnected without sending a response.")
    assert is_retryable_error(exc) is True


def test_is_retryable_error_handles_connect_eof() -> None:
    exc = RuntimeError("ConnectError: [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol")
    assert is_retryable_error(exc) is True


def test_retry_call_caps_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr("rag_system.utils.time.sleep", lambda sec: slept.append(sec))

    attempts = {"n": 0}

    def _always_fail() -> None:
        attempts["n"] += 1
        raise RuntimeError("temporary timeout")

    with pytest.raises(RuntimeError):
        retry_call(
            _always_fail,
            max_retries=3,
            base_backoff_sec=10.0,
            max_backoff_sec=2.0,
            retry_if=lambda _exc: True,
        )

    assert attempts["n"] == 4
    assert slept == [2.0, 2.0, 2.0]
