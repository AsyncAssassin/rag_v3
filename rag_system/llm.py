"""LLM helpers for query rewriting and grounded answer generation."""

from __future__ import annotations

import json
import re
from typing import Any

from .config import Settings
from .types import RetrievedChunk
from .utils import normalize_whitespace, retry_call


JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


class GigaLLMClient:
    """Wrapper around GigaChat for rewrite + answer generation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._llm = None

    def _load_llm(self):
        """Lazy-load GigaChat model client."""
        if self._llm is not None:
            return self._llm

        if not self.settings.giga_api_key:
            raise RuntimeError("GIGA_API_KEY is not configured")

        try:
            from langchain_gigachat.chat_models import GigaChat
        except Exception as exc:
            raise RuntimeError("langchain_gigachat chat model import failed") from exc

        self._llm = GigaChat(
            credentials=self.settings.giga_api_key,
            verify_ssl_certs=False,
            scope=self.settings.giga_scope,
            model=self.settings.giga_chat_model,
            temperature=0.0,
            top_p=0.6,
            timeout=float(self.settings.giga_http_timeout_sec),
        )
        return self._llm

    def rewrite_query(self, query: str, n: int = 3) -> list[str]:
        """Generate query rewrites for better recall."""
        prompt = (
            "Сгенерируй альтернативные формулировки для поиска по документам.\n"
            f"Нужно {n} переформулировок на том же языке, включая исходный смысл.\n"
            "Верни ТОЛЬКО JSON-массив строк без пояснений.\n"
            f"Запрос: {query}"
        )

        try:
            llm = self._load_llm()
            response = retry_call(
                lambda: llm.invoke(prompt),
                max_retries=self.settings.api_max_retries,
                base_backoff_sec=self.settings.api_retry_backoff_sec,
            )
            text = getattr(response, "content", str(response))
            parsed = self._parse_json_list(text)
            if parsed:
                uniq = []
                seen = set()
                for item in [query, *parsed]:
                    norm = normalize_whitespace(item)
                    if norm and norm not in seen:
                        seen.add(norm)
                        uniq.append(norm)
                return uniq[: max(1, n)]
        except Exception:
            pass

        return [query]

    def generate_answer(self, query: str, context_chunks: list[RetrievedChunk]) -> str:
        """Generate grounded answer with explicit citation tags."""
        if not context_chunks:
            return "Не удалось найти подтверждающую информацию в документах."

        context_parts: list[str] = []
        for i, ch in enumerate(context_chunks, start=1):
            page_str = f", page={ch.page}" if ch.page is not None else ""
            context_parts.append(
                f"[S{i}] source={ch.source_path}{page_str}\n{ch.text}"
            )
        context = "\n\n".join(context_parts)

        prompt = (
            "Ты помощник по вопросам на основе документов.\n"
            "Правила:\n"
            "1) Отвечай только на основе контекста ниже.\n"
            "2) Если информации недостаточно, ответь строго: "
            "\"Не нашел подтверждения в документах корпуса.\".\n"
            "3) Добавляй ссылки на источники в формате [S1], [S2].\n"
            "4) Не выдумывай факты.\n\n"
            f"Вопрос: {query}\n\n"
            f"Контекст:\n{context}\n\n"
            "Ответ:"
        )

        try:
            llm = self._load_llm()
            response = retry_call(
                lambda: llm.invoke(prompt),
                max_retries=self.settings.api_max_retries,
                base_backoff_sec=self.settings.api_retry_backoff_sec,
            )
            text = getattr(response, "content", str(response))
            return str(text).strip()
        except Exception as exc:
            return (
                "Не удалось получить ответ от LLM. "
                f"Ошибка: {exc}.\n\n"
                "Черновой контекст найден, но генерация отключена."
            )

    @staticmethod
    def _parse_json_list(text: str) -> list[str]:
        """Extract JSON list from model output."""
        text = text.strip()
        candidates: list[str] = []

        # Direct parse.
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [str(x) for x in data if str(x).strip()]
        except Exception:
            pass

        # Parse first JSON-array-like block.
        m = JSON_ARRAY_RE.search(text)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                candidates = [str(x) for x in data if str(x).strip()]
        except Exception:
            return []
        return candidates
