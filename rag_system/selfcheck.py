"""Runtime preflight checks for GigaChat/GigaEmbeddings access."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Settings
from .utils import now_ms, retry_call


@dataclass(slots=True)
class CheckResult:
    """Single preflight check result."""

    name: str
    ok: bool
    latency_ms: float
    detail: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-friendly dict."""
        return {
            "name": self.name,
            "ok": self.ok,
            "latency_ms": round(self.latency_ms, 2),
            "detail": self.detail,
        }


def _safe_exc(exc: Exception) -> str:
    """Return concise error text for diagnostics."""
    text = f"{type(exc).__name__}: {exc}"
    return text[:600]


def _model_id(item: Any) -> str | None:
    """Extract model id from SDK model object safely."""
    for key in ("id_", "id"):
        value = getattr(item, key, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def run_preflight(
    settings: Settings,
    *,
    check_chat: bool = True,
    check_embeddings: bool = True,
) -> dict[str, Any]:
    """Run preflight checks and return structured diagnostics."""
    from gigachat import GigaChat

    checks: list[CheckResult] = []
    available_models: list[str] = []

    client = GigaChat(
        credentials=settings.giga_api_key,
        scope=settings.giga_scope,
        verify_ssl_certs=False,
        timeout=float(settings.giga_http_timeout_sec),
    )

    # Models API.
    t0 = now_ms()
    try:
        models = retry_call(
            lambda: client.get_models(),
            max_retries=settings.api_max_retries,
            base_backoff_sec=settings.api_retry_backoff_sec,
        )
        available_models = sorted({m for m in (_model_id(x) for x in (models.data or [])) if m})
        missing: list[str] = []
        if check_chat and settings.giga_chat_model not in available_models:
            missing.append(settings.giga_chat_model)
        if check_embeddings and settings.giga_embedding_model not in available_models:
            missing.append(settings.giga_embedding_model)
        if missing:
            checks.append(
                CheckResult(
                    name="models",
                    ok=False,
                    latency_ms=now_ms() - t0,
                    detail=f"Required model(s) are unavailable: {', '.join(missing)}",
                )
            )
        else:
            checks.append(
                CheckResult(
                    name="models",
                    ok=True,
                    latency_ms=now_ms() - t0,
                    detail=f"ok ({len(available_models)} models)",
                )
            )
    except Exception as exc:
        checks.append(
            CheckResult(
                name="models",
                ok=False,
                latency_ms=now_ms() - t0,
                detail=_safe_exc(exc),
            )
        )

    # Chat completion API.
    if check_chat:
        t1 = now_ms()
        try:
            response = retry_call(
                lambda: client.chat(
                    {
                        "model": settings.giga_chat_model,
                        "messages": [{"role": "user", "content": "Ответь одним словом: ок"}],
                        "temperature": 0.0,
                        "max_tokens": 8,
                    }
                ),
                max_retries=settings.api_max_retries,
                base_backoff_sec=settings.api_retry_backoff_sec,
            )
            text = ""
            if getattr(response, "choices", None):
                text = str(response.choices[0].message.content).strip()
            checks.append(
                CheckResult(
                    name="chat",
                    ok=True,
                    latency_ms=now_ms() - t1,
                    detail=f"ok ({text[:80] if text else 'empty response'})",
                )
            )
        except Exception as exc:
            checks.append(
                CheckResult(
                    name="chat",
                    ok=False,
                    latency_ms=now_ms() - t1,
                    detail=_safe_exc(exc),
                )
            )

    # Embeddings API.
    if check_embeddings:
        t2 = now_ms()
        try:
            emb = retry_call(
                lambda: client.embeddings(
                    texts=["selfcheck embeddings probe"],
                    model=settings.giga_embedding_model,
                ),
                max_retries=settings.api_max_retries,
                base_backoff_sec=settings.api_retry_backoff_sec,
            )
            dim = len(emb.data[0].embedding) if getattr(emb, "data", None) else 0
            checks.append(
                CheckResult(
                    name="embeddings",
                    ok=dim > 0,
                    latency_ms=now_ms() - t2,
                    detail=f"ok (dim={dim})" if dim > 0 else "empty embeddings response",
                )
            )
        except Exception as exc:
            checks.append(
                CheckResult(
                    name="embeddings",
                    ok=False,
                    latency_ms=now_ms() - t2,
                    detail=_safe_exc(exc),
                )
            )

    overall_ok = all(c.ok for c in checks)
    failed = [c.to_dict() for c in checks if not c.ok]
    return {
        "ok": overall_ok,
        "scope": settings.giga_scope,
        "chat_model": settings.giga_chat_model,
        "embedding_model": settings.giga_embedding_model,
        "available_models": available_models,
        "checks": [c.to_dict() for c in checks],
        "failed_checks": failed,
    }
