"""HuggingFace-based rerankers for multiple model backends."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseReranker


@dataclass(slots=True)
class _ModelRuntime:
    """Holds loaded model runtime objects."""

    mode: str
    model: Any
    tokenizer: Any | None = None
    device: str | None = None


class HuggingFaceCrossEncoderReranker(BaseReranker):
    """Cross-encoder reranker with robust fallback between libraries."""

    backend = "huggingface"

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        batch_size: int = 16,
        trust_remote_code: bool = False,
        max_length: int = 512,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__(model_name=model_name)
        self.device = device
        self.batch_size = batch_size
        self.trust_remote_code = trust_remote_code
        self.max_length = max_length
        self.cache_dir = cache_dir
        self._runtime: _ModelRuntime | None = None
        self._configure_cache_env()

    def _configure_cache_env(self) -> None:
        """Apply optional cache directory for HF runtime artifacts."""
        if not self.cache_dir:
            return
        root = Path(self.cache_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(root))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(root / "transformers"))
        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(root / "sentence_transformers"))

    def _load_runtime(self) -> _ModelRuntime:
        """Lazy-load scoring runtime."""
        if self._runtime is not None:
            return self._runtime

        # First try sentence-transformers CrossEncoder.
        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(
                self.model_name,
                device=self.device,
                trust_remote_code=self.trust_remote_code,
                max_length=self.max_length,
            )
            self._runtime = _ModelRuntime(mode="st_cross_encoder", model=model, device=self.device)
            return self._runtime
        except Exception:
            pass

        # Fallback to raw transformers sequence classification.
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )
            if self.device:
                model = model.to(self.device)
            elif torch.cuda.is_available():
                model = model.to("cuda")
                self.device = "cuda"
            else:
                self.device = "cpu"
            model.eval()

            self._runtime = _ModelRuntime(
                mode="transformers_seq_cls",
                model=model,
                tokenizer=tokenizer,
                device=self.device,
            )
            return self._runtime
        except Exception as exc:
            raise RuntimeError(f"Failed to load reranker model '{self.model_name}': {exc}") from exc

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Compute scores for query-passages."""
        runtime = self._load_runtime()
        self._warmed = True
        if not passages:
            return []

        if runtime.mode == "st_cross_encoder":
            pairs = [(query, p) for p in passages]
            raw = runtime.model.predict(pairs, batch_size=self.batch_size)
            arr = np.asarray(raw, dtype=np.float32)
            if arr.ndim == 1:
                scores = arr
            elif arr.ndim == 2 and arr.shape[1] == 1:
                scores = arr[:, 0]
            elif arr.ndim == 2 and arr.shape[1] >= 2:
                # Classification-like output: use positive-class column.
                scores = arr[:, -1]
            else:
                raise RuntimeError(f"Unsupported CrossEncoder output shape: {arr.shape}")
            return scores.reshape(-1).tolist()

        return self._score_with_transformers(runtime, query, passages)

    def warmup(self) -> None:
        """Force model runtime loading ahead of first request."""
        self._load_runtime()
        self._warmed = True

    def _score_with_transformers(self, runtime: _ModelRuntime, query: str, passages: list[str]) -> list[float]:
        """Compute rerank scores with transformers model."""
        import torch

        model = runtime.model
        tokenizer = runtime.tokenizer
        assert tokenizer is not None

        scores: list[float] = []
        for i in range(0, len(passages), self.batch_size):
            batch = passages[i : i + self.batch_size]
            encoded = tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(runtime.device or "cpu") for k, v in encoded.items()}
            with torch.no_grad():
                logits = model(**encoded).logits

            # Handle binary/1-logit/multi-logit heads robustly.
            if logits.ndim == 1:
                batch_scores = logits.detach().cpu().numpy()
            elif logits.shape[-1] == 1:
                batch_scores = logits[:, 0].detach().cpu().numpy()
            else:
                # Use positive class probability-like score.
                probs = torch.softmax(logits, dim=-1)
                batch_scores = probs[:, -1].detach().cpu().numpy()

            scores.extend(float(x) for x in batch_scores)

        return scores


class AmberoadReranker(HuggingFaceCrossEncoderReranker):
    """Amberoad multilingual reranker backend."""

    backend = "amberoad"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(model_name="amberoad/bert-multilingual-passage-reranking-msmarco", **kwargs)


class BgeReranker(HuggingFaceCrossEncoderReranker):
    """BAAI bge-reranker-v2-m3 backend."""

    backend = "bge_m3"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(model_name="BAAI/bge-reranker-v2-m3", **kwargs)


class JinaReranker(HuggingFaceCrossEncoderReranker):
    """Jina multilingual reranker backend."""

    backend = "jina_multilingual"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            model_name="jinaai/jina-reranker-v2-base-multilingual",
            trust_remote_code=True,
            **kwargs,
        )
