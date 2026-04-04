"""Fast markdown-first extractor using PyMuPDF4LLM."""

from __future__ import annotations

import re

from .base import ChunkingOptions, DocumentExtractor, chunk_long_text, ensure_pdf
from ..types import DocumentChunk


class PyMuPDF4LLMExtractor(DocumentExtractor):
    """Fast extractor for text-heavy PDFs, with page-level metadata."""

    name = "pymupdf4llm"

    def __init__(
        self,
        table_strategy: str = "lines",
        extract_images: bool = False,
        chunking: ChunkingOptions | None = None,
    ) -> None:
        self.table_strategy = table_strategy
        self.extract_images = extract_images
        self.chunking = chunking or ChunkingOptions()

    @staticmethod
    def _looks_like_table(text: str) -> bool:
        """Heuristic table detector for page markdown."""
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        if len(lines) < 5:
            return False
        pipe_lines = sum(1 for ln in lines[:80] if "|" in ln)
        multi_space_lines = sum(1 for ln in lines[:80] if len([p for p in re.split(r"\s{2,}", ln) if p]) >= 3)
        return pipe_lines >= 3 or multi_space_lines >= 6

    def extract(self, file_path: str) -> list[DocumentChunk]:
        """Extract PDF as page-level markdown, then chunk."""
        ensure_pdf(file_path)

        try:
            from langchain_pymupdf4llm import PyMuPDF4LLMLoader
        except Exception as exc:
            raise RuntimeError("PyMuPDF4LLM loader import failed") from exc

        loader = PyMuPDF4LLMLoader(
            file_path,
            mode="page",
            table_strategy=self.table_strategy,
            extract_images=self.extract_images,
        )
        docs = loader.load()

        output: list[DocumentChunk] = []
        for doc in docs:
            text = (doc.page_content or "").strip()
            if not text:
                continue
            meta = dict(doc.metadata or {})
            page = meta.get("page")
            element_type = "table" if self._looks_like_table(text) else "text"
            output.extend(
                chunk_long_text(
                    text=text,
                    source_path=file_path,
                    page=int(page) + 1 if page is not None else None,
                    element_type=element_type,
                    metadata={"extractor": self.name, **meta},
                    chunking=self.chunking,
                )
            )

        return output
