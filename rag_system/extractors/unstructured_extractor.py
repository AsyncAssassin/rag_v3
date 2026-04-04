"""Unstructured PDF extractor fallback."""

from __future__ import annotations

from pathlib import Path
import tempfile

from .base import ChunkingOptions, DocumentExtractor, chunk_long_text, ensure_pdf
from ..types import DocumentChunk


class UnstructuredExtractor(DocumentExtractor):
    """Fallback extractor using UnstructuredPDFLoader in hi_res mode."""

    name = "unstructured_hi_res"

    def __init__(
        self,
        languages: tuple[str, ...] = ("rus", "eng"),
        strategy: str = "hi_res",
        chunking: ChunkingOptions | None = None,
        target_pages: list[int] | None = None,
    ) -> None:
        self.languages = languages
        self.strategy = strategy
        self.chunking = chunking or ChunkingOptions()
        cleaned: set[int] = set()
        for page in target_pages or []:
            try:
                page_no = int(page)
            except Exception:
                continue
            if page_no > 0:
                cleaned.add(page_no)
        self.target_pages = sorted(cleaned)

    def extract(self, file_path: str) -> list[DocumentChunk]:
        """Extract PDF elements and convert to chunks."""
        ensure_pdf(file_path)

        if self.target_pages:
            return self._extract_targeted_pages(file_path)
        return self._extract_full_file(file_path)

    def _extract_full_file(self, file_path: str) -> list[DocumentChunk]:
        """Extract all pages through Unstructured loader."""
        docs = self._load_docs(file_path)
        return self._docs_to_chunks(docs, source_path=file_path, page_map=None)

    def _extract_targeted_pages(self, file_path: str) -> list[DocumentChunk]:
        """Extract only selected pages by creating temporary subset PDF."""
        try:
            import fitz  # type: ignore
        except Exception as exc:
            raise RuntimeError("PyMuPDF is required for targeted Unstructured fallback") from exc

        src_doc = fitz.open(file_path)
        try:
            total_pages = int(src_doc.page_count)
            selected = [p for p in self.target_pages if 1 <= p <= total_pages]
            if not selected:
                return []

            page_map: dict[int, int] = {}
            with tempfile.TemporaryDirectory(prefix="rag_unstructured_pages_") as tmp_dir:
                subset_path = Path(tmp_dir) / "subset.pdf"
                dst_doc = fitz.open()
                try:
                    for idx, page_no in enumerate(selected, start=1):
                        dst_doc.insert_pdf(src_doc, from_page=page_no - 1, to_page=page_no - 1)
                        page_map[idx] = page_no
                    dst_doc.save(str(subset_path))
                finally:
                    dst_doc.close()

                docs = self._load_docs(str(subset_path))
                return self._docs_to_chunks(docs, source_path=file_path, page_map=page_map)
        finally:
            src_doc.close()

    def _load_docs(self, file_path: str):
        """Load documents from unstructured loader."""
        try:
            from langchain_community.document_loaders import UnstructuredPDFLoader
        except Exception as exc:
            raise RuntimeError("Unstructured loader import failed") from exc

        loader = UnstructuredPDFLoader(
            file_path,
            mode="elements",
            strategy=self.strategy,
            languages=list(self.languages),
        )
        return loader.load()

    def _docs_to_chunks(self, docs, source_path: str, page_map: dict[int, int] | None) -> list[DocumentChunk]:
        """Convert loaded elements to normalized chunks."""
        output: list[DocumentChunk] = []
        for doc in docs:
            text = (doc.page_content or "").strip()
            if not text:
                continue

            meta = dict(doc.metadata or {})
            page = meta.get("page_number") or meta.get("page")
            if page is not None and page_map:
                try:
                    page = page_map.get(int(page), int(page))
                except Exception:
                    page = page
            element_type = str(meta.get("category") or meta.get("element_type") or "text")
            table_html = meta.get("text_as_html") if element_type.lower() == "table" else None
            bbox = meta.get("coordinates")

            new_chunks = chunk_long_text(
                text=text,
                source_path=source_path,
                page=int(page) if page is not None else None,
                element_type=element_type,
                metadata={
                    "extractor": self.name,
                    "targeted_pages": list(self.target_pages) if self.target_pages else None,
                    **meta,
                },
                chunking=self.chunking,
            )
            if table_html and new_chunks:
                new_chunks[0].table_html = str(table_html)
            if bbox and new_chunks:
                try:
                    new_chunks[0].bbox = dict(bbox)
                except Exception:
                    pass
            output.extend(new_chunks)

        return output
