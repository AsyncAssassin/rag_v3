"""Docling-based extraction implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..logging_utils import get_logger
from .base import ChunkingOptions, DocumentExtractor, UnsupportedFileTypeError, chunk_long_text, ensure_pdf
from ..types import DocumentChunk, normalize_source_path


LOGGER = get_logger()


class DoclingExtractor(DocumentExtractor):
    """Primary PDF extractor using Docling with optional full-page OCR."""

    name = "docling"

    def __init__(
        self,
        languages: tuple[str, ...] = ("rus", "eng"),
        full_page_ocr: bool = False,
        ocr_backend: str = "easyocr",
        ocr_fallbacks: tuple[str, ...] = ("easyocr", "tesseract", "rapidocr", "none"),
        easyocr_langs: tuple[str, ...] = ("ru", "en"),
        tesseract_langs: tuple[str, ...] = ("rus", "eng"),
        chunking: ChunkingOptions | None = None,
    ) -> None:
        self.languages = languages
        self.full_page_ocr = full_page_ocr
        self.ocr_backend = (ocr_backend or "easyocr").strip().lower()
        self.ocr_fallbacks = tuple((x or "").strip().lower() for x in ocr_fallbacks if (x or "").strip())
        self.easyocr_langs = tuple(x.strip().lower() for x in easyocr_langs if x.strip())
        self.tesseract_langs = tuple(x.strip().lower() for x in tesseract_langs if x.strip())
        self.chunking = chunking or ChunkingOptions()

    def extract(self, file_path: str) -> list[DocumentChunk]:
        """Extract chunks from PDF using Docling-first strategy."""
        ensure_pdf(file_path)

        chunks = self._extract_with_docling_core(file_path)
        if chunks:
            return chunks

        chunks = self._extract_with_langchain_docling(file_path)
        if chunks:
            return chunks

        raise RuntimeError("Docling returned no chunks")

    def _ocr_backend_sequence(self) -> list[str]:
        """Build deterministic OCR backend sequence with requested backend first."""
        supported = {"easyocr", "rapidocr", "tesseract", "none"}
        ordered: list[str] = []
        if self.ocr_backend in supported:
            ordered.append(self.ocr_backend)
        for item in self.ocr_fallbacks:
            if item in supported and item not in ordered:
                ordered.append(item)
        if not ordered:
            ordered = ["easyocr", "tesseract", "rapidocr", "none"]
        return ordered

    def _build_ocr_options(self, backend: str):
        """Create docling OCR options object for selected backend."""
        from docling.datamodel.pipeline_options import (
            EasyOcrOptions,
            RapidOcrOptions,
            TesseractCliOcrOptions,
        )

        if backend == "easyocr":
            return EasyOcrOptions(
                lang=list(self.easyocr_langs or ("ru", "en")),
                force_full_page_ocr=bool(self.full_page_ocr),
            )
        if backend == "rapidocr":
            return RapidOcrOptions(
                backend="onnxruntime",
                force_full_page_ocr=bool(self.full_page_ocr),
            )
        if backend == "tesseract":
            return TesseractCliOcrOptions(
                lang=list(self.tesseract_langs or ("rus", "eng")),
                force_full_page_ocr=bool(self.full_page_ocr),
            )
        return None

    def _extract_with_docling_core(self, file_path: str) -> list[DocumentChunk]:
        """Try extraction via native Docling API."""
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except Exception as exc:
            raise RuntimeError("docling core import failed") from exc

        errors: dict[str, str] = {}
        attempts: list[str] = []

        for backend in self._ocr_backend_sequence():
            attempts.append(backend)
            try:
                pdf_options = PdfPipelineOptions()
                pdf_options.do_table_structure = True
                if backend == "none":
                    pdf_options.do_ocr = False
                else:
                    pdf_options.do_ocr = True
                    if hasattr(pdf_options, "ocr_options"):
                        pdf_options.ocr_options = self._build_ocr_options(backend)
                if hasattr(pdf_options, "force_full_page_ocr"):
                    setattr(pdf_options, "force_full_page_ocr", bool(self.full_page_ocr))

                converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
                    }
                )
                result = converter.convert(file_path)
                output = self._convert_docling_result_to_chunks(
                    result=result,
                    file_path=file_path,
                    ocr_backend_effective=backend,
                    ocr_fallback_path=list(attempts),
                )
                if output:
                    return output
                errors[backend] = "no_chunks"
            except Exception as exc:
                errors[backend] = str(exc)
                LOGGER.warning("Docling backend '%s' failed for %s: %s", backend, file_path, exc)

        err_text = "; ".join(f"{k}: {v}" for k, v in errors.items()) or "unknown"
        raise RuntimeError(f"docling backends exhausted: {err_text}")

    def _convert_docling_result_to_chunks(
        self,
        *,
        result,
        file_path: str,
        ocr_backend_effective: str,
        ocr_fallback_path: list[str],
    ) -> list[DocumentChunk]:
        """Convert docling conversion result into standardized chunks."""
        doc = result.document
        output: list[DocumentChunk] = []
        meta_base = {
            "extractor": self.name,
            "full_page_ocr": bool(self.full_page_ocr),
            "ocr_backend_effective": ocr_backend_effective,
            "ocr_fallback_path": list(ocr_fallback_path),
        }

        pages = getattr(doc, "pages", None)
        if pages:
            for page in pages:
                page_no = getattr(page, "page_no", None) or getattr(page, "page_number", None)
                page_md = ""
                for attr in ("export_to_markdown", "to_markdown"):
                    if hasattr(page, attr):
                        try:
                            page_md = getattr(page, attr)()
                        except Exception:
                            page_md = ""
                        break
                if not page_md:
                    page_md = str(page)
                output.extend(
                    chunk_long_text(
                        text=page_md,
                        source_path=file_path,
                        page=int(page_no) if page_no is not None else None,
                        element_type="text",
                        metadata=meta_base,
                        chunking=self.chunking,
                    )
                )

        if output:
            return output

        markdown = ""
        for attr in ("export_to_markdown", "to_markdown"):
            if hasattr(doc, attr):
                try:
                    markdown = getattr(doc, attr)()
                except Exception:
                    markdown = ""
                break
        if not markdown:
            markdown = str(doc)
        if not markdown.strip():
            return []

        return chunk_long_text(
            text=markdown,
            source_path=file_path,
            page=None,
            element_type="text",
            metadata=meta_base,
            chunking=self.chunking,
        )

    def _extract_with_langchain_docling(self, file_path: str) -> list[DocumentChunk]:
        """Try extraction through LangChain DoclingLoader integration."""
        try:
            from langchain_community.document_loaders import DoclingLoader
        except Exception as exc:
            raise RuntimeError("langchain DoclingLoader import failed") from exc

        kwargs: dict[str, Any] = {
            "file_path": file_path,
        }

        # Different versions expose different init signatures.
        # We pass conservative keys and let Loader ignore unsupported ones when possible.
        for candidate in (
            {"pipeline": "standard"},
            {"pipeline": "vlm", "vlm_model": "granite_docling"},
            {},
        ):
            try:
                loader = DoclingLoader(**kwargs, **candidate)
                docs = loader.load()
                break
            except TypeError:
                continue
            except Exception:
                docs = []
                break
        else:
            docs = []

        output: list[DocumentChunk] = []
        for doc in docs:
            text = (doc.page_content or "").strip()
            if not text:
                continue

            metadata = dict(doc.metadata or {})
            page = metadata.get("page") or metadata.get("page_number")
            element_type = metadata.get("element_type") or metadata.get("category") or "text"
            table_html = metadata.get("text_as_html") if str(element_type).lower() == "table" else None

            output.extend(
                chunk_long_text(
                    text=text,
                    source_path=file_path,
                    page=int(page) if page is not None else None,
                    element_type=str(element_type),
                    metadata={"extractor": self.name, "full_page_ocr": self.full_page_ocr, **metadata},
                    chunking=self.chunking,
                )
            )

            if table_html and output:
                output[-1].table_html = str(table_html)

        return output


class PlainTextExtractor(DocumentExtractor):
    """Simple extractor for TXT and CSV files."""

    name = "plain_text"

    def __init__(self, chunking: ChunkingOptions | None = None) -> None:
        self.chunking = chunking or ChunkingOptions()

    def extract(self, file_path: str) -> list[DocumentChunk]:
        """Extract chunks from plain text files."""
        suffix = Path(file_path).suffix.lower()
        if suffix not in {".txt", ".csv", ".md"}:
            raise UnsupportedFileTypeError(f"Unsupported file type: {suffix}")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        return chunk_long_text(
            text=text,
            source_path=normalize_source_path(file_path),
            page=None,
            element_type="text",
            metadata={"extractor": self.name},
            chunking=self.chunking,
        )
