"""PDF ingestion regression script for project corpus."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rag_system.extractors.factory import ExtractorOrchestrator



def discover_pdfs(data_dir: str) -> list[str]:
    """Discover PDF files recursively."""
    root = Path(data_dir).expanduser().resolve()
    return [str(p) for p in sorted(root.rglob("*.pdf")) if p.is_file()]



def main() -> None:
    """Entrypoint for PDF regression checks."""
    parser = argparse.ArgumentParser(description="Run PDF extraction regression")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--extractor", type=str, default="docling")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--out-csv", type=str, default="pdf_regression_report.csv")
    args = parser.parse_args()

    orchestrator = ExtractorOrchestrator(languages=("rus", "eng"))
    files = discover_pdfs(args.data_dir)

    rows: list[dict] = []
    for fp in files:
        try:
            out = orchestrator.extract_with_policy(fp, preferred=args.extractor, fast_mode=args.fast)
            rows.append(
                {
                    "file_path": fp,
                    "status": "ok",
                    "extractor_used": out.extractor_used,
                    "chunks": out.stats.total_chunks,
                    "chars_per_page": round(out.stats.chars_per_page, 2),
                    "empty_page_ratio": round(out.stats.empty_page_ratio, 4),
                    "short_chunk_ratio": round(out.stats.short_chunk_ratio, 4),
                    "has_table_elements": out.stats.has_table_elements,
                    "notes": " | ".join(out.notes),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "file_path": fp,
                    "status": "failed",
                    "extractor_used": "n/a",
                    "chunks": 0,
                    "chars_per_page": 0.0,
                    "empty_page_ratio": 1.0,
                    "short_chunk_ratio": 1.0,
                    "has_table_elements": False,
                    "notes": str(exc),
                }
            )

    print(f"Checked {len(rows)} files")
    print(f"Failed: {sum(1 for r in rows if r['status'] == 'failed')}")

    out_path = Path(args.out_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_path",
                "status",
                "extractor_used",
                "chunks",
                "chars_per_page",
                "empty_page_ratio",
                "short_chunk_ratio",
                "has_table_elements",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved report: {out_path}")


if __name__ == "__main__":
    main()
