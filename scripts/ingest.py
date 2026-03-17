#!/usr/bin/env python
"""CLI entry point for document ingestion.

Usage:
    python scripts/ingest.py data/sample.pdf          # single file
    python scripts/ingest.py data/                     # all PDFs in directory
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from geo_pipeline.ingestion.document_ingester import ingest_pdf

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into geo-pipeline")
    parser.add_argument(
        "path",
        type=str,
        help="Path to a PDF file or directory of PDFs",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Skip Qdrant embedding (store in MongoDB only)",
    )
    args = parser.parse_args()

    target = Path(args.path)
    embed = not args.no_embed

    if target.is_file():
        pdf_files = [target]
    elif target.is_dir():
        pdf_files = sorted(target.glob("*.pdf"))
        if not pdf_files:
            console.print(f"[red]No PDFs found in {target}[/red]")
            sys.exit(1)
    else:
        console.print(f"[red]Path not found: {target}[/red]")
        sys.exit(1)

    console.print(f"[bold]Ingesting {len(pdf_files)} PDF(s)...[/bold]\n")

    results = []
    for pdf in pdf_files:
        try:
            result = ingest_pdf(pdf, embed=embed)
            results.append(result)
        except Exception as exc:
            console.print(f"[red]ERROR:[/red] {pdf.name} — {exc}")
            results.append({"status": "error", "filename": pdf.name, "error": str(exc)})

    # Summary
    console.print("\n[bold]Summary:[/bold]")
    ingested = [r for r in results if r["status"] == "ingested"]
    skipped = [r for r in results if r["status"] == "skipped"]
    errors = [r for r in results if r["status"] == "error"]
    console.print(f"  Ingested: {len(ingested)}")
    console.print(f"  Skipped:  {len(skipped)}")
    if errors:
        console.print(f"  [red]Errors:   {len(errors)}[/red]")


if __name__ == "__main__":
    main()
