#!/usr/bin/env python
"""CLI for querying ingested documents via Qdrant similarity search.

Usage:
    python scripts/query.py "restoration project near Cascade Creek"
    python scripts/query.py "PLSS township range" --top-k 10
"""
from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from geo_pipeline.storage.mongo_client import get_database
from geo_pipeline.storage.qdrant_client import ensure_collection, search

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Query geo-pipeline documents")
    parser.add_argument("query", type=str, help="Search query text")
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of results (default: 5)"
    )
    args = parser.parse_args()

    ensure_collection()
    results = search(args.query, top_k=args.top_k)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    db = get_database()

    table = Table(title=f"Results for: '{args.query}'")
    table.add_column("Score", style="cyan", width=8)
    table.add_column("File", style="green")
    table.add_column("Page", width=6)
    table.add_column("Content", max_width=80)

    for hit in results:
        chunk_id = hit.get("chunk_id", "")
        score = f"{hit['score']:.3f}"
        page = str(hit.get("page_number", "?"))

        # Fetch chunk content from MongoDB
        from bson import ObjectId

        chunk = db.chunks.find_one({"_id": ObjectId(chunk_id)}) if chunk_id else None
        if chunk:
            content = chunk.get("content", "")[:200]
            filename = chunk.get("provenance", {}).get("filename", "?")
        else:
            content = "(chunk not found in MongoDB)"
            filename = "?"

        table.add_row(score, filename, page, content)

    console.print(table)


if __name__ == "__main__":
    main()
