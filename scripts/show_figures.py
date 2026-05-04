#!/usr/bin/env python
"""Browse described figure chunks in MongoDB.

Usage:
    python scripts/show_figures.py                         # all described
    python scripts/show_figures.py --type map
    python scripts/show_figures.py --type annotated_imagery
    python scripts/show_figures.py --spatial                # only spatial ones
    python scripts/show_figures.py --type map --spatial
"""
from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from geo_pipeline.storage.mongo_client import get_database

console = Console()

IMAGE_TYPES = ["map", "annotated_imagery", "aerial_imagery", "diagram", "chart", "photograph", "table_image", "other"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Browse described figure chunks")
    parser.add_argument("--type", choices=IMAGE_TYPES, default=None, help="Filter by image type")
    parser.add_argument("--spatial", action="store_true", help="Only show figures with spatial data")
    parser.add_argument("--summary", action="store_true", help="Show type counts only")
    args = parser.parse_args()

    db = get_database()

    query: dict = {"chunk_type": "figure", "figure_classification": {"$exists": True}}
    if args.type:
        query["figure_classification.image_type"] = args.type
    if args.spatial:
        query["figure_classification.contains_spatial_data"] = True

    chunks = list(db.chunks.find(query).sort("_id", 1))

    if args.summary or not chunks:
        from collections import Counter
        all_chunks = list(db.chunks.find({"chunk_type": "figure", "figure_classification": {"$exists": True}}))
        types = Counter(c["figure_classification"].get("image_type", "unknown") for c in all_chunks)
        spatial = sum(1 for c in all_chunks if c["figure_classification"].get("contains_spatial_data"))
        pending = db.chunks.count_documents({"chunk_type": "figure", "content": "[figure — VLM description pending]"})

        t = Table(title="Figure Classification Summary")
        t.add_column("Type", style="cyan")
        t.add_column("Count", justify="right")
        for img_type, count in types.most_common():
            t.add_row(img_type, str(count))
        t.add_row("─" * 20, "─" * 5)
        t.add_row("contains spatial data", str(spatial))
        t.add_row("still pending", str(pending))
        console.print(t)
        return

    console.print(f"Showing [cyan]{len(chunks)}[/cyan] figure(s)\n")
    for c in chunks:
        cls = c["figure_classification"]
        p = c["provenance"]
        spatial_flag = "[green]spatial[/green]" if cls.get("contains_spatial_data") else ""
        console.print(f"[bold]{p['filename']}[/bold]  page {p['page_number']}  [{cls.get('image_type','?')}] {spatial_flag}")
        console.print(f"  {cls.get('description', '')}")
        if cls.get("spatial_info"):
            console.print(f"  [dim]Spatial: {cls['spatial_info']}[/dim]")
        console.print()


if __name__ == "__main__":
    main()
