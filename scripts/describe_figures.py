#!/usr/bin/env python
"""Describe pending figure chunks using the VLM and embed descriptions in Qdrant.

For each chunk with content='[figure — VLM description pending]', this script:
  1. Re-extracts the image from the source PDF using Docling
  2. Sends the image to the VLM for classification and description
  3. Updates the MongoDB chunk with the VLM output
  4. Embeds the description in Qdrant so it becomes searchable

Usage:
    python scripts/describe_figures.py                       # all pending
    python scripts/describe_figures.py --folder data/35564/
    python scripts/describe_figures.py --limit 10
    python scripts/describe_figures.py --dry-run             # show what would run
    python scripts/describe_figures.py --model qwen2.5vl:7b
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx
from bson import ObjectId
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from geo_pipeline.storage.mongo_client import get_database
from geo_pipeline.storage.qdrant_client import embed_and_upsert, ensure_collection

load_dotenv()
console = Console()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("FIGURE_MODEL", "qwen2.5vl:7b")
PIPELINE_VERSION = os.getenv("PIPELINE_VERSION", "0.1.0")
PENDING_MARKER = "[figure — VLM description pending]"

FIGURE_PROMPT = """You are analyzing an image extracted from an environmental restoration document.

First, classify this image as exactly one of:
- map — shows geographic area, watershed, project site, or spatial layout
- annotated_imagery — aerial or satellite photo with labels, arrows, or overlays
- aerial_imagery — aerial or satellite photo with no annotations
- diagram — technical drawing, cross-section, elevation view, or engineering schematic
- chart — bar chart, line graph, pie chart, or other data visualization
- photograph — ground-level photo of site conditions, habitat, or construction
- table_image — a table rendered as an image rather than text
- other — anything that does not fit the above

Then provide:
1. image_type: the classification label (one of the options above)
2. description: 2-3 sentences describing what the image shows
3. spatial_info: any coordinates, place names, scale bars, north arrows, watershed names, or spatial references visible (empty string if none)
4. contains_spatial_data: true if the image contains any spatial or geographic information, false otherwise

Respond as a JSON object with exactly these four keys."""


def _extract_images_from_pdf(pdf_path: str | Path) -> list[tuple[int, bytes]]:
    """Re-extract images from a PDF using Docling. Returns (page_number, png_bytes) list."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.generate_picture_images = True

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    result = converter.convert(str(pdf_path))

    images = []
    for picture in result.document.pictures:
        try:
            img = picture.get_image(result.document)
            if img:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                page_num = getattr(picture, "page_no", 0) or 0
                images.append((page_num, buf.getvalue()))
        except Exception:
            pass
    return images


def _call_vlm(model: str, image_bytes: bytes, prompt: str) -> str:
    """Send an image to the Ollama VLM and return the response text."""
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    with httpx.Client(timeout=300.0) as client:
        resp = client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt, "images": [b64]}],
                "stream": False,
            },
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text[:300]}")
        return resp.json()["message"]["content"]


def _parse_vlm_response(text: str) -> dict:
    """Extract JSON from VLM response, with fallback to raw description."""
    import json

    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Couldn't parse JSON — store raw text as description
    return {
        "image_type": "other",
        "description": stripped[:500],
        "spatial_info": "",
        "contains_spatial_data": False,
    }


def _build_chunk_content(parsed: dict) -> str:
    """Format the VLM output as the chunk's searchable text content."""
    parts = [f"[{parsed.get('image_type', 'figure')}]", parsed.get("description", "")]
    if parsed.get("spatial_info"):
        parts.append(f"Spatial info: {parsed['spatial_info']}")
    return " ".join(p for p in parts if p)


def _get_pending_chunks(db, folder: str | None, limit: int | None) -> list[dict]:
    """Fetch figure chunks that still have the pending marker as content."""
    query: dict = {"chunk_type": "figure", "content": PENDING_MARKER}

    if folder:
        resolved = str(Path(folder).resolve())
        docs = list(db.documents.find({"source_path": {"$regex": f"^{re.escape(resolved.rstrip('/') + '/')}"}}))
        if not docs:
            console.print(f"[yellow]No documents found in folder: {folder}[/yellow]")
            return []
        doc_ids = [str(d["_id"]) for d in docs]
        query["source_doc_id"] = {"$in": doc_ids}

    cursor = db.chunks.find(query)
    if limit:
        cursor = cursor.limit(limit)
    return list(cursor)


def main() -> None:
    parser = argparse.ArgumentParser(description="Describe pending figure chunks with VLM")
    parser.add_argument("--folder", type=str, default=None, help="Scope to documents from this folder")
    parser.add_argument("--limit", type=int, default=None, help="Max figures to process")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Ollama vision model (default: {DEFAULT_MODEL})")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed without calling the VLM")
    args = parser.parse_args()

    db = get_database()
    ensure_collection()

    pending = _get_pending_chunks(db, args.folder, args.limit)
    if not pending:
        console.print("[green]No pending figure chunks found.[/green]")
        return

    console.print(f"Found [cyan]{len(pending)}[/cyan] pending figure chunks")

    if args.dry_run:
        for chunk in pending:
            p = chunk.get("provenance", {})
            console.print(f"  • {p.get('filename', '?')} page {p.get('page_number', '?')} [{chunk['source_doc_id']}]")
        return

    # Group chunks by source document to avoid re-converting the same PDF multiple times
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for chunk in pending:
        by_doc[chunk["source_doc_id"]].append(chunk)

    ok = failed = 0

    for doc_id, chunks in by_doc.items():
        doc = db.documents.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            console.print(f"[red]Document not found:[/red] {doc_id}")
            failed += len(chunks)
            continue

        source_path = doc.get("source_path", "")
        if not Path(source_path).exists():
            console.print(f"[red]PDF not found:[/red] {source_path}")
            failed += len(chunks)
            continue

        console.print(f"\n[bold]{doc['filename']}[/bold] — {len(chunks)} figure(s)")
        console.print(f"  Re-extracting images from [dim]{source_path}[/dim]...")

        try:
            images = _extract_images_from_pdf(source_path)
        except Exception as exc:
            console.print(f"  [red]Docling failed:[/red] {exc}")
            failed += len(chunks)
            continue

        if not images:
            console.print("  [yellow]No images extracted from PDF[/yellow]")
            failed += len(chunks)
            continue

        console.print(f"  Extracted {len(images)} image(s) from PDF")

        # Build an ordered position map for ALL figure chunks in this document (by _id insertion order).
        # This lets us find the correct image index for each pending chunk even when some were
        # already described in a previous run and the pending subset doesn't start at index 0.
        all_figure_ids = [
            str(c["_id"])
            for c in db.chunks.find(
                {"source_doc_id": doc_id, "chunk_type": "figure"},
                sort=[("_id", 1)],
            )
        ]
        position_of = {chunk_id: i for i, chunk_id in enumerate(all_figure_ids)}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("  Describing figures...", total=len(chunks))

            for chunk in chunks:
                chunk_id = str(chunk["_id"])
                idx = position_of.get(chunk_id)

                if idx is None or idx >= len(images):
                    console.print(f"  [yellow]No image for chunk {chunk_id} (position {idx}, {len(images)} images extracted)[/yellow]")
                    failed += 1
                    progress.advance(task)
                    continue

                page_num = chunk.get("provenance", {}).get("page_number", 0)
                img_bytes = images[idx][1]
                chunk_id = str(chunk["_id"])

                try:
                    raw = _call_vlm(args.model, img_bytes, FIGURE_PROMPT)
                    parsed = _parse_vlm_response(raw)
                    content = _build_chunk_content(parsed)

                    # Embed in Qdrant
                    point_id = embed_and_upsert(
                        chunk_text=content,
                        chunk_id=chunk_id,
                        source_doc_id=doc_id,
                        page_number=page_num,
                        content_type="figure",
                    )

                    # Update MongoDB chunk
                    db.chunks.update_one(
                        {"_id": ObjectId(chunk_id)},
                        {
                            "$set": {
                                "content": content,
                                "figure_classification": parsed,
                                "embedding_id": point_id,
                                "described_at": datetime.now(timezone.utc),
                                "description_model": args.model,
                                "pipeline_version": PIPELINE_VERSION,
                            }
                        },
                    )
                    ok += 1
                except Exception as exc:
                    console.print(f"  [red]VLM error (page {page_num}):[/red] {exc}")
                    failed += 1

                progress.advance(task)

    console.print(f"\n[green]Done:[/green] {ok} described, [red]{failed} failed[/red]")


if __name__ == "__main__":
    main()
