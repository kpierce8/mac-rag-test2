from __future__ import annotations

import hashlib
import io
import logging
import os
from pathlib import Path

from rich.console import Console

from geo_pipeline.schema.documents import DocumentRecord
from geo_pipeline.schema.spatial import DocumentChunk, Provenance
from geo_pipeline.storage.mongo_client import (
    ensure_indexes,
    get_database,
    get_document_by_hash,
    insert_chunks,
    upsert_document,
)
from geo_pipeline.storage.qdrant_client import (
    embed_and_upsert,
    ensure_collection,
)

logger = logging.getLogger(__name__)
console = Console()


def get_file_hash(path: str | Path) -> str:
    """SHA-256 of file contents — stable across renames."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _convert_pdf(pdf_path: str | Path):
    """Use Docling to decompose a PDF. Returns the conversion result."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.generate_picture_images = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    return converter.convert(str(pdf_path))


def _extract_images(result) -> list[tuple[int, bytes]]:
    """Extract images from Docling result as (page_number, png_bytes) pairs."""
    images = []
    try:
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
    except Exception:
        pass
    return images


def _chunk_markdown(markdown: str, max_chars: int = 4000) -> list[str]:
    """Split markdown into chunks, preferring paragraph boundaries."""
    if len(markdown) <= max_chars:
        return [markdown]
    chunks = []
    while markdown:
        if len(markdown) <= max_chars:
            chunks.append(markdown)
            break
        split_at = markdown.rfind("\n\n", 0, max_chars)
        if split_at == -1:
            split_at = max_chars
        chunks.append(markdown[:split_at])
        markdown = markdown[split_at:].lstrip()
    return chunks


def ingest_pdf(
    pdf_path: str | Path,
    embed: bool = True,
) -> dict:
    """Ingest a single PDF: decompose, store chunks in MongoDB, embed in Qdrant.

    Returns a summary dict with counts.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    db = get_database()
    ensure_indexes(db)

    file_hash = get_file_hash(pdf_path)

    # Skip if already ingested
    existing = get_document_by_hash(db, file_hash)
    if existing:
        console.print(f"[yellow]SKIP:[/yellow] {pdf_path.name} (already ingested)")
        return {"status": "skipped", "filename": pdf_path.name}

    console.print(f"[bold]Processing:[/bold] {pdf_path.name}")

    # Convert with Docling
    result = _convert_pdf(pdf_path)
    markdown = result.document.export_to_markdown()
    try:
        total_pages = result.document.num_pages()
    except Exception:
        total_pages = max(1, markdown.count("\n---\n") + 1)

    # Store document record
    doc_record = DocumentRecord(
        filename=pdf_path.name,
        file_hash=file_hash,
        total_pages=total_pages,
        file_size_bytes=pdf_path.stat().st_size,
        source_path=str(pdf_path.resolve()),
    )
    doc_id = upsert_document(db, doc_record)
    console.print(f"  Document stored: [cyan]{doc_id}[/cyan]")

    # Split into text chunks
    text_chunks = _chunk_markdown(markdown)
    console.print(f"  Text chunks: {len(text_chunks)}")

    # Create DocumentChunk records
    chunk_models = []
    for i, text in enumerate(text_chunks):
        chunk = DocumentChunk(
            source_doc_id=doc_id,
            chunk_type="text",
            content=text,
            provenance=Provenance(
                source_doc_id=doc_id,
                filename=pdf_path.name,
                page_number=i + 1,  # approximate — Docling doesn't give per-chunk pages
                bbox=None,
                content_type="text",
            ),
        )
        chunk_models.append(chunk)

    # Extract and store figure descriptions
    images = _extract_images(result)
    if images:
        console.print(f"  Figures extracted: {len(images)}")
        for page_num, _img_bytes in images:
            chunk = DocumentChunk(
                source_doc_id=doc_id,
                chunk_type="figure",
                content="[figure — VLM description pending]",
                provenance=Provenance(
                    source_doc_id=doc_id,
                    filename=pdf_path.name,
                    page_number=page_num,
                    bbox=None,
                    content_type="figure",
                ),
            )
            chunk_models.append(chunk)

    # Insert all chunks into MongoDB
    chunk_ids = insert_chunks(db, chunk_models)
    console.print(f"  Chunks stored in MongoDB: {len(chunk_ids)}")

    # Embed text chunks in Qdrant
    vectors_created = 0
    if embed:
        ensure_collection()
        for chunk_id, chunk_model in zip(chunk_ids, chunk_models):
            if chunk_model.chunk_type == "text" and chunk_model.content.strip():
                point_id = embed_and_upsert(
                    chunk_text=chunk_model.content,
                    chunk_id=chunk_id,
                    source_doc_id=doc_id,
                    page_number=chunk_model.provenance.page_number,
                    content_type=chunk_model.chunk_type,
                )
                vectors_created += 1
        console.print(f"  Vectors in Qdrant: {vectors_created}")

    # Update document status
    db.documents.update_one(
        {"file_hash": file_hash},
        {"$set": {"extraction_status": "complete"}},
    )

    summary = {
        "status": "ingested",
        "filename": pdf_path.name,
        "doc_id": doc_id,
        "chunks": len(chunk_ids),
        "vectors": vectors_created,
        "figures": len(images),
    }
    console.print(f"[green]Done:[/green] {pdf_path.name} — {len(chunk_ids)} chunks, {vectors_created} vectors")
    return summary
