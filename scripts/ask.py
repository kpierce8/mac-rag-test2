#!/usr/bin/env python
"""CLI for RAG query with LLM answering — retrieves context and generates answers.

Usage:
    python scripts/ask.py "What coordinates are mentioned?" --folder data/35564/
    python scripts/ask.py "What restoration methods?" --file wdfw00760.pdf
    python scripts/ask.py "Summarize the project goals" --top-k 10
"""
from __future__ import annotations

import argparse
import os

import httpx
from bson import ObjectId
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from geo_pipeline.storage.mongo_client import get_database, get_docs_by_folder
from geo_pipeline.storage.qdrant_client import ensure_collection, search

load_dotenv()
console = Console()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("ASK_MODEL", "qwen2.5-vl")

SYSTEM_PROMPT = (
    "You are a document analyst. Answer the question using ONLY the provided context. "
    "Cite specific documents and page numbers. If the context doesn't contain enough "
    "information to answer, say so clearly."
)


def _resolve_doc_ids(db, folder: str | None, file: str | None) -> list[str] | None:
    """Resolve folder/file filters to a list of source_doc_id strings."""
    if folder:
        from pathlib import Path

        resolved = str(Path(folder).resolve())
        docs = get_docs_by_folder(db, resolved)
        if not docs:
            console.print(f"[yellow]No documents found in folder: {folder}[/yellow]")
            return []
        ids = [str(d["_id"]) for d in docs]
        console.print(f"Scoped to {len(ids)} document(s) from [cyan]{folder}[/cyan]")
        return ids

    if file:
        docs = list(db.documents.find({"filename": {"$regex": file, "$options": "i"}}))
        if not docs:
            console.print(f"[yellow]No documents matching: {file}[/yellow]")
            return []
        ids = [str(d["_id"]) for d in docs]
        console.print(f"Scoped to {len(ids)} document(s) matching [cyan]{file}[/cyan]")
        return ids

    return None


def _build_context(db, hits: list[dict]) -> str:
    """Build a numbered context string from search hits with source citations."""
    sections = []
    for i, hit in enumerate(hits, 1):
        chunk_id = hit.get("chunk_id", "")
        score = hit.get("score", 0.0)
        chunk = db.chunks.find_one({"_id": ObjectId(chunk_id)}) if chunk_id else None
        if chunk:
            filename = chunk.get("provenance", {}).get("filename", "unknown")
            page = chunk.get("provenance", {}).get("page_number", "?")
            content = chunk.get("content", "")
        else:
            filename = "unknown"
            page = "?"
            content = "(chunk not found)"
        sections.append(
            f"--- Source {i}: {filename}, page {page} (score: {score:.2f}) ---\n{content}"
        )
    return "\n\n".join(sections)


def _call_ollama(model: str, system: str, user_message: str) -> str:
    """Call Ollama chat completion and return the assistant message content."""
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


def _print_sources(db, hits: list[dict]) -> None:
    """Print a compact sources list below the answer."""
    console.print("\n[bold]Sources:[/bold]")
    for i, hit in enumerate(hits, 1):
        chunk_id = hit.get("chunk_id", "")
        chunk = db.chunks.find_one({"_id": ObjectId(chunk_id)}) if chunk_id else None
        if chunk:
            filename = chunk.get("provenance", {}).get("filename", "unknown")
            page = chunk.get("provenance", {}).get("page_number", "?")
        else:
            filename = "unknown"
            page = "?"
        score = hit.get("score", 0.0)
        console.print(f"  {i}. {filename}, page {page} (score: {score:.2f})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask questions about ingested documents (RAG + LLM)"
    )
    parser.add_argument("query", type=str, help="Question to ask")
    parser.add_argument("--folder", type=str, default=None, help="Scope to documents from this folder")
    parser.add_argument("--file", type=str, default=None, help="Scope to documents matching this filename")
    parser.add_argument("--top-k", type=int, default=8, help="Number of chunks to retrieve (default: 8)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    db = get_database()
    ensure_collection()

    # Resolve document scope
    doc_ids = _resolve_doc_ids(db, args.folder, args.file)
    if doc_ids is not None and len(doc_ids) == 0:
        return

    # Retrieve relevant chunks
    hits = search(args.query, top_k=args.top_k, source_doc_ids=doc_ids)
    if not hits:
        console.print("[yellow]No relevant chunks found.[/yellow]")
        return

    console.print(f"Retrieved {len(hits)} chunks, calling [cyan]{args.model}[/cyan]...")

    # Build context and ask LLM
    context = _build_context(db, hits)
    user_message = f"Context:\n{context}\n\nQuestion: {args.query}"

    answer = _call_ollama(args.model, SYSTEM_PROMPT, user_message)

    console.print()
    console.print(Markdown(answer))
    _print_sources(db, hits)


if __name__ == "__main__":
    main()
