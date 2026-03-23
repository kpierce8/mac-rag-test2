#!/usr/bin/env python
"""CLI for RAG query with LLM answering — retrieves context and generates answers.

Usage with a config file (recommended):
    python scripts/ask.py queries/permit_extract.toml
    python scripts/ask.py queries/species_qa.toml
    python scripts/ask.py queries/permit_extract.toml --top-k 20   # override a field

Usage with CLI args (still works):
    python scripts/ask.py --query "What coordinates are mentioned?" --folder data/35564/
    python scripts/ask.py --query "Summarize the project goals" --top-k 10
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tomllib

import httpx
from bson import ObjectId
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from geo_pipeline.storage.mongo_client import get_database, get_docs_by_folder, insert_extraction
from geo_pipeline.storage.qdrant_client import ensure_collection, search

load_dotenv()
console = Console()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("ASK_MODEL", "qwen2.5vl:7b")
PIPELINE_VERSION = os.getenv("PIPELINE_VERSION", "0.1.0")

SYSTEM_PROMPT = (
    "You are a document analyst. Answer the question using ONLY the provided context. "
    "Cite specific documents and page numbers. If the context doesn't contain enough "
    "information to answer, say so clearly."
)

DEFAULT_STRUCTURED_PROMPT = (
    "You are a structured data extraction assistant. Extract the requested information "
    "from the provided context and return it as a valid JSON object. Use null for any "
    "fields you cannot determine from the context. Do not include explanatory text "
    "outside the JSON."
)


def _load_config(args: argparse.Namespace) -> dict:
    """Merge TOML config (if provided) with CLI overrides. CLI flags win."""
    config = {
        "query": None,
        "folder": None,
        "file": None,
        "top_k": 8,
        "model": DEFAULT_MODEL,
        "structured": False,
        "system_prompt": None,
    }

    # Load TOML base if provided
    if args.config:
        with open(args.config, "rb") as f:
            toml_data = tomllib.load(f)
        for key in config:
            if key in toml_data:
                config[key] = toml_data[key]
        console.print(f"Loaded config from [cyan]{args.config}[/cyan]")

    # CLI overrides — only apply if the user actually passed the flag
    if args.query is not None:
        config["query"] = args.query
    if args.folder is not None:
        config["folder"] = args.folder
    if args.file is not None:
        config["file"] = args.file
    if args.top_k is not None:
        config["top_k"] = args.top_k
    if args.model is not None:
        config["model"] = args.model
    if args.structured:
        config["structured"] = True
    if args.system_prompt is not None:
        config["system_prompt"] = args.system_prompt

    if not config["query"]:
        console.print("[red]Error: query is required (set in TOML or pass --query)[/red]")
        sys.exit(1)

    return config


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
    with httpx.Client(timeout=600.0) as client:
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
        if resp.status_code != 200:
            console.print(f"[red]Ollama error {resp.status_code}:[/red] {resp.text[:500]}")
            resp.raise_for_status()
        return resp.json()["message"]["content"]


def _extract_json(text: str) -> dict:
    """Parse JSON from LLM response, handling markdown fences."""
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

    raise ValueError(f"Could not parse JSON from LLM response:\n{stripped[:500]}")


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
    parser.add_argument("config", nargs="?", default=None, help="Path to a TOML config file")
    parser.add_argument("--query", type=str, default=None, help="Question to ask")
    parser.add_argument("--folder", type=str, default=None, help="Scope to documents from this folder")
    parser.add_argument("--file", type=str, default=None, help="Scope to documents matching this filename")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve (default: 8)")
    parser.add_argument("--model", type=str, default=None, help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--structured", action="store_true", help="Parse LLM response as JSON and store in MongoDB")
    parser.add_argument("--system-prompt", type=str, default=None, help="Override the system prompt sent to the LLM")

    args = parser.parse_args()
    cfg = _load_config(args)

    db = get_database()
    ensure_collection()

    # Resolve document scope
    doc_ids = _resolve_doc_ids(db, cfg["folder"], cfg["file"])
    if doc_ids is not None and len(doc_ids) == 0:
        return

    # Retrieve relevant chunks
    hits = search(cfg["query"], top_k=cfg["top_k"], source_doc_ids=doc_ids)
    if not hits:
        console.print("[yellow]No relevant chunks found.[/yellow]")
        return

    console.print(f"Retrieved {len(hits)} chunks, calling [cyan]{cfg['model']}[/cyan]...")

    # Build context and ask LLM
    context = _build_context(db, hits)
    user_message = f"Context:\n{context}\n\nQuestion: {cfg['query']}"

    # Select system prompt
    if cfg["system_prompt"]:
        system_prompt = cfg["system_prompt"]
    elif cfg["structured"]:
        system_prompt = DEFAULT_STRUCTURED_PROMPT
    else:
        system_prompt = SYSTEM_PROMPT

    answer = _call_ollama(cfg["model"], system_prompt, user_message)

    if cfg["structured"]:
        try:
            result = _extract_json(answer)
        except ValueError as exc:
            console.print(f"[red]JSON parse failed:[/red] {exc}")
            console.print("\n[bold]Raw response:[/bold]")
            console.print(answer)
            return

        # Store in MongoDB
        chunk_ids = [h.get("chunk_id", "") for h in hits]
        extraction = {
            "query": cfg["query"],
            "system_prompt": system_prompt,
            "source_doc_ids": doc_ids or [],
            "folder": cfg["folder"],
            "result": result,
            "model": cfg["model"],
            "chunk_ids": chunk_ids,
            "pipeline_version": PIPELINE_VERSION,
        }
        extraction_id = insert_extraction(db, extraction)
        console.print(f"\nStored extraction [green]{extraction_id}[/green] in MongoDB")

        console.print()
        console.print_json(json.dumps(result))
        _print_sources(db, hits)
    else:
        console.print()
        console.print(Markdown(answer))
        _print_sources(db, hits)


if __name__ == "__main__":
    main()
