# Geo-Pipeline: File Reference and Query Guide

This document describes every file in the repository, how they fit together,
and how to use the pipeline to query a folder of documents for spatial and
textual information.

---

## Overview: How the Pipeline Works

```
PDFs on disk
    ↓
ingest.py           — parse PDFs, store text chunks in MongoDB + Qdrant
    ↓
describe_figures.py — send extracted images to VLM, store descriptions
    ↓
ask.py              — run targeted queries, extract structured data to MongoDB
query.py            — raw similarity search, inspect what was retrieved
show_figures.py     — browse classified figure chunks
```

Documents flow in one direction: ingest first, describe figures second, then
query. Re-ingesting a file that has already been processed is a no-op (detected
by SHA-256 hash).

---

## Step-by-Step Workflow

### 1. Ingest a folder of PDFs

```bash
conda run -n geo-pipeline python scripts/ingest.py data/35564/
```

For each PDF this:
- Parses text and tables using Docling, splits into ~4000-character chunks
- Extracts embedded images (figures, maps, diagrams) as PNG
- Stores all chunks in MongoDB (`geo_pipeline.chunks`)
- Embeds text chunks in Qdrant (`text_chunks` collection) using sentence-transformers
- Figure chunks are stored with a `[figure — VLM description pending]` placeholder

### 2. Describe figures with the VLM

```bash
conda run -n geo-pipeline python scripts/describe_figures.py --folder data/35564/
```

For each pending figure chunk this:
- Re-extracts the image from the source PDF using Docling
- Sends the image to `qwen2.5vl:7b` via Ollama
- Classifies the image as: map, annotated_imagery, aerial_imagery, diagram,
  chart, photograph, table_image, or other
- Records description, spatial_info, and contains_spatial_data in MongoDB
- Embeds the description text in Qdrant so figures become searchable

### 3. Query documents

**Structured extraction (recommended):**
```bash
conda run -n geo-pipeline python scripts/ask.py queries/permit_extract.toml
```

**Plain Q&A:**
```bash
conda run -n geo-pipeline python scripts/ask.py queries/species_qa.toml
```

**Raw similarity search (debugging/exploration):**
```bash
conda run -n geo-pipeline python scripts/query.py "bankfull width hydraulic design"
```

**Browse classified figures:**
```bash
conda run -n geo-pipeline python scripts/show_figures.py --type map --spatial
```

---

## Scripts

### `scripts/ingest.py`
CLI entry point for PDF ingestion. Accepts a single file or a directory.
Skips files already in MongoDB (by hash). Use `--no-embed` to skip Qdrant
if you only want MongoDB storage.

### `scripts/ask.py`
The main query interface. Reads a TOML config file that defines what to search
for and how to extract it. Supports two modes:

- **Single-query mode** (`query = "..."` in TOML): one vector search, LLM
  answers as plain text or JSON depending on `structured = true`
- **Multi-field mode** (`[[fields]]` entries in TOML): each field runs its own
  targeted vector search, results are deduplicated, one LLM call fills the full
  schema. Structured results are stored in `geo_pipeline.extractions` in MongoDB.

### `scripts/describe_figures.py`
Processes figure chunks that have not yet been described by the VLM. Groups
work by source document to avoid re-converting the same PDF multiple times.
Matches images to their MongoDB chunks by insertion order (position-stable
across partial runs). Supports `--dry-run`, `--limit`, `--folder`, `--model`.

### `scripts/query.py`
Low-level Qdrant similarity search. Takes a query string, returns the top-k
most similar chunks as a table showing filename, page, score, and content
preview. Useful for understanding what the vector index actually contains and
whether a query retrieves the right material before building a TOML config.

### `scripts/show_figures.py`
Browses described figure chunks in MongoDB. Filter by `--type` (map, diagram,
photograph, etc.) or `--spatial` (figures where the VLM found geographic
information). Use `--summary` for a count breakdown by type.

---

## Query Config Files (`queries/`)

TOML files that define what `ask.py` extracts. Add new files here for each
document type or extraction task.

### `queries/permit_extract.toml`
Multi-field structured extraction for environmental permitting documents.
Each `[[fields]]` block defines one output field with its own targeted vector
search query, a plain-English description of what to extract, and a `top_k`
for how many chunks to retrieve. Scoped to `data/35564/`.

Fields extracted: project_name, permit_type, project_type, project_description,
species, bankfull_width_ft, high_flow_cfs, stream_simulation_design,
climate_change_impacts, habitat_improvements.

### `queries/species_qa.toml`
Single-query plain-text Q&A scoped to a specific file. Returns a narrative
answer rather than structured JSON. A good template for open-ended questions
where you want a summary rather than extracted values.

---

## Source Library (`src/geo_pipeline/`)

### `schema/documents.py`
`DocumentRecord` — Pydantic model for source document metadata stored in the
`documents` collection. Fields: filename, file_hash (SHA-256), total_pages,
file_size_bytes, ingested_at, extraction_status, source_path.

### `schema/spatial.py`
Three Pydantic models:
- `Provenance` — where a chunk came from (doc id, filename, page, bbox, content_type)
- `SpatialRef` — a spatial reference extracted from a chunk (latlon, PLSS, WKT,
  address, named_place) with geometry and confidence
- `DocumentChunk` — the core unit of storage: typed content (text, table, figure,
  aerial_image) with provenance, spatial_refs list, cross-modal links, and
  Qdrant embedding_id

### `schema/entities.py`
Placeholder for typed named entity models (species, parties, methods). Not yet
populated — target for Phase 1 extraction work.

### `ingestion/document_ingester.py`
Core ingestion logic. `ingest_pdf()` is the main entry point:
1. Hashes the file, skips if already in MongoDB
2. Converts with Docling (OCR off, picture extraction on)
3. Exports to markdown, splits into chunks at paragraph boundaries
4. Extracts embedded images as PNG bytes
5. Stores `DocumentRecord` and all `DocumentChunk` records in MongoDB
6. Embeds text chunks in Qdrant (figures are stored but not embedded here —
   that happens in `describe_figures.py` after VLM description)

### `storage/mongo_client.py`
MongoDB connection and helper functions. Database: `geo_pipeline`. Collections
used: `documents`, `chunks`, `extractions`, `spatial_refs`, `escalation_queue`.
Key helpers: `get_docs_by_folder()` (folder-scoped document lookup),
`insert_chunks()`, `insert_extraction()`, `upsert_document()`.

### `storage/qdrant_client.py`
Qdrant connection and vector search. Collection: `text_chunks`. Embedding model:
`all-MiniLM-L6-v2` (384-dimensional, CPU). Key functions: `search()` (with
optional `source_doc_ids` filter for folder/file scoping), `embed_and_upsert()`.
Payload fields stored per point: chunk_id, source_doc_id, page_number,
content_type, lat/lon (when resolved).

### `extraction/`, `retrieval/`, `escalation/`
Stub modules — not yet implemented. Targets for upcoming phases:
- `extraction/` — spatial entity extraction (PLSS, GPS, WKT) using pydantic-ai + Ollama
- `retrieval/` — Spatial RAG (geo-indexed queries) and Graph RAG (entity linking)
- `escalation/` — human-review queue for documents that fail local extraction,
  with optional cloud batch processing via Anthropic API

---

## Infrastructure

### `docker-compose.yaml`
Starts MongoDB and Qdrant as local services. Run once before ingesting:
```bash
docker compose up -d
```

### `environment.yml`
Conda environment spec for `geo-pipeline` (Python 3.11). Always update this
when adding a dependency, then install with:
```bash
conda run -n geo-pipeline pip install <package>
```

### `data/35564/`
Test document set — a folder of PDFs related to the Keystone restoration
project (permit applications, engineering reports, SEPA decisions, site maps).
This is the primary test corpus for Phase 0–1 development.

---

## MongoDB Collections Reference

| Collection | Contents |
|---|---|
| `documents` | One record per ingested PDF (metadata, hash, status) |
| `chunks` | All content chunks — text, figures, tables. Primary query target. |
| `extractions` | Results from structured `ask.py` runs (JSON output + provenance) |
| `spatial_refs` | Extracted spatial references with GeoJSON geometry (Phase 1+) |
| `escalation_queue` | Documents flagged for human review before cloud processing |

---

## Adding a New Query Config

1. Copy `queries/species_qa.toml` (plain Q&A) or `queries/permit_extract.toml`
   (structured extraction) as a starting point
2. Set `folder` or `file` to scope the search
3. For structured extraction, add one `[[fields]]` block per output field.
   Write the `query` as search terms likely to appear near that data in the
   documents, and write `description` as a plain-English instruction to the LLM
4. Test with `--dry-run` equivalent by running `query.py` first to verify the
   vector search retrieves the right chunks
5. Run with `ask.py` and check the JSON output; tune `top_k` and query wording
   if fields come back null
