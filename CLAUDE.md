# Geospatial Knowledge Extraction Pipeline — Claude Code Context

## Project Overview

This system processes tens of thousands of documents (PDFs containing mixed text,
tables, figures, aerial/satellite imagery, and maps) and produces a queryable
knowledge graph with spatial awareness and citation provenance.

**Domain:** Geospatial and environmental restoration  
**Document types:** Restoration agreements, SRFB documents, environmental reports,
survey materials  
**Primary goal:** Extract typed, queryable spatial values (GPS coordinates, PLSS
descriptions, spatial references) for use with geometry APIs (BLM, NHD) and
build a Spatial + Graph RAG retrieval system.

---

## Non-Negotiable Architecture Rules

These decisions are final. Do not suggest alternatives without a strong reason.

1. **Local-first processing with user-approved cloud escalation.** All document
   ingestion, extraction, and embedding runs locally by default. The system must
   function fully with `cloud.enabled = false` (no cloud dependency). Documents
   that fail local extraction are placed in an **escalation queue** with reason
   codes — cloud processing happens only when a user explicitly reviews and
   approves queued documents. Cloud calls are batch-oriented (use Anthropic
   Batch API for 50% cost savings), budget-controlled, and logged to an audit
   trail. See the Escalation Queue section below for details.

2. **No system Python.** Always use the `geo-pipeline` conda environment. The
   interpreter is at `~/miniconda3/envs/geo-pipeline/bin/python`. Use
   `conda run -n geo-pipeline` or activate the env before running any script.

3. **MongoDB + Qdrant are the only stores.** Do not introduce additional databases,
   caches, or file-based stores without explicit approval. SQLite is not acceptable
   as a substitute.

4. **Pydantic for all schema validation.** Every extracted entity, spatial ref,
   and ingested document chunk must pass through a Pydantic model before storage.
   No raw dict insertion into MongoDB.

5. **Two-pass table extraction.** Docling handles structural parsing; the VLM
   handles visual validation. Always reconcile both passes and flag mismatches
   for human review — do not silently pick one.

6. **Cross-modal context links must be preserved.** When a figure and surrounding
   text share a page, they must be linked in MongoDB via `related_content_ids`.
   Never discard cross-modal relationships.

---

## Hardware Fleet

| Machine | OS | GPU | RAM | Role |
|---|---|---|---|---|
| MacBook M5 | macOS | Apple Silicon (unified 32 GB) | 32 GB | Prototyping, dev, MLX inference |
| zaphod | Ubuntu 24.04 | RTX 3090 (24 GB) | 128 GB | Primary inference (Qwen2.5-VL-32B) |
| Z2 | Windows 11 | RTX 3090 (24 GB) | 128 GB | CNN training — do not reassign |
| Dual-CPU Tower | Linux | Titan V (12 GB HBM2) | 256 GB | MongoDB, Qdrant, Granite-Docling-258M |
| A6000 towers (×2) | Linux | RTX A6000 (48 GB) | 256 GB | Offline ~6 months, Phase 3+ target |

**Current phase target: MacBook M5.** Zaphod is the secondary target for inference
scale-up. Do not write code that assumes zaphod is available during Phase 0–1.

---

## Model Stack

| Role | Model | Runtime | Hardware |
|---|---|---|---|
| Document decomposition | Docling (+ Granite-Docling-258M for validation) | Python lib / CPU | Dual-CPU Tower |
| Vision-first retrieval | ColQwen2 | colpali-engine | M5 / zaphod |
| Unified VLM (extraction + validation) | Qwen2.5-VL-7B (Phase 0–1), -32B (Phase 2+) | Ollama (all machines) | M5 / zaphod |
| Text embeddings (interim) | sentence-transformers (CPU) | sentence-transformers lib | M5 |
| Agent framework | pydantic-ai | Python lib | All machines |
| Cloud escalation (optional) | Claude Haiku 4.5 / Sonnet 4.6 | Anthropic Batch API | N/A (cloud) |

**Ollama endpoint on M5:** `http://localhost:11434` (Metal-accelerated on Apple Silicon)
**Ollama endpoint on zaphod:** `http://zaphod:11434`

**VLM runtime decision:** Ollama runs natively on macOS with Metal acceleration and
provides full feature parity (structured output, tool calling, vision) across all
machines. This eliminates the need for separate MLX vs Ollama code paths. If tighter
Apple Silicon optimization is needed later, `vllm-mlx` can serve an OpenAI-compatible
API that pydantic-ai talks to without code changes.

**Cloud API calls are forbidden in the default pipeline path.** Cloud models are only
invoked for user-approved escalation queue items (see Escalation Queue section).

---

## Conda Environment

**Environment name:** `geo-pipeline`  
**Python version:** 3.11

Core dependencies (always keep these in sync with `environment.yml`):
```
pydantic-ai[ollama]  # agent framework — structured output, tools, retry, Ollama provider
docling
pymongo
qdrant-client
pydantic>=2.0
httpx
rich
sentence-transformers
colpali-engine
rasterio             # for GeoTIFF CRS parsing
shapely
pyproj
tenacity             # retry logic for non-agent VLM calls
anthropic            # optional — only for escalation queue batch processing
```

When adding a new dependency, update `environment.yml` first, then install with:
```bash
conda run -n geo-pipeline pip install <package>
```

Never use `pip install` outside the conda env.

---

## Codebase Structure

```
geo-pipeline/
├── CLAUDE.md                  # this file
├── environment.yml            # conda env spec
├── .agents/
│   └── plans/                 # phase implementation plans
├── .claude/
│   └── skills/                # reusable agent skill definitions
├── src/
│   geo_pipeline/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── document_ingester.py   # Docling decomposition + storage
│   │   ├── image_ingester.py      # standalone aerial/satellite imagery
│   │   └── table_extractor.py     # two-pass table extraction
│   ├── extraction/
│   │   ├── spatial_extractor.py   # PLSS, GPS, WKT extraction
│   │   ├── entity_extractor.py    # species, habitat, parties, methods
│   │   ├── vlm_client.py          # pydantic-ai agent + Ollama VLM interface
│   │   └── validators.py          # structural validation (geocoding, PLSS parse)
│   ├── storage/
│   │   ├── mongo_client.py        # MongoDB connection + helpers
│   │   └── qdrant_client.py       # Qdrant collection management
│   ├── retrieval/
│   │   ├── spatial_rag.py         # geo-indexed retrieval (Phase 3)
│   │   └── graph_rag.py           # entity graph traversal (Phase 4)
│   ├── escalation/
│   │   ├── queue.py               # EscalationItem model + queue operations
│   │   └── cloud_processor.py     # Anthropic Batch API client (optional)
│   └── schema/
│       ├── documents.py           # MongoDB document models
│       ├── spatial.py             # SpatialRef and geometry models
│       └── entities.py            # typed entity models
├── scripts/
│   ├── ingest.py              # CLI entry point
│   ├── query.py               # retrieval CLI
│   ├── review_queue.py        # review & approve escalation queue items
│   └── process_approved.py    # send approved items to cloud API (batch)
└── tests/
    └── unit/
```

When creating new files, follow this structure exactly. Do not create top-level
scripts outside `scripts/` or ad-hoc modules outside `src/geo_pipeline/`.

---

## Core Schemas

These are the canonical schemas. Do not alter field names or types without
updating this file and all downstream consumers.

### Provenance
```python
class Provenance(BaseModel):
    source_doc_id: str        # MongoDB ObjectId as string
    filename: str
    page_number: int
    bbox: list[float] | None  # [x0, y0, x1, y1] normalized 0–1
    content_type: Literal["text", "table", "figure", "aerial_image"]
```

### SpatialRef
```python
class SpatialRef(BaseModel):
    ref_type: Literal["latlon", "plss", "wkt", "address", "named_place"]
    raw_text: str             # original string verbatim from document
    parsed: dict              # normalized form (type-specific structure)
    geometry: dict | None     # GeoJSON geometry once resolved via API
    confidence: float         # 0.0–1.0
    provenance: Provenance
    resolved: bool = False    # True once BLM/NHD API has been called
```

### DocumentChunk
```python
class DocumentChunk(BaseModel):
    source_doc_id: str
    chunk_type: Literal["text", "table", "figure", "aerial_image"]
    content: str              # text content or VLM description
    provenance: Provenance
    spatial_refs: list[SpatialRef] = []
    related_content_ids: list[str] = []  # cross-modal links
    embedding_id: str | None = None      # Qdrant point ID
```

---

## VLM Interaction Patterns

### Use pydantic-ai agents for all VLM extraction

All extraction uses pydantic-ai `Agent` with `OllamaProvider`. This gives you
structured output validation, automatic retry on parse failure, and tool calling
(e.g., geocoding) through a single abstraction.

```python
from pydantic_ai import Agent
from pydantic_ai.providers.ollama import OllamaProvider

extraction_agent = Agent(
    'ollama:qwen2.5-vl',
    output_type=SpatialRefList,
    system_prompt="Extract all spatial references...",
    retries=3,
)

result = await extraction_agent.run(chunk_text)
spatial_refs = result.output
```

For operations that don't need agent features (simple VLM calls without tools),
use the lower-level `vlm_client.py` with `tenacity` retry + Pydantic validation.

### Structured output
pydantic-ai registers the output Pydantic model as a tool schema with the LLM.
Ollama enforces the schema at generation time (native structured output since
pydantic-ai v1.67). Do not hand-write JSON schema prompts — let pydantic-ai
handle schema injection.

### Retry on validation failure
pydantic-ai automatically feeds Pydantic validation errors back to the LLM for
retry (up to `retries` attempts). On persistent failure, store the chunk with
`extraction_status: "failed"` and add to the escalation queue — never crash.

### Image input
For VLM calls that include page images, pass the image as base64 via the Ollama
multimodal API. Always include the page's extracted text alongside the image for
cross-modal grounding.

### Tool calling
Geocoding tools (Nominatim, BLM PLSS) are registered as pydantic-ai `@agent.tool`
functions. The VLM can call them during extraction to resolve named places and
legal land descriptions inline, without a separate post-processing step.

---

## MongoDB Conventions

**Database name:** `geo_pipeline`  
**Collections:**
- `documents` — source document metadata
- `chunks` — all DocumentChunk records (text, table, figure, image)
- `spatial_refs` — all SpatialRef records with 2dsphere index on `geometry`
- `entities` — typed named entities (Phase 4)
- `escalation_queue` — documents needing user review for cloud escalation
- `cloud_audit_log` — record of every cloud API call (cost, tokens, results)

**Always include:**
- `created_at: datetime` on insert
- `pipeline_version: str` to track which version of the code produced the record
- `extraction_status: Literal["pending", "complete", "failed", "needs_review"]`

**Geospatial index** (create on startup if not exists):
```python
db.spatial_refs.create_index([("geometry", "2dsphere")])
```

---

## Qdrant Conventions

**Collections:**
- `text_chunks` — sentence-transformer embeddings of text/table content
- `page_embeddings` — ColQwen2 page-level vision embeddings (Phase 2+)

**Payload fields required on every point:**
- `chunk_id` — MongoDB ObjectId as string
- `source_doc_id`
- `page_number`
- `content_type`
- `lat`, `lon` — if a spatial ref is resolved (enables geo-radius filter)

---

## External API Rules

### BLM GeoCommunicator (PLSS resolution)
- Endpoint: `https://geoengine.nationalmap.gov/arcgis/rest/services/`
- Rate limit: respect 1 req/sec, use `httpx.AsyncClient` with semaphore
- Cache resolved geometries in MongoDB — never re-call for the same PLSS string
- API key stored in `.env` as `BLM_API_KEY` — never hardcode

### NHD Plus (watershed resolution)
- Use the USGS NHD REST API for HUC boundary polygons
- Store resolved watershed polygons in `spatial_refs` collection with
  `ref_type: "watershed_boundary"`

### `.env` file
All secrets and config live in `.env`. Use `python-dotenv` to load. Never
commit `.env`. The `.env.example` in the repo shows all required keys.

---

## Escalation Queue (Local-First Cloud Fallback)

### Design Principles

The pipeline runs fully locally. Documents that fail local extraction are queued
for review — **never auto-escalated to cloud**. A human reviews the queue and
approves which documents to send to a cloud model.

### Flow

```
PDF → [Qwen2.5-VL-7B local] → Structural Validator
                                 ├─ PASS → accept, store normally
                                 ├─ PARTIAL → retry with adjusted prompt (1x)
                                 │              ├─ PASS → accept
                                 │              └─ FAIL → escalation queue
                                 └─ TOTAL FAIL → escalation queue
```

### Structural Validation (confidence signal)

Do not rely on VLM self-reported confidence. Use deterministic checks:
- **Coordinates:** Do they geocode to a valid location within Washington State?
- **PLSS:** Does the description parse to a valid Section/Township/Range?
- **Named places:** Does the name resolve in a gazetteer (Nominatim)?
- **Required fields:** Are all non-optional SpatialRef fields populated?

If any check fails, the extraction is flagged.

### EscalationItem Schema

```python
class EscalationItem(BaseModel):
    document_id: str              # MongoDB ObjectId as string
    filename: str
    file_hash: str
    reason: Literal[
        "coordinates_failed_geocoding",
        "plss_parse_error",
        "named_place_unresolved",
        "extraction_returned_empty",
        "validation_failed_all_retries",
        "low_spatial_ref_count",
    ]
    local_result: dict | None     # what the local model produced (if anything)
    failed_chunks: list[int]      # page numbers that failed
    queued_at: datetime
    status: Literal["pending_review", "approved", "rejected", "completed"]
    cloud_result: dict | None = None
    cloud_model: str | None = None
    cost_usd: float | None = None
    reviewed_at: datetime | None = None
    completed_at: datetime | None = None
```

### MongoDB Collection

- **Collection:** `escalation_queue`
- **Unique index:** `(file_hash, reason)` — one entry per document per failure type
- **Status flow:** `pending_review` → user approves → `approved` → batch sends →
  `completed` (or user rejects → `rejected`)

### Cloud Processing (user-approved only)

```toml
# .env or config.toml
CLOUD_ENABLED=false              # system works fully without this
CLOUD_PROVIDER=anthropic
CLOUD_TRIAGE_MODEL=claude-haiku-4.5
CLOUD_FULL_MODEL=claude-sonnet-4.6
CLOUD_DAILY_LIMIT_USD=5.00
CLOUD_MONTHLY_LIMIT_USD=50.00
CLOUD_USE_BATCH_API=true         # 50% discount, 24hr turnaround
```

When `CLOUD_ENABLED=false`, approved items are simply marked `approved` and await
the user enabling cloud processing. The pipeline never blocks on cloud availability.

### Audit Trail

Every cloud call is logged:
- `document_id`, `filename`, `reason` (why it was escalated)
- `local_result` vs `cloud_result` (what changed)
- `cloud_model`, `input_tokens`, `output_tokens`, `cost_usd`
- `reviewed_at`, `completed_at` (timestamps for the full lifecycle)

### Scripts

- `scripts/review_queue.py` — CLI to list pending items, approve/reject in bulk
- `scripts/process_approved.py` — sends approved items to cloud API (batch mode)

---

## Testing Standards

- Unit tests live in `tests/unit/` and use `pytest`
- Every Pydantic schema must have at least one round-trip test (serialize → validate)
- Every extractor function must have tests against fixture documents in `tests/fixtures/`
- Use `pytest-mock` to mock VLM calls in unit tests — never call a real model in CI
- Run tests with: `conda run -n geo-pipeline pytest tests/unit/ -v`

---

## Phase Summary

| Phase | Focus | Hardware | Exit Criterion |
|---|---|---|---|
| 0 | Foundation — Docling, MongoDB, Qdrant, pipeline skeleton | M5 | One PDF → stored chunks + queryable vectors |
| 1 | Spatial entity extraction — VLM-7B via Ollama, PLSS, GPS, schema design | M5 | >80% extraction accuracy on 10-doc sample |
| 2 | Multimodal pipeline — ColQwen2, figures, aerial imagery, cross-modal context | M5 → zaphod | Mixed doc set fully ingested with cross-modal links |
| 3 | Spatial RAG — BLM + NHD APIs, MongoDB geo index, spatial retrieval | zaphod primary | Polygon query returns proximate docs, <2s latency |
| 4 | Graph RAG — entity linking, knowledge graph, hybrid Spatial + Graph RAG | zaphod + Tower | Multi-hop queries via entity graph |

**Escalation queue** is a cross-cutting concern active from Phase 1 onward: structural
validation → user review → optional cloud batch processing. It is not a separate phase.

**Current phase: 0**

---

## What NOT To Do

- Do not suggest Elasticsearch, Pinecone, Weaviate, or ChromaDB as replacements
  for MongoDB + Qdrant.
- Do not use `langchain` or `llamaindex` — these are heavyweight frameworks
  with excessive abstraction. Use `pydantic-ai` for agent orchestration (structured
  output, tool calling, retry/validation). All pipeline-level orchestration
  beyond individual VLM calls is custom Python.
- Do not write async code with `asyncio.run()` inside sync functions — use
  `asyncio.gather()` at the top-level entry point only.
- Do not log sensitive document content to stdout in production mode — use
  `rich` with a log level check.
- Do not create new conda environments without updating `environment.yml`.
- Do not use relative imports (`from ..module`) — always use absolute imports
  from `geo_pipeline.*`.
- Do not add type: ignore comments without a comment explaining why.
