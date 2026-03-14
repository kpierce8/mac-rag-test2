# Getting Started — Phase 0 & Phase 1

This guide walks through setting up the development environment on MacBook M5
and completing Phases 0 and 1 of the geo-pipeline.

**Phase 0 exit criterion:** One PDF ingested → stored chunks in MongoDB + queryable vectors in Qdrant
**Phase 1 exit criterion:** >80% spatial extraction accuracy on a 10-document sample

---

## Prerequisites

- macOS with Homebrew
- Docker Desktop (for MongoDB + Qdrant)
- Ollama (for Qwen2.5-VL)
- Miniconda or Miniforge

---

## 1. Environment Setup

### 1a. Install Ollama

Download from https://ollama.com or:
```bash
brew install ollama
```

Start Ollama (or use Ollama.app):
```bash
ollama serve
```

Pull the VLM model (~4.7 GB Q4):
```bash
ollama pull qwen2.5-vl:7b
```

Verify it works:
```bash
ollama run qwen2.5-vl:7b "What spatial reference system uses Township, Range, and Section?"
```

### 1b. Create the conda environment

```bash
conda create -n geo-pipeline python=3.11 -y
conda activate geo-pipeline
```

Install core dependencies:
```bash
pip install "pydantic-ai[ollama]" docling pymongo qdrant-client "pydantic>=2.0" \
    httpx rich sentence-transformers tenacity pdf2image pillow pytest pytest-asyncio
```

Save the environment spec:
```bash
conda env export --from-history > environment.yml
```

### 1c. Start databases via Docker

Create `docker-compose.yaml` in the project root:
```yaml
services:
  mongodb:
    image: mongo:7
    container_name: geo-mongodb
    ports:
      - "27017:27017"
    volumes:
      - ./mongo_data:/data/db

  qdrant:
    image: qdrant/qdrant:latest
    container_name: geo-qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_data:/qdrant/storage
```

```bash
docker compose up -d
```

Verify both are running:
```bash
# MongoDB
python -c "from pymongo import MongoClient; print(MongoClient().server_info()['version'])"

# Qdrant
python -c "from qdrant_client import QdrantClient; print(QdrantClient('http://localhost:6333').get_collections())"
```

### 1d. Create the .env file

```bash
cat > .env << 'EOF'
# Required
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5-vl:7b
MONGODB_URL=mongodb://localhost:27017/
QDRANT_URL=http://localhost:6333

# Optional — cloud escalation (disabled by default)
CLOUD_ENABLED=false
# ANTHROPIC_API_KEY=sk-ant-...
# CLOUD_TRIAGE_MODEL=claude-haiku-4.5
# CLOUD_FULL_MODEL=claude-sonnet-4.6
# CLOUD_DAILY_LIMIT_USD=5.00
# CLOUD_MONTHLY_LIMIT_USD=50.00
EOF
```

### 1e. Prepare test documents

Copy a small set of PDFs (5-10 restoration/SRFB documents) into `data/`:
```bash
mkdir -p data
# Copy PDFs from mac-rag-test or another source:
# cp /path/to/restoration-docs/*.pdf data/
```

If you have the mac-rag-test corpus available:
```bash
cp ../mac-rag-test/data/*.pdf data/
```

---

## 2. Phase 0 — Foundation

**Goal:** Build the pipeline skeleton so that one PDF produces stored chunks in
MongoDB and queryable vectors in Qdrant.

### Step 0.1: Create the package structure

```bash
mkdir -p src/geo_pipeline/{ingestion,extraction,storage,retrieval,escalation,schema}
mkdir -p scripts tests/unit tests/fixtures
touch src/geo_pipeline/__init__.py
touch src/geo_pipeline/{ingestion,extraction,storage,retrieval,escalation,schema}/__init__.py
```

### Step 0.2: Implement Pydantic schemas

Create the canonical schemas in `src/geo_pipeline/schema/`:

**`src/geo_pipeline/schema/documents.py`** — Document metadata model:
```python
from __future__ import annotations
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field

class DocumentRecord(BaseModel):
    filename: str
    file_hash: str                    # SHA-256 of file contents
    total_pages: int
    file_size_bytes: int
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    extraction_status: Literal["pending", "complete", "failed", "needs_review"] = "pending"
    pipeline_version: str = "0.1.0"
```

**`src/geo_pipeline/schema/spatial.py`** — Provenance, SpatialRef, DocumentChunk
(copy the schemas exactly as defined in CLAUDE.md).

Write round-trip tests:
```bash
# tests/unit/test_schemas.py
# Verify each schema can serialize and deserialize without data loss
```

### Step 0.3: Implement MongoDB client

**`src/geo_pipeline/storage/mongo_client.py`**:
- Connect using `MONGODB_URL` from `.env`
- Database: `geo_pipeline`
- Create collections and indexes on startup (including 2dsphere on `spatial_refs.geometry`)
- Helper functions: `upsert_document()`, `insert_chunks()`, `get_document_by_hash()`

### Step 0.4: Implement document ingester

**`src/geo_pipeline/ingestion/document_ingester.py`**:
- Accept a PDF path
- Compute SHA-256 file hash (skip if already ingested)
- Use Docling to decompose PDF into text, tables, and figures
- Create `DocumentChunk` records with `Provenance` for each chunk
- Store chunks in MongoDB `chunks` collection
- Store document metadata in `documents` collection

Cherry-pick the Docling + hash-based caching logic from mac-rag-test's
`test_analyst.py` (`get_or_cache_markdown()` function, ~30 lines).

### Step 0.5: Implement Qdrant client

**`src/geo_pipeline/storage/qdrant_client.py`**:
- Connect using `QDRANT_URL` from `.env`
- Create `text_chunks` collection (sentence-transformer vector size)
- `embed_and_upsert()`: take chunk text, compute embedding, upsert to Qdrant
  with required payload fields (`chunk_id`, `source_doc_id`, `page_number`, `content_type`)
- `search()`: query by text, return top-k results

### Step 0.6: Create the ingest CLI

**`scripts/ingest.py`**:
```bash
conda run -n geo-pipeline python scripts/ingest.py data/sample.pdf
```

This should:
1. Decompose the PDF via Docling
2. Store document metadata + chunks in MongoDB
3. Embed text chunks and upsert to Qdrant
4. Print a summary (chunks stored, vectors created)

### Step 0.7: Create the query CLI

**`scripts/query.py`**:
```bash
conda run -n geo-pipeline python scripts/query.py "Cascade Creek restoration"
```

This should:
1. Embed the query text
2. Search Qdrant for top-k similar chunks
3. Fetch chunk details from MongoDB
4. Print results with provenance (filename, page number, content snippet)

### Phase 0 exit check

Run against one PDF and verify:
```bash
# Ingest
conda run -n geo-pipeline python scripts/ingest.py data/sample.pdf

# Verify MongoDB
python -c "
from pymongo import MongoClient
db = MongoClient()['geo_pipeline']
print(f'Documents: {db.documents.count_documents({})}')
print(f'Chunks: {db.chunks.count_documents({})}')
"

# Verify Qdrant
python -c "
from qdrant_client import QdrantClient
c = QdrantClient('http://localhost:6333')
info = c.get_collection('text_chunks')
print(f'Vectors: {info.points_count}')
"

# Query
conda run -n geo-pipeline python scripts/query.py "restoration project location"
```

If all three return data, Phase 0 is complete.

---

## 3. Phase 1 — Spatial Entity Extraction

**Goal:** Extract spatial references (coordinates, PLSS, named places) from
document chunks using Qwen2.5-VL via pydantic-ai, with >80% accuracy on a
10-document sample.

### Step 1.1: Implement the VLM client

**`src/geo_pipeline/extraction/vlm_client.py`**:
- Direct Ollama calls via httpx for simple tasks (image description, yes/no)
- `tenacity` retry decorator (3 attempts, exponential backoff)
- See the vlm-extraction skill (`.claude/skills/vlm-extraction.md`) for the
  canonical implementation

### Step 1.2: Implement the spatial extraction agent

**`src/geo_pipeline/extraction/spatial_extractor.py`**:

```python
from pydantic_ai import Agent
from geo_pipeline.schema.spatial import SpatialRef
from pydantic import RootModel

class SpatialRefList(RootModel):
    root: list[SpatialRef]

spatial_agent = Agent(
    'ollama:qwen2.5-vl:7b',
    output_type=SpatialRefList,
    retries=3,
    system_prompt="You are a geospatial data extraction specialist..."
)
```

Register geocoding tools on the agent:
- `@spatial_agent.tool_plain` for `geocode_place()` (Nominatim)
- `@spatial_agent.tool_plain` for `lookup_plss()` (BLM API)

Cherry-pick the geometry builder functions from mac-rag-test's `test_analyst.py`:
- `lookup_plss()` (~40 lines)
- `lookup_watershed()` (~30 lines)
- `lookup_river_mile()` (~30 lines)
- `point_from_direction()` (~20 lines)

### Step 1.3: Implement structural validators

**`src/geo_pipeline/extraction/validators.py`**:
- `validate_coordinates()` — check lat/lon within Washington State bounds
  (45.5-49.0 N, 125.0-116.9 W)
- `validate_plss()` — regex parse for Township/Range/Section format
- `validate_named_place()` — Nominatim geocode check
- `validate_spatial_refs()` — run all validators, return (valid, failures)

### Step 1.4: Implement the escalation queue

**`src/geo_pipeline/escalation/queue.py`**:
- `EscalationItem` Pydantic model (from CLAUDE.md)
- `add_to_queue()` — insert into `escalation_queue` collection
- `get_pending()` — list items with `status: "pending_review"`
- `approve_items()` / `reject_items()` — batch status updates

**`scripts/review_queue.py`**:
```bash
# List pending items
conda run -n geo-pipeline python scripts/review_queue.py list

# Approve specific items
conda run -n geo-pipeline python scripts/review_queue.py approve --ids doc123 doc456

# Reject items
conda run -n geo-pipeline python scripts/review_queue.py reject --ids doc789
```

### Step 1.5: Integrate extraction into the ingest pipeline

Update `scripts/ingest.py` to add an `--extract` flag:
```bash
# Ingest + extract spatial refs
conda run -n geo-pipeline python scripts/ingest.py data/ --extract
```

The extraction flow for each chunk:
1. Run `spatial_agent.run(chunk_text)` to extract spatial refs
2. Run `validate_spatial_refs()` on the results
3. Store valid refs in `spatial_refs` collection with `extraction_status: "complete"`
4. For failed validations, add to escalation queue
5. Update chunk's `extraction_status` accordingly

### Step 1.6: Build the 10-document test set

Select 10 PDFs from your corpus that have known spatial references. Create a
ground truth file:

**`tests/fixtures/ground_truth.json`**:
```json
[
  {
    "filename": "cascade_creek_agreement.pdf",
    "expected_refs": [
      {"ref_type": "plss", "raw_text": "T17N R2W S14"},
      {"ref_type": "latlon", "raw_text": "46.9876°N 122.8543°W"},
      {"ref_type": "named_place", "raw_text": "Cascade Creek"}
    ]
  }
]
```

### Step 1.7: Measure extraction accuracy

Create an evaluation script:

**`scripts/evaluate.py`**:
```bash
conda run -n geo-pipeline python scripts/evaluate.py tests/fixtures/ground_truth.json
```

For each document:
1. Ingest + extract
2. Compare extracted refs against ground truth
3. Compute precision, recall, F1 per document
4. Report overall accuracy

**Accuracy metric:** A spatial ref is "correct" if:
- `ref_type` matches
- `raw_text` contains the expected text (fuzzy substring match)
- For coordinates: within 0.01 degrees of expected values
- For PLSS: exact Township/Range/Section match

### Phase 1 exit check

```bash
conda run -n geo-pipeline python scripts/evaluate.py tests/fixtures/ground_truth.json
```

Target: **>80% F1 score** across the 10-document sample. If below 80%:
- Review failed extractions in MongoDB (`extraction_status: "failed"`)
- Check the escalation queue for patterns
- Adjust system prompts or add few-shot examples
- Consider if certain document types need specialized prompts

---

## 4. Cherry-Pick Reference (from mac-rag-test)

These specific functions from mac-rag-test v1 are worth porting:

| v1 Location | v2 Target | What to copy |
|---|---|---|
| `test_analyst.py:get_or_cache_markdown()` | `ingestion/document_ingester.py` | Docling conversion + SHA-256 hash caching in MongoDB |
| `test_analyst.py:lookup_plss()` | `extraction/spatial_extractor.py` (as agent tool) | BLM PLSS API call + geometry parsing |
| `test_analyst.py:lookup_watershed()` | `extraction/spatial_extractor.py` (as agent tool) | NHD watershed boundary lookup |
| `test_analyst.py:lookup_river_mile()` | `extraction/spatial_extractor.py` (as agent tool) | NHD+ flowline point lookup |
| `test_analyst.py:point_from_direction()` | `extraction/spatial_extractor.py` (as agent tool) | "X miles direction of place" → calculated point |
| `test_analyst.py:build_geometry()` | `extraction/spatial_extractor.py` | Geometry dispatcher (ref_type → builder function) |
| `embed.py:get_model()` + `embed_images()` | `storage/qdrant_client.py` (Phase 2) | ColQwen2 lazy loading + MPS batch embedding |
| `docker-compose.yaml` | `docker-compose.yaml` | MongoDB + Qdrant service definitions |

**Do not copy:**
- pydantic-ai `Agent` class usage (v1 pattern) — use v2's pydantic-ai pattern instead
- Claude Haiku escalation logic — replaced by the escalation queue
- `llamaindex_app/` — not used
- Session log files — development notes only

---

## 5. Daily Workflow

```bash
# Terminal 1: databases
cd /Users/ken/DLRepos/mac-rag-test2
docker compose up -d

# Terminal 2: Ollama (or use Ollama.app)
ollama serve

# Terminal 3: development
conda activate geo-pipeline
cd /Users/ken/DLRepos/mac-rag-test2

# Run tests
pytest tests/unit/ -v

# Ingest a document
python scripts/ingest.py data/sample.pdf --extract

# Check the escalation queue
python scripts/review_queue.py list

# Query
python scripts/query.py "restoration project near Cascade Creek"
```

---

## Troubleshooting

**Ollama not responding:**
```bash
# Check if running
curl http://localhost:11434/api/tags
# If not, start it
ollama serve
```

**MongoDB connection refused:**
```bash
docker compose ps    # check container status
docker compose up -d # restart if needed
```

**Qdrant connection refused:**
```bash
curl http://localhost:6333/collections  # check health
docker compose logs qdrant              # check logs
```

**pydantic-ai structured output failures:**
- Check Ollama model version: `ollama list`
- Ensure you're using `qwen2.5-vl:7b` (not a non-vision variant)
- Try lowering temperature or increasing `retries` on the agent
- Check the raw output: `result.all_messages()` shows the full conversation

**Import errors (`ModuleNotFoundError: geo_pipeline`):**
```bash
# Install the package in development mode
cd /Users/ken/DLRepos/mac-rag-test2
pip install -e .
```

This requires a `pyproject.toml`:
```toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "geo-pipeline"
version = "0.1.0"
requires-python = ">=3.11"

[tool.setuptools.packages.find]
where = ["src"]
```
