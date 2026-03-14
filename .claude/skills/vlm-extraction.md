# Skill: vlm-extraction

## Purpose

This skill defines the canonical pattern for all VLM-based extraction in the
geo-pipeline. Use it whenever writing or modifying code that calls Qwen2.5-VL
(via Ollama on any machine) to extract structured data from text or images.

The pattern uses **pydantic-ai** for agent-based extraction (structured output,
tool calling, retry/validation) and **tenacity** for simpler non-agent VLM calls.
Every extractor in `src/geo_pipeline/extraction/` must follow these patterns.

---

## 1. pydantic-ai Agent Interface

All extraction agents are defined in `src/geo_pipeline/extraction/` modules.
The VLM backend is always Ollama (runs on all machines with Metal/CUDA acceleration).

```python
# src/geo_pipeline/extraction/spatial_extractor.py

from __future__ import annotations

import os

from pydantic import BaseModel, RootModel
from pydantic_ai import Agent
from pydantic_ai.providers.ollama import OllamaProvider

from geo_pipeline.schema.spatial import SpatialRef

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-vl")


class SpatialRefList(RootModel):
    root: list[SpatialRef]

    def __iter__(self):
        return iter(self.root)


spatial_agent = Agent(
    f"ollama:{OLLAMA_MODEL}",
    output_type=SpatialRefList,
    retries=3,
    system_prompt=(
        "You are a geospatial data extraction specialist. Extract ALL spatial "
        "references from the provided content. Include GPS coordinates, PLSS "
        "township/range/section descriptions, named waterbodies, watershed "
        "identifiers, and any other location references. Be thorough — do not "
        "skip references even if they seem minor."
    ),
)
```

### Configuring the Ollama provider

pydantic-ai auto-detects Ollama from the `ollama:` model prefix. To customize
the endpoint:

```python
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

model = OpenAIChatModel(
    model_name='qwen2.5-vl',
    provider=OllamaProvider(base_url=OLLAMA_HOST + '/v1')
)
agent = Agent(model, output_type=SpatialRefList, retries=3)
```

---

## 2. Tool Registration

Register geocoding tools directly on the agent. pydantic-ai auto-generates
the JSON schema from function signatures and docstrings.

```python
import httpx


@spatial_agent.tool_plain
async def geocode_place(place_name: str) -> dict:
    """Look up GPS coordinates for a named place using Nominatim.

    Args:
        place_name: The name of the place to geocode (city, river, landmark).

    Returns:
        Dict with lat, lon, and display_name. Empty dict if not found.
    """
    skip_terms = {"not specified", "n/a", "unknown", "various", "multiple"}
    if place_name.lower().strip() in skip_terms:
        return {}

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": place_name, "format": "json", "limit": 1},
            headers={"User-Agent": "geo-pipeline/0.1"},
        )
        results = resp.json()
        if not results:
            return {}
        return {
            "lat": float(results[0]["lat"]),
            "lon": float(results[0]["lon"]),
            "display_name": results[0]["display_name"],
        }


@spatial_agent.tool_plain
async def lookup_plss(description: str) -> dict:
    """Resolve a PLSS legal land description to a GeoJSON polygon via BLM API.

    Args:
        description: PLSS string like "T12N R4E S14" or "Township 12 North, Range 4 East, Section 14".

    Returns:
        Dict with geometry (GeoJSON) and parsed fields. Empty dict if resolution fails.
    """
    # Implementation calls BLM GeoCommunicator REST API
    # Cache results in MongoDB to avoid re-calling for the same PLSS string
    ...
```

**Rules for tools:**
- Tools are plain async functions — no special base classes
- Use `@agent.tool_plain` for tools that don't need `RunContext`
- Use `@agent.tool` with `RunContext[Dependencies]` for tools needing DB access
- Google-style docstrings are mandatory — pydantic-ai extracts the schema from them
- Rate-limit external API calls with `httpx.AsyncClient` + semaphore

---

## 3. Running Extraction

```python
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

async def extract_spatial_refs(
    chunk_text: str,
    filename: str,
    page_number: int,
    image_path: Path | None = None,
) -> list[SpatialRef]:
    """Extract spatial references from a document chunk."""

    user_prompt = (
        f"Document: {filename}, page {page_number}\n\n"
        f"Content:\n{chunk_text[:6000]}"
    )

    try:
        result = await spatial_agent.run(user_prompt)
        return list(result.output)
    except Exception as exc:
        logger.warning("Extraction failed for %s p%d: %s", filename, page_number, exc)
        return []
```

### Handling failures

pydantic-ai retries automatically when Pydantic validation fails (feeds error
back to the LLM). If all retries exhaust:

```python
result = await spatial_agent.run(user_prompt)
refs = list(result.output)

if not refs:
    # No spatial refs found — may be a non-spatial page (normal) or extraction failure
    # Only escalate if structural validation suggests content was missed
    if has_spatial_keywords(chunk_text):
        await add_to_escalation_queue(
            document_id=doc_id,
            filename=filename,
            reason="extraction_returned_empty",
            failed_chunks=[page_number],
        )
```

---

## 4. Lower-Level VLM Client (non-agent calls)

For simple VLM calls that don't need tools or agent orchestration (e.g., image
description, yes/no classification), use the direct client with tenacity retry:

```python
# src/geo_pipeline/extraction/vlm_client.py

import base64
import json
import logging
from pathlib import Path

import httpx
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-vl")


class VLMResponse(BaseModel):
    raw_text: str
    parsed_json: dict | list | None = None
    model: str


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def call_vlm(
    prompt: str,
    image_path: Path | None = None,
    schema: dict | None = None,
) -> VLMResponse:
    """Direct VLM call via Ollama API. Use for non-agent tasks only."""
    messages = [{"role": "user", "content": prompt}]

    if image_path:
        img_b64 = base64.b64encode(image_path.read_bytes()).decode()
        messages[0]["images"] = [img_b64]

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
    }
    if schema:
        payload["format"] = schema  # JSON Schema object for constrained output

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{OLLAMA_HOST}/api/chat", json=payload)
        resp.raise_for_status()
        raw = resp.json()["message"]["content"]

    parsed = None
    if schema:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("VLM returned non-JSON despite schema constraint")

    return VLMResponse(raw_text=raw, parsed_json=parsed, model=OLLAMA_MODEL)
```

**When to use agent vs direct client:**
- **Agent** (`spatial_agent.run()`): extraction tasks needing structured output,
  tool calling, or retry with error feedback to LLM
- **Direct client** (`call_vlm()`): image description, content classification,
  simple yes/no questions

---

## 5. Image Grounding Pattern

When a chunk includes a page image, pass both the image and extracted text.
With pydantic-ai, use `BinaryContent` for image input:

```python
from pydantic_ai import BinaryContent

async def extract_from_page(
    chunk_text: str,
    page_image_path: Path,
    filename: str,
    page_number: int,
) -> list[SpatialRef]:
    image_data = page_image_path.read_bytes()

    result = await spatial_agent.run([
        BinaryContent(data=image_data, media_type="image/png"),
        f"Document: {filename}, page {page_number}\n"
        f"Page type: document scan with text and possibly maps/figures.\n"
        f"Extracted text from this page:\n{chunk_text[:6000]}\n\n"
        f"Extract ALL spatial references visible in either the image or text.",
    ])
    return list(result.output)
```

---

## 6. Structural Validation (post-extraction)

After extraction, validate results structurally before accepting. This feeds
the escalation queue:

```python
# src/geo_pipeline/extraction/validators.py

async def validate_spatial_refs(
    refs: list[SpatialRef],
    chunk_text: str,
    document_id: str,
    filename: str,
    page_number: int,
) -> tuple[list[SpatialRef], list[str]]:
    """Validate extracted refs. Returns (valid_refs, failure_reasons)."""
    valid = []
    failures = []

    for ref in refs:
        if ref.ref_type == "latlon":
            lat, lon = ref.parsed.get("lat"), ref.parsed.get("lon")
            if not _in_washington_state(lat, lon):
                failures.append("coordinates_failed_geocoding")
                continue

        elif ref.ref_type == "plss":
            if not _valid_plss(ref.raw_text):
                failures.append("plss_parse_error")
                continue

        elif ref.ref_type == "named_place":
            geo = await geocode_place(ref.raw_text)
            if not geo:
                failures.append("named_place_unresolved")
                continue

        valid.append(ref)

    return valid, failures


def _in_washington_state(lat: float | None, lon: float | None) -> bool:
    if lat is None or lon is None:
        return False
    return 45.5 <= lat <= 49.0 and -125.0 <= lon <= -116.9
```

---

## 7. Testing This Pattern

Every extractor must have these test cases in `tests/unit/`:

```python
# tests/unit/test_spatial_extraction.py

import pytest
from unittest.mock import AsyncMock, patch
from pydantic_ai.models.test import TestModel

from geo_pipeline.extraction.spatial_extractor import spatial_agent, SpatialRefList
from geo_pipeline.schema.spatial import SpatialRef


@pytest.mark.asyncio
async def test_extraction_returns_spatial_refs():
    """Agent extracts valid spatial references from text."""
    test_data = SpatialRefList(root=[
        SpatialRef(
            ref_type="latlon",
            raw_text="47.2°N 122.4°W",
            parsed={"lat": 47.2, "lon": -122.4},
            confidence=0.95,
            resolved=False,
            provenance={
                "source_doc_id": "abc",
                "filename": "test.pdf",
                "page_number": 1,
                "bbox": None,
                "content_type": "text",
            },
        )
    ])

    with spatial_agent.override(model=TestModel(custom_output_args=test_data)):
        result = await spatial_agent.run("Extract spatial refs from: 47.2°N 122.4°W")
        refs = list(result.output)
        assert len(refs) == 1
        assert refs[0].ref_type == "latlon"


@pytest.mark.asyncio
async def test_extraction_returns_empty_for_non_spatial():
    """Agent returns empty list for content without spatial references."""
    with spatial_agent.override(model=TestModel(custom_output_args=SpatialRefList(root=[]))):
        result = await spatial_agent.run("The meeting was productive.")
        assert list(result.output) == []


@pytest.mark.asyncio
async def test_validation_rejects_out_of_state_coordinates():
    """Structural validator rejects coordinates outside Washington State."""
    from geo_pipeline.extraction.validators import validate_spatial_refs

    ref = SpatialRef(
        ref_type="latlon",
        raw_text="35.0°N 118.0°W",
        parsed={"lat": 35.0, "lon": -118.0},  # California, not WA
        confidence=0.9,
        resolved=False,
        provenance={
            "source_doc_id": "abc",
            "filename": "test.pdf",
            "page_number": 1,
            "bbox": None,
            "content_type": "text",
        },
    )
    valid, failures = await validate_spatial_refs(
        [ref], "some text", "doc123", "test.pdf", 1
    )
    assert len(valid) == 0
    assert "coordinates_failed_geocoding" in failures
```

---

## When to Use This Skill

Invoke this skill any time you are:
- Writing a new extractor in `src/geo_pipeline/extraction/`
- Modifying `vlm_client.py` or any pydantic-ai agent definition
- Adding a new Pydantic schema that will be used as a VLM extraction target
- Registering new tools on an extraction agent
- Writing tests for any extraction function
- Debugging extraction failures in MongoDB (`extraction_status: "failed"` records)
- Working with the escalation queue (validation → queue → review flow)

Do not use this skill for:
- Docling document decomposition (see `skill: docling-ingestion` when available)
- Qdrant embedding upserts (see `skill: qdrant-storage` when available)
- BLM/NHD API resolution (see `skill: spatial-resolution` when available)
- Cloud escalation processing (see `skill: cloud-escalation` when available)
