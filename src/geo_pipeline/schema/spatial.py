from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Provenance(BaseModel):
    """Tracks where a piece of extracted content came from."""

    source_doc_id: str  # MongoDB ObjectId as string
    filename: str
    page_number: int
    bbox: list[float] | None  # [x0, y0, x1, y1] normalized 0-1
    content_type: Literal["text", "table", "figure", "aerial_image"]


class SpatialRef(BaseModel):
    """A spatial reference extracted from a document chunk."""

    ref_type: Literal["latlon", "plss", "wkt", "address", "named_place"]
    raw_text: str  # original string verbatim from document
    parsed: dict  # normalized form (type-specific structure)
    geometry: dict | None = None  # GeoJSON geometry once resolved via API
    confidence: float  # 0.0-1.0
    provenance: Provenance
    resolved: bool = False  # True once BLM/NHD API has been called


class DocumentChunk(BaseModel):
    """A chunk of content extracted from a source document."""

    source_doc_id: str
    chunk_type: Literal["text", "table", "figure", "aerial_image"]
    content: str  # text content or VLM description
    provenance: Provenance
    spatial_refs: list[SpatialRef] = []
    related_content_ids: list[str] = []  # cross-modal links
    embedding_id: str | None = None  # Qdrant point ID
