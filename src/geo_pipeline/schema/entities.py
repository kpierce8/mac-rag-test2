from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from geo_pipeline.schema.spatial import Provenance


class Entity(BaseModel):
    """A typed named entity extracted from a document (Phase 4)."""

    entity_type: Literal[
        "species", "habitat", "organization", "person", "method", "regulation"
    ]
    name: str
    aliases: list[str] = []
    provenance: Provenance
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    pipeline_version: str = "0.1.0"
