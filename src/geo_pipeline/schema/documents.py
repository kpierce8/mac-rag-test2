from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    """Source document metadata stored in MongoDB `documents` collection."""

    filename: str
    file_hash: str  # SHA-256 of file contents
    total_pages: int
    file_size_bytes: int
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extraction_status: Literal["pending", "complete", "failed", "needs_review"] = (
        "pending"
    )
    pipeline_version: str = "0.1.0"
    source_path: str | None = None  # absolute path at ingest time
