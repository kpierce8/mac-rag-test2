from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from pymongo import MongoClient, GEOSPHERE
from pymongo.collection import Collection
from pymongo.database import Database

from geo_pipeline.schema.documents import DocumentRecord
from geo_pipeline.schema.spatial import DocumentChunk

load_dotenv()
logger = logging.getLogger(__name__)

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017/")
DB_NAME = "geo_pipeline"


def get_database() -> Database:
    client = MongoClient(MONGODB_URL)
    return client[DB_NAME]


def ensure_indexes(db: Database) -> None:
    """Create required indexes if they don't exist."""
    db.documents.create_index("file_hash", unique=True)
    db.chunks.create_index("source_doc_id")
    db.chunks.create_index("provenance.page_number")
    db.spatial_refs.create_index("source_doc_id")
    try:
        db.spatial_refs.create_index([("geometry", GEOSPHERE)])
    except Exception as exc:
        logger.warning("2dsphere index skipped (may need geometry data first): %s", exc)
    db.escalation_queue.create_index(
        [("file_hash", 1), ("reason", 1)], unique=True
    )


def get_document_by_hash(db: Database, file_hash: str) -> dict | None:
    return db.documents.find_one({"file_hash": file_hash})


def upsert_document(db: Database, doc: DocumentRecord) -> str:
    """Insert or update a document record. Returns the MongoDB _id as string."""
    data = doc.model_dump()
    result = db.documents.update_one(
        {"file_hash": doc.file_hash},
        {"$set": data},
        upsert=True,
    )
    if result.upserted_id:
        return str(result.upserted_id)
    existing = db.documents.find_one({"file_hash": doc.file_hash})
    return str(existing["_id"])


def insert_chunks(db: Database, chunks: list[DocumentChunk]) -> list[str]:
    """Insert document chunks. Returns list of inserted _id strings."""
    if not chunks:
        return []
    docs = [chunk.model_dump() for chunk in chunks]
    for d in docs:
        d["created_at"] = datetime.now(timezone.utc)
    result = db.chunks.insert_many(docs)
    return [str(oid) for oid in result.inserted_ids]


def get_chunks_by_doc(db: Database, source_doc_id: str) -> list[dict]:
    return list(db.chunks.find({"source_doc_id": source_doc_id}))
