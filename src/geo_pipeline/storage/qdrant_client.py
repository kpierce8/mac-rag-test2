from __future__ import annotations

import logging
import os
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()
logger = logging.getLogger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
TEXT_COLLECTION = "text_chunks"

_client: QdrantClient | None = None
_embedder = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
    return _client


def _get_embedder():
    """Lazy-load sentence-transformers model."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def ensure_collection(vector_size: int = 384) -> None:
    """Create the text_chunks collection if it doesn't exist."""
    client = get_client()
    collections = [c.name for c in client.get_collections().collections]
    if TEXT_COLLECTION not in collections:
        client.create_collection(
            collection_name=TEXT_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s'", TEXT_COLLECTION)


def embed_text(text: str) -> list[float]:
    """Compute embedding for a text string."""
    model = _get_embedder()
    return model.encode(text).tolist()


def embed_and_upsert(
    chunk_text: str,
    chunk_id: str,
    source_doc_id: str,
    page_number: int,
    content_type: str,
    lat: float | None = None,
    lon: float | None = None,
) -> str:
    """Embed text and upsert to Qdrant. Returns the point ID."""
    client = get_client()
    embedding = embed_text(chunk_text)
    point_id = str(uuid.uuid4())

    payload = {
        "chunk_id": chunk_id,
        "source_doc_id": source_doc_id,
        "page_number": page_number,
        "content_type": content_type,
    }
    if lat is not None and lon is not None:
        payload["lat"] = lat
        payload["lon"] = lon

    client.upsert(
        collection_name=TEXT_COLLECTION,
        points=[
            PointStruct(id=point_id, vector=embedding, payload=payload)
        ],
    )
    return point_id


def search(query_text: str, top_k: int = 5) -> list[dict]:
    """Search for similar chunks by text query."""
    client = get_client()
    query_vector = embed_text(query_text)
    results = client.query_points(
        collection_name=TEXT_COLLECTION,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "id": str(hit.id),
            "score": hit.score,
            **hit.payload,
        }
        for hit in results.points
    ]
