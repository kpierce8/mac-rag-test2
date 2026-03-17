"""Round-trip tests for all Pydantic schemas."""
from datetime import datetime, timezone

from geo_pipeline.schema.documents import DocumentRecord
from geo_pipeline.schema.entities import Entity
from geo_pipeline.schema.spatial import DocumentChunk, Provenance, SpatialRef


def _make_provenance(**overrides) -> dict:
    defaults = {
        "source_doc_id": "abc123",
        "filename": "test.pdf",
        "page_number": 1,
        "bbox": None,
        "content_type": "text",
    }
    defaults.update(overrides)
    return defaults


class TestProvenance:
    def test_round_trip(self):
        data = _make_provenance(bbox=[0.1, 0.2, 0.8, 0.9])
        p = Provenance(**data)
        dumped = p.model_dump()
        restored = Provenance(**dumped)
        assert restored == p

    def test_nullable_bbox(self):
        p = Provenance(**_make_provenance())
        assert p.bbox is None


class TestDocumentRecord:
    def test_round_trip(self):
        doc = DocumentRecord(
            filename="test.pdf",
            file_hash="abc123def456",
            total_pages=10,
            file_size_bytes=5000,
        )
        dumped = doc.model_dump()
        restored = DocumentRecord(**dumped)
        assert restored.filename == "test.pdf"
        assert restored.extraction_status == "pending"
        assert restored.pipeline_version == "0.1.0"

    def test_all_statuses(self):
        for status in ("pending", "complete", "failed", "needs_review"):
            doc = DocumentRecord(
                filename="f.pdf",
                file_hash="h",
                total_pages=1,
                file_size_bytes=100,
                extraction_status=status,
            )
            assert doc.extraction_status == status


class TestSpatialRef:
    def test_round_trip(self):
        ref = SpatialRef(
            ref_type="latlon",
            raw_text="47.6°N 122.3°W",
            parsed={"lat": 47.6, "lon": -122.3},
            confidence=0.95,
            provenance=Provenance(**_make_provenance()),
        )
        dumped = ref.model_dump()
        restored = SpatialRef(**dumped)
        assert restored.ref_type == "latlon"
        assert restored.resolved is False
        assert restored.geometry is None

    def test_with_geometry(self):
        geojson = {"type": "Point", "coordinates": [-122.3, 47.6]}
        ref = SpatialRef(
            ref_type="latlon",
            raw_text="47.6, -122.3",
            parsed={"lat": 47.6, "lon": -122.3},
            geometry=geojson,
            confidence=1.0,
            resolved=True,
            provenance=Provenance(**_make_provenance()),
        )
        assert ref.geometry["type"] == "Point"
        assert ref.resolved is True

    def test_all_ref_types(self):
        for ref_type in ("latlon", "plss", "wkt", "address", "named_place"):
            ref = SpatialRef(
                ref_type=ref_type,
                raw_text="test",
                parsed={},
                confidence=0.5,
                provenance=Provenance(**_make_provenance()),
            )
            assert ref.ref_type == ref_type


class TestDocumentChunk:
    def test_round_trip(self):
        chunk = DocumentChunk(
            source_doc_id="doc1",
            chunk_type="text",
            content="Some restoration text about Cascade Creek.",
            provenance=Provenance(**_make_provenance()),
        )
        dumped = chunk.model_dump()
        restored = DocumentChunk(**dumped)
        assert restored.content == chunk.content
        assert restored.spatial_refs == []
        assert restored.related_content_ids == []
        assert restored.embedding_id is None

    def test_with_spatial_refs(self):
        ref = SpatialRef(
            ref_type="named_place",
            raw_text="Cascade Creek",
            parsed={"name": "Cascade Creek"},
            confidence=0.8,
            provenance=Provenance(**_make_provenance()),
        )
        chunk = DocumentChunk(
            source_doc_id="doc1",
            chunk_type="text",
            content="Near Cascade Creek.",
            provenance=Provenance(**_make_provenance()),
            spatial_refs=[ref],
            related_content_ids=["fig1", "fig2"],
        )
        assert len(chunk.spatial_refs) == 1
        assert chunk.related_content_ids == ["fig1", "fig2"]

    def test_all_chunk_types(self):
        for ct in ("text", "table", "figure", "aerial_image"):
            chunk = DocumentChunk(
                source_doc_id="doc1",
                chunk_type=ct,
                content="test",
                provenance=Provenance(**_make_provenance(content_type=ct)),
            )
            assert chunk.chunk_type == ct


class TestEntity:
    def test_round_trip(self):
        entity = Entity(
            entity_type="species",
            name="Chinook salmon",
            aliases=["king salmon", "Oncorhynchus tshawytscha"],
            provenance=Provenance(**_make_provenance()),
        )
        dumped = entity.model_dump()
        restored = Entity(**dumped)
        assert restored.name == "Chinook salmon"
        assert len(restored.aliases) == 2
