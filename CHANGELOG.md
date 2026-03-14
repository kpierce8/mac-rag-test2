# Changelog

All notable changes to the Geospatial Knowledge Extraction Pipeline are documented here.

## [0.1.0] — 2026-03-14

### Added
- CLAUDE.md project specification with architecture rules, schemas, and phase roadmap
- pydantic-ai with Ollama as the unified VLM runtime across all machines (replacing MLX-only path)
- Local-first processing with user-approved cloud escalation queue architecture
- EscalationItem schema with structural validation (geocoding, PLSS parse, gazetteer checks)
- Escalation queue MongoDB collection and audit trail design
- Budget-controlled Anthropic Batch API integration (optional, `CLOUD_ENABLED=false` by default)
- VLM extraction skill (`.claude/skills/vlm-extraction.md`) with pydantic-ai agent patterns
- Codebase structure for `src/geo_pipeline/` package with ingestion, extraction, storage,
  retrieval, escalation, and schema modules
- Scripts: `ingest.py`, `query.py`, `review_queue.py`, `process_approved.py`
- VERSION and CHANGELOG tracking (semver 0.1.x, patch increment per commit)
