# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

## [0.3.0] - 2026-08-05

### Added

- Added Word, PowerPoint, Excel, OpenDocument, RTF, EPUB, and CSV ingestion through Firecrawl Anydoc. #7

## [0.2.0] - 2026-08-05

### Added

- Added document similarity query endpoint. #5
- Added local file ingestion for document embeddings. #6
- Added Docker build and GitHub Container Registry push scripts.

## [0.1.0] - 2026-08-04

### Added

- Added JWT authentication for protected API endpoints. #1
- Added document upload, retrieval, download, deletion, and pgvector ingestion with local Ollama embeddings. #3

### Changed

- Split the API and PostgreSQL services into separate Docker Compose configurations. #2
