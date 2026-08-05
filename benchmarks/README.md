# Local RAG retrieval benchmark

This benchmark is intentionally local-only. It uses the private `cv.pdf` corpus,
the approved Ragas JSONL data, generated candidates, reports, and baselines under
`benchmarks/local/`. Git ignores all of them.

It measures Ragas `ContextPrecision` and `ContextRecall` for k=1, 3, and 5. It
also reports end-to-end authenticated HTTP latency and pgvector-only latency.
The latter calls the shared `app.services.vector_store.search_document_vectors`
function, so it measures the same search operation as the API without changing
the production response schema.

## Setup

1. Keep your private `cv.pdf` at the repository root.
2. Copy `.env.benchmark.example` to `.env.benchmark`, then set values that match
   your local PostgreSQL instance. `VECTOR_COLLECTION_NAME` and
   `UPLOAD_DIRECTORY` must be unused benchmark-specific names.
3. Pull the local judge model:

   ```bash
   ollama pull qwen3.5:4b
   ```

4. Start the isolated API resources. The explicit shell environment overrides
   the normal Compose defaults while `.env` continues to provide normal API
   settings such as `JWT_SECRET_KEY`:

   ```bash
   set -a
   source .env
   source .env.benchmark
   set +a
   docker compose up --build
   ```

## Generate and approve candidates

Generate more samples than the final dataset needs:

```bash
uv run --project benchmarks \
  python -m benchmarks.generate_ragas_testset \
  --source-pdf cv.pdf \
  --output benchmarks/local/candidates.jsonl \
  --metadata-output benchmarks/local/corpus-metadata.json \
  --judge-model "$RAGAS_JUDGE_MODEL" \
  --testset-size 40
```

Review candidates manually. Save 15-30 unambiguous records to
`benchmarks/local/approved.jsonl`, one JSON object per line:

```json
{"question":"...","reference_answer":"...","reference_contexts":["..."]}
```

Do not approve generated records with incorrect answers, ambiguous questions,
or irrelevant reference contexts.

## Evaluate

With the same environment still exported, run:

```bash
uv run --project benchmarks \
  python -m benchmarks.evaluate_ragas_retrieval \
  --source-pdf cv.pdf \
  --dataset benchmarks/local/approved.jsonl \
  --report benchmarks/local/reports/latest.json \
  --judge-model "$RAGAS_JUDGE_MODEL"
```

The command verifies API readiness, uploads the corpus through the authenticated
API, evaluates all approved questions, writes JSON and Markdown reports, then
deletes the uploaded document. It leaves the benchmark collection itself intact.
For the first baseline, copy `latest.json` to `benchmarks/local/baseline.json`.
For later runs, add `--baseline benchmarks/local/baseline.json` to report metric
deltas. Results are informational only and do not run in CI.

## Compatibility

Ragas currently requires an older LangChain stack than the application. The
benchmark is an isolated, locked uv project in `benchmarks/`; run `uv sync
--project benchmarks --locked` to pre-install it. Its `uv.lock` and
`pyproject.toml` prevent it from altering the API's locked dependencies.
