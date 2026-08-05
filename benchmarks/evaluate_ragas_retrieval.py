"""Runs private retrieval-quality and latency benchmarks against the local API."""

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import median, quantiles
from time import perf_counter
from typing import Any
from uuid import UUID

import httpx
import jwt
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from openai import OpenAI
from ragas import EvaluationDataset, evaluate
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision, ContextRecall
from sqlalchemy.ext.asyncio import create_async_engine

from app.services.vector_store import search_document_vectors
from benchmarks.benchmark_models import BenchmarkRecord, load_records


def parse_arguments() -> argparse.Namespace:
    """Parses local benchmark settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pdf", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--api-base-url", default="http://localhost:8000")
    parser.add_argument(
        "--jwt-secret-key", default=os.environ.get("JWT_SECRET_KEY")
    )
    parser.add_argument(
        "--database-url", default=os.environ.get("DATABASE_URL")
    )
    parser.add_argument(
        "--collection-name", default=os.environ.get("VECTOR_COLLECTION_NAME")
    )
    parser.add_argument("--judge-model", default="qwen3.5:4b")
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get(
            "EMBEDDING_MODEL",
            "hf.co/LiquidAI/LFM2.5-Embedding-350M-GGUF",
        ),
    )
    parser.add_argument(
        "--ollama-base-url",
        default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--baseline", type=Path)
    return parser.parse_args()


def create_access_token(*, secret_key: str) -> str:
    """Creates a short-lived local token for authenticated benchmark calls."""
    return jwt.encode(
        {
            "sub": "local-benchmark",
            "exp": datetime.now(UTC) + timedelta(hours=1),
        },
        secret_key,
        algorithm="HS256",
    )


def percentile_summary(*, durations: list[float]) -> dict[str, float]:
    """Summarizes seconds as millisecond percentiles."""
    if not durations:
        raise ValueError("Cannot summarize an empty duration list")
    if len(durations) == 1:
        percentile_95 = durations[0]
    else:
        percentile_95 = quantiles(durations, n=100, method="inclusive")[94]
    return {
        "count": len(durations),
        "p50_ms": round(median(durations) * 1000, 3),
        "p95_ms": round(percentile_95 * 1000, 3),
    }


def upload_document(
    *, client: httpx.Client, headers: dict[str, str], source_pdf: Path
) -> UUID:
    """Uploads the private corpus through the production ingestion endpoint."""
    with source_pdf.open("rb") as source_file:
        response = client.post(
            "/documents/embed",
            headers=headers,
            files={"files": (source_pdf.name, source_file, "application/pdf")},
        )
    response.raise_for_status()
    return UUID(response.json()["documents"][0]["id"])


def query_document(
    *,
    client: httpx.Client,
    headers: dict[str, str],
    document_id: UUID,
    question: str,
    limit: int,
) -> tuple[list[str], float]:
    """Queries the public API and returns contexts with end-to-end duration."""
    started_at = perf_counter()
    response = client.post(
        "/documents/query",
        headers=headers,
        json={"query": question, "file_id": str(document_id), "limit": limit},
    )
    duration_seconds = perf_counter() - started_at
    response.raise_for_status()
    return (
        [item["content"] for item in response.json()["results"]],
        duration_seconds,
    )


async def measure_vector_searches(
    *,
    records: list[BenchmarkRecord],
    document_id: UUID,
    database_url: str,
    collection_name: str,
    embedding_model_name: str,
    ollama_base_url: str,
    limit: int,
    repetitions: int,
) -> list[float]:
    """Measures only pgvector similarity searches using the shared helper."""
    engine = create_async_engine(database_url)
    embedding_model = OllamaEmbeddings(
        model=embedding_model_name,
        base_url=ollama_base_url,
    )
    vector_store = PGVector(
        embeddings=embedding_model,
        connection=engine,
        collection_name=collection_name,
        use_jsonb=True,
        create_extension=False,
        async_mode=True,
    )
    await vector_store.__apost_init__()
    try:
        durations = []
        for record in records:
            embedding = await asyncio.to_thread(
                embedding_model.embed_query, record.question
            )
            for _ in range(repetitions):
                result = await search_document_vectors(
                    vector_store=vector_store,
                    embedding=embedding,
                    document_id=document_id,
                    limit=limit,
                )
                durations.append(result.duration_seconds)
        return durations
    finally:
        await engine.dispose()


def evaluate_quality(
    *,
    records: list[BenchmarkRecord],
    retrieved_contexts: list[list[str]],
    judge_model: Any,
) -> dict[str, float]:
    """Calculates the agreed Ragas retrieval-context metrics."""
    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": record.question,
                "reference": record.reference_answer,
                "reference_contexts": record.reference_contexts,
                "retrieved_contexts": contexts,
            }
            for record, contexts in zip(
                records, retrieved_contexts, strict=True
            )
        ]
    )
    result = evaluate(
        dataset=dataset,
        metrics=[
            ContextPrecision(llm=judge_model),
            ContextRecall(llm=judge_model),
        ],
        raise_exceptions=True,
        allow_nest_asyncio=False,
    )
    row_scores = result.to_pandas().to_dict(orient="records")
    metric_names = ("context_precision", "context_recall")
    return {
        metric_name: round(
            sum(float(row[metric_name]) for row in row_scores)
            / len(row_scores),
            4,
        )
        for metric_name in metric_names
    }


def compare_baseline(
    *, report: dict[str, Any], baseline_path: Path
) -> dict[str, Any]:
    """Calculates metric deltas from an earlier local report."""
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_quality = baseline["quality_by_limit"]
    quality_delta = {
        limit: {
            metric_name: round(
                value - float(baseline_quality[limit][metric_name]), 4
            )
            for metric_name, value in metrics.items()
            if limit in baseline_quality
            and metric_name in baseline_quality[limit]
        }
        for limit, metrics in report["quality_by_limit"].items()
    }
    baseline_latency = baseline["latency_by_limit"]
    latency_delta = {
        limit: {
            layer: {
                percentile: round(
                    values[percentile]
                    - float(baseline_latency[limit][layer][percentile]),
                    3,
                )
                for percentile in ("p50_ms", "p95_ms")
                if limit in baseline_latency
                and layer in baseline_latency[limit]
            }
            for layer, values in layers.items()
        }
        for limit, layers in report["latency_by_limit"].items()
    }
    return {"quality": quality_delta, "latency": latency_delta}


def render_markdown(*, report: dict[str, Any]) -> str:
    """Renders a concise human-readable local benchmark report."""
    lines = [
        "# Local RAG retrieval benchmark",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Quality",
        "",
        "| k | Context Precision | Context Recall |",
        "| - | - | - |",
    ]
    for limit, values in report["quality_by_limit"].items():
        lines.append(
            f"| {limit} | {values['context_precision']:.4f} | "
            f"{values['context_recall']:.4f} |"
        )
    lines.extend(["", "## Latency", ""])
    for limit, values in report["latency_by_limit"].items():
        lines.extend(
            [
                f"### k={limit}",
                "",
                "| Layer | p50 ms | p95 ms | Samples |",
                "| - | - | - | - |",
                (
                    f"| End-to-end | {values['end_to_end']['p50_ms']} | "
                    f"{values['end_to_end']['p95_ms']} | "
                    f"{values['end_to_end']['count']} |"
                ),
                (
                    f"| Vector search | {values['vector_search']['p50_ms']} | "
                    f"{values['vector_search']['p95_ms']} | "
                    f"{values['vector_search']['count']} |"
                ),
                "",
            ]
        )
    if report.get("baseline_delta"):
        lines.extend(["## Baseline delta", ""])
        for limit, values in report["baseline_delta"]["quality"].items():
            for name, value in values.items():
                lines.append(f"- k={limit} {name}: {value:+.4f}")
        for limit, layers in report["baseline_delta"]["latency"].items():
            for layer, values in layers.items():
                for name, value in values.items():
                    lines.append(f"- k={limit} {layer} {name}: {value:+.3f}")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Runs the complete private local benchmark workflow."""
    arguments = parse_arguments()
    if not arguments.source_pdf.is_file():
        raise FileNotFoundError(
            f"Source PDF does not exist: {arguments.source_pdf}"
        )
    if not arguments.jwt_secret_key:
        raise ValueError("jwt-secret-key is required")
    if not arguments.database_url:
        raise ValueError("database-url is required")
    if not arguments.collection_name:
        raise ValueError("collection-name is required")
    if arguments.repetitions < 1:
        raise ValueError("repetitions must be greater than zero")

    records = load_records(path=arguments.dataset)
    headers = {
        "Authorization": f"Bearer {create_access_token(secret_key=arguments.jwt_secret_key)}"
    }
    judge_model = llm_factory(
        arguments.judge_model,
        client=OpenAI(
            base_url=f"{arguments.ollama_base_url.rstrip('/')}/v1",
            api_key="ollama",
        ),
    )
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "configuration": {
            "judge_model": arguments.judge_model,
            "embedding_model": arguments.embedding_model,
            "collection_name": arguments.collection_name,
            "repetitions": arguments.repetitions,
        },
        "quality_by_limit": {},
        "latency_by_limit": {},
    }
    with httpx.Client(base_url=arguments.api_base_url, timeout=120) as client:
        readiness = client.get("/check/")
        readiness.raise_for_status()
        document_id: UUID | None = None
        try:
            document_id = upload_document(
                client=client,
                headers=headers,
                source_pdf=arguments.source_pdf,
            )
            for limit in (1, 3, 5):
                retrieved_contexts: list[list[str]] = []
                end_to_end_durations: list[float] = []
                for record in records:
                    contexts, duration = query_document(
                        client=client,
                        headers=headers,
                        document_id=document_id,
                        question=record.question,
                        limit=limit,
                    )
                    retrieved_contexts.append(contexts)
                    end_to_end_durations.append(duration)
                    for _ in range(arguments.repetitions - 1):
                        _, duration = query_document(
                            client=client,
                            headers=headers,
                            document_id=document_id,
                            question=record.question,
                            limit=limit,
                        )
                        end_to_end_durations.append(duration)
                report["quality_by_limit"][str(limit)] = evaluate_quality(
                    records=records,
                    retrieved_contexts=retrieved_contexts,
                    judge_model=judge_model,
                )
                vector_durations = asyncio.run(
                    measure_vector_searches(
                        records=records,
                        document_id=document_id,
                        database_url=arguments.database_url,
                        collection_name=arguments.collection_name,
                        embedding_model_name=arguments.embedding_model,
                        ollama_base_url=arguments.ollama_base_url,
                        limit=limit,
                        repetitions=arguments.repetitions,
                    )
                )
                report["latency_by_limit"][str(limit)] = {
                    "end_to_end": percentile_summary(
                        durations=end_to_end_durations
                    ),
                    "vector_search": percentile_summary(
                        durations=vector_durations
                    ),
                }
        finally:
            if document_id is not None:
                response = client.delete(
                    f"/documents/{document_id}", headers=headers
                )
                response.raise_for_status()

    if arguments.baseline:
        report["baseline_delta"] = compare_baseline(
            report=report, baseline_path=arguments.baseline
        )
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    arguments.report.with_suffix(".md").write_text(
        render_markdown(report=report), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
