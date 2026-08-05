"""Generates private Ragas benchmark candidates from a local PDF."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.testset import TestsetGenerator

from benchmarks.benchmark_models import BenchmarkRecord, write_records


def parse_arguments() -> argparse.Namespace:
    """Parses command-line options for local candidate generation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
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
    parser.add_argument("--testset-size", type=int, default=40)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def source_metadata(*, source_pdf: Path) -> dict[str, str]:
    """Builds provenance metadata without copying private source content."""
    with source_pdf.open("rb") as source_file:
        checksum = hashlib.file_digest(source_file, "sha256").hexdigest()
    return {
        "source": "local private file",
        "filename": source_pdf.name,
        "sha256": checksum,
    }


def main() -> None:
    """Generates candidates that require manual approval before evaluation."""
    arguments = parse_arguments()
    if not arguments.source_pdf.is_file():
        raise FileNotFoundError(
            f"Source PDF does not exist: {arguments.source_pdf}"
        )
    if arguments.testset_size < 1:
        raise ValueError("testset-size must be greater than zero")
    if arguments.output.exists() and not arguments.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {arguments.output}; use --overwrite"
        )

    documents = PyPDFLoader(str(arguments.source_pdf)).load()
    judge_model = ChatOllama(
        model=arguments.judge_model,
        base_url=arguments.ollama_base_url,
        temperature=0,
        format="json",
    )
    embedding_model = OllamaEmbeddings(
        model=arguments.embedding_model,
        base_url=arguments.ollama_base_url,
    )
    generator = TestsetGenerator.from_langchain(judge_model, embedding_model)
    generated_testset = generator.generate_with_langchain_docs(
        documents, testset_size=arguments.testset_size
    )
    generated_records = generated_testset.to_pandas().to_dict(orient="records")
    records = [
        BenchmarkRecord(
            question=record["user_input"],
            reference_answer=record["reference"],
            reference_contexts=record["reference_contexts"],
        )
        for record in generated_records
    ]
    write_records(path=arguments.output, records=records)
    arguments.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.metadata_output.write_text(
        json.dumps(source_metadata(source_pdf=arguments.source_pdf), indent=2)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
