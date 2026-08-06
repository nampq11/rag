"""Generates private benchmark candidates from a PDF with DeepSeek."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, TypeAdapter

from benchmarks.benchmark_models import BenchmarkRecord, write_records


class GeneratedCandidate(BaseModel):
    """A candidate returned by DeepSeek before its source context is attached."""

    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    context_indexes: list[int] = Field(min_length=1)


def parse_arguments() -> argparse.Namespace:
    """Parses command-line options for candidate generation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("RAGAS_JUDGE_MODEL", "deepseek-v4-flash"),
    )
    parser.add_argument(
        "--deepseek-api-key", default=os.environ.get("DEEPSEEK_API_KEY")
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


def build_generation_prompt(
    *, contexts: list[str], testset_size: int
) -> str:
    """Builds the single-turn, JSON-only candidate-generation request."""
    formatted_contexts = "\n\n".join(
        f"[Context {index}]\n{context}"
        for index, context in enumerate(contexts, start=1)
    )
    return f"""Create exactly {testset_size} diverse retrieval-benchmark candidates.
Use only the source contexts below. Each question must be answerable from its cited
contexts alone. Do not invent facts, and avoid ambiguous, opinion-based, duplicate,
or multi-part questions. Reference answers must be concise and factual.

Return JSON only, with this exact schema:
{{"records":[{{"question":"...","reference_answer":"...","context_indexes":[1]}}]}}

Source contexts:
{formatted_contexts}
"""


def build_records(*, response_content: str, contexts: list[str]) -> list[BenchmarkRecord]:
    """Validates DeepSeek output and attaches exact source contexts."""
    try:
        payload = json.loads(response_content)
        candidates = TypeAdapter(list[GeneratedCandidate]).validate_python(
            payload["records"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("DeepSeek returned invalid candidate JSON") from error

    records = []
    for candidate in candidates:
        if any(
            index < 1 or index > len(contexts)
            for index in candidate.context_indexes
        ):
            raise ValueError("DeepSeek returned an invalid context index")
        records.append(
            BenchmarkRecord(
                question=candidate.question,
                reference_answer=candidate.reference_answer,
                reference_contexts=[
                    contexts[index - 1] for index in candidate.context_indexes
                ],
            )
        )
    return records


def main() -> None:
    """Generates candidates that require manual approval before evaluation."""
    arguments = parse_arguments()
    if not arguments.source_pdf.is_file():
        raise FileNotFoundError(
            f"Source PDF does not exist: {arguments.source_pdf}"
        )
    if arguments.testset_size < 1:
        raise ValueError("testset-size must be greater than zero")
    if not arguments.judge_model.strip():
        raise ValueError("judge-model must not be empty")
    if not arguments.deepseek_api_key:
        raise ValueError("DEEPSEEK_API_KEY or --deepseek-api-key is required")
    if arguments.output.exists() and not arguments.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {arguments.output}; use --overwrite"
        )

    contexts = [
        document.page_content.strip()
        for document in PyPDFLoader(str(arguments.source_pdf)).load()
        if document.page_content.strip()
    ]
    if not contexts:
        raise ValueError("Source PDF does not contain extractable text")

    judge_model = ChatOpenAI(
        model=arguments.judge_model,
        base_url="https://api.deepseek.com",
        api_key=arguments.deepseek_api_key,
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
        extra_body={"thinking": {"type": "disabled"}},
    )
    response = judge_model.invoke(
        build_generation_prompt(
            contexts=contexts, testset_size=arguments.testset_size
        )
    )
    if not isinstance(response.content, str):
        raise TypeError("DeepSeek returned a non-text candidate response")
    records = build_records(response_content=response.content, contexts=contexts)
    if not records:
        raise ValueError("DeepSeek returned no benchmark candidates")

    write_records(path=arguments.output, records=records)
    arguments.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.metadata_output.write_text(
        json.dumps(source_metadata(source_pdf=arguments.source_pdf), indent=2)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
