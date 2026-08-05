"""Validated local benchmark data models."""

from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class BenchmarkRecord(BaseModel):
    """Defines one approved retrieval evaluation sample."""

    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    reference_contexts: list[str] = Field(min_length=1)

    @field_validator("reference_contexts")
    @classmethod
    def validate_contexts(cls, contexts: list[str]) -> list[str]:
        if any(not context.strip() for context in contexts):
            raise ValueError("Reference contexts must not contain blank values")
        return contexts


def load_records(*, path: Path) -> list[BenchmarkRecord]:
    """Loads approved JSONL benchmark records."""
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark dataset does not exist: {path}")

    records = [
        BenchmarkRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("Benchmark dataset must contain at least one record")
    return records


def write_records(*, path: Path, records: list[BenchmarkRecord]) -> None:
    """Writes benchmark records as newline-delimited JSON."""
    if not records:
        raise ValueError("Benchmark dataset must contain at least one record")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(record.model_dump_json() for record in records) + "\n",
        encoding="utf-8",
    )
