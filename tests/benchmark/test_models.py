from pathlib import Path

import pytest

from benchmarks.benchmark_models import (
    BenchmarkRecord,
    load_records,
    write_records,
)


def test_benchmark_records_round_trip_as_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "approved.jsonl"
    record = BenchmarkRecord(
        question="What is the candidate's primary skill?",
        reference_answer="Python",
        reference_contexts=["Primary skill: Python"],
    )

    write_records(path=path, records=[record])

    assert load_records(path=path) == [record]


def test_benchmark_records_require_non_blank_reference_contexts() -> None:
    with pytest.raises(ValueError, match="blank"):
        BenchmarkRecord(
            question="What is the candidate's primary skill?",
            reference_answer="Python",
            reference_contexts=[" "],
        )
