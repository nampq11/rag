from app.services.vector_store import create_vector_store


def test_vector_store_uses_configured_ollama_embeddings(monkeypatch) -> None:
    captured_arguments: dict[str, object] = {}

    class FakeEmbeddings:
        def __init__(self, **arguments: object) -> None:
            captured_arguments.update(arguments)

    class FakePgVector:
        def __init__(self, **_: object) -> None:
            pass

    monkeypatch.setattr(
        "app.services.vector_store.OllamaEmbeddings", FakeEmbeddings
    )
    monkeypatch.setattr("app.services.vector_store.PGVector", FakePgVector)

    engine, _ = create_vector_store(
        "postgresql+asyncpg://user:password@localhost:5434/database",
        "documents",
        "nomic-embed-text",
        "http://localhost:11434",
    )

    assert captured_arguments == {
        "model": "nomic-embed-text",
        "base_url": "http://localhost:11434",
    }
    engine.sync_engine.dispose()
