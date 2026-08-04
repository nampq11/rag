from fastapi import FastAPI

app = FastAPI(title="RAG API")


@app.get("/")
async def health_check() -> dict[str, str]:
    return {"message": "RAG API is running"}
