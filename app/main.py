from fastapi import FastAPI

from app.config import get_settings

settings = get_settings()

app = FastAPI(title="Career Intelligence Assistant")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm_provider": settings.llm_provider}
