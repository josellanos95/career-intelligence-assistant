from fastapi import FastAPI

from app.api.routes import chat, documents
from app.config import get_settings

settings = get_settings()

app = FastAPI(title="Career Intelligence Assistant")
app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm_provider": settings.llm_provider}
