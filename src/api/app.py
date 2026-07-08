import logging

from fastapi import FastAPI

from api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

app = FastAPI(
    title="Edge RAG API",
    description="On-device RAG using Qdrant Edge, Qwen3 embeddings, and Gemma4",
    version="0.1.0",
)

app.include_router(router)


@app.get("/")
def root() -> dict:
    return {"message": "Edge RAG API", "docs": "/docs"}
