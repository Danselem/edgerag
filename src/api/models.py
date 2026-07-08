from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question to ask the RAG")
    shard_type: str = Field(
        default="dense",
        pattern="^(dense|hybrid|quantized)$",
        description="Index type to query against",
    )
    limit: int = Field(default=2, ge=1, le=10, description="Number of context chunks to retrieve")


class QueryResponse(BaseModel):
    answer: str
    context: list[str]
    metadata: dict


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"


class ConfigResponse(BaseModel):
    embedding_model: str
    llm_model: str
    document: str
    shard_types: list[str]
