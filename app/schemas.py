from pydantic import BaseModel, Field
from typing import List, Optional

class SearchRequest(BaseModel):
    query: str = Field(..., example="What is dense vector retrieval?")
    top_k_candidates: Optional[int] = Field(default=10, description="FAISS candidate pool")
    top_k_final: Optional[int] = Field(default=5, description="Final results after re-ranking")
    rerank: Optional[bool] = Field(default=True, description="Enable cross-encoder stage")

class DocumentHit(BaseModel):
    id: str
    text: str
    score: float

class SearchResponse(BaseModel):
    query: str
    stage: str
    latency_ms: float
    results: List[DocumentHit]
