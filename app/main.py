from fastapi import FastAPI, HTTPException
from app.schemas import SearchRequest, SearchResponse, DocumentHit
from app.service import NeuralSearchService

app = FastAPI(
    title="Dense Semantic Search & Neural Retrieval Microservice",
    version="1.0.0",
    description="Two-stage dense retrieval pipeline utilizing FAISS vector indexing and Cross-Encoder re-ranking."
)

service: NeuralSearchService = None

@app.on_event("startup")
def startup_event():
    global service
    service = NeuralSearchService(data_path="data/sample_corpus.json")

@app.get("/health")
def health():
    return {"status": "healthy", "service": "neural-retrieval-api"}

@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    hits, stage, latency_ms = service.query(
        text=request.query,
        top_k_candidates=request.top_k_candidates,
        top_k_final=request.top_k_final,
        rerank=request.rerank
    )
    
    return SearchResponse(
        query=request.query,
        stage=stage,
        latency_ms=round(latency_ms, 2),
        results=[DocumentHit(**h) for h in hits]
    )
