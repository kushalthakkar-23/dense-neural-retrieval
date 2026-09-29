import pytest
from app.service import NeuralSearchService

@pytest.fixture(scope="module")
def search_service():
    return NeuralSearchService(data_path="data/sample_corpus.json")

def test_faiss_retrieval_returns_hits(search_service):
    query = "FAISS similarity indexing"
    hits, stage, latency = search_service.query(query, top_k_candidates=3, rerank=False)
    
    assert len(hits) > 0
    assert "doc_2" in [h["id"] for h in hits]
    assert latency > 0

def test_cross_encoder_reranking(search_service):
    query = "Quantization to INT8"
    hits, stage, latency = search_service.query(query, top_k_candidates=5, top_k_final=1, rerank=True)
    
    assert len(hits) == 1
    assert hits[0]["id"] == "doc_5"
    assert stage == "BiEncoder_FAISS_Plus_CrossEncoder_Rerank"
