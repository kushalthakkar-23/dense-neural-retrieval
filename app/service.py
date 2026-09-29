import os
import json
import time
from typing import Tuple, List, Dict
from src.bi_encoder import BiEncoderEmbedder
from src.indexer import FAISSIndexManager
from src.cross_encoder import CrossEncoderReranker

class NeuralSearchService:
    def __init__(self, data_path: str = "data/sample_corpus.json"):
        self.bi_encoder = BiEncoderEmbedder()
        self.indexer = FAISSIndexManager(dimension=self.bi_encoder.dimension)
        self.reranker = CrossEncoderReranker()
        
        with open(data_path, "r", encoding="utf-8") as f:
            documents = json.load(f)
            
        embeddings = self.bi_encoder.encode([d["text"] for d in documents])
        self.indexer.build_index(embeddings, documents)

    def query(self, text: str, top_k_candidates: int = 10, top_k_final: int = 5, rerank: bool = True) -> Tuple[List[Dict], str, float]:
        start = time.perf_counter()
        query_vec = self.bi_encoder.encode(text)
        candidates = self.indexer.search(query_vec, top_k=top_k_candidates)
        
        stage = "FAISS_Dense_Retrieval"
        if rerank:
            candidate_docs = [hit[0] for hit in candidates]
            reranked = self.reranker.rerank(text, candidate_docs, top_k=top_k_final)
            final_hits = [{"id": hit[0]["id"], "text": hit[0]["text"], "score": round(hit[1], 4)} for hit in reranked]
            stage = "BiEncoder_FAISS_Plus_CrossEncoder_Rerank"
        else:
            final_hits = [{"id": hit[0]["id"], "text": hit[0]["text"], "score": round(hit[1], 4)} for hit in candidates[:top_k_final]]
            
        latency = (time.perf_counter() - start) * 1000
        return final_hits, stage, latency
