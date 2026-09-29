import torch
from sentence_transformers import CrossEncoder
from typing import List, Dict, Tuple

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = CrossEncoder(model_name, device=self.device)

    def rerank(self, query: str, candidate_docs: List[Dict[str, str]], top_k: int = 5) -> List[Tuple[Dict[str, str], float]]:
        if not candidate_docs:
            return []
            
        pairs = [[query, doc["text"]] for doc in candidate_docs]
        scores = self.model.predict(pairs)
        
        scored_docs = list(zip(candidate_docs, [float(s) for s in scores]))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs[:top_k]
