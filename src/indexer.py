import os
import json
import faiss
import numpy as np
from typing import List, Dict, Tuple

class FAISSIndexManager:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(self.dimension)
        self.doc_store: List[Dict[str, str]] = []

    def build_index(self, embeddings: np.ndarray, documents: List[Dict[str, str]]):
        if embeddings.shape[0] != len(documents):
            raise ValueError("Mismatched counts between embeddings and documents.")
        
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.doc_store = documents

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[Dict[str, str], float]]:
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)
            
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1 and idx < len(self.doc_store):
                results.append((self.doc_store[idx], float(score)))
        return results

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "corpus.json"), "w", encoding="utf-8") as f:
            json.dump(self.doc_store, f, indent=2)

    def load(self, directory: str):
        index_path = os.path.join(directory, "index.faiss")
        corpus_path = os.path.join(directory, "corpus.json")
        
        if not os.path.exists(index_path) or not os.path.exists(corpus_path):
            raise FileNotFoundError("Index or corpus metadata file not found.")
            
        self.index = faiss.read_index(index_path)
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.doc_store = json.load(f)
