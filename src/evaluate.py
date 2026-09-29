import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict
from src.bi_encoder import BiEncoderEmbedder
from src.indexer import FAISSIndexManager
from src.cross_encoder import CrossEncoderReranker

def compute_mrr_at_k(relevant_doc_id: str, retrieved_doc_ids: List[str], k: int = 10) -> float:
    retrieved_k = retrieved_doc_ids[:k]
    if relevant_doc_id in retrieved_k:
        rank = retrieved_k.index(relevant_doc_id) + 1
        return 1.0 / rank
    return 0.0

def compute_ndcg_at_k(relevant_doc_id: str, retrieved_doc_ids: List[str], k: int = 10) -> float:
    retrieved_k = retrieved_doc_ids[:k]
    if relevant_doc_id in retrieved_k:
        rank = retrieved_k.index(relevant_doc_id) + 1
        return 1.0 / np.log2(rank + 1)
    return 0.0

def evaluate_systems():
    corpus = [
        {"id": "doc_1", "text": "Dense passage retrieval uses continuous representations to index documents for open-domain question answering."},
        {"id": "doc_2", "text": "FAISS is a library for efficient similarity search and clustering of dense vectors, developed by Meta AI."},
        {"id": "doc_3", "text": "Cross-encoders perform full self-attention over query-passage pairs for ranking."},
        {"id": "doc_4", "text": "BM25 is a term-frequency inverted index baseline for lexical retrieval."},
        {"id": "doc_5", "text": "Quantization converts 32-bit floating point representations to INT8 for low-latency inference."}
    ]
    
    test_queries = [
        {"query": "vector similarity search library Meta", "gold_id": "doc_2"},
        {"query": "full self-attention reranking pair", "gold_id": "doc_3"},
        {"query": "probabilistic term matching baseline", "gold_id": "doc_4"},
        {"query": "low latency integer precision acceleration", "gold_id": "doc_5"}
    ]
    
    tokenized_corpus = [doc["text"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    bi_enc = BiEncoderEmbedder()
    indexer = FAISSIndexManager(dimension=bi_enc.dimension)
    corpus_embeddings = bi_enc.encode([d["text"] for d in corpus])
    indexer.build_index(corpus_embeddings, corpus)
    reranker = CrossEncoderReranker()
    
    results = {
        "BM25": {"mrr": [], "ndcg": []},
        "Bi-Encoder (FAISS)": {"mrr": [], "ndcg": []},
        "Bi-Encoder + Cross-Encoder": {"mrr": [], "ndcg": []}
    }
    
    for item in test_queries:
        q = item["query"]
        gold = item["gold_id"]
        
        bm25_scores = bm25.get_scores(q.lower().split())
        bm25_ranked_idx = np.argsort(bm25_scores)[::-1]
        bm25_ranked_ids = [corpus[i]["id"] for i in bm25_ranked_idx]
        results["BM25"]["mrr"].append(compute_mrr_at_k(gold, bm25_ranked_ids, 10))
        results["BM25"]["ndcg"].append(compute_ndcg_at_k(gold, bm25_ranked_ids, 10))
        
        q_emb = bi_enc.encode(q)
        faiss_hits = indexer.search(q_emb, top_k=10)
        faiss_ranked_ids = [hit[0]["id"] for hit in faiss_hits]
        results["Bi-Encoder (FAISS)"]["mrr"].append(compute_mrr_at_k(gold, faiss_ranked_ids, 10))
        results["Bi-Encoder (FAISS)"]["ndcg"].append(compute_ndcg_at_k(gold, faiss_ranked_ids, 10))
        
        candidate_docs = [hit[0] for hit in faiss_hits]
        reranked_hits = reranker.rerank(q, candidate_docs, top_k=10)
        reranked_ids = [hit[0]["id"] for hit in reranked_hits]
        results["Bi-Encoder + Cross-Encoder"]["mrr"].append(compute_mrr_at_k(gold, reranked_ids, 10))
        results["Bi-Encoder + Cross-Encoder"]["ndcg"].append(compute_ndcg_at_k(gold, reranked_ids, 10))
        
    print("\n--- Comparative IR Evaluation Results (Top-10) ---")
    for method, metrics in results.items():
        print(f"Method: {method:<28} | MRR@10: {np.mean(metrics['mrr']):.4f} | NDCG@10: {np.mean(metrics['ndcg']):.4f}")

if __name__ == "__main__":
    evaluate_systems()
