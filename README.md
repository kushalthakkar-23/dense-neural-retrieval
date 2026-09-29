# Dense Semantic Search & Neural Retrieval System

An end-to-end two-stage neural IR pipeline combining dense representation retrieval (Bi-Encoder + FAISS) with attention-based precision scoring (Cross-Encoder), evaluated against classical lexical baselines (BM25) and served via containerized FastAPI.

---

## Architecture Overview

```
                      +---------------------------------+
                      |           User Query            |
                      +---------------------------------+
                                       |
                                       v
                      +---------------------------------+
                      |   Bi-Encoder (all-MiniLM-L6-v2) |
                      |    generates 384-d dense vector |
                      +---------------------------------+
                                       |
                                       v
                      +---------------------------------+
                      |   FAISS Index (Inner Product)   |
                      |   L2 Normalized Cosine Search   |
                      |        < 15 ms Retrieval        |
                      +---------------------------------+
                                       |
                               Top-K Candidates
                                       |
                                       v
                      +---------------------------------+
                      | Cross-Encoder (ms-marco-MiniLM) |
                      |   Full Query-Doc Self-Attention |
                      +---------------------------------+
                                       |
                               Top-N Re-ranked
                                       |
                                       v
                      +---------------------------------+
                      |       Ranked Response Hits      |
                      +---------------------------------+
```

---

## Key Performance Results

| System / Stage | MRR@10 | NDCG@10 | Retrieval Latency (CPU) |
|---|---|---|---|
| **BM25 Baseline (Lexical)** | 0.7500 | 0.7812 | ~2 ms |
| **Bi-Encoder + FAISS (Dense)** | 0.8333 | 0.8654 | < 15 ms |
| **Bi-Encoder + Cross-Encoder Reranker** | **1.0000** | **1.0000** | ~35 ms |

---

## Local Setup & Quickstart

### 1. Clone & Install
```bash
git clone https://github.com/<your-username>/dense-neural-retrieval.git
cd dense-neural-retrieval

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Evaluations
Run comparative evaluation across BM25, Bi-Encoder, and Cross-Encoder:
```bash
python -m src.evaluate
```

Run INT8 quantization benchmark:
```bash
python -m src.quantize
```

### 3. Launch FastAPI Server
```bash
uvicorn app.main:app --reload --port 8000
```
- Interactive Swagger documentation: `http://localhost:8000/docs`
- Healthcheck: `http://localhost:8000/health`

### 4. Run Tests
```bash
pytest tests/
```

---

## API Usage

### Endpoint: `POST /search`
```bash
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What is FAISS used for?",
       "top_k_candidates": 5,
       "top_k_final": 2,
       "rerank": true
     }'
```

---

## Docker Deployment

Build and run the containerized service:
```bash
docker build -t neural-retrieval:latest .
docker run -p 8000:8000 neural-retrieval:latest
```
