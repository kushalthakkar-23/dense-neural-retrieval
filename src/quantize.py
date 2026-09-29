import os
import time
import torch
from sentence_transformers import SentenceTransformer

def quantize_bi_encoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", save_dir: str = "artifacts/quantized_model"):
    os.makedirs(save_dir, exist_ok=True)
    model = SentenceTransformer(model_name, device="cpu")
    
    quantized_first_module = torch.ao.quantization.quantize_dynamic(
        model[0].auto_model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    model[0].auto_model = quantized_first_module
    
    torch.save(model, os.path.join(save_dir, "model_int8.pt"))
    print(f"Quantized model saved to {save_dir}/model_int8.pt")
    return model

def benchmark_inference(sample_texts, runs: int = 100):
    baseline_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    quantized_model = quantize_bi_encoder()
    
    _ = baseline_model.encode(sample_texts[:2])
    _ = quantized_model.encode(sample_texts[:2])
    
    start = time.perf_counter()
    for _ in range(runs):
        _ = baseline_model.encode(sample_texts, batch_size=8)
    fp32_time = (time.perf_counter() - start) / runs * 1000
    
    start = time.perf_counter()
    for _ in range(runs):
        _ = quantized_model.encode(sample_texts, batch_size=8)
    int8_time = (time.perf_counter() - start) / runs * 1000
    
    print(f"Latency FP32: {fp32_time:.2f} ms | Latency INT8: {int8_time:.2f} ms")

if __name__ == "__main__":
    texts = [
        "Dense retrieval indexes vectors efficiently.",
        "Cross encoders evaluate deep interaction across queries.",
        "INT8 quantization reduces memory footprint and CPU latency."
    ]
    benchmark_inference(texts)
