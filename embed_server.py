"""
GPU embedding + FAISS search server for BrowseComp+.

Endpoints:
  POST /encode   {"inputs": ["text1", ...]}
                 -> {"embeddings": [[float, ...], ...]}

  POST /search   {"query": "...", "top_k": 10}
                 -> {"results": [{"score": float, "doc_id": str, "text": str}, ...]}

  GET  /health   -> {"status": "ok"}

Usage on GPU server:
  python embed_server.py \
      --faiss-index ~/faiss_cache/_cached.faiss \
      --faiss-ids   ~/faiss_cache/_cached_ids.pkl \
      --corpus-dataset Tevatron/browsecomp-plus-corpus
"""

import argparse
import os
import pickle

import faiss
import numpy as np
import torch
import uvicorn
from datasets import load_dataset
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", default=os.path.expanduser("~/models/qwen3-embed-8b"))
parser.add_argument("--faiss-index", default=os.path.expanduser("~/faiss_cache/_cached.faiss"))
parser.add_argument("--faiss-ids",   default=os.path.expanduser("~/faiss_cache/_cached_ids.pkl"))
parser.add_argument("--corpus-dataset", default="Tevatron/browsecomp-plus-corpus")
parser.add_argument("--port", type=int, default=8001)
parser.add_argument("--max-length", type=int, default=512)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# Load embedding model
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading embedding model from {args.model_path}...")
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(
    args.model_path, torch_dtype=torch.float16, trust_remote_code=True
).to(DEVICE).eval()
print(f"Model ready. Vector dim: {model.config.hidden_size}")

# ─────────────────────────────────────────────────────────────────────────────
# Load FAISS index
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading FAISS index from {args.faiss_index}...")
_cpu_index = faiss.read_index(args.faiss_index)

# Move to GPU if available
num_gpus = faiss.get_num_gpus()
if num_gpus > 0:
    print(f"Moving FAISS index to {num_gpus} GPU(s)...")
    if num_gpus == 1:
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        res = faiss.StandardGpuResources()
        faiss_index = faiss.index_cpu_to_gpu(res, 0, _cpu_index, co)
    else:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        faiss_index = faiss.index_cpu_to_all_gpus(_cpu_index, co, ngpu=num_gpus)
    print("FAISS on GPU.")
else:
    faiss_index = _cpu_index
    print("No GPU found for FAISS — using CPU.")

with open(args.faiss_ids, "rb") as f:
    faiss_lookup: list[str] = pickle.load(f)

print(f"FAISS ready: {faiss_index.ntotal:,} vectors, dim={faiss_index.d}")

# ─────────────────────────────────────────────────────────────────────────────
# Load corpus (docid → text)
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading corpus from {args.corpus_dataset}...")
_ds = load_dataset(args.corpus_dataset, split="train")
corpus: dict[str, str] = {row["docid"]: row["text"] for row in _ds}
print(f"Corpus ready: {len(corpus):,} documents")

# ─────────────────────────────────────────────────────────────────────────────
# Embedding helper
# ─────────────────────────────────────────────────────────────────────────────

QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query\nQuery: "
)


def embed_texts(texts: list[str]) -> np.ndarray:
    with torch.no_grad():
        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(DEVICE)
        out = model(**batch)
        seq_len = batch["attention_mask"].sum(dim=1) - 1
        vecs = out.last_hidden_state[range(len(texts)), seq_len]
        vecs = torch.nn.functional.normalize(vecs, dim=-1)
    return vecs.cpu().float().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="BrowseComp+ GPU Search Server")


class EncodeRequest(BaseModel):
    inputs: list[str]


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


class DocumentRequest(BaseModel):
    doc_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/encode")
def encode(req: EncodeRequest):
    vecs = embed_texts(req.inputs)
    return {"embeddings": vecs.tolist()}


@app.post("/search")
def search(req: SearchRequest):
    q_vec = embed_texts([QUERY_INSTRUCTION + req.query])
    scores, indices = faiss_index.search(q_vec, req.top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        doc_id = faiss_lookup[idx]
        text = corpus.get(doc_id, "")
        results.append({
            "score": round(float(score), 4),
            "doc_id": doc_id,
            "text": text[:2000],
        })
    return {"results": results}


@app.post("/document")
def get_document(req: DocumentRequest):
    text = corpus.get(req.doc_id)
    if text is None:
        return {"error": f"doc_id '{req.doc_id}' not found", "doc_id": req.doc_id, "text": ""}
    return {"doc_id": req.doc_id, "text": text}


if __name__ == "__main__":
    print(f"Starting server on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
