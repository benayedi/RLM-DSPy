"""
GPU embedding + FAISS search server with cross-encoder reranking.

Drop-in replacement for embed_server_new.py. The /search endpoint fetches
--first-stage-k candidates from FAISS, reranks them with a cross-encoder,
and returns the top_k results. All other endpoints (/encode, /document,
/health) are identical.

Usage:
  python embed_server_reranking.py \
      --faiss-index ~/faiss_cache/_cached.faiss \
      --faiss-ids   ~/faiss_cache/_cached_ids.pkl \
      --corpus-dataset Tevatron/browsecomp-plus-corpus \
      --reranker-model BAAI/bge-reranker-v2-m3 \
      --first-stage-k 50
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
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--model-path",          default=os.path.expanduser("~/models/qwen3-embed-8b"))
parser.add_argument("--faiss-index",         default=os.path.expanduser("~/faiss_cache/_cached.faiss"))
parser.add_argument("--faiss-ids",           default=os.path.expanduser("~/faiss_cache/_cached_ids.pkl"))
parser.add_argument("--nprobe",              type=int, default=128)
parser.add_argument("--corpus-dataset",      default="Tevatron/browsecomp-plus-corpus")
parser.add_argument("--port",                type=int, default=8001)
parser.add_argument("--max-length",          type=int, default=512)
parser.add_argument("--reranker-model",      default="BAAI/bge-reranker-v2-m3")
parser.add_argument("--reranker-max-length", type=int, default=512)
parser.add_argument("--first-stage-k",       type=int, default=50)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# Load embedding model  (Qwen3-Embed-8B)
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading embedding model from {args.model_path} ...")
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(
    args.model_path, dtype=torch.float16, trust_remote_code=True
).to(DEVICE).eval()
print(f"Embedding model ready. Vector dim: {model.config.hidden_size}")

# ─────────────────────────────────────────────────────────────────────────────
# Load cross-encoder reranker
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading reranker from {args.reranker_model} ...")
reranker_tokenizer = AutoTokenizer.from_pretrained(args.reranker_model)
reranker_model = AutoModelForSequenceClassification.from_pretrained(
    args.reranker_model, dtype=torch.float16
).to(DEVICE).eval()
print("Reranker ready.")

# ─────────────────────────────────────────────────────────────────────────────
# Load FAISS index
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading FAISS index from {args.faiss_index} ...")
_cpu_index = faiss.read_index(args.faiss_index)
if hasattr(_cpu_index, "nprobe"):
    _cpu_index.nprobe = args.nprobe

num_gpus = faiss.get_num_gpus()
if num_gpus > 0:
    print(f"Moving FAISS index to {num_gpus} GPU(s) ...")
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
    if hasattr(faiss_index, "nprobe"):
        faiss_index.nprobe = args.nprobe
    print("FAISS on GPU.")
else:
    faiss_index = _cpu_index
    print("No GPU found for FAISS — using CPU.")

with open(args.faiss_ids, "rb") as f:
    faiss_lookup: list[str] = pickle.load(f)

print(f"FAISS ready: {faiss_index.ntotal:,} vectors, dim={faiss_index.d}")

# ─────────────────────────────────────────────────────────────────────────────
# Load corpus  (docid → text)
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading corpus from {args.corpus_dataset} ...")
_ds = load_dataset(args.corpus_dataset, split="train")
corpus: dict[str, str] = {row["docid"]: row["text"] for row in _ds}
print(f"Corpus ready: {len(corpus):,} documents")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
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


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Score (query, doc_text) pairs with the cross-encoder, return top_k sorted by score."""
    pairs = [[query, c["text"]] for c in candidates]
    with torch.no_grad():
        enc = reranker_tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=args.reranker_max_length,
            return_tensors="pt",
        ).to(DEVICE)
        scores = reranker_model(**enc, return_dict=True).logits.squeeze(-1).float().cpu()
    order = scores.argsort(descending=True).tolist()
    results = []
    for i in order[:top_k]:
        doc = dict(candidates[i])
        doc["rerank_score"] = round(float(scores[i]), 4)
        results.append(doc)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="BrowseComp+ GPU Search Server (with reranking)")


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
    # first stage: fetch more candidates than needed
    first_k = max(req.top_k * 5, args.first_stage_k)
    q_vec = embed_texts([QUERY_INSTRUCTION + req.query])
    scores, indices = faiss_index.search(q_vec, first_k)

    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        doc_id = faiss_lookup[idx]
        text = corpus.get(doc_id, "")
        candidates.append({
            "score": round(float(score), 4),
            "doc_id": doc_id,
            "text": text,          # full text for reranker
        })

    # second stage: cross-encoder reranking
    results = rerank(req.query, candidates, req.top_k)

    # truncate text in the response (full text was used for reranking above)
    for r in results:
        r["text"] = r["text"][:2000]

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
