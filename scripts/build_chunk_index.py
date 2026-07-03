"""
Build a chunked FAISS index from the BrowseComp+ corpus.

Run this on the GPU server (raid03) where Qwen3-Embed-8B is available.

Usage:
    python build_chunk_index.py \
        --out-dir ~/faiss_cache_chunked \
        --chunk-size 1500 \
        --overlap 200 \
        --batch-size 128 \
        --model Qwen/Qwen3-Embed-8B

Outputs (in --out-dir):
    chunks.faiss       FAISS IndexFlatIP, one vector per chunk
    chunk_ids.pkl      list[str] — chunk IDs in FAISS index order
                       format: "{doc_id}__chunk_{i}"
    chunk_to_doc.pkl   dict[str, str] — chunk_id → doc_id
    chunk_texts.pkl    dict[str, str] — chunk_id → chunk text
    doc_texts.pkl      dict[str, str] — doc_id → full text (for get_document)
"""

import argparse
import os
import pickle
from pathlib import Path

import faiss
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping character-level chunks."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ── Embedding ─────────────────────────────────────────────────────────────────

INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query\nQuery: "
)


def embed_batch(texts: list[str], tokenizer, model, device, max_length: int) -> np.ndarray:
    """Embed a batch of texts, return L2-normalised float32 numpy array."""
    # For document chunks we do NOT prepend the query instruction
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output = model(**encoded)

    # Last non-padding token hidden state (Qwen3-Embed style)
    attention_mask = encoded["attention_mask"]
    last_token_idx = attention_mask.sum(dim=1) - 1
    hidden = output.last_hidden_state
    vecs = hidden[torch.arange(hidden.size(0)), last_token_idx].float().cpu().numpy()

    # L2 normalise
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-9)
    return vecs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="~/faiss_cache_chunked")
    parser.add_argument("--chunk-size", type=int, default=1500,
                        help="Characters per chunk (~375 tokens)")
    parser.add_argument("--overlap", type=int, default=200,
                        help="Character overlap between consecutive chunks")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Embedding batch size (reduce if OOM)")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max tokens per chunk for the embedding model")
    parser.add_argument("--model", default="Qwen/Qwen3-Embed-8B")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load corpus ───────────────────────────────────────────────────────────
    print("Loading BrowseComp+ corpus...")
    corpus = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")
    print(f"  {len(corpus):,} documents loaded")

    # ── Chunk ─────────────────────────────────────────────────────────────────
    print(f"\nChunking (size={args.chunk_size}, overlap={args.overlap})...")
    chunk_ids: list[str] = []
    chunk_to_doc: dict[str, str] = {}
    chunk_texts: dict[str, str] = {}
    doc_texts: dict[str, str] = {}

    for row in tqdm(corpus, desc="Chunking"):
        doc_id = row["docid"]
        text = row["text"]
        doc_texts[doc_id] = text

        chunks = chunk_text(text, args.chunk_size, args.overlap)
        for i, chunk in enumerate(chunks):
            cid = f"{doc_id}__chunk_{i}"
            chunk_ids.append(cid)
            chunk_to_doc[cid] = doc_id
            chunk_texts[cid] = chunk

    print(f"  {len(chunk_ids):,} chunks from {len(doc_texts):,} documents")
    print(f"  avg chunks/doc: {len(chunk_ids)/len(doc_texts):.1f}")

    # ── Load embedding model ──────────────────────────────────────────────────
    print(f"\nLoading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16)
    model = model.to(args.device).eval()
    print(f"  Model loaded on {args.device}")

    # ── Embed all chunks ──────────────────────────────────────────────────────
    dim = model.config.hidden_size
    print(f"\nEmbedding {len(chunk_ids):,} chunks (dim={dim})...")

    all_texts = [chunk_texts[cid] for cid in chunk_ids]
    all_vecs = []

    for start in tqdm(range(0, len(all_texts), args.batch_size), desc="Embedding"):
        batch = all_texts[start : start + args.batch_size]
        vecs = embed_batch(batch, tokenizer, model, args.device, args.max_length)
        all_vecs.append(vecs)

    all_vecs = np.vstack(all_vecs).astype("float32")
    print(f"  Embedding matrix: {all_vecs.shape}")

    # ── Build FAISS index ─────────────────────────────────────────────────────
    print("\nBuilding FAISS IndexFlatIP...")
    index = faiss.IndexFlatIP(dim)
    index.add(all_vecs)
    print(f"  Index has {index.ntotal:,} vectors")

    # Keep on CPU — chunk index is ~34GB, exceeds GPU VRAM after model is loaded.
    # Embedding still runs on GPU batch-by-batch; only the final index lives in RAM.
    print("  Keeping index in CPU RAM (GPU VRAM reserved for embedding model)")

    # ── Save outputs ──────────────────────────────────────────────────────────
    print(f"\nSaving to {out_dir}...")

    faiss.write_index(index, str(out_dir / "chunks.faiss"))
    print(f"  chunks.faiss saved  ({index.ntotal:,} vectors)")

    with open(out_dir / "chunk_ids.pkl", "wb") as f:
        pickle.dump(chunk_ids, f)
    print(f"  chunk_ids.pkl saved ({len(chunk_ids):,} entries)")

    with open(out_dir / "chunk_to_doc.pkl", "wb") as f:
        pickle.dump(chunk_to_doc, f)
    print(f"  chunk_to_doc.pkl saved")

    with open(out_dir / "chunk_texts.pkl", "wb") as f:
        pickle.dump(chunk_texts, f)
    print(f"  chunk_texts.pkl saved")

    with open(out_dir / "doc_texts.pkl", "wb") as f:
        pickle.dump(doc_texts, f)
    print(f"  doc_texts.pkl saved")

    print("\nDone. Update embed_server_new.py to use:")
    print(f"  --faiss-index {out_dir}/chunks.faiss")
    print(f"  --faiss-ids   {out_dir}/chunk_ids.pkl")
    print(f"  --chunk-to-doc {out_dir}/chunk_to_doc.pkl")
    print(f"  --chunk-texts  {out_dir}/chunk_texts.pkl")
    print(f"  --doc-texts    {out_dir}/doc_texts.pkl")


if __name__ == "__main__":
    main()
