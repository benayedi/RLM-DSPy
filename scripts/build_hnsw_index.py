"""
Build an HNSW FAISS index from an existing IndexFlatIP index.

HNSW keeps exact (uncompressed) vectors, so recall is near-perfect (~99%+
at ef_search=64) unlike IVFPQ's lossy 52.9%. Search is ~10-100x faster
than brute-force flat search because it uses a graph-based approximate NN.

Usage:
    python scripts/build_hnsw_index.py \
        --flat-index  path/to/_cached.faiss \
        --output      path/to/_cached_hnsw.faiss \
        [--M 32] [--ef-construction 200] [--ef-search 64]

Memory: HNSW still stores raw vectors (same as flat) + a small graph
  (~4 × M × N bytes extra). At 100K docs × 4096 dims: ~1.64 GB + ~25 MB.
"""

import argparse
import time
from pathlib import Path

import faiss
import numpy as np


def build_hnsw(flat_path: str, output_path: str, M: int, ef_construction: int, ef_search: int):
    print(f"Loading flat index from {flat_path} ...")
    flat = faiss.read_index(flat_path)
    n, d = flat.ntotal, flat.d
    print(f"  {n:,} vectors  dim={d}")

    print(f"Extracting {n:,} vectors ...")
    t0 = time.time()
    vectors = flat.reconstruct_n(0, n)
    print(f"  Done in {time.time()-t0:.1f}s  ({vectors.nbytes/1e6:.0f} MB)")

    print(f"Building HNSW (M={M}, ef_construction={ef_construction}) ...")
    index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction

    t0 = time.time()
    index.add(vectors)
    print(f"  Build done in {time.time()-t0:.1f}s")

    # Set ef_search before saving — embed server can override at query time
    index.hnsw.efSearch = ef_search
    print(f"  ef_search set to {ef_search} (controls recall/speed tradeoff at query time)")

    print(f"Saving to {output_path} ...")
    faiss.write_index(index, output_path)
    size_mb = Path(output_path).stat().st_size / 1e6
    flat_mb = n * d * 4 / 1e6
    print(f"  Saved ({size_mb:.1f} MB vs {flat_mb:.0f} MB flat, +{size_mb-flat_mb:.0f} MB for graph)")

    print(f"\nRecall test (ef_search={ef_search}) ...")
    rng = np.random.default_rng(42)
    q = vectors[rng.integers(0, n, 200)].copy().astype("float32")
    faiss.normalize_L2(q)

    _, flat_ids  = flat.search(q, 10)
    _, hnsw_ids  = index.search(q, 10)

    recalls = [len(set(fi) & set(hi)) / 10 for fi, hi in zip(flat_ids, hnsw_ids)]
    print(f"  Recall@10 vs flat (ef_search={ef_search}): {sum(recalls)/len(recalls):.4f}")

    # Quick sweep over ef_search values
    print("\n  ef_search sweep:")
    for ef in [16, 32, 64, 128, 256]:
        index.hnsw.efSearch = ef
        _, ids = index.search(q, 10)
        r = sum(len(set(fi) & set(hi)) / 10 for fi, hi in zip(flat_ids, ids)) / len(flat_ids)
        print(f"    ef_search={ef:4d} → recall@10={r:.4f}")

    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat-index", required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--M",              type=int, default=32,
                        help="HNSW graph degree — higher=better recall, more memory (default 32)")
    parser.add_argument("--ef-construction",type=int, default=200,
                        help="Build-time graph quality (default 200)")
    parser.add_argument("--ef-search",     type=int, default=64,
                        help="Query-time beam width — higher=better recall, slower (default 64)")
    args = parser.parse_args()

    build_hnsw(args.flat_index, args.output, args.M, args.ef_construction, args.ef_search)
