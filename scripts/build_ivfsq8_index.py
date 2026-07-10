"""
Build an IVFScalarQuantizer (8-bit) FAISS index from an existing IndexFlatIP.

IVF clusters vectors so only nprobe clusters are searched at query time.
SQ8 compresses each float32 dimension to 8-bit — much less distortion than PQ
because it quantizes per-dimension (not per-subvector group).

Memory: ~4x smaller than flat (1 byte vs 4 bytes per float).
GPU: compatible (unlike HNSW).
Recall: typically 95-99%+ with nprobe=128, vs 52.9% for IVFPQ.

Usage:
    python scripts/build_ivfsq8_index.py \
        --flat-index  path/to/_cached.faiss \
        --output      path/to/_cached_ivfsq8.faiss \
        [--nlist 1024] [--nprobe 128]
"""

import argparse
import time
from pathlib import Path

import faiss
import numpy as np


def build_ivfsq8(flat_path: str, output_path: str, nlist: int, nprobe: int):
    print(f"Loading flat index from {flat_path} ...")
    flat = faiss.read_index(flat_path)
    n, d = flat.ntotal, flat.d
    print(f"  {n:,} vectors  dim={d}")

    print(f"Extracting {n:,} vectors ...")
    t0 = time.time()
    vectors = flat.reconstruct_n(0, n)
    print(f"  Done in {time.time()-t0:.1f}s  ({vectors.nbytes/1e6:.0f} MB)")

    print(f"Building IVFScalarQuantizer SQ8 (nlist={nlist}) ...")
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFScalarQuantizer(
        quantizer, d, nlist,
        faiss.ScalarQuantizer.QT_8bit,
        faiss.METRIC_INNER_PRODUCT,
    )

    print(f"  Training on {n:,} vectors ...")
    t0 = time.time()
    index.train(vectors)
    print(f"  Training done in {time.time()-t0:.1f}s")

    print("  Adding vectors ...")
    t0 = time.time()
    index.add(vectors)
    print(f"  Adding done in {time.time()-t0:.1f}s")

    index.nprobe = nprobe
    print(f"  nprobe set to {nprobe}")

    print(f"Saving to {output_path} ...")
    faiss.write_index(index, output_path)
    size_mb = Path(output_path).stat().st_size / 1e6
    flat_mb = n * d * 4 / 1e6
    print(f"  Saved ({size_mb:.1f} MB vs {flat_mb:.0f} MB flat, {flat_mb/size_mb:.1f}x smaller)")

    print(f"\nRecall test vs flat (nprobe={nprobe}) ...")
    rng = np.random.default_rng(42)
    q = vectors[rng.integers(0, n, 200)].copy().astype("float32")
    faiss.normalize_L2(q)

    _, flat_ids   = flat.search(q, 10)
    _, ivfsq8_ids = index.search(q, 10)

    recalls = [len(set(fi) & set(hi)) / 10 for fi, hi in zip(flat_ids, ivfsq8_ids)]
    print(f"  Recall@10 vs flat (nprobe={nprobe}): {sum(recalls)/len(recalls):.4f}")

    print("\n  nprobe sweep:")
    for np_ in [8, 16, 32, 64, 128, 256, 512]:
        index.nprobe = np_
        _, ids = index.search(q, 10)
        r = sum(len(set(fi) & set(hi)) / 10 for fi, hi in zip(flat_ids, ids)) / len(flat_ids)
        print(f"    nprobe={np_:4d} → recall@10={r:.4f}")

    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat-index", required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--nlist",  type=int, default=1024,
                        help="Number of IVF clusters (default 1024)")
    parser.add_argument("--nprobe", type=int, default=128,
                        help="Clusters to search at query time (default 128)")
    args = parser.parse_args()

    build_ivfsq8(args.flat_index, args.output, args.nlist, args.nprobe)
