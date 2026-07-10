"""
Build an IVFPQ FAISS index from an existing IndexFlatIP index.

Usage:
    python scripts/build_ivfpq_index.py \
        --flat-index  path/to/_cached.faiss \
        --output      path/to/_cached_ivfpq.faiss \
        [--nlist 1024] [--M 64] [--nbits 8] [--nprobe 64]

The output file can be dropped alongside _cached.faiss on the GPU server
and selected with --index-type ivfpq when starting embed_server_new.py.
"""

import argparse
import time
from pathlib import Path

import faiss
import numpy as np


def build_ivfpq(flat_path: str, output_path: str, nlist: int, M: int, nbits: int):
    print(f"Loading flat index from {flat_path} ...")
    flat = faiss.read_index(flat_path)
    n, d = flat.ntotal, flat.d
    print(f"  {n:,} vectors  dim={d}")

    assert d % M == 0, f"dim={d} must be divisible by M={M}"

    print(f"Extracting {n:,} vectors ...")
    t0 = time.time()
    vectors = flat.reconstruct_n(0, n)
    print(f"  Done in {time.time()-t0:.1f}s  ({vectors.nbytes/1e6:.0f} MB)")

    print(f"Building IVFPQ (nlist={nlist}, M={M}, nbits={nbits}) ...")
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits,
                              faiss.METRIC_INNER_PRODUCT)

    print(f"  Training on {n:,} vectors ...")
    t0 = time.time()
    index.train(vectors)
    print(f"  Training done in {time.time()-t0:.1f}s")

    print("  Adding vectors ...")
    t0 = time.time()
    index.add(vectors)
    print(f"  Adding done in {time.time()-t0:.1f}s")

    print(f"Saving to {output_path} ...")
    faiss.write_index(index, output_path)
    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"  Saved ({size_mb:.1f} MB vs {n*d*4/1e6:.0f} MB flat)")
    print(f"  Compression ratio: {n*d*4 / Path(output_path).stat().st_size:.0f}x")

    print("\nQuick recall test (nprobe=64) ...")
    index.nprobe = 64
    rng = np.random.default_rng(42)
    q = vectors[rng.integers(0, n, 100)]
    faiss.normalize_L2(q)

    _, flat_ids  = flat.search(q, 10)
    _, ivfpq_ids = index.search(q, 10)

    recalls = []
    for fi, ii in zip(flat_ids, ivfpq_ids):
        recalls.append(len(set(fi) & set(ii)) / 10)
    print(f"  Recall@10 vs flat (nprobe={index.nprobe}): {sum(recalls)/len(recalls):.3f}")

    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat-index", required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--nlist",  type=int, default=1024,
                        help="Number of IVF cells (default 1024)")
    parser.add_argument("--M",      type=int, default=64,
                        help="PQ subquantizers — dim must be divisible by M (default 64)")
    parser.add_argument("--nbits",  type=int, default=8,
                        help="Bits per subquantizer (default 8)")
    parser.add_argument("--nprobe", type=int, default=64,
                        help="Cells to probe at search time (default 64, test only)")
    args = parser.parse_args()

    build_ivfpq(args.flat_index, args.output, args.nlist, args.M, args.nbits)
