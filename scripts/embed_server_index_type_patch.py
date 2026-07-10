"""
Patch to apply to embed_server_new.py on raid03 to support --index-type flag.

Replace the index-loading section in embed_server_new.py with the
load_index() function below, and add --index-type to argparse.

Changes needed in embed_server_new.py:
  1. Add --index-type argument (flat / hnsw / ivfpq)
  2. Add optional --ef-search argument (only used for hnsw, default 64)
  3. Replace the faiss.read_index(...) call with load_index()

--- ARGPARSE ADDITIONS ---

    parser.add_argument("--index-type", default="flat",
                        choices=["flat", "hnsw", "ivfpq"],
                        help="FAISS index type to load")
    parser.add_argument("--ef-search", type=int, default=64,
                        help="HNSW ef_search parameter (only used with --index-type hnsw)")

--- REPLACE INDEX LOADING WITH ---
"""

import os
import faiss


def load_index(index_dir: str, index_type: str = "flat", ef_search: int = 64):
    """Load a FAISS index by type from the index directory."""
    filenames = {
        "flat":  "_cached.faiss",
        "hnsw":  "_cached_hnsw.faiss",
        "ivfpq": "_cached_ivfpq.faiss",
    }
    filename = filenames[index_type]
    path = os.path.join(index_dir, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Index file not found: {path}\n"
            f"  flat  → {os.path.join(index_dir, '_cached.faiss')}\n"
            f"  hnsw  → {os.path.join(index_dir, '_cached_hnsw.faiss')}\n"
            f"  ivfpq → {os.path.join(index_dir, '_cached_ivfpq.faiss')}"
        )

    print(f"Loading {index_type.upper()} index from {path} ...")
    index = faiss.read_index(path)

    if index_type == "hnsw":
        index.hnsw.efSearch = ef_search
        print(f"  HNSW ef_search={ef_search}")
    elif index_type == "ivfpq":
        index.nprobe = 128  # reasonable default; flat beats it anyway
        print(f"  IVFPQ nprobe={index.nprobe}  (note: recall ~53%, not recommended)")

    print(f"  Loaded {index.ntotal:,} vectors  dim={index.d}")
    return index


# Example usage in embed_server_new.py startup:
#
#   index = load_index(
#       index_dir  = args.index_dir,   # e.g. /path/to/qwen3-embedding-8b/
#       index_type = args.index_type,  # "flat" | "hnsw" | "ivfpq"
#       ef_search  = args.ef_search,
#   )
