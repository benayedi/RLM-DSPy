"""FAISS retrieval tools for RAG-enabled RLM using the BrowseComp+ corpus."""

from __future__ import annotations

import glob
import os
import pickle
from typing import Any, Callable

import faiss
import numpy as np

QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query\nQuery: "
)


def make_embedding_fn(server_url: str) -> Callable[[str], list[float]]:
    """Create an embedding function backed by the Qwen3-Embed-8B GPU server."""
    import requests

    def embed(text: str) -> list[float]:
        r = requests.post(
            f"{server_url.rstrip('/')}/encode",
            json={"inputs": [text]},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["embeddings"][0]

    return embed


class FaissRetriever:
    """
    FAISS retriever for the BrowseComp+ corpus (Qwen3-Embed-8B, dim=4096).

    Loads pre-built pickle shards, builds an inner-product (cosine) index,
    and lazily loads document text from the HuggingFace corpus dataset.

    On first load, saves a native FAISS cache (_cached.faiss + _cached_ids.pkl)
    alongside the shards so subsequent runs load in seconds.
    """

    def __init__(
        self,
        index_pattern: str,
        embedding_fn: Callable[[str], list[float]],
        k: int = 10,
        corpus_dataset: str = "Tevatron/browsecomp-plus-corpus",
    ) -> None:
        """
        Args:
            index_pattern:  Glob for pickle shards, e.g. "/path/to/*.pkl"
            embedding_fn:   callable(text) -> float list (GPU server)
            k:              Default top-k results per query
            corpus_dataset: HuggingFace dataset name for full document text
        """
        self.embedding_fn = embedding_fn
        self.k = k
        self._docid_to_text: dict[str, str] | None = None
        self._corpus_dataset = corpus_dataset

        cache_index = index_pattern.replace("*.pkl", "_cached.faiss")
        cache_ids = index_pattern.replace("*.pkl", "_cached_ids.pkl")

        # Exclude the cache file itself from the shard list
        shard_files = sorted(
            f for f in glob.glob(index_pattern) if not f.endswith("_cached_ids.pkl")
        )
        if not shard_files:
            raise FileNotFoundError(f"No shard files matching: {index_pattern}")

        if (
            os.path.exists(cache_index)
            and os.path.exists(cache_ids)
            and os.path.getmtime(cache_index) > max(os.path.getmtime(f) for f in shard_files)
        ):
            print("Loading FAISS index from cache…")
            self._index = faiss.read_index(cache_index)
            with open(cache_ids, "rb") as f:
                self._lookup = pickle.load(f)
            print(f"FAISS ready: {self._index.ntotal:,} vectors, dim={self._index.d}")
        else:
            print(f"Loading {len(shard_files)} FAISS shard(s)…")
            all_vecs, all_ids = [], []
            for path in shard_files:
                with open(path, "rb") as f:
                    reps, lookup = pickle.load(f)
                all_vecs.append(np.array(reps, dtype=np.float32))
                all_ids.extend(lookup)
                print(f"  {path} ({len(lookup):,} docs)")

            matrix = np.concatenate(all_vecs, axis=0)
            self._lookup = all_ids
            self._index = faiss.IndexFlatIP(matrix.shape[1])
            self._index.add(matrix)
            print(f"FAISS ready: {self._index.ntotal:,} vectors, dim={matrix.shape[1]}")

            print("Saving FAISS cache…")
            faiss.write_index(self._index, cache_index)
            with open(cache_ids, "wb") as f:
                pickle.dump(self._lookup, f)
            print("Cache saved.")

    def _corpus(self) -> dict[str, str]:
        if self._docid_to_text is None:
            from datasets import load_dataset

            print("Loading BrowseComp+ corpus from HuggingFace…")
            ds = load_dataset(self._corpus_dataset, split="train")
            self._docid_to_text = {row["docid"]: row["text"] for row in ds}
            print(f"Corpus loaded: {len(self._docid_to_text):,} docs")
        return self._docid_to_text

    def search_index(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """
        Search the FAISS index for passages relevant to the query.

        Args:
            query:  Natural-language search query.
            top_k:  Number of results (overrides default k if given).

        Returns:
            List of dicts: {score: float, doc_id: str, text: str (≤2000 chars)}.
        """
        vec = self.embedding_fn(QUERY_INSTRUCTION + query)
        q = np.array([vec], dtype=np.float32)
        scores, indices = self._index.search(q, top_k if top_k is not None else self.k)
        corpus = self._corpus()

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc_id = self._lookup[idx]
            text = corpus.get(doc_id, "")
            results.append(
                {"score": round(float(score), 4), "doc_id": doc_id, "text": text[:2000]}
            )
        return results

    def get_document(self, doc_id: str) -> dict[str, Any]:
        """
        Fetch the full text of a document by its doc_id.

        Args:
            doc_id: The doc_id string from a search_index result.

        Returns:
            {doc_id: str, text: str} or {error: str, doc_id: str, text: ""}.
        """
        corpus = self._corpus()
        text = corpus.get(doc_id)
        if text is None:
            return {"error": f"doc_id '{doc_id}' not found", "doc_id": doc_id, "text": ""}
        return {"doc_id": doc_id, "text": text}
