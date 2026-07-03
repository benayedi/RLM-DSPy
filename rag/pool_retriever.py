"""
PoolRetriever: searches within a pre-defined 1K-doc pool using local vectors.

Used for the BrowseComp+(1K) evaluation setup (replicating the RLM paper).
Drop-in replacement for RemoteRetriever — same interface (search_index,
get_document), so the agent code in agent.py is unchanged.

Search flow:
  1. Call GPU /encode endpoint with the query (Qwen3-Embed-8B)
  2. Local cosine similarity: pool_vectors @ query_vec  (~1ms on CPU)
  3. Return top-k results with text snippets from local corpus cache
"""

from __future__ import annotations

import numpy as np
import requests


_QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query\nQuery: "
)


class PoolRetriever:
    """Retrieves from a fixed 1K-doc pool using local cosine similarity.

    Args:
        pool_doc_ids:    List of doc IDs in the pool (order matches pool_vectors rows).
        pool_vectors:    np.ndarray shape (pool_size, 4096), L2-normalised float32.
        corpus_texts:    {doc_id: full_text} for all docs in the pool.
        embed_server_url: GPU server URL with /encode endpoint.
        snippet_len:     Max chars for text snippets returned by search_index.
        timeout:         HTTP timeout in seconds.
    """

    def __init__(
        self,
        pool_doc_ids: list[str],
        pool_vectors: np.ndarray,
        corpus_texts: dict[str, str],
        embed_server_url: str,
        snippet_len: int = 2000,
        timeout: int = 60,
    ) -> None:
        self.pool_doc_ids = pool_doc_ids
        self.pool_vectors = pool_vectors.astype("float32")
        self.corpus_texts = corpus_texts
        self.base = embed_server_url.rstrip("/")
        self.snippet_len = snippet_len
        self.timeout = timeout

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a query using the GPU server's /encode endpoint."""
        text_with_instruction = _QUERY_INSTRUCTION + query
        r = requests.post(
            f"{self.base}/encode",
            json={"inputs": [text_with_instruction]},
            timeout=self.timeout,
        )
        r.raise_for_status()
        vec = np.array(r.json()["embeddings"][0], dtype="float32")
        # L2-normalise
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        return vec

    def search_index(self, query: str, top_k: int = 10) -> list[dict]:
        """Search the 1K pool. Returns [{score, doc_id, text}, ...]."""
        query_vec = self._embed_query(query)

        # Cosine similarity = dot product (vectors are L2-normalised)
        scores = self.pool_vectors @ query_vec

        top_k = min(top_k, len(self.pool_doc_ids))
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        results = []
        for idx in top_indices:
            doc_id = self.pool_doc_ids[idx]
            text = self.corpus_texts.get(doc_id, "")
            results.append({
                "score": float(scores[idx]),
                "doc_id": doc_id,
                "text": text[: self.snippet_len],
            })
        return results

    def get_document(self, doc_id: str) -> dict:
        """Fetch the full text of a document from the local corpus cache."""
        text = self.corpus_texts.get(doc_id)
        if text is None:
            return {"doc_id": doc_id, "text": "Not Found"}
        return {"doc_id": doc_id, "text": text}
