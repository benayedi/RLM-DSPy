"""FAISS retrieval tools — search and document fetch both on GPU server."""

from __future__ import annotations

from typing import Any

import requests


class FaissRetriever:
    """
    Retriever backed entirely by the GPU search server.

    Both search_index (embed + FAISS) and get_document (corpus lookup)
    are served by the GPU, so no local corpus or index loading is needed.
    """

    def __init__(self, server_url: str, k: int = 10) -> None:
        """
        Args:
            server_url: Base URL of the GPU server, e.g. "http://localhost:8001"
            k:          Default top-k results per query
        """
        self._url = server_url.rstrip("/")
        self.k = k

    def search_index(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """
        Search the corpus via the GPU server (embed + FAISS on GPU).

        Returns list of {score: float, doc_id: str, text: str (≤2000 chars)}.
        """
        payload = {"query": query, "top_k": top_k if top_k is not None else self.k}
        r = requests.post(f"{self._url}/search", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["results"]

    def get_document(self, doc_id: str) -> dict[str, Any]:
        """
        Fetch the full text of a document from the GPU server.

        Returns {doc_id: str, text: str}.
        """
        r = requests.post(f"{self._url}/document", json={"doc_id": doc_id}, timeout=60)
        r.raise_for_status()
        return r.json()
