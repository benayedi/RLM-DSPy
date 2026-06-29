"""Remote retrieval tools — delegates FAISS search and doc fetch to the GPU server."""

from __future__ import annotations

from typing import Any

import requests


QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query\nQuery: "
)


class RemoteRetriever:
    """HTTP client for the GPU embed+search server (/search, /document)."""

    def __init__(self, server_url: str, timeout: int = 60) -> None:
        self.base = server_url.rstrip("/")
        self.timeout = timeout

    def search_index(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        r = requests.post(
            f"{self.base}/search",
            json={"query": query, "top_k": top_k},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["results"]

    def get_document(self, doc_id: str) -> dict[str, Any]:
        r = requests.post(
            f"{self.base}/document",
            json={"doc_id": doc_id},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()


# Keep old name as alias so any external references don't break
FaissRetriever = RemoteRetriever


def make_embedding_fn(server_url: str):
    """Embedding function backed by the GPU server's /encode endpoint."""
    def embed(text: str) -> list[float]:
        r = requests.post(
            f"{server_url.rstrip('/')}/encode",
            json={"inputs": [text]},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["embeddings"][0]
    return embed
