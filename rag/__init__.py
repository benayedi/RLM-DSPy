"""RAG-enabled dspy.RLM with FAISS retrieval and delegation for BrowseComp+."""

from .agent import RunMetrics, build_lm, build_rlm_agent, run_question
from .signatures import BrowseCompSignature, ChildBrowseCompSignature
from .tools import FaissRetriever, RemoteRetriever, make_embedding_fn
from .pool_retriever import PoolRetriever

__all__ = [
    "RemoteRetriever",
    "FaissRetriever",
    "PoolRetriever",
    "make_embedding_fn",
    "BrowseCompSignature",
    "ChildBrowseCompSignature",
    "RunMetrics",
    "build_lm",
    "build_rlm_agent",
    "run_question",
]
