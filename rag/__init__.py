"""RAG-enabled dspy.RLM with FAISS retrieval and delegation for BrowseComp+."""

from .agent import RunMetrics, build_lm, build_rlm_agent, run_question
from .signatures import BrowseCompSignature, ChildBrowseCompSignature
from .tools import FaissRetriever, RemoteRetriever, make_embedding_fn

__all__ = [
    "RemoteRetriever",
    "FaissRetriever",
    "make_embedding_fn",
    "BrowseCompSignature",
    "ChildBrowseCompSignature",
    "RunMetrics",
    "build_lm",
    "build_rlm_agent",
    "run_question",
]
