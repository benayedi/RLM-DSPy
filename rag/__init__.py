"""RAG-enabled dspy.RLM with FAISS retrieval and delegation for BrowseComp+."""

from .agent import RunMetrics, build_lm, build_rlm_agent, run_question
from .tools import FaissRetriever, make_embedding_fn

__all__ = [
    "FaissRetriever",
    "make_embedding_fn",
    "RunMetrics",
    "build_lm",
    "build_rlm_agent",
    "run_question",
]
