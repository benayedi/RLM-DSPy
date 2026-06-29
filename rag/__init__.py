"""RAG-enabled dspy.RLM with GPU FAISS retrieval and delegation for BrowseComp+."""

from .agent import RunMetrics, build_lm, build_rlm_agent, run_question
from .tools import FaissRetriever

__all__ = [
    "FaissRetriever",
    "RunMetrics",
    "build_lm",
    "build_rlm_agent",
    "run_question",
]
