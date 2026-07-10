"""
RAG RLM agent built on dspy.RLM with FAISS retrieval and delegation.

Usage:
    from rag import build_lm, build_rlm_agent, FaissRetriever, make_embedding_fn

    lm = build_lm()
    dspy.configure(lm=lm)

    retriever = FaissRetriever(index_pattern, embedding_fn)
    rlm, metrics = build_rlm_agent(retriever)

    result = rlm(question="Who invented the telephone?")
    print(result.answer)
    print(metrics)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import dspy
from dotenv import load_dotenv
from dspy.utils.usage_tracker import track_usage

from .signatures import BrowseCompSignature
from .tools import RemoteRetriever as FaissRetriever

load_dotenv()


@dataclass
class RunMetrics:
    """Per-question metrics accumulated across all REPL turns and delegations."""

    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    iterations: int = 0
    delegation_calls: int = 0
    search_calls: int = 0
    get_document_calls: int = 0
    retrieved_doc_ids: list = field(default_factory=list)
    _query_cache: dict = field(default_factory=dict)
    _gold_ids: set = field(default_factory=set)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def unique_docs_retrieved(self) -> int:
        return len(set(self.retrieved_doc_ids))

    def gold_recall(self, gold_ids: list[str]) -> float:
        if not gold_ids:
            return 0.0
        retrieved = set(self.retrieved_doc_ids)
        return sum(1 for g in gold_ids if g in retrieved) / len(gold_ids)

    def __repr__(self) -> str:
        return (
            f"RunMetrics(lat={self.latency_s:.1f}s, "
            f"tok={self.input_tokens}/{self.output_tokens}, "
            f"iters={self.iterations}, "
            f"search={self.search_calls}, getdoc={self.get_document_calls}, "
            f"deleg={self.delegation_calls}, "
            f"docs={self.unique_docs_retrieved})"
        )


def build_lm() -> dspy.LM:
    """Configure Azure OpenAI LM from .env / environment variables."""
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    model = os.environ.get("AZURE_OPENAI_MODEL", "gpt-5.4-mini")

    return dspy.LM(
        model=f"azure/{model}",
        api_base=endpoint,
        api_key=api_key,
        api_version=api_version,
        temperature=1.0,
        max_tokens=int(os.environ.get("AZURE_OPENAI_MAX_TOKENS", "16000")),
        cache=False,
    )


def build_rlm_agent(
    retriever: FaissRetriever,
    depth: int = 0,
    max_depth: int = 5,
    max_iterations: int = 25,
    max_search_calls: int = 50,
    metrics: RunMetrics | None = None,
    verbose: bool = False,
    default_top_k: int = 10,
) -> tuple[dspy.RLM, RunMetrics]:
    """
    Build a dspy.RLM agent with FAISS retrieval and delegation tools.

    Args:
        retriever:      Loaded FaissRetriever instance (shared across delegations).
        depth:          Current recursion depth (0 = root agent).
        max_depth:      Maximum delegation depth.
        max_iterations: Max REPL iterations per agent call.
        metrics:        Shared RunMetrics (created fresh if None at depth=0).
        verbose:        Enable dspy.RLM verbose logging.
        default_top_k:  Default number of results returned by search_index.

    Returns:
        (rlm, metrics) — call rlm(question=...) to run.
    """
    if metrics is None:
        metrics = RunMetrics()

    # ------------------------------------------------------------------ #
    # Tools                                                                #
    # ------------------------------------------------------------------ #

    def search_index(query: str, top_k: int = default_top_k) -> list[dict]:
        """Search the BrowseComp+ corpus for relevant passages.

        Returns list of {score: float, doc_id: str, text: str (≤2000 chars)}.
        Search for specific named entities, titles, or proper nouns.
        Tip: scores below 0.30 are usually off-topic — try a different query.
        """
        if metrics.search_calls >= max_search_calls:
            print(f"    [search BLOCKED — limit {max_search_calls} reached] {query[:60]!r}")
            return [{"score": 0.0, "doc_id": "LIMIT", "text": (
                f"Search limit of {max_search_calls} calls reached. "
                "Stop searching and provide your best answer based on what you already retrieved."
            )}]

        if query in metrics._query_cache:
            print(f"    [search CACHED] {query[:60]!r}")
            return metrics._query_cache[query]

        t0 = time.perf_counter()
        results = retriever.search_index(query, top_k=top_k)
        elapsed = time.perf_counter() - t0
        metrics.retrieved_doc_ids.extend(r["doc_id"] for r in results)
        snippet_len = int(os.environ.get("SEARCH_SNIPPET_LEN", "500"))
        for r in results:
            if "text" in r and len(r["text"]) > snippet_len:
                r["text"] = r["text"][:snippet_len] + "…"
        metrics.search_calls += 1
        metrics._query_cache[query] = results
        doc_ids = [r["doc_id"] for r in results]
        gold_hits = sum(1 for d in doc_ids if d in metrics._gold_ids)
        hit_str = f"  ★{gold_hits}" if gold_hits else ""
        print(f"    [search #{metrics.search_calls}] {query[:60]!r}  → {len(results)} docs{hit_str}  ({elapsed*1000:.0f}ms)")
        return results

    def get_document(doc_id: str) -> dict:
        """Fetch the full text of a document.

        Args:
            doc_id: The doc_id string from a search_index result.
        Returns:
            {doc_id: str, text: str}.
        Only call when the snippet is truncated at a relevant point.
        """
        t0 = time.perf_counter()
        result = retriever.get_document(doc_id)
        elapsed = time.perf_counter() - t0
        metrics.get_document_calls += 1
        print(f"    [getdoc #{metrics.get_document_calls}] {doc_id}  ({elapsed*1000:.0f}ms)")
        return result

    tools = [search_index, get_document]

    if depth < max_depth:

        def delegate(sub_question: str, sub_context: str = "") -> str:
            """Launch an independent child agent to answer a sub-question.

            Each delegate() has its own fresh retrieval session, letting you
            investigate multiple clues without contaminating each other.

            Args:
                sub_question: The specific sub-question to investigate.
                sub_context:  Optional extra context string for the child agent.
            Returns:
                The child agent's answer as a string.
            """
            metrics.delegation_calls += 1
            child_rlm, _ = build_rlm_agent(
                retriever=retriever,
                depth=depth + 1,
                max_depth=max_depth,
                max_iterations=max_iterations,
                max_search_calls=max_search_calls,
                metrics=metrics,
                verbose=verbose,
            )
            question = sub_question
            if sub_context:
                question = f"{sub_context}\n\n{sub_question}"
            result = child_rlm(question=question)
            return str(getattr(result, "answer", result))

        tools.append(delegate)

    rlm = dspy.RLM(
        signature=BrowseCompSignature,
        tools=tools,
        max_iterations=max_iterations,
        verbose=verbose,
    )

    return rlm, metrics


def run_question(
    retriever: FaissRetriever,
    question: str,
    max_depth: int = 5,
    max_iterations: int = 25,
    max_search_calls: int = 50,
    verbose: bool = False,
    default_top_k: int = 10,
    gold_ids: set | None = None,
) -> tuple[str, RunMetrics]:
    """
    Convenience wrapper: build a fresh agent, run one question, return (answer, metrics).

    Tracks latency and token usage from the DSPy LM history.
    """
    rlm, metrics = build_rlm_agent(
        retriever=retriever,
        max_depth=max_depth,
        max_iterations=max_iterations,
        max_search_calls=max_search_calls,
        verbose=verbose,
        default_top_k=default_top_k,
    )
    if gold_ids:
        metrics._gold_ids = set(gold_ids)

    with track_usage() as tracker:
        t0 = time.time()
        result = rlm(question=question)
        metrics.latency_s = time.time() - t0

    for usage in tracker.get_total_tokens().values():
        metrics.input_tokens += usage.get("prompt_tokens", 0)
        metrics.output_tokens += usage.get("completion_tokens", 0)

    # Count REPL iterations from trajectory
    trajectory = getattr(result, "trajectory", None) or {}
    # dspy.RLM stores trajectory as dict with iteration keys
    if isinstance(trajectory, dict):
        metrics.iterations = len([k for k in trajectory if k.startswith("iteration_")])
    elif hasattr(trajectory, "__len__"):
        metrics.iterations = len(trajectory)

    answer = getattr(result, "answer", str(result))
    return str(answer), metrics
