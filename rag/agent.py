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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import dspy
from dotenv import load_dotenv
from dspy.utils.usage_tracker import track_usage

from .signatures import BrowseCompSignature, ChildBrowseCompSignature
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
    retrieved_doc_ids: list = field(default_factory=list)

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
        max_tokens=16000,
        cache=False,
    )


CHILD_MAX_ITERATIONS = 6  # children are focused tasks, not full searches


def build_rlm_agent(
    retriever: FaissRetriever,
    depth: int = 0,
    max_depth: int = 5,
    max_iterations: int = 25,
    metrics: RunMetrics | None = None,
    verbose: bool = False,
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

    Returns:
        (rlm, metrics) — call rlm(question=...) to run.
    """
    if metrics is None:
        metrics = RunMetrics()

    # ------------------------------------------------------------------ #
    # Tools                                                                #
    # ------------------------------------------------------------------ #

    is_child = depth > 0

    def search_index(query: str, top_k: int = 10) -> list[dict]:
        """Search the BrowseComp+ corpus for relevant passages.

        Returns list of {score: float, doc_id: str, text: str (≤2000 chars)}.
        Search for specific named entities, titles, or proper nouns.
        Tip: scores below 0.30 are usually off-topic — try a different query.
        """
        results = retriever.search_index(query, top_k=top_k)
        metrics.search_calls += 1
        metrics.retrieved_doc_ids.extend(r["doc_id"] for r in results)
        return results

    def get_document(doc_id: str) -> dict:
        """Fetch the full text of a document.

        Args:
            doc_id: The doc_id string from a search_index result.
        Returns:
            {doc_id: str, text: str}.
        Only call when the snippet is truncated at a relevant point.
        """
        return retriever.get_document(doc_id)

    tools = [search_index, get_document]

    if depth < max_depth and not is_child:

        def delegate(sub_question: str, sub_context: str = "") -> str:
            """Launch an independent child agent to answer a sub-question.

            The child receives the documents you pass as context and reasons
            over them without searching the corpus again.

            Args:
                sub_question: The specific sub-question to investigate.
                sub_context:  Documents / text to pass to the child as context.
            Returns:
                The child agent's answer as a string.
            """
            metrics.delegation_calls += 1
            child_rlm, _ = build_rlm_agent(
                retriever=retriever,
                depth=depth + 1,
                max_depth=max_depth,
                max_iterations=CHILD_MAX_ITERATIONS,
                metrics=metrics,
                verbose=verbose,
            )
            result = child_rlm(context=sub_context, query=sub_question)
            return str(getattr(result, "answer", result))

        tools.append(delegate)

        def delegate_batch(tasks: list[dict]) -> list[str]:
            """Spawn multiple child agents IN PARALLEL, one per task.

            Each task is a dict with keys:
                query        - the sub-question for the child
                context      - full document text to pass to the child (can be "")
                parent_query - the original question for broader orientation

            Returns answers in the same order as tasks.
            All children run concurrently — total time ≈ slowest child, not sum.
            Use when you have 2+ independent sub-questions to investigate.
            """
            def run_task(idx_task):
                idx, task = idx_task
                sub_question = task.get("query", "")
                sub_context = task.get("context", "")
                parent_q = task.get("parent_query", "")
                if parent_q and sub_context:
                    sub_context = f"Parent question: {parent_q}\n\n{sub_context}"
                elif parent_q:
                    sub_question = f"Parent question: {parent_q}\n\n{sub_question}"
                return idx, delegate(sub_question, sub_context)

            results = [None] * len(tasks)
            with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
                futures = {pool.submit(run_task, (i, t)): i for i, t in enumerate(tasks)}
                for future in as_completed(futures):
                    idx, answer = future.result()
                    results[idx] = answer
            return results

        tools.append(delegate_batch)

    signature = ChildBrowseCompSignature if is_child else BrowseCompSignature

    rlm = dspy.RLM(
        signature=signature,
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
    verbose: bool = False,
) -> tuple[str, RunMetrics]:
    """
    Convenience wrapper: build a fresh agent, run one question, return (answer, metrics).

    Tracks latency and token usage from the DSPy LM history.
    """
    rlm, metrics = build_rlm_agent(
        retriever=retriever,
        max_depth=max_depth,
        max_iterations=max_iterations,
        verbose=verbose,
    )

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
