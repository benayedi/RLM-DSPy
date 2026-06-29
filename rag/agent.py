"""
RAG RLM agent built on dspy.RLM with FAISS retrieval and delegation.

Usage:
    from rag import build_lm, build_rlm_agent, FaissRetriever, make_embedding_fn

    lm = build_lm()
    dspy.configure(lm=lm)

    retriever = FaissRetriever(server_url="http://localhost:8001")
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

from .signatures import BrowseCompSignature, ChildRLMSignature
from .tools import FaissRetriever

load_dotenv()


@dataclass
class RunMetrics:
    """Per-question metrics accumulated across all REPL turns and delegations."""

    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    iterations: int = 0
    delegation_calls: int = 0
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


def build_rlm_agent(
    retriever: FaissRetriever,
    depth: int = 0,
    max_depth: int = 5,
    max_iterations: int = 20,
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

    def search_index(query: str, top_k: int = 10) -> list[dict]:
        """Search the BrowseComp+ corpus for relevant passages.

        Returns list of {score: float, doc_id: str, text: str (≤2000 chars)}.
        Search for specific named entities, titles, or proper nouns.
        Tip: scores below 0.30 are usually off-topic — try a different query.
        """
        results = retriever.search_index(query, top_k=top_k)
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

    if depth < max_depth:

        def delegate_batch(tasks: list[dict], mode: str = "extract") -> list[str]:
            """Delegate multiple sub-tasks to child agents. Returns one answer per task.

            mode="extract": each child reads its task["context"] doc and extracts task["query"].
                            Fast parallel execution — preferred for most multi-doc questions.
            mode="orchestrate": each child is a full RLM that can reason iteratively.
                                Use only for genuinely complex sub-tasks.

            Task format:
                {
                    "query":        "specific extraction question",
                    "context":      "<full document text from get_document()>",
                    "doc_id":       "doc_id for tracking",
                    "parent_query": "<the original question>",
                }
            """
            import concurrent.futures

            metrics.delegation_calls += len(tasks)
            lm = dspy.settings.lm

            if mode == "extract":
                def extract_one(task: dict) -> str:
                    prompt = (
                        f"Parent question: {task.get('parent_query', '')}\n\n"
                        f"Your task: {task['query']}\n\n"
                        f"Document [doc_id={task.get('doc_id', '?')}]:\n"
                        f"{task.get('context', '')[:60000]}\n\n"
                        "Extract ONLY the exact value asked for (name, date, title, number). "
                        "If not found in this document, say None."
                    )
                    response = lm(messages=[{"role": "user", "content": prompt}])
                    return response[0] if isinstance(response, list) else str(response)

                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(tasks), 8)) as ex:
                    return list(ex.map(extract_one, tasks))

            else:  # orchestrate — full child RLM per task
                def orchestrate_one(task: dict) -> str:
                    child_rlm = dspy.RLM(
                        signature=ChildRLMSignature,
                        tools=[],
                        max_iterations=min(max_iterations, 8),
                        verbose=verbose,
                    )
                    result = child_rlm(
                        context=task.get("context", ""),
                        query=task["query"],
                    )
                    return str(getattr(result, "answer", result))

                return [orchestrate_one(t) for t in tasks]

        tools.append(delegate_batch)

        def delegate(sub_question: str, sub_context: str = "") -> str:
            """Spawn an independent child RLM with its own fresh search session.

            The child can search the corpus independently — results don't mix with
            the parent's searches. Use for clues about different entities that would
            contaminate each other if searched in the same session.

            Args:
                sub_question: The specific sub-question for the child to investigate.
                sub_context:  Optional hint or pre-fetched context for the child.
            Returns:
                The child's answer as a string.
            """
            metrics.delegation_calls += 1

            def child_search_index(query: str, top_k: int = 10) -> list[dict]:
                """Search the BrowseComp+ corpus for relevant passages."""
                results = retriever.search_index(query, top_k=top_k)
                metrics.retrieved_doc_ids.extend(r["doc_id"] for r in results)
                return results

            def child_get_document(doc_id: str) -> dict:
                """Fetch the full text of a document by its doc_id."""
                return retriever.get_document(doc_id)

            child_rlm = dspy.RLM(
                signature=ChildRLMSignature,
                tools=[child_search_index, child_get_document],
                max_iterations=min(max_iterations, 8),
                verbose=verbose,
            )
            result = child_rlm(context=sub_context or "", query=sub_question)
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
    max_iterations: int = 20,
    verbose: bool = False,
) -> tuple[str, RunMetrics]:
    """
    Convenience wrapper: build a fresh agent, run one question, return (answer, metrics).

    Tracks latency and token usage from the DSPy LM history.
    """
    lm = dspy.settings.lm
    history_before = len(getattr(lm, "history", []) or [])

    rlm, metrics = build_rlm_agent(
        retriever=retriever,
        max_depth=max_depth,
        max_iterations=max_iterations,
        verbose=verbose,
    )

    t0 = time.time()
    result = rlm(question=question)
    metrics.latency_s = time.time() - t0

    # Count tokens from new LM history entries
    history = getattr(lm, "history", []) or []
    for entry in history[history_before:]:
        usage = entry.get("usage") or {}
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
