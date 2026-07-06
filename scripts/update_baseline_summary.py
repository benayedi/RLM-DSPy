"""
Regenerate logs/baseline_summary.txt from all *baseline*.json files in logs/.
Run after every baseline eval:
    python scripts/update_baseline_summary.py
"""

import json
import os
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path(__file__).parent.parent / "logs"
OUT_FILE = LOGS_DIR / "baseline_summary.txt"


def load_all_baseline_results():
    files = sorted(LOGS_DIR.glob("*baseline*.json"))
    all_results = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        all_results.extend(data)
    # Sort by q_idx and deduplicate (keep last seen for each q_idx)
    seen = {}
    for r in all_results:
        seen[r["q_idx"]] = r
    return [seen[k] for k in sorted(seen)]


def write_summary(results):
    n = len(results)
    if n == 0:
        print("No baseline results found.")
        return

    correct = sum(r["correct"] for r in results)
    avg = lambda k: sum(r[k] for r in results) / n
    gold_recalls = [r["gold_recall"] for r in results]
    evid_recalls = [r["evid_recall"] for r in results]

    q_min = results[0]["q_idx"] + 1
    q_max = results[-1]["q_idx"] + 1

    # Chunk into ranges of ~30 for per-range stats
    ranges = []
    chunk = []
    for r in results:
        chunk.append(r)
        if len(chunk) == 30 or r is results[-1]:
            ranges.append(chunk)
            chunk = []

    pure_miss  = [r for r in results if not r["correct"] and r["gold_recall"] == 0.0]
    gold_found = [r for r in results if not r["correct"] and r["gold_recall"] == 1.0]
    partial    = [r for r in results if not r["correct"] and 0 < r["gold_recall"] < 1.0]

    lines = []
    lines.append("=" * 72)
    lines.append("BrowseComp+ Baseline Evaluation — RAG RLM")
    lines.append("Model: gpt-5.4-mini (Azure)  |  Retriever: FAISS (Qwen3-Embed-8B)")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Configuration")
    lines.append("-" * 40)
    lines.append("  Model              : gpt-5.4-mini (Azure OpenAI)")
    lines.append("  Embeddings         : Qwen3-Embed-8B (Tevatron pre-built, 4096-dim)")
    lines.append("  Retriever          : FAISS IndexFlatIP (100,195 vectors, GPU)")
    lines.append("  Default top_k      : 10")
    lines.append("  Max iterations     : 25")
    lines.append("  Max depth          : 5")
    lines.append("  Temperature        : 1.0 (gpt-5.4-mini only supports 1.0)")
    lines.append("  Dataset            : Tevatron/browsecomp-plus (test split, 830 questions)")
    lines.append(f"  Questions evaluated: {n}  (Q{q_min}–Q{q_max})")
    lines.append("")
    lines.append("=" * 72)
    lines.append(f"OVERALL SUMMARY  (Q{q_min}–Q{q_max}, {n} questions)")
    lines.append("=" * 72)
    lines.append(f"  Accuracy           : {correct}/{n} ({100*correct/n:.1f}%)")
    lines.append(f"  Avg latency        : {avg('latency_s'):.1f}s")
    lines.append(f"  Avg tokens in      : {avg('input_tokens'):,.0f}")
    lines.append(f"  Avg tokens out     : {avg('output_tokens'):,.0f}")
    lines.append(f"  Avg iterations     : {avg('iterations'):.1f}")
    lines.append(f"  Avg delegation calls: {avg('delegation_calls'):.2f}")
    lines.append(f"  Avg unique docs    : {avg('unique_docs'):.1f}")
    lines.append(f"  Avg gold recall    : {sum(gold_recalls)/n:.3f}")
    lines.append(f"  Avg evid recall    : {sum(evid_recalls)/n:.3f}")
    lines.append("")
    lines.append(f"Failure breakdown ({n - correct} wrong):")
    lines.append(f"  Pure retrieval miss (gold_recall=0.00) : {len(pure_miss)} questions")
    lines.append(f"  Gold found, wrong answer (recall=1.00) : {len(gold_found)} questions")
    lines.append(f"  Partial recall (0 < recall < 1)        : {len(partial)} questions")
    lines.append("")

    lines.append("=" * 72)
    lines.append("BY RANGE (groups of 30)")
    lines.append("=" * 72)
    start_idx = 0
    for chunk in ranges:
        nc = sum(r["correct"] for r in chunk)
        nr = len(chunk)
        q0 = chunk[0]["q_idx"] + 1
        q1 = chunk[-1]["q_idx"] + 1
        lines.append(f"  Q{q0}–Q{q1}: {nc}/{nr} ({100*nc/nr:.1f}%)")
        lines.append(f"    Avg latency   : {sum(r['latency_s'] for r in chunk)/nr:.1f}s")
        lines.append(f"    Avg tok in    : {sum(r['input_tokens'] for r in chunk)/nr:,.0f}")
        lines.append(f"    Avg tok out   : {sum(r['output_tokens'] for r in chunk)/nr:,.0f}")
        lines.append(f"    Avg iters     : {sum(r['iterations'] for r in chunk)/nr:.1f}")
        lines.append(f"    Avg gold rec  : {sum(r['gold_recall'] for r in chunk)/nr:.3f}")
        lines.append("")

    lines.append("=" * 72)
    lines.append("PER-QUESTION RESULTS")
    lines.append("=" * 72)
    lines.append(
        f"  {'Q':<6} {'✓/✗':<5} {'Predicted':<40} {'Gold':<40}"
        f" {'Lat(s)':>7} {'TokIn':>9} {'TokOut':>8} {'Iters':>6} {'GRec':>6} {'ERec':>6}"
    )
    lines.append("  " + "-" * 130)
    for r in results:
        sym = "✓" if r["correct"] else "✗"
        pred = r["predicted"][:38] + ".." if len(r["predicted"]) > 40 else r["predicted"]
        gold = r["gold"][:38] + ".." if len(r["gold"]) > 40 else r["gold"]
        lines.append(
            f"  Q{r['q_idx']+1:03d}  {sym:<5} {pred:<40} {gold:<40}"
            f" {r['latency_s']:>7.1f} {r['input_tokens']:>9,} {r['output_tokens']:>8,}"
            f" {r['iterations']:>6} {r['gold_recall']:>6.2f} {r['evid_recall']:>6.2f}"
        )

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Updated {OUT_FILE}  ({n} questions, {correct}/{n} = {100*correct/n:.1f}%)")


if __name__ == "__main__":
    results = load_all_baseline_results()
    write_summary(results)
