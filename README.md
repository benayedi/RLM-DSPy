# RLM-DSPy — BrowseComp+ RAG Evaluation

A RAG agent built on [DSPy RLM](https://github.com/stanfordnlp/dspy) evaluated on the BrowseComp+ benchmark (830 hard multi-constraint questions, 100K documents).

## Setup

```bash
pip install -e .
cp .env.template .env  # fill in Azure credentials
```

Start the embedding server (must be running before any eval):

```bash
curl $EMBEDDING_SERVER_URL/health  # verify it's up
```

---

## Running the BrowseComp+ Evaluation

### Standard full run (GPT-5 + IVFSQ8)

```bash
python3 examples/browsecomp_eval.py \
  --indices $(seq 0 829 | tr '\n' ',') \
  --out logs/my_run \
  --max-iters 16 \
  --max-search 35
```

### Run a specific question range (e.g. Q120–Q200, 0-based indices)

```bash
python3 examples/browsecomp_eval.py \
  --indices $(seq 119 199 | tr '\n' ',') \
  --out logs/run_q120_q200_gpt5_ivfsq8 \
  --max-iters 16 \
  --max-search 35
```

### Rerun specific questions

```bash
python3 examples/browsecomp_eval.py \
  --indices 9,19,42 \
  --out logs/rerun_q10_q20_q43 \
  --max-iters 16 \
  --max-search 35
```

### Run with watchdog (auto-skips stuck questions)

```bash
python3 -u examples/browsecomp_eval.py --indices ... --out logs/my_run > logs/my_run.log 2>&1 &
python3 scripts/watchdog.py logs/my_run.log
```

---

## Key settings

| Parameter | Default | Notes |
|---|---|---|
| `AZURE_OPENAI_MODEL` | `ismail-gpt-5` | Agent model |
| `AZURE_OPENAI_JUDGE_MODEL` | `gpt-5.4-mini` | Grading model — do not change |
| `AZURE_OPENAI_MAX_TOKENS` | `12000` | Reasoning budget |
| `AZURE_OPENAI_REASONING_EFFORT` | `low` | `low` is fastest; `high` is ~5x slower with marginal gains |
| `--max-iters` | `16` | Max REPL turns per question |
| `--max-search` | `35` | Max `search_index` calls per question |
| `BROWSECOMP_TIMEOUT` | `800` | Per-question timeout in seconds |

---

## Index

Two FAISS index variants:
- **Flat** — exact nearest-neighbour search, no quantization loss
- **IVFSQ8** — `IVFScalarQuantizer`, `nprobe=64`, 428 MB, GPU — used for all main results

Build the IVFSQ8 index from a flat index:

```bash
python3 scripts/build_ivfsq8_index.py \
  --flat-index path/to/_cached.faiss \
  --output path/to/_cached_ivfsq8.faiss
```

---

## Results (Q1–Q830, GPT-5 + IVFSQ8)

| Metric | Value |
|---|---|
| Accuracy | 573/830 **(69.0%)** |
| Total runtime | 239,794s (66.6 hours) |
| Avg latency | 125.8s/question (excl. >600s outliers) |
| Avg gold recall | 0.724 |
| Avg search calls | 11.2 / question |
| Total search calls | 9,295 |

Detailed per-question results and failure breakdown: [`docs/gpt5_ivfsq8_q1_q830_summary.txt`](docs/gpt5_ivfsq8_q1_q830_summary.txt)

---

## Output files

Each run produces:
- `logs/<name>.json` — structured results per question
- `logs/<name>.txt` — human-readable trace with search queries
- `logs/<name>.log` — full stdout

JSON fields per question: `q_idx`, `question`, `gold`, `pred`, `correct`, `latency_s`,
`input_tokens`, `output_tokens`, `iterations`, `search_calls`, `get_document_calls`,
`unique_docs`, `gold_recall`, `evid_recall`.
