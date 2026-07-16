# RLM-DSPy — BrowseComp+ RAG Evaluation

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # fill in Azure credentials
```

Key `.env` variables:

```
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://gpt5openai.openai.azure.com/
EMBEDDING_SERVER_URL=http://...
```

---

## Running the BrowseComp+ Evaluation

### ⚠️ Important: always use `ismail-gpt-5`, not the default `gpt-5.4-mini`

The `.env` default (`AZURE_OPENAI_MODEL=gpt-5.4-mini`) is the **judge/grading model**.  
The **agent model** must be overridden to `ismail-gpt-5` for all evaluation runs.

### Standard full run (GPT-5 + IVFSQ8)

```bash
AZURE_OPENAI_MODEL=ismail-gpt-5 \
AZURE_OPENAI_MAX_TOKENS=12000 \
AZURE_OPENAI_REASONING_EFFORT=low \
python3 examples/browsecomp_eval.py \
  --indices 0 1 2 ... \
  --out logs/my_run \
  --max-iters 16 \
  --max-search 35 \
  --timeout 800
```

### Run a specific question range (e.g. Q120–Q200, 0-based indices)

```bash
AZURE_OPENAI_MODEL=ismail-gpt-5 \
AZURE_OPENAI_MAX_TOKENS=12000 \
AZURE_OPENAI_REASONING_EFFORT=low \
python3 examples/browsecomp_eval.py \
  --indices $(seq 119 199 | tr '\n' ' ') \
  --out logs/run_q120_q200_gpt5_ivfsq8 \
  --max-iters 16 \
  --max-search 35 \
  --timeout 800
```

### Rerun specific failing questions

```bash
AZURE_OPENAI_MODEL=ismail-gpt-5 \
AZURE_OPENAI_MAX_TOKENS=12000 \
AZURE_OPENAI_REASONING_EFFORT=low \
python3 examples/browsecomp_eval.py \
  --indices 9 19 42 \
  --out logs/rerun_q10_q20_q43_gpt5 \
  --max-iters 16 \
  --max-search 35 \
  --timeout 800
```

---

## Key settings

| Parameter | Value | Notes |
|---|---|---|
| `AZURE_OPENAI_MODEL` | `ismail-gpt-5` | Agent model — always override this |
| `AZURE_OPENAI_MAX_TOKENS` | `12000` | Required for GPT-5 reasoning budget |
| `AZURE_OPENAI_REASONING_EFFORT` | `low` | `low` is sufficient and much faster |
| `--max-iters` | `16` | Max REPL turns per question |
| `--max-search` | `35` | Max `search_index` calls per question |
| `--timeout` | `800` | Per-question timeout in seconds (also set via `BROWSECOMP_TIMEOUT` in `.env`) |
| Judge model | `gpt-5.4-mini` | Set via `AZURE_OPENAI_JUDGE_MODEL` in `.env` |

---

## Index server

The embedding server must be running before starting any eval.  
Check health:

```bash
curl $EMBEDDING_SERVER_URL/health
```

Two index variants tested:
- **FAISS Flat** — exact search, slower
- **FAISS IVFSQ8** — `IVFScalarQuantizer`, `nprobe=64`, 428 MB, GPU — used for main results

---

## Results (Q1–Q119, ismail-gpt-5 + IVFSQ8)

| Metric | Value |
|---|---|
| Accuracy | 78/119 = **65.5%** |
| Avg latency | ~60s/question |
| Max search calls | 35 per question |
| Max iterations | 16 per question |

---

## Output files

Each run produces:
- `logs/<name>.json` — structured results per question
- `logs/<name>.txt` — human-readable trace with search queries
- `logs/<name>.log` — full stdout (if redirected)

JSON fields per question: `q_idx`, `question`, `gold`, `pred`, `correct`, `latency_s`,
`input_tokens`, `output_tokens`, `iterations`, `search_calls`, `get_document_calls`,
`unique_docs`, `gold_recall`, `evid_recall`.
