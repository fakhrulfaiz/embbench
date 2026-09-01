# Proposal: keep scoring in embbench (MTEB), generation in another service

We will split the work into two repos. This document is about **embbench**: why we score with MTEB instead of writing our own benchmark, and how that still covers our own data.

The other service (doc extract, question generation, STS pairing) is out of scope here. It only needs to **export files**. Schema for that store: [docs/dataset-store.md](docs/dataset-store.md). embbench does not ingest documents, chunk them, or invent labels.

```
other service  →  data/retrieval/<name>/ and data/sts/<name>/
embbench       →  encode + exact search + nDCG / Recall / Spearman
```

## What embbench is for

embbench is the exam, not the question writer.

- **Retrieval:** nDCG@10/30, Recall@10/30
- **STS:** cosine Spearman
- **Languages we already run:** English, Chinese, Malay (Malay is retrieval-only on public MTEB)
- **Models (one GPU process at a time):** BCE (production baseline), Voyage-4-nano, Harrier, Qwen3-Embedding-0.6B, bge-m3 (dense only)

Public MTEB/C-MTEB tasks run today. Any internal corpus is a folder drop, scored with the **same** `mteb.evaluate` path. Quality scores are exact cosine in RAM. Qdrant is optional and only for serving numbers (`--profile-ops`: p95, index size, ANN vs exact). It is not how we decide which model is better.

## Why MTEB, not a custom benchmark

A homemade loop (“embed texts, cosine, pick top-k, print recall”) looks simple. It is how we would quietly get the ranking wrong.

### 1. The score has to mean the same thing as everyone else’s

nDCG@k and Recall@k have a standard definition (BEIR / MTEB). Spearman on STS is the same. If we invent our own cut-offs, pooling, or “close enough” search, we cannot compare BCE to Voyage on FiQA, or our SOP set to STSBenchmark, with a straight face. MTEB is that protocol. Our jsonl is just another task in the same exam.

### 2. Wrappers: some models are not a single `encode()`

This is the main reason not to roll our own encoder calls.

Several models we care about were trained with a **prefix** on the text, and **query ≠ document**:

| Model | Recipe |
|---|---|
| Voyage-4-nano | `encode_query` / `encode_document` with different “Represent the …” prefixes |
| Harrier, Qwen3 | Instruction prefix, often `Instruct: <task>\nQuery: ` |
| BCE, bge-m3 | No prefix. Plain `encode()` is correct |

MTEB keeps a **registry** of those recipes. `mteb.get_model(hf_name)` loads the official wrapper so retrieval uses `encode_query` on questions and `encode_document` on passages.

If we called `SentenceTransformer.encode()` on Voyage ourselves, the weights would still run. The vectors would **not** match how it was trained. nDCG would come out low and we would pick the wrong model.

That is why `configs/models.yaml` has two loaders:

- `loader: mteb` — registered / instruction-aware (Voyage, Harrier, Qwen3)
- `loader: sentence_transformers` — not in the registry, or no prefixes (BCE, bge-m3)
- `loader: openai_api` — model already served (vLLM pooling); this process is an HTTP client

`uv run embbench check-model <hf_name>` tells us which one to use. Registration is not a leaderboard badge. It is the answer key for **how to embed**. Unregistered models still sit the same exam; we just own a plain `encode()`.

bge-m3 is in the registry but `use_instructions: false`. Dense retrieval is the same as SentenceTransformer (CLS + L2). We do not need a custom encode class for it. Hybrid (sparse + ColBERT) is a later backend, not this scoring path.

### 3. Exact search, not our vector database

MTEB retrieval is:

1. Embed **every** corpus row
2. Embed **every** query
3. Exact cosine in memory
4. Compare top-k to qrels

Each model re-embeds from scratch. Stored production vectors cannot be reused (they belong to one model). Search through ANN would mix index error into nDCG. A custom “benchmark against Qdrant” would measure the serving stack, not the embedding model. We already isolate that: quality = MTEB exact; serving = `--profile-ops` only.

A 40k-chunk dump is in the same band as public FiQA. VRAM follows batch size and sequence length (we pin 512), not corpus size. Wall time grows; the protocol does not change.

### 4. Public tasks and our tasks share one runner

`run_job(JobSpec)` is the only execution path. Public tasks come from `configs/tasks.yaml`. Local tasks are folders:

```
data/retrieval/<name>/   meta.yaml, corpus.jsonl, queries.jsonl, qrels.tsv
data/sts/<name>/         meta.yaml, pairs.jsonl
```

No new Python module per dataset. `--no-include-mteb --task-names <name>` runs only our set. Same metrics, same wrappers, same cache (`revision` in `meta.yaml` is the cache key).

Building a second scorer for “our data only” would duplicate encode, pooling, k-cut-offs, and prediction dumps, then drift from the public numbers we already trust.

## What we will do

**embbench (this repo)**

- Keep MTEB as the scorer (`mteb.evaluate`, official wrappers where registered)
- Compare the five models on public EN / ZH / MS tasks, plus any folder the other service drops
- Add models with a yaml row + `check-model`, not a new eval framework
- Keep Qdrant off the quality path

**Other repo / service (not designed here)**

- Extract docs, generate questions, build STS pairs
- Export the jsonl/tsv layout above and bump `revision` when files change
- Does not load embedding models or compute nDCG

**Interface between them**

A script or job in the other service writes the folder. embbench only reads it.

```bash
uv run embbench run \
  --model all \
  --languages eng \
  --task-types Retrieval \
  --no-include-mteb \
  --task-names <folder-name>
```

## What we will not do in embbench

- Document ingestion, chunking, or label generation
- Scoring through a production vector store
- A hand-rolled encode/rank loop “so we don’t depend on MTEB”
- Per-dataset Python adapters
- bge-m3 hybrid in this phase

## Decision

Use MTEB because it is the **encode recipe + metric protocol**, not because we need the public leaderboard. The wrappers are what stop us from under-scoring Voyage/Harrier/Qwen. A custom benchmark would reimplement that badly and split our internal set from the public one.

Generation stays elsewhere. This repo stays the harness.
