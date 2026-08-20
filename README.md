# Embedding benchmark harness

Compare five embedding models on **retrieval (nDCG@k, Recall@k)** and **STS**. Public MTEB tasks run now. Any future dataset (Malay STS, your own SOP corpus) is a folder drop, not a code change.

Scoring is exact in-memory search. Qdrant is only for serving-path numbers (p95 latency, index size, ANN vs exact recall delta).

## Setup

```bash
cp .env.example .env
uv sync
```

`.env` pins HuggingFace caches to `/mnt/c/ml-cache/huggingface`. The process refuses to start if `HF_HOME` is not under `/mnt/c`.

Optional, for serving-path metrics:

```bash
docker compose up -d qdrant
```

## Run

One model (what you want while iterating):

```bash
uv run embbench run --model bce-embedding-base_v1 --languages eng --task-types Retrieval
```

All five models, **one subprocess each**, fail-soft, resumable:

```bash
uv run embbench run --model all --profile-ops
```

This resumes by default. A model that already has a completed result **for the same scope** is skipped, no matter which run produced it, so re-running after a crash only picks up what is left. Models that failed (bge-m3 on this 8GB card) are retried. To redo everything, including MTEB's own score cache:

```bash
uv run embbench run --model all --force
```

Scope means languages, task types, task names, and the include flags. Change any of them and the old result is not reused, so `--languages eng` after a full run genuinely re-runs rather than quietly reporting stale numbers.

Smoke test (baseline on SciFact):

```bash
uv run embbench smoke
```

```bash
uv run embbench list-models
uv run embbench list-datasets
uv run embbench report
```

## View results

`embbench report` writes `results/report.md`. For anything more than the headline table, use the dashboard:

```bash
uv run embbench dashboard          # http://localhost:8501
uv run embbench dashboard --port 8600
```

It is read-only. It renders what a run already wrote to `results/` and never re-scores, writes, or touches the GPU, so it is safe to leave open while a benchmark runs. Six pages:

| Page | What it answers |
|---|---|
| Overview | Which model wins per language, by how much against the current production baseline, and what failed |
| Retrieval | nDCG and Recall at k, per task, as absolute scores or as a gap vs baseline |
| Similarity | The same for STS, including which languages MTEB has no dataset for |
| Metric explorer | The full MTEB metric family at every cut-off. The k-sweep shows whether a weak nDCG@10 is a ranking problem a reranker could fix or a recall miss it cannot |
| Ops and cost | Peak VRAM against the 8GB budget, encode time, and query latency |
| Artifacts | Rendered reports, every file on disk with its size and purpose, raw JSON, and a per-query drill-down |

Prediction files run to hundreds of megabytes per model, so the drill-down lists them by size and parses one only when you ask for it.

## Add a dataset

Drop a folder into `data/`. The harness wraps it as an MTEB task and scores it exactly like public FiQA or STSBenchmark. No Python edits, no `configs/tasks.yaml` change.

| Kind | Folder | Files | Measures |
|---|---|---|---|
| Retrieval | `data/retrieval/<name>/` | `meta.yaml`, `corpus.jsonl`, `queries.jsonl`, `qrels.tsv` | nDCG@k, Recall@k |
| STS | `data/sts/<name>/` | `meta.yaml`, `pairs.jsonl` | cosine Spearman |

Retrieval uses the BEIR layout. `corpus.jsonl` is one document per line (`_id`, `text`, optional `title`), `queries.jsonl` the same shape, and `qrels.tsv` the relevance judgements (`query-id`, `corpus-id`, `score`, where `0` is not relevant). STS is one pair per line: `sentence1`, `sentence2`, `score`.

`meta.yaml` needs a `language` that matches your `--languages` filter (`eng`, `cmn`, `zsm` and their aliases), plus a `revision` you bump whenever the data changes, otherwise MTEB serves the cached score.

Confirm it was discovered, then run only that folder:

```bash
uv run embbench list-datasets --languages zsm    # local rows show source=local

uv run embbench run \
  --model bce-embedding-base_v1 \
  --languages eng --task-types Retrieval \
  --no-include-mteb --task-names sop-handbook-v1
```

Full field reference, language aliases, mixing local with public tasks, and troubleshooting: [docs/custom-datasets.md](docs/custom-datasets.md).

## Add a model

Ask the harness which loader to use, then paste its output into `configs/models.yaml`:

```bash
uv run embbench check-model intfloat/multilingual-e5-large
```

It reports whether MTEB has the model registered and prints a ready-to-use config block.

| MTEB knows it | Use | Why |
|---|---|---|
| Yes | `loader: mteb` | Keeps the model's own query and document prompts |
| No | `loader: sentence_transformers` | Plain `encode()` |

This is not cosmetic. Instruction-aware models (Voyage, Harrier, Qwen3) are trained with a prefix on queries and documents. MTEB's wrapper applies them; a plain `encode()` does not, and reports a score below the model's real quality. `check-model` also warns about the two traps we hit: a vendor API extra that local weights do not need, and Hub code written for an older Transformers.

Then smoke it on one task before committing to the full run:

```bash
uv run embbench run --model multilingual-e5-large \
  --languages eng --task-types Retrieval \
  --no-include-mteb --task-names SciFact
```

Field reference, `trust_remote_code`, and fitting an 8GB card: [docs/adding-models.md](docs/adding-models.md).

## Architecture

How a run is sequenced, how nDCG is computed without Qdrant, and where files land: [docs/benchmarking.md](docs/benchmarking.md).

How to print Malay / public / local texts: [docs/inspect-datasets.md](docs/inspect-datasets.md).

Plain-language account of the first full run, including why Voyage needed Transformers patches and why encoding bars repeat: [docs/what-happened.md](docs/what-happened.md).

`results/` is gitignored. Only the empty directory is committed, so a fresh clone has somewhere to write. Regenerate with `uv run embbench run --model all` then `uv run embbench report`.

`run_job(JobSpec) -> JobResult` is the only execution path. The CLI and the FastAPI stub both build a `JobSpec`.

Phase 2 stub (not used for the GPU run):

```bash
uv run uvicorn embbench.api.main:app --port 8000
# POST /jobs          persist a spec (default) or ?execute=true
# GET  /jobs/{id}
# GET  /results
```

## Models

| id | role |
|---|---|
| bce-embedding-base_v1 | baseline |
| voyage-4-nano | predicted winner |
| harrier-oss-v1-0.6b | backup |
| Qwen3-Embedding-0.6B | popular reference |
| bge-m3 | dense-only hybrid check |

Malay STS does not exist in MTEB. The run logs that gap until you drop a folder into `data/sts/`.
