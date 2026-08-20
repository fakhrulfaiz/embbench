# How to look at a dataset

Two places data can come from:

| Source | Where | How you inspect it |
|---|---|---|
| Public MTEB | HuggingFace Hub, cached under `/mnt/c/ml-cache/huggingface/datasets` | `list-datasets`, then `datasets.load_dataset` |
| Local drop-in | `data/retrieval/<name>/` or `data/sts/<name>/` | open the jsonl / tsv, or `list-datasets` |

The harness does not copy Hub datasets into `data/`. `data/` is only for folders you add. Diagrams of the run path: [benchmarking.md](benchmarking.md).

## 1. See what a run would load

```bash
uv run embbench list-datasets
uv run embbench list-datasets --languages zsm
uv run embbench list-datasets --languages eng --include-heavy
```

`source=mteb` is public. `source=local` is a folder under `data/`. If a local row is missing, `meta.yaml` `language` does not match `--languages`.

## 2. Public MTEB (including Malay)

Always call `bootstrap_env()` first so Hub files land on `/mnt/c`, not `~/.cache/huggingface`.

### Malay retrieval

```bash
uv run python - <<'PY'
from embbench.core.config import bootstrap_env
bootstrap_env()
from datasets import load_dataset

# Belebele: 900 Malay reading-comprehension rows
bele = load_dataset("mteb/belebele", name="zsm_Latn", split="test")
print("Belebele", len(bele), bele.column_names)
print("Q:", bele[0]["question"])
print("D:", bele[0]["flores_passage"][:400])

# WebFAQ: Malay FAQ queries + corpus + qrels
q = load_dataset("mteb/WebFAQRetrieval", name="msa-queries", split="test")
c = load_dataset("mteb/WebFAQRetrieval", name="msa-corpus", split="test")
r = load_dataset("mteb/WebFAQRetrieval", name="msa-qrels", split="test")
print("WebFAQ queries", len(q), "corpus", len(c), "qrels", len(r))
print("Q:", q[0])
print("qrel:", r[0])
PY
```

Hub configs:

| Task | Config / subset | Columns you care about |
|---|---|---|
| BelebeleRetrieval | `zsm_Latn` | `question`, `flores_passage` |
| WebFAQRetrieval | `msa-queries`, `msa-corpus`, `msa-qrels` | `id`/`text`, `query-id`/`corpus-id`/`score` |
| MKQARetrieval | `msa` (heavy) | same BEIR-style split as other MKQA langs |

### English / Chinese example

```bash
uv run python - <<'PY'
from embbench.core.config import bootstrap_env
bootstrap_env()
import mteb

task = mteb.get_task("SciFact")  # or FiQA2018, STSBenchmark, CovidRetrieval, ...
task.load_data()
print(type(task.dataset), getattr(task, "hf_subsets", None))

# Retrieval: nested {subset: {split: RetrievalSplitData}}
ds = task.dataset
if isinstance(ds, dict):
    subset = next(iter(ds))
    split = next(iter(ds[subset]))
    payload = ds[subset][split]
    queries = getattr(payload, "queries", None) or payload.get("queries")
    corpus = getattr(payload, "corpus", None) or payload.get("corpus")
    print("subset", subset, "split", split)
    print("n queries", len(queries) if queries is not None else None)
    print("first query", next(iter(queries)))
PY
```

`mteb.get_task("BelebeleRetrieval", languages=["zsm"])` is what the harness uses. Belebele also keeps `zsm_Latn-eng_Latn` pairs, so a raw `load_dataset(..., "zsm_Latn")` is the simplest way to read Malay text.

## 3. Local drop-in folders

```bash
ls data/retrieval data/sts
```

Retrieval:

```bash
head -n 3 data/retrieval/<name>/corpus.jsonl
head -n 3 data/retrieval/<name>/queries.jsonl
head -n 5 data/retrieval/<name>/qrels.tsv
cat data/retrieval/<name>/meta.yaml
```

STS:

```bash
head -n 5 data/sts/<name>/pairs.jsonl
cat data/sts/<name>/meta.yaml
```

`list-datasets` only sees a folder if `meta.yaml` exists and `language` aliases to `eng`, `cmn`, or `zsm`.

## 4. After a run: predictions, not the corpus

MTEB writes ranked doc ids, not the original passages.

```
results/<run-id>-<model>/predictions/<TaskName>_predictions.json
results/<run-id>-<model>/result.json
results/mteb-cache/results/<hf_org>__<model>/<revision>/<TaskName>.json
```

`result.json` has nDCG / Recall / Spearman. To read the Malay (or English) sentences again, go back to step 2 or 3.

## 5. Cache location

| What | Path |
|---|---|
| Model weights | `/mnt/c/ml-cache/huggingface/hub` |
| Public datasets | `/mnt/c/ml-cache/huggingface/datasets` |
| MTEB score cache | `results/mteb-cache/` |
| Job outputs | `results/<job-id>/` |

If `list-datasets` shows a task but `load_dataset` errors, check `.env`: `HF_HOME` must start with `/mnt/c`.
