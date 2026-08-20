# Custom tasks with custom datasets

Drop a folder into `data/`. The harness wraps it as an MTEB task and scores it the same way as public FiQA / STSBenchmark. No Python adapter, no `configs/tasks.yaml` edit.

Two kinds of local task:

| Kind | Folder | What it measures |
|---|---|---|
| Retrieval | `data/retrieval/<name>/` | nDCG@k, Recall@k (k = 10, 30) |
| STS | `data/sts/<name>/` | cosine Spearman correlation |

Public MTEB tasks stay in `configs/tasks.yaml`. Local folders are extra, not a replacement, unless you turn MTEB off (see [Run only the custom task](#run-only-the-custom-task)).

## 1. Create the folder

### Retrieval (BEIR layout)

```
data/retrieval/sop-handbook-v1/
  meta.yaml
  corpus.jsonl
  queries.jsonl
  qrels.tsv
```

`meta.yaml`:

```yaml
name: sop-handbook-v1
language: eng-Latn
revision: "2026-08-19"
description: Internal SOP handbook retrieval.
prompt: Given a question, retrieve the handbook passage that answers it.
```

`corpus.jsonl` — one document per line. `_id` / `id` and `text` / `contents` are accepted. `title` is optional.

```json
{"_id": "d1", "title": "Leave policy", "text": "Annual leave is 14 days."}
{"_id": "d2", "title": "VPN", "text": "Connect through the company VPN before accessing the intranet."}
```

`queries.jsonl` — `_id` / `id` and `text` / `query`.

```json
{"_id": "q1", "text": "How many days of annual leave do I get?"}
```

`qrels.tsv` — relevance judgements. Header row is skipped. Tab, comma, or space is fine. Three columns (`query-id`, `doc-id`, `score`) or TREC four columns (`qid`, `iter`, `docid`, `rel`):

```
query-id	corpus-id	score
q1	d1	1
```

Use score `0` for non-relevant and `≥1` for relevant. Retrieval main score is nDCG@10.

### STS (sentence pairs)

```
data/sts/malay-sts-v1/
  meta.yaml
  pairs.jsonl
```

`meta.yaml`:

```yaml
name: malay-sts-v1
language: zsm-Latn
min_score: 0
max_score: 5
revision: "2026-08-19"
description: Malay semantic textual similarity pairs.
```

`pairs.jsonl`:

```json
{"sentence1": "Kereta itu pantas.", "sentence2": "Mobil tersebut laju.", "score": 4.5}
```

`score` must sit in `[min_score, max_score]`. STS main score is cosine Spearman.

MTEB has no Malay STS. A folder under `data/sts/` with `language: zsm-Latn` (or `ms`, `msa`, `zlm`) fills that slot.

## 2. Language must match the run filter

`uv run embbench run --languages eng,cmn,zsm` only loads local folders whose `language` aliases to one of those codes.

| `--languages` value | Accepted in `meta.yaml` |
|---|---|
| `eng` | `eng`, `en`, `eng-Latn` |
| `cmn` | `cmn`, `zh`, `zho`, `cmn-Hans`, `zho-Hans` |
| `zsm` | `zsm`, `msa`, `zlm`, `ms`, `may`, `zsm-Latn`, `msa-Latn` |

If `language` is missing, the folder is tagged `und` and skipped. Folders without `meta.yaml` are ignored.

`name` in `meta.yaml` is the task id in results. It defaults to the folder name.

## 3. Confirm discovery

```bash
uv run embbench list-datasets
```

Local rows show `source=local`. Narrow the language list if needed:

```bash
uv run embbench list-datasets --languages zsm
uv run embbench list-datasets --languages eng
```

If the row is missing, the language in `meta.yaml` does not match `--languages`, or a required file (`pairs.jsonl` / `corpus.jsonl` / `queries.jsonl` / `qrels.tsv`) is absent. Discovery only requires `meta.yaml`; load fails later if the data files are missing.

## 4. Run

Local datasets are on by default (`--include-local`). A default `run` still also executes every public MTEB task for those languages. Use the flags below to avoid that.

### Run only the custom task

`--no-include-mteb` drops public FiQA / STSBenchmark / etc. `--task-names` must equal `name` in `meta.yaml`. `--languages` must still match.

```bash
# English retrieval folder only, baseline model
uv run embbench run \
  --model bce-embedding-base_v1 \
  --languages eng \
  --task-types Retrieval \
  --no-include-mteb \
  --task-names sop-handbook-v1

# Malay STS folder only
uv run embbench run \
  --model bce-embedding-base_v1 \
  --languages zsm \
  --task-types STS \
  --no-include-mteb \
  --task-names malay-sts-v1
```

Same dataset on every model (one subprocess each):

```bash
uv run embbench run \
  --model all \
  --languages eng \
  --task-types Retrieval \
  --no-include-mteb \
  --task-names sop-handbook-v1
```

### Custom + public MTEB

Omit `--no-include-mteb`. The local folder is appended after the public list for that language.

```bash
uv run embbench run --model bce-embedding-base_v1 --languages eng --task-types Retrieval
```

`--task-names` can mix public and local names:

```bash
uv run embbench run \
  --model bce-embedding-base_v1 \
  --languages eng \
  --task-types Retrieval \
  --task-names SciFact,sop-handbook-v1
```

### Compare models on the same custom set

Run each model with the **same** `--languages`, `--task-types`, `--task-names`, and `--no-include-mteb`. Then:

```bash
uv run embbench report
```

Per-job scores live in `results/<run-id>-<model-id>/result.json`. Local tasks have `"source": "local"`. `report.md` concatenates every job under `results/`, including earlier public MTEB runs, so compare on matching task names rather than the whole table.

Re-score after changing the files:

```bash
uv run embbench run ... --overwrite
```

## 5. Cache: bump `revision` when files change

MTEB caches scores under `results/mteb-cache/` keyed by dataset `revision`. If you edit `pairs.jsonl` / `corpus.jsonl` and keep the same `revision`, the old score is reused.

1. Change the data files.
2. Set a new `revision` in `meta.yaml` (date string is enough).
3. Re-run. Add `--overwrite` if the job folder already completed.

## 6. What this does not do

- No document ingestion or chunking. Pass already-chunked `corpus.jsonl`.
- No translation pipeline. Write the target language into the jsonl yourself.
- No hybrid sparse / ColBERT path. Local retrieval is dense exact search.
- Qdrant is serving-path only (p95, index size). Quality scores do not go through ANN.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `list-datasets` has no `local` row | Missing `meta.yaml`, or `language` does not match `--languages` |
| Task listed, then load error | Missing `pairs.jsonl` (STS) or `corpus.jsonl` / `queries.jsonl` / `qrels.tsv` (retrieval) |
| Job runs 20 public tasks instead of your folder | Forgot `--no-include-mteb` |
| `--task-names` runs nothing | Name must match `meta.yaml` `name`, not only the folder name |
| Scores identical after you edited the jsonl | `revision` not bumped, or `--overwrite` not passed |
| Malay STS warning in the log | No `data/sts/` folder whose language aliases to `zsm` |
