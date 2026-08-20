# Drop-in retrieval datasets (BEIR layout)

Each subdirectory is one task. Required files:

- `meta.yaml` — name, language, revision, optional prompt
- `corpus.jsonl` — `{"_id","title","text"}`
- `queries.jsonl` — `{"_id","text"}`
- `qrels.tsv` — `query-id<TAB>corpus-id<TAB>score`
