# Drop-in STS datasets

Each subdirectory is one task. Required files:

- `meta.yaml` — name, language, min_score, max_score, revision
- `pairs.jsonl` — one `{"sentence1","sentence2","score"}` object per line

See the repo README for a full example. Malay STS belongs here once synthesized.
