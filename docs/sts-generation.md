# STS generation (planning)

Planning notes for the **generation service**, not for embbench. Chunks already live in Postgres (`embbench`, profile `bce_500_50_min100`). Retrieval chunking stays **~500 tokens / 50-token overlap**. STS is a separate labelling job on those rows. Output is `pairs.jsonl` (`sentence1`, `sentence2`, `score`) as in [dataset-store.md](dataset-store.md) and [custom-datasets.md](custom-datasets.md).

Chunk quality (letterhead, empty tables) is **out of scope for this phase**. Use current rows as seeds. A keep/skip filter can come later.

embbench only scores the export. It does not generate pairs or open Postgres.

## Problem

SOP/PDF windows are not paraphrases of each other. The same letterhead and markdown pipes dominate TF-IDF. Neighbours often share a **header or a 50-token overlap**, not the same rule.

Scores 4.0 and 5.0 almost never exist as two distinct chunks. They must be **written from one seed**.

## What we keep vs what we generate

| Job | Input | Output |
|---|---|---|
| Chunker | Extracted markdown | ~500 / 50, profile `bce_500_50_min100` → `chunks` |
| STS | One chunk + target score | `sentence1`, `sentence2`, `score` |
| Retrieval questions | Chunks | Separate bank (`queries.jsonl` / qrels) |

Do not re-chunk on headings for RAG. Do not use two raw windows as a 4–5 pair.

## Pair construction

### Scores 3.0 / 4.0 / 5.0 — both sides from the LLM

1. Take one chunk as **seed** (`source_chunk_id`, e.g. `b7bdd4aa-1135-4d10-9341-41d5ac716e6f`).
2. Prompt: extract a short **gist** (2–8 sentences of actual procedure; no letterhead, no empty tables).
3. Emit JSON: `{"sentence1", "sentence2", "score"}`.
   - `sentence1` = gist of the seed (entailed by the chunk).
   - `sentence2` = generated at the target score.
4. `pair_kind`: `paraphrase` (5) or `generated` (3–4).

| Score | `sentence2` rule |
|---|---|
| 5.0 | Same meaning; paraphrase only |
| 4.0 | Same meaning; at most one extra or omitted detail |
| 3.0 | Same SOP **topic**, **different** step or adjacent rule (leave vs sick leave; proforma vs PI approval). Not a paraphrase |

Reject if either side still contains `Page N of`, `SOP-Number`, or a pipe-only table.

### Scores 0.0 / 1.0 / 2.0 — two cleaned passages or two chunks

| Score | Source |
|---|---|
| 2.0 | Same `doc_id`, **non-adjacent** windows (skip the 50-token overlap partner) |
| 1.0 | Same doc, weak overlap, or TF-IDF on **filtered** text after header strip |
| 0.0 | Different `doc_id`, or random chunks |

TF-IDF / cosine is allowed **only** in this band. It must not be the source of 3–5.

`pair_kind`: `chunk_chunk`. Optional: LLM still rewrites each side into a short gist so `pairs.jsonl` is not 500-token header soup. Lineage keeps both chunk ids if needed; export still uses `sentence1` / `sentence2` strings.

## Target mix

Generate **counts**, do not hope mined pairs land on a score.

| Score | Share |
|---|---|
| 0.0 | 10% |
| 1.0 | 20% |
| 2.0 | 20% |
| 3.0 | 20% |
| 4.0 | 20% |
| 5.0 | 10% |

`score` must sit in `[min_score, max_score]` in STS `meta.yaml` (default 0–5).

## Example (procurement SOP seed)

Seed gist: investigator obtains a USD/TZS proforma, checks type and count, includes bank/SWIFT.

- **5.0** — rewrite of that sentence only.
- **4.0** — same, plus “compare about three quotes when possible.”
- **3.0** — PI checks the pack against the grant budget (later step, same SOP).
- **0.0** — gist from a different document (not procurement).

## Out of scope for this job

- Embedding models, Qdrant, nDCG
- Changing the 500 / 50 chunker
- Cleaning extractor markdown (deferred)
- Using extractor `download_url` as long-term storage
- Asking the LLM to **label** two raw page-slices as 4.0 or 5.0
- Committing database passwords; use env on the generation host

## Export

`data/sts/<name>/pairs.jsonl` plus `meta.yaml` (`name`, `language`, `revision`). Bump `revision` when pairs change. embbench: `--task-types STS --no-include-mteb --task-names <name>`.
