# What happened during this benchmark

This note is for people who watched the terminal and saw things like `create_causal_mask`, Transformers patches, and encoding batches looping for hours. It is not the score table. Scores live in `results/report.md`, which is generated locally by `uv run embbench report` and is not committed.

## What we were trying to do

Compare five embedding models on the same public tasks:

| Model | Role | Result |
|---|---|---|
| `bce-embedding-base_v1` | current production (baseline) | finished 20 tasks |
| `voyage-4-nano` | predicted winner | finished 20 tasks, after code fixes |
| `harrier-oss-v1-0.6b` | backup | finished 20 tasks |
| `Qwen3-Embedding-0.6B` | popular reference | finished 20 tasks |
| `bge-m3` | hybrid-search check | **did not run** (GPU out of memory) |

Tasks: retrieval (nDCG / Recall) and STS (semantic similarity) in English, Chinese, and Malay. Malay has retrieval only. MTEB has no Malay STS dataset.

The machine is an RTX 3070 with 8GB VRAM, so **only one model is loaded at a time**. When one finishes, the process exits so VRAM is fully released, then the next model starts.

## Why you saw encoding batches over and over

That progress bar (`Batches: 38% | 1200/3125`) is **not a bug and not the same job stuck**.

Each retrieval task has two encode steps:

1. Encode every document in that task’s corpus.
2. Encode every query.

MTEB then compares queries to documents in memory (exact search). That is the score. Qdrant is not used for scoring.

We ran **20 tasks per model**. FiQA alone has tens of thousands of documents, so one task can show `3125` batches. When that bar hits 100%, the next **task** starts and the bar goes back to 0%. That looks like “encoding again from the start.” It is a new dataset.

Then the **next model** does the same 20 tasks. So you see FiQA encode four times if four models finish: once for BCE, once for Harrier, once for Qwen, once for Voyage. The vectors are not reusable across models. Each model produces different numbers.

Voyage was also **retried** after it failed to load, so some of its batches ran more than once.

Rough picture:

```
Model A
  Task 1  encode corpus ████████  encode queries ██  score
  Task 2  encode corpus ████      encode queries ██  score
  ...
  Task 20
process exits (VRAM freed)
Model B
  Task 1  encode corpus ████████   ← this is why the bar restarts
  ...
```

## Why Transformers patches and “causal mask” showed up

This part is **only about Voyage**. BCE, Harrier, and Qwen did not need it.

Voyage-4-nano is not a normal BERT file. The Hugging Face repo ships custom Python (`modeling_qwen3_bidirectional.py`) that builds a Qwen3-based embedding network. Loading it requires `trust_remote_code=True` so that file actually runs on your machine.

That file was written against an **older Transformers API**. This environment has **Transformers 5.15** (Sentence-Transformers 6 requires Transformers 5, so we cannot downgrade to 4.x).

Two crashes came from that mismatch:

1. **`NoneType has no attribute '__name__'`**  
   On load, Transformers 5 registers the custom class and reads `model_class.config_class.__name__`. Voyage’s class leaves `config_class` as `None`. That is a one-line API change in Transformers, not a GPU problem.

2. **`create_causal_mask() got an unexpected keyword argument 'input_embeds'`**  
   “Causal mask” is the attention mask inside the Transformer (which tokens may look at which). Voyage’s file still calls `create_causal_mask(input_embeds=..., cache_position=...)`. Transformers 5 renamed `input_embeds` → `inputs_embeds` and dropped `cache_position` from that function. Again: old model code, new library. Not something we invented for the benchmark.

The patches in `src/embbench/core/encoders.py` do **not** change Voyage’s embedding recipe. They only:

- fill in `config_class` so AutoModel can register the custom class
- rename/drop keyword arguments so the mask function still runs

Without those, Voyage never produces a vector. With them, `encode_query("hello")` returns a 2048-d vector and MTEB can score it.

Harrier and Qwen also use instructions, but they are loaded through `mteb.get_model`, whose wrappers already match Transformers 5. BCE is a plain SentenceTransformer with no custom modeling file.

## Why Voyage was not loaded the same way as BCE

BCE: `SentenceTransformer("maidalun1020/bce-embedding-base_v1")` then `encode(text)`.

Voyage’s own model card says **not** to use a single `encode()` for retrieval. It wants:

- `encode_query(...)` → prepends `Represent the query for retrieving supporting documents: `
- `encode_document(...)` → prepends `Represent the document for retrieval: `

If we only called `encode()`, Voyage would still run, but the scores would not match how it was trained. MTEB’s wrapper (`SentenceTransformerEncoderWrapper`) is what calls query vs document encode during evaluation.

First we tried raw SentenceTransformer (same as BCE). Load crashed (`config_class`). Then `mteb.get_model("voyageai/voyage-4-nano")` refused because MTEB tags Voyage with a `voyageai` **API** extra we do not need for local weights. The working path is: same MTEB wrapper, skip the API extra, plus the two Transformers 5 shims above.

## What failed, and what we did about it

| What you saw | Cause | Action |
|---|---|---|
| Voyage dies in ~10s with `'NoneType' ... __name__` | Transformers 5 vs Voyage custom class | Patch AutoModel.register |
| Voyage loads, then every task errors on `create_causal_mask` | Transformers 5 vs Voyage mask call | Patch kwargs (alias + drop unknown) |
| `mteb.get_model` asks for `voyageai` extra | MTEB metadata meant for the Voyage **API** | Load the local wrapper instead of installing the API SDK |
| `bge-m3` CUDA OOM | 8GB card, model did not fit on load | Left failed. Not retried. |
| Qdrant `object has no attribute 'search'` | qdrant-client 1.19 removed `.search` | Code now uses `.query_points`. BCE ops already ran with in-memory fallback. |
| Encoding bar restarts many times | New task or new model | Expected |

## Timeline (compressed)

1. Install stack, cache HuggingFace weights on `/mnt/c`.
2. Smoke test: BCE on SciFact. Worked. ~0.82 GiB VRAM.
3. Full run, one subprocess per model:
   - BCE: 20/20.
   - Voyage: failed to load (Transformers). Orchestrator continued (fail-soft).
   - Harrier: 20/20 (slowest; 0.6B decoder).
   - Qwen: 20/20.
   - bge-m3: OOM, stop.
4. Voyage retried with `mteb.get_model` → blocked on unused `voyageai` extra.
5. Voyage retried with MTEB’s SentenceTransformer wrapper → loaded, then all 20 tasks failed on `create_causal_mask`.
6. Mask kwargs patched. Voyage rerun from scratch: **20/20 scored**.

## What the numbers actually mean (one paragraph)

Retrieval nDCG/Recall: “if the user asks a question, do the right passages land in the top 10 / top 30?” Higher is better. That is the metric that matters for RAG without a reranker.

STS: “do two sentences that humans marked as similar get similar vectors?” Useful sanity check, not a substitute for retrieval.

On **this** task set, Voyage leads English and Malay retrieval. Qwen leads Chinese retrieval and English STS. BCE is last on every retrieval language. That is why replacing the production model is still on the table; which replacement depends on whether Chinese retrieval is a hard requirement.

Peak VRAM while scoring: BCE ~0.8 GiB, Voyage ~1.3, Harrier ~1.3, Qwen ~2.2. All fine on 8GB except bge-m3.

## What we did *not* do

- Did not ingest your SOP/handbook PDFs. Public MTEB datasets only.
- Did not generate Malay STS. The slot is empty until you drop files into `data/sts/`.
- Did not score through Qdrant. Exact in-memory search only. Mixing ANN into nDCG would hide whether the model or the index was weak.
- Did not change Voyage’s prompts or architecture. The patches only make its own file runnable on Transformers 5.

## Where to look

| File | What it is |
|---|---|
| `results/report.md` | Scores, averages, takeaways (generated, not committed) |
| `20260819T102212Z-<model>/result.json` | Per-task numbers for one model |
| `src/embbench/core/encoders.py` | How models are loaded; Voyage patches live here |
| `configs/models.yaml` | Which loader each model uses |
| `src/embbench/evaluation/runner.py` | One job = one model, tasks in sequence |
