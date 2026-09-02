# Benchmarking another model

Adding a model is a `configs/models.yaml` entry. No Python edit, unless the model ships broken custom code (see [Custom Hub code](#5-custom-hub-code-trust_remote_code)).

The only real decision is **which loader**, and the CLI answers it for you (except vLLM: that is `loader: openai_api`).

## 1. Ask the CLI

```bash
uv run embbench check-model intfloat/multilingual-e5-large
```

It reports whether MTEB knows the model, prints its wrapper, dimensions, and flags, warns about the traps below, and emits a paste-ready `models.yaml` block.

## 2. Pick the loader

| | `loader: mteb` | `loader: sentence_transformers` | `loader: openai_api` |
|---|---|---|---|
| When | MTEB has the model registered | It does not | Weights are already served (vLLM pooling or OpenAI-compatible `/v1/embeddings`) |
| How it encodes | MTEB's own wrapper, with the model's query/document prompts | plain `SentenceTransformer.encode()` | HTTP client; this process does not load the GPU |
| Use for | instruction-aware models | plain bi-encoders | production-style serving, models too large to load here |

**This choice changes the score, not just the plumbing.** Instruction-aware models are trained with a prefix on each side:

- Voyage: `Represent the query for retrieving supporting documents: ` vs `Represent the document for retrieval: `
- Harrier and Qwen3: `Instruct: <task>\nQuery: `

MTEB's wrapper calls `encode_query` and `encode_document` separately and applies those prefixes. A plain `encode()` runs the same weights with no prefix and reports a number below the model's real quality. So prefer `mteb` whenever the model is registered.

For a model with `use_instructions: false` (bge-m3, for example) both loaders produce the same vectors, so either is fine.

`check-model` says which applies. It is decided in `load_encoder`:

```20:31:src/embbench/core/encoders.py
def load_encoder(config: ModelConfig) -> Encoder:
    """Load exactly one encoder. Caller owns GPU lifetime (one process per model)."""
    encode_kwargs = {"batch_size": config.batch_size}
    if config.trust_remote_code:
        _patch_remote_code_config_class()
        _patch_create_causal_mask_alias()

    if config.loader == "mteb":
        model = _load_mteb_model(config)
```

## 3. Add the entry

```yaml
  - id: multilingual-e5-large
    hf_name: intfloat/multilingual-e5-large
    role: candidate  # omit it; candidate is the default
    loader: mteb
    trust_remote_code: false
    use_instructions: true
    max_seq_length: 512
    batch_size: 16
    description: Strong multilingual baseline, check against Qwen3.
```

| Field | Notes |
|---|---|
| `id` | What you pass to `--model` and what appears in every report |
| `role` | `baseline` for the incumbent (exactly one), `candidate` for everything else. Defaults to `candidate`. It is the only field the delta maths reads; put the reason you added the model in `description` |
| `loader` | `mteb`, `sentence_transformers`, or `openai_api` |
| `endpoint_url` | `openai_api` only. Defaults to `EMBBENCH_EMBEDDINGS_URL` |
| `use_chat_template` | `openai_api` only. `false` for pooling models with no chat template (bge-m3, BCE). `true` only for chat/VLM embedders |
| `instruction_template` | Prefix applied to queries, e.g. `Instruct: {instruction}\nQuery: `. Pair with `apply_instruction_to_documents` |
| `apply_instruction_to_documents` | `false` for Qwen3/Harrier (queries only), `true` for Voyage (both sides) |
| `trust_remote_code` | Required if the repo ships its own modeling `.py` (local loaders) |
| `use_instructions` | Documentation only; the wrapper decides the actual prompting |
| `batch_size` | Lower this first when you hit OOM |
| `max_seq_length` | 512 across the board so models are compared on equal footing |

## 4. Smoke it before the full run

One task, one language, a few minutes:

```bash
uv run embbench run \
  --model multilingual-e5-large \
  --languages eng --task-types Retrieval \
  --no-include-mteb --task-names SciFact
```

If nDCG@10 is near zero, the model loaded but is being prompted wrong. Check the loader. Then run everything:

```bash
uv run embbench run --model all --profile-ops
```

Existing models are skipped because they already finished, so only the new one actually runs. See [resuming](../README.md#run).

## 5. Custom Hub code (`trust_remote_code`)

Some repos ship their own modeling file that runs on your machine. Two failure modes, both seen with `voyage-4-nano`:

| Error | Cause | Handled by |
|---|---|---|
| `'NoneType' object has no attribute '__name__'` | Custom class leaves `config_class` as `None`; Transformers 5 reads it during `AutoModel.register` | `_patch_remote_code_config_class()` |
| `create_causal_mask() got an unexpected keyword argument 'input_embeds'` | Model code written against an older Transformers; the argument is now `inputs_embeds` and `cache_position` is gone | `_patch_create_causal_mask_alias()` |

Both patches live in `src/embbench/core/encoders.py` and apply only when `trust_remote_code: true`. Neither changes the model's embedding recipe; they only make its own file runnable on Transformers 5. Background: [what-happened.md](what-happened.md).

A new model with the same problem may need a similar shim. The pattern is to drop unknown keyword arguments using `inspect.signature` rather than pinning an old Transformers, because Sentence-Transformers 6 requires Transformers 5.

## 6. The `extra_requirements_groups` trap

MTEB sometimes tags an open-weights model with an extra meant for the vendor's **hosted API**:

```
Model voyageai/voyage-4-nano is missing required dependencies:
voyageai<2.0.0,>0.3.0; extra == "voyageai".
```

You do not need it to run local weights. Do not install it and do not set `MTEB_AUTO_INSTALL_EXTRAS=1`. `_load_mteb_model` already catches this and falls back to the same wrapper MTEB would have used:

```57:75:src/embbench/core/encoders.py
def _load_mteb_model(config: ModelConfig):
    """Use MTEB's registered wrapper so query/document prompts are applied.

    Some open-weight models (voyage-4-nano) are gated on an API extra we do not
    need. Fall back to the same SentenceTransformerEncoderWrapper MTEB would use.
    """
    import mteb

    try:
        return mteb.get_model(config.hf_name)
    except Exception as exc:
        if "missing required dependencies" not in str(exc):
            raise
```

`check-model` flags this before you run.

## 7. Serve with vLLM (`loader: openai_api`)

This repo scores. vLLM (or another OpenAI-compatible server) holds the weights.

```bash
vllm serve BAAI/bge-m3 --runner pooling --port 8000
```

```yaml
  - id: bge-m3-vllm
    hf_name: BAAI/bge-m3
    role: candidate
    loader: openai_api
    endpoint_url: http://127.0.0.1:8000
    use_chat_template: false
    use_instructions: false
    max_seq_length: 512
    batch_size: 16
    description: Same dense model as bge-m3, encoded through vLLM.
```

`hf_name` is the model id the server advertises (`/v1/models`). Pin `--max-model-len` (or equivalent) to 512 on the server so truncation matches the local runs.

vLLM `/v1/embeddings` has no query vs document route. Prefixes belong on the client (`OpenAIAPIEncodeWrapper`): Qwen/Harrier use `Instruct: {instruction}\nQuery: ` on queries only; Voyage uses the two “Represent the …” prefixes. BCE and dense bge-m3 take the raw text.

Do not add this row to a `--model all` run that also loads Voyage/Harrier in-process on the same GPU. Run the vLLM model alone:

```bash
uv run embbench run \
  --model bge-m3-vllm \
  --languages eng --task-types Retrieval \
  --no-include-mteb --task-names SciFact
```

Plain pooling embedders (BCE, bge-m3) match local `encode()`. Instruction models still need prefixes on the HTTP client (`instruction_template`, and `apply_instruction_to_documents: false` for Qwen/Harrier). Voyage’s query/document strings are applied in `encoders.py`.

`--profile-ops` still expects a local `encode(list[str])`. HTTP models may skip or fail that step; nDCG still comes from `mteb.evaluate`.

For the managed path (`launch.sh` option 3), you do not run `vllm serve` by hand. Serve flags per model live in `set_embed_flags()`; see the next section.

## 8. Per-model parameters

Every model needs two sets of parameters, and they live in different files:

| Layer | File | Controls |
|---|---|---|
| Client | `configs/models.yaml`, one entry per `id` | How embbench calls the server: batch size, query/document prefixes, sequence length, endpoint |
| Server | `set_embed_flags()` in [`launch.sh`](../launch.sh) | How vLLM loads the weights: dtype, pooling, CUDA graphs, prefix caching, token budget |

The server layer flows through env vars, not command-line arguments:

```
launch.sh set_embed_flags()  →  EMBED_* env  →  docker-compose.yml (embed)  →  embed-serve.sh  →  vllm serve
```

`embed-serve.sh` is bind-mounted into the container, so changing a flag needs a container recreate, never an image rebuild.

### Server knobs

Set these per model. Defaults are assigned above the `case`, so a branch only lists what differs.

| Env var | vLLM flag | Default | Set it when |
|---|---|---|---|
| `EMBED_DTYPE` | `--dtype` | `float16` | Qwen3-family weights ship bf16; use `bfloat16` |
| `EMBED_POOLER_CONFIG` | `--pooler-config` | `{"pooling_type":"CLS"}` | `LAST` for causal decoders, `MEAN` for Voyage, `CLS` for BERT-style |
| `EMBED_CONVERT` | `--convert` | `embed` | Rarely; `embed` is what pooling retrieval needs |
| `EMBED_HF_OVERRIDES` | `--hf-overrides` | unset | The repo's `config.json` names an architecture vLLM should not use |
| `EMBED_ENFORCE_EAGER` | `--enforce-eager` | `0` | The model misbehaves under `torch.compile` / CUDA graphs |
| `EMBED_PREFIX_CACHE` | `--no-enable-prefix-caching` when `0` | `1` | Causal pooling model stalls mid-corpus (see below) |
| `EMBED_MAX_BATCHED_TOKENS` | `--max-num-batched-tokens` | unset (vLLM decides) | A pooling request must not be split across scheduler steps |

These are shared by every model and come from `.env`, so do not put them in a branch:

| Env var | Meaning |
|---|---|
| `EMBED_PORT` | Host port for the pooling server (`8001`) |
| `EMBED_MAX_LEN` | `--max-model-len`, pinned to 512 so all models truncate alike |
| `EMBED_MAX_REQ` | `--max-num-seqs`, concurrent sequences in the engine |
| `EMBED_GPU_MEMORY` | `--gpu-memory-utilization`, a **fraction of the whole card** |
| `EMBED_HF_CACHE` | Host Hugging Face cache, mounted so weights download once |

`EMBED_GPU_MEMORY` is the usual source of "why is a 0.6B model using 10 GB". At `0.6` on a 16 GB card, vLLM reserves ~9.8 GB up front as a KV/activation pool regardless of weight size. Lower it if you want the embedder to sit lighter; it is not a measure of the model.

### Adding a preset

Add a branch keyed on the `id` from `configs/models.yaml`:

```bash
        my-new-embedder)
            EMBED_DTYPE=bfloat16
            EMBED_POOLER_CONFIG='{"pooling_type":"LAST"}'
            ;;
```

An id with no branch still runs; `launch.sh` prints `No vLLM pooling preset for '<id>'; using CLS float16.` That default is wrong for any instruction-tuned decoder, so treat the message as a prompt to add a branch.

Current presets:

| id | dtype | pooling | Other |
|---|---|---|---|
| bce-embedding-base_v1 | float16 | CLS | — |
| bge-m3 | float16 | CLS | dense only; no sparse/ColBERT override |
| voyage-4-nano | bfloat16 | MEAN | eager; `--hf-overrides` to the bidirectional architecture |
| harrier-oss-v1-0.6b | bfloat16 | LAST | prefix caching off; batched-token budget pinned |
| Qwen3-Embedding-0.6B | bfloat16 | LAST | prefix caching off; batched-token budget pinned |

### Check what actually got served

`launch.sh` echoes the resolved flags before starting the container:

```
embed flags: dtype=bfloat16 pooler={"pooling_type":"LAST"} eager=0 prefix_cache=0 max_batched_tokens=4096
```

Confirm the server agrees, since a stale container keeps its old flags:

```bash
curl -s http://127.0.0.1:8001/v1/models
docker logs embbench-embed-1 2>&1 | grep 'non-default args'
```

### Worked example: a model that stalls mid-corpus

Harrier encoded 204 query batches at 25 it/s, then documents crawled to a 4h45m ETA with `Request timeout (attempt 1/3)` from MTEB every five minutes. It was not slow compute:

```
Avg prompt throughput: 13791.4 tokens/s, Running: 1 reqs   ← healthy
Avg prompt throughput:     0.0 tokens/s, Running: 1 reqs   ← wedged, GPU at 0%
```

One request sat in the engine with the GPU idle, and only the client timeout cleared it. Queries (~40 tokens) never stalled; documents (~430 tokens) always did.

Only Harrier and Qwen3 are exposed, and vLLM's own gating explains why. `ModelConfig.is_prefix_caching_supported` (and the identical `is_chunked_prefill_supported`) turn both features **off** for bidirectional attention, and also off for causal models pooling with `MEAN`, `CLS`, or `STEP`. Causal attention plus `LAST` pooling is the one combination that leaves them **on**:

| id | Architecture | Attention | Pooling | Prefix caching + chunked prefill |
|---|---|---|---|---|
| bce-embedding-base_v1 | `XLMRobertaModel` | encoder_only | CLS | off |
| bge-m3 | `XLMRobertaModel` | encoder_only | CLS | off |
| voyage-4-nano | overridden to `VoyageQwen3BidirectionalEmbedModel` | encoder_only | MEAN | off |
| harrier-oss-v1-0.6b | `Qwen3Model` | decoder | LAST | **on** |
| Qwen3-Embedding-0.6B | `Qwen3ForCausalLM` | decoder | LAST | **on** |

A long `LAST`-pooling request that gets split across scheduler steps has no way to emit its vector, which matches the wedge exactly. Any future decoder-backbone embedder with `LAST` pooling lands in that same cell, so give it these two knobs from the start.

The fix was two server knobs, no change to the embedding recipe:

```bash
EMBED_PREFIX_CACHE=0
EMBED_MAX_BATCHED_TOKENS=$((EMBED_MAX_REQ * EMBED_MAX_LEN))
```

The budget equals the engine's concurrent prefill capacity, so no single request can be chunked. Reach for `--enforce-eager` only if a stall survives both, and confirm it with the logs above rather than assuming CUDA graphs.

## 9. Fitting an 8GB card

One model per subprocess, so only one set of weights is resident at a time. When a model still will not fit:

1. Lower `batch_size` (16 → 8 → 4).
2. Keep `dtype: float16`.
3. Lower `max_seq_length`, but then it is no longer comparable to the others.

`bge-m3` OOMs on load at 8GB even at `batch_size: 16`. Measured peaks on this card: BCE 0.82, Voyage 1.27, Harrier 1.35, Qwen3 2.22 GiB. The dashboard's Ops page plots these against the detected GPU budget.

A job that dies with CUDA OOM is recorded as `failed` and the orchestrator continues to the next model, so one bad model never sinks the run.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `Unknown model 'x'. Known ids: ...` | `id` in `models.yaml` does not match `--model` |
| Scores far below the model card | Instruction model loaded with `sentence_transformers`; switch to `mteb` |
| `missing required dependencies ... extra ==` | Vendor API extra; already handled, but verify `loader: mteb` |
| `'NoneType' object has no attribute '__name__'` | Needs `trust_remote_code: true` |
| `create_causal_mask() got an unexpected keyword` | Hub code predates Transformers 5; see section 5 |
| CUDA out of memory on load | Model does not fit at 8GB; lower `batch_size` or skip it |
| `openai_api` connection refused | vLLM is not up, or `endpoint_url` / `EMBBENCH_EMBEDDINGS_URL` is wrong |
| vLLM 400 about chat template | Set `use_chat_template: false` for non-chat pooling models |
| Model skipped on `run --model all` | It already completed at this scope; pass `--force` |
| `No vLLM pooling preset for '<id>'` | No branch in `set_embed_flags()`; CLS float16 is being guessed. See section 8 |
| Encoding stalls mid-corpus, GPU 0%, MTEB `Request timeout` | Causal pooling model with prefix caching on; set `EMBED_PREFIX_CACHE=0` and pin the token budget. See section 8 |
| VRAM far larger than the weights | `EMBED_GPU_MEMORY` is a fraction of the whole card, not the model size |
| Serve flag edit had no effect | Container still running the old flags; recreate it and re-check `non-default args` |
