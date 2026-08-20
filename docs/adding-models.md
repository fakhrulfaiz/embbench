# Benchmarking another model

Adding a model is a `configs/models.yaml` entry. No Python edit, unless the model ships broken custom code (see [Custom Hub code](#5-custom-hub-code-trust_remote_code)).

The only real decision is **which loader**, and the CLI answers it for you.

## 1. Ask the harness

```bash
uv run embbench check-model intfloat/multilingual-e5-large
```

It reports whether MTEB knows the model, prints its wrapper, dimensions, and flags, warns about the traps below, and emits a paste-ready `models.yaml` block.

## 2. Pick the loader

| | `loader: mteb` | `loader: sentence_transformers` |
|---|---|---|
| When | MTEB has the model registered | It does not |
| How it encodes | MTEB's own wrapper, with the model's query/document prompts | plain `SentenceTransformer.encode()` |
| Use for | instruction-aware models | plain bi-encoders |

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
    role: candidate
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
| `role` | Free text. `baseline` is special: the dashboard measures every other model against it, so exactly one model should carry it |
| `loader` | `mteb` or `sentence_transformers` |
| `trust_remote_code` | Required if the repo ships its own modeling `.py` |
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

## 7. Fitting an 8GB card

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
| Model skipped on `run --model all` | It already completed at this scope; pass `--force` |
