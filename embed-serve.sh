#!/usr/bin/env bash
set -euo pipefail

args=(
    "${EMBED_SERVE}"
    --runner pooling
    --convert "${EMBED_CONVERT}"
    --port 8000
    --served-model-name "${EMBED_NAME}"
    --max-model-len "${EMBED_MAX_LEN}"
    --gpu-memory-utilization "${EMBED_GPU_MEMORY}"
    --max-num-seqs "${EMBED_MAX_REQ}"
    --dtype "${EMBED_DTYPE}"
    --trust-remote-code
)
if [[ -n "${EMBED_POOLER_CONFIG:-}" ]]; then
    args+=(--pooler-config "${EMBED_POOLER_CONFIG}")
fi
if [[ -n "${EMBED_HF_OVERRIDES:-}" ]]; then
    args+=(--hf-overrides "${EMBED_HF_OVERRIDES}")
fi
if [[ "${EMBED_ENFORCE_EAGER:-0}" == "1" ]]; then
    args+=(--enforce-eager)
fi
# Causal embedders (Qwen3 family) hang mid-corpus with these on: a long pooling
# request splits across scheduler steps and never emits its vector, so the GPU
# idles until the client times out. Bidirectional/encoder-only models
# (voyage, BCE, bge-m3) get prefix caching disabled by vLLM anyway.
if [[ "${EMBED_PREFIX_CACHE:-1}" == "0" ]]; then
    args+=(--no-enable-prefix-caching)
fi
if [[ -n "${EMBED_MAX_BATCHED_TOKENS:-}" ]]; then
    args+=(--max-num-batched-tokens "${EMBED_MAX_BATCHED_TOKENS}")
fi
exec vllm serve "${args[@]}"
