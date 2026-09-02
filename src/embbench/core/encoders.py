"""Encoder protocol and loaders.

Instruction-aware models go through `mteb.get_model` so Harrier/Qwen keep their
per-task prompt dictionaries. Voyage uses the same MTEB SentenceTransformer
wrapper (encode_query / encode_document). BCE/BGE use raw SentenceTransformer.
`openai_api` is an HTTP client to a vLLM (or OpenAI-compatible) embeddings
server; this process does not load those weights onto the GPU.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from embbench.core.config import get_settings
from embbench.core.registry import ModelConfig


@runtime_checkable
class Encoder(Protocol):
    def encode(self, sentences: list[str], **kwargs: Any) -> Any: ...


def load_encoder(config: ModelConfig) -> Encoder:
    """Load exactly one encoder. Caller owns GPU lifetime (one process per model).

    `openai_api` does not occupy the GPU; the embeddings server does.
    """
    encode_kwargs = {"batch_size": config.batch_size}
    if config.loader == "openai_api":
        model = _load_openai_api_model(config)
        model._embbench_encode_kwargs = encode_kwargs
        return model

    if config.trust_remote_code:
        _patch_remote_code_config_class()
        _patch_create_causal_mask_alias()

    if config.loader == "mteb":
        model = _load_mteb_model(config)
        _pin_max_seq_length(model, config.max_seq_length)
        model._embbench_encode_kwargs = encode_kwargs
        return model

    from sentence_transformers import SentenceTransformer

    st_kwargs: dict[str, Any] = {"trust_remote_code": config.trust_remote_code}
    model = None
    last_error: Exception | None = None
    attempts = [
        dict(model_kwargs={"torch_dtype": _torch_dtype(config.dtype)}, **st_kwargs),
        dict(model_kwargs={"dtype": _torch_dtype(config.dtype)}, **st_kwargs),
        dict(**st_kwargs),
    ]
    for kwargs in attempts:
        try:
            model = SentenceTransformer(config.hf_name, **kwargs)
            break
        except Exception as exc:
            last_error = exc
    if model is None:
        raise RuntimeError(f"Failed to load {config.hf_name}: {last_error}") from last_error

    model.max_seq_length = config.max_seq_length
    _pin_max_seq_length(model, config.max_seq_length)
    return model


def _openai_wrapper_cls():
    try:
        from mteb.models import OpenAIAPIEncodeWrapper
    except ImportError:
        try:
            from mteb.models.openai_wrappers import OpenAIAPIEncodeWrapper
        except ImportError as exc:
            raise RuntimeError(
                "This mteb install has no OpenAIAPIEncodeWrapper. Upgrade mteb "
                "to use loader: openai_api (vLLM / OpenAI-compatible /v1/embeddings)."
            ) from exc
    return OpenAIAPIEncodeWrapper


def _voyage_instruction_template(instruction: str, prompt_type: Any = None) -> str:
    """Voyage has no /v1 query vs document endpoint; prepend the card prefixes here."""
    name = getattr(prompt_type, "value", prompt_type)
    if name in ("document", "passage"):
        return "Represent the document for retrieval: "
    return "Represent the query for retrieving supporting documents: "


def _load_openai_api_model(config: ModelConfig):
    """HTTP client to a pooling server. Weights stay in vLLM (or the remote API)."""
    settings = get_settings()
    endpoint = config.endpoint_url or settings.embeddings_url
    api_key = config.api_key or settings.embeddings_api_key or None
    kwargs: dict[str, Any] = {
        "endpoint_url": endpoint.rstrip("/"),
        "model_name": config.hf_name,
        "api_key": api_key,
        "max_length": config.max_seq_length,
        "use_chat_template": config.use_chat_template,
        "modalities": ["text"],
    }
    if config.use_instructions:
        kwargs["use_instructions"] = True
        # Wrapper only applies instructions when prompt_dict is not None.
        kwargs["prompt_dict"] = config.prompt_dict or {}
        kwargs["apply_instruction_to_documents"] = config.apply_instruction_to_documents
        if config.id == "voyage-4-nano" or config.hf_name.startswith("voyageai/"):
            kwargs["instruction_template"] = _voyage_instruction_template
        elif config.instruction_template:
            kwargs["instruction_template"] = config.instruction_template
    return _openai_wrapper_cls()(**kwargs)


def _load_mteb_model(config: ModelConfig):
    """Use MTEB's registered wrapper so query/document prompts are applied.
    """
    import mteb

    try:
        return mteb.get_model(config.hf_name)
    except Exception as exc:
        if "missing required dependencies" not in str(exc):
            raise
        from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

        return SentenceTransformerEncoderWrapper(
            model=config.hf_name,
            trust_remote_code=config.trust_remote_code,
        )


def _patch_remote_code_config_class() -> None:
    from transformers.models.auto.auto_factory import _BaseAutoModelClass

    if getattr(_BaseAutoModelClass.register, "_embbench_patched", False):
        return

    @classmethod
    def register(cls, config_class, model_class, exist_ok=False):
        current = getattr(model_class, "config_class", None)
        if current is None:
            try:
                model_class.config_class = config_class
            except Exception:
                pass
            current = getattr(model_class, "config_class", None)
        if current is not None and getattr(current, "__name__", None) != getattr(
            config_class, "__name__", None
        ):
            raise ValueError(
                "The model class you are passing has a `config_class` attribute that is not "
                f"consistent with the config class you passed (model has {current} and you "
                f"passed {config_class}. Fix one of those so they match!"
            )
        cls._model_mapping.register(config_class, model_class, exist_ok=exist_ok)

    register._embbench_patched = True  # type: ignore[attr-defined]
    _BaseAutoModelClass.register = register


def _patch_create_causal_mask_alias() -> None:
    """Voyage remote code calls an older create_causal_mask signature."""
    import inspect

    import transformers.masking_utils as masking_utils

    original = masking_utils.create_causal_mask
    if getattr(original, "_embbench_patched", False):
        return

    accepted = set(inspect.signature(original).parameters)
    aliases = {"input_embeds": "inputs_embeds"}

    def create_causal_mask(*args, **kwargs):
        for old, new in aliases.items():
            if old in kwargs and new not in kwargs:
                kwargs[new] = kwargs.pop(old)
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        return original(*args, **kwargs)

    create_causal_mask._embbench_patched = True  # type: ignore[attr-defined]
    masking_utils.create_causal_mask = create_causal_mask


def _torch_dtype(name: str):
    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(name, torch.float16)


def _pin_max_seq_length(model: Any, max_seq_length: int) -> None:
    for attr in ("max_seq_length", "max_seq_len"):
        if hasattr(model, attr):
            try:
                setattr(model, attr, max_seq_length)
            except Exception:
                pass
    inner = getattr(model, "model", None)
    if inner is not None and inner is not model:
        _pin_max_seq_length(inner, max_seq_length)
    st = getattr(model, "st_model", None) or getattr(model, "_model", None)
    if st is not None and st is not model and hasattr(st, "max_seq_length"):
        try:
            st.max_seq_length = max_seq_length
        except Exception:
            pass
