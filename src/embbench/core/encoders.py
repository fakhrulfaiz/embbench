"""Encoder protocol and loaders.

Instruction-aware models go through `mteb.get_model` so Harrier/Qwen keep their
per-task prompt dictionaries. Voyage uses the same MTEB SentenceTransformer
wrapper (encode_query / encode_document). BCE/BGE use raw SentenceTransformer.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from embbench.core.registry import ModelConfig


@runtime_checkable
class Encoder(Protocol):
    def encode(self, sentences: list[str], **kwargs: Any) -> Any: ...


def load_encoder(config: ModelConfig) -> Encoder:
    """Load exactly one encoder. Caller owns GPU lifetime (one process per model)."""
    encode_kwargs = {"batch_size": config.batch_size}
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
        from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

        return SentenceTransformerEncoderWrapper(
            model=config.hf_name,
            trust_remote_code=config.trust_remote_code,
        )


def _patch_remote_code_config_class() -> None:
    """Transformers 5 crashes on remote models whose `config_class` is None.

    Voyage-4-nano's Qwen3BidirectionalModel inherits PreTrainedModel.config_class
    as None. AutoModel.register then does `None.__name__`. Fill it in before the check.
    """
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
