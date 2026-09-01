"""openai_api loader talks to an embeddings HTTP server; no GPU in this process."""

from __future__ import annotations

from types import SimpleNamespace

from embbench.core.encoders import load_encoder
from embbench.core.registry import ModelConfig


class _FakeOpenAIWrapper:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _patch_openai(monkeypatch, captured: dict):
    def fake_cls():
        def ctor(**kwargs):
            captured.update(kwargs)
            return _FakeOpenAIWrapper(**kwargs)

        return ctor

    monkeypatch.setattr("embbench.core.encoders._openai_wrapper_cls", fake_cls)
    monkeypatch.setattr(
        "embbench.core.encoders.get_settings",
        lambda: SimpleNamespace(
            embeddings_url="http://127.0.0.1:8000",
            embeddings_api_key=None,
        ),
    )


def test_openai_api_loader_does_not_import_sentence_transformers(monkeypatch):
    captured: dict = {}
    _patch_openai(monkeypatch, captured)

    cfg = ModelConfig(
        id="bge-m3-vllm",
        hf_name="BAAI/bge-m3",
        role="candidate",
        loader="openai_api",
        endpoint_url="http://127.0.0.1:8000/",
        use_chat_template=False,
        max_seq_length=512,
        batch_size=16,
    )
    model = load_encoder(cfg)
    assert isinstance(model, _FakeOpenAIWrapper)
    assert captured["endpoint_url"] == "http://127.0.0.1:8000"
    assert captured["model_name"] == "BAAI/bge-m3"
    assert captured["max_length"] == 512
    assert captured["use_chat_template"] is False
    assert captured["modalities"] == ["text"]
    assert captured.get("use_instructions") is None


def test_openai_api_passes_instruction_template(monkeypatch):
    captured: dict = {}
    _patch_openai(monkeypatch, captured)

    cfg = ModelConfig(
        id="instruct-vllm",
        hf_name="Qwen/Qwen3-Embedding-0.6B",
        role="candidate",
        loader="openai_api",
        endpoint_url="http://127.0.0.1:8000",
        use_instructions=True,
        instruction_template="Instruct: {instruction}\nQuery: ",
    )
    load_encoder(cfg)
    assert captured["use_instructions"] is True
    assert "{instruction}" in captured["instruction_template"]


def test_openai_api_uses_settings_url_when_endpoint_omitted(monkeypatch):
    captured: dict = {}
    _patch_openai(monkeypatch, captured)

    cfg = ModelConfig(
        id="remote",
        hf_name="BAAI/bge-m3",
        role="candidate",
        loader="openai_api",
    )
    load_encoder(cfg)
    assert captured["endpoint_url"] == "http://127.0.0.1:8000"
