from __future__ import annotations

import pytest
import torch

from dflasher import hf


class FakeTokenizer:
    eos_token = "<eos>"
    bos_token = None

    def __init__(self) -> None:
        self.pad_token_id = None
        self._pad_token = None

    @property
    def pad_token(self):
        return self._pad_token

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value
        if value is not None:
            self.pad_token_id = 0


def test_dtype_from_name_rejects_unknown_dtype():
    assert hf.dtype_from_name("float16") is torch.float16
    assert hf.dtype_from_name("bf16") is torch.bfloat16
    with pytest.raises(ValueError, match="Unsupported torch dtype"):
        hf.dtype_from_name("not_a_dtype")
    with pytest.raises(ValueError, match="Unsupported torch dtype"):
        hf.dtype_from_name("Tensor")


def test_resolve_device_rejects_unavailable_requested_accelerator(monkeypatch):
    monkeypatch.setattr(hf.torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="CUDA"):
        hf.resolve_device("cuda")

    class FakeMps:
        @staticmethod
        def is_available():
            return False

    monkeypatch.setattr(hf.torch.backends, "mps", FakeMps())
    with pytest.raises(ValueError, match="MPS"):
        hf.resolve_device("mps")


def test_load_tokenizer_sets_pad_token_from_eos(monkeypatch):
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(
        hf.AutoTokenizer,
        "from_pretrained",
        lambda source_model, trust_remote_code=False: tokenizer,
    )

    result = hf.load_tokenizer("tiny", trust_remote_code=True)

    assert result.pad_token == "<eos>"
