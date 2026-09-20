# Copyright 2026 The LatentForge Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared fixtures building a duplex model small enough to assert on exactly."""

import pytest
import torch
from transformers import PersonaPlexConfig, PersonaPlexForConditionalGeneration

from polyplex_research.models.polyplex import PolyplexConfig, PolyplexForConditionalGeneration
from polyplex_research.utils import quantization


BASE_VOCAB_SIZE = 48
VOCAB_SIZE = 64
AUDIO_VOCAB_SIZE = 16
NUM_CODEBOOKS = 4
PRETRAINED_SIZE = 40
INTERMEDIATE_SIZE = 48


def build_config(predicts_user_stream: bool = False) -> PersonaPlexConfig:
    r"""
    Args:
        predicts_user_stream (`bool`, *optional*, defaults to `False`):
            Whether the depth decoder carries heads for the user stream as well as the assistant one.
            The released checkpoints carry only the assistant stream.

    Returns:
        `PersonaPlexConfig`: A configuration of the real architecture, sized so that a test can enumerate
        every row of every table it asserts on.
    """
    depth_codebooks = 2 * NUM_CODEBOOKS if predicts_user_stream else NUM_CODEBOOKS
    return PersonaPlexConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        ffn_dim=64,
        audio_vocab_size=AUDIO_VOCAB_SIZE,
        num_codebooks=NUM_CODEBOOKS,
        max_position_embeddings=64,
        sliding_window=64,
        depth_decoder_config={
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "ffn_dim": 32,
            "num_codebooks": depth_codebooks,
        },
    )


def build_model(predicts_user_stream: bool = False, seed: int = 0):
    r"""
    Args:
        predicts_user_stream (`bool`, *optional*, defaults to `False`):
            Passed to [`build_config`].
        seed (`int`, *optional*, defaults to 0):
            Seed of the weight initialization.

    Returns:
        `PersonaPlexForConditionalGeneration`: A freshly initialized duplex model.
    """
    torch.manual_seed(seed)
    return PersonaPlexForConditionalGeneration(build_config(predicts_user_stream)).eval()


def build_polyplex_config(**overrides) -> PolyplexConfig:
    r"""
    Returns:
        `PolyplexConfig`: The configuration of `conftest.build_config` with a DualFFN of width 48 whose first
        40 neurons come from Helium.
    """
    values = dict(
        vocab_size=VOCAB_SIZE,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        intermediate_size=INTERMEDIATE_SIZE,
        duplex_pretrained_size=PRETRAINED_SIZE,
        audio_vocab_size=AUDIO_VOCAB_SIZE,
        num_codebooks=NUM_CODEBOOKS,
        max_position_embeddings=64,
        sliding_window=64,
        depth_decoder_config={
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "ffn_dim": 32,
            "num_codebooks": NUM_CODEBOOKS,
        },
    )
    values.update(overrides)
    return PolyplexConfig(**values)


def build_polyplex(seed: int = 0, **overrides) -> PolyplexForConditionalGeneration:
    torch.manual_seed(seed)
    return PolyplexForConditionalGeneration(build_polyplex_config(**overrides)).eval()


class StandInNVFP4(torch.Tensor):
    r"""
    A CPU stand-in for an NVFP4 tensor: values rounded to a grid of 1/12 of the global amax.

    The grid follows the global scale the way the real format's values do, so a scale read off other rows
    changes how a row is rounded here as well.
    """

    @staticmethod
    def quantize(tensor: torch.Tensor, amax=None) -> "StandInNVFP4":
        amax = tensor.float().abs().max() if amax is None else torch.tensor(float(amax))
        step = amax.clamp(min=1e-12) / 12
        quantized = (torch.round(tensor.float() / step) * step).as_subclass(StandInNVFP4)
        quantized.orig_dtype = tensor.dtype
        return quantized

    def dequantize(self, dtype):
        return self.as_subclass(torch.Tensor).to(dtype)


def stand_in_matmul(a, b, scale, bias):
    out = a.as_subclass(torch.Tensor) @ b.as_subclass(torch.Tensor)
    return out if bias is None else out + bias.float()


def stand_in_to_mxfp8(tensor):
    r"""
    A CPU stand-in for the MXFP8 cast: each block of 32 along the last axis is scaled by the power of two that brings
    its amax under the E4M3 maximum and cast to E4M3, and returned as the float values it holds.
    """
    pad = -tensor.shape[-1] % 32
    tensor = torch.nn.functional.pad(tensor.float(), (0, pad))
    blocks = tensor.reshape(*tensor.shape[:-1], -1, 32)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448)))
    return ((blocks / scale).to(torch.float8_e4m3fn).float() * scale).reshape(tensor.shape)


@pytest.fixture
def cpu_nvfp4(monkeypatch):
    monkeypatch.setattr(quantization, "_to_nvfp4", StandInNVFP4.quantize)
    monkeypatch.setattr(quantization, "_to_nvfp4_stochastic", StandInNVFP4.quantize)
    monkeypatch.setattr(quantization, "_addmm_nvfp4_dispatch", stand_in_matmul)
    monkeypatch.setattr(quantization, "_to_mxfp8", stand_in_to_mxfp8)


@pytest.fixture
def base_model():
    return build_model()


@pytest.fixture(params=[False, True], ids=["assistant_heads", "both_heads"])
def any_base_model(request):
    return build_model(request.param)


@pytest.fixture
def text_batch():
    r"""
    Returns:
        `dict`: A Phase 1 batch whose first two frames are an unscored prompt.
    """
    torch.manual_seed(1)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 6))
    text_labels = input_ids.clone()
    text_labels[:, :2] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones(2, 6, dtype=torch.bool),
        "text_labels": text_labels,
    }


@pytest.fixture
def audio_batch():
    r"""
    Returns:
        `dict`: A Phase 2 batch of acoustic codes with a silent user stream.
    """
    torch.manual_seed(2)
    codes = torch.randint(0, AUDIO_VOCAB_SIZE, (2, NUM_CODEBOOKS, 6))
    return {
        "input_ids": torch.randint(0, VOCAB_SIZE, (2, 6)),
        "attention_mask": torch.ones(2, 6, dtype=torch.bool),
        "assistant_audio_codes": codes,
        "user_audio_codes": torch.full_like(codes, AUDIO_VOCAB_SIZE),
        "audio_labels": codes.clone(),
    }
