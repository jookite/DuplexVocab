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
"""Polyplex model configuration."""

from huggingface_hub.dataclasses import strict
from transformers import PersonaPlexConfig


@strict
class PolyplexConfig(PersonaPlexConfig):
    r"""
    This is the configuration class to store the configuration of a [`PolyplexForConditionalGeneration`]. The
    audio streams, the depth decoder and the attention are PersonaPlex's; each backbone layer carries a DualFFN,
    the Helium FFN widened to `intermediate_size` beside an FFN grafted from a text LLM.

    Args:
        intermediate_size (`int`, *optional*, defaults to 12288):
            Width of both FFNs of a layer.
        duplex_pretrained_size (`int`, *optional*, defaults to 11264):
            Number of duplex FFN neurons taken from Helium. The neurons above it are reserved: the assembly
            writes them as zeros, and a Phase 1 adapter hands a slice of them to a language.
        duplex_ffn_languages (`dict`, *optional*):
            Language name mapped to the `[start, end)` slice of reserved neurons it was given.
        language_ffn_layers (`list[int]`, *optional*):
            For each backbone layer, the layer of the text LLM its language FFN was taken from.
        language_rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon of the normalizations the text LLM was trained with, before its FFNs and its LM head.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether the attention projections carry a bias.
        use_qk_norm (`bool`, *optional*, defaults to `False`):
            Whether the queries and keys are normalized per head before the rotary embedding. Helium has no
            such normalization.
        text_composition_size (`int`, *optional*):
            Number of base tokens each text token's shared composition coefficients cover (SPEC S12.3a). The text
            input table, the LM head and the depth decoder's text table then add to each stored row the combination
            of their first `base_vocab_size` rows the coefficients give. Without it, the stored rows are read as
            they are.
        base_vocab_size (`int`, *optional*):
            Number of leading text tokens taken from the duplex checkpoint. Their rows stay fixed in training and
            are the basis of the shared composition.
        appended_layers (`list[int]`, *optional*):
            Layers appended after the duplex checkpoint's own, copies of its last layers that start as the identity
            and train (SPEC S11.4a).
        language_scale (`bool`, *optional*, defaults to `False`):
            Whether each layer multiplies its language FFN output by a learned per-dimension scale (SPEC S11.2c).
        language_stream (`bool`, *optional*, defaults to `False`):
            Whether the language FFNs read and write a second residual stream in the text LLM's basis, joined to the
            backbone's by a fixed rotation per layer (SPEC S11.2d). Needs `language_scale`.
        language_stream_folded (`bool`, *optional*, defaults to `False`):
            Whether the rotation and scale joining the two streams are folded into the language FFN weights (SPEC
            S11.2h). The language stream then runs in the backbone's basis and holds no rotation of its own.
        language_stream_from_text (`bool`, *optional*, defaults to `True`):
            Whether the language stream starts from the text embeddings alone (SPEC S11.2e). `False` starts it from
            the whole input, text and audio embeddings, as checkpoints trained under SPEC S11.2d read it.
        language_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the language stream starts from a text table of its own in the text LLM's basis (SPEC S11.2f),
            in place of the text embeddings moved by the first layer's rotation.
        language_lora_rank (`int`, *optional*, defaults to 0):
            Rank of the low rank corrections beside each language FFN projection (SPEC S11.2g). 0 adds none.
        layer_types (`list[str]`, *optional*):
            Attention type of each layer. Every layer is a sliding window layer when `sliding_window` is set.
    """

    model_type = "polyplex"
    base_model_tp_plan = {}

    intermediate_size: int = 12288
    duplex_pretrained_size: int = 11264
    duplex_ffn_languages: dict | None = None
    language_ffn_layers: list[int] | None = None
    language_rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    use_qk_norm: bool = False
    text_composition_size: int | None = None
    base_vocab_size: int | None = None
    appended_layers: list[int] | None = None
    language_scale: bool = False
    language_stream: bool = False
    language_stream_folded: bool = False
    language_stream_from_text: bool = True
    language_embeddings: bool = False
    language_lora_rank: int = 0
    layer_types: list[str] | None = None

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            kind = "sliding_attention" if self.sliding_window is not None else "full_attention"
            self.layer_types = [kind] * self.num_hidden_layers
        self.duplex_ffn_languages = {
            name: list(bounds) for name, bounds in (self.duplex_ffn_languages or {}).items()
        }
        self.appended_layers = list(self.appended_layers or [])
        if self.language_stream and not self.language_scale:
            raise ValueError("`language_stream` needs `language_scale`, the scale its output enters the backbone with.")
        if not 0 < self.duplex_pretrained_size <= self.intermediate_size:
            raise ValueError(
                f"`duplex_pretrained_size` {self.duplex_pretrained_size} must lie in (0, "
                f"`intermediate_size` {self.intermediate_size}]."
            )
        taken = []
        for name, (start, end) in self.duplex_ffn_languages.items():
            if not self.duplex_pretrained_size <= start < end <= self.intermediate_size:
                raise ValueError(
                    f"Language {name!r} has neurons [{start}, {end}) outside the reserved range "
                    f"[{self.duplex_pretrained_size}, {self.intermediate_size})."
                )
            if any(start < other_end and other_start < end for other_start, other_end in taken):
                raise ValueError(f"Language {name!r} overlaps the reserved neurons of another language.")
            taken.append((start, end))
        if self.language_ffn_layers is not None and len(self.language_ffn_layers) != self.num_hidden_layers:
            raise ValueError(
                f"`language_ffn_layers` names {len(self.language_ffn_layers)} layers for a backbone of "
                f"{self.num_hidden_layers}."
            )
        if self.text_composition_size and not self.base_vocab_size:
            raise ValueError("A shared text composition needs `base_vocab_size`, the tokens it is written over.")
        super().__post_init__(**kwargs)

    @property
    def reserved_end(self) -> int:
        r"""
        Returns:
            `int`: The first reserved neuron no language holds yet.
        """
        return max([end for _, end in self.duplex_ffn_languages.values()], default=self.duplex_pretrained_size)


__all__ = ["PolyplexConfig"]
