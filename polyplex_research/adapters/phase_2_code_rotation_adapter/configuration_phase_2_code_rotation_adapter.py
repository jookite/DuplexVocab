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
"""Code rotation adapter configuration, the Phase 2 comparison arm."""

from dataclasses import dataclass


@dataclass
class CodeRotationAdapterConfig:
    r"""
    Settings of the Phase 2 acoustic adapter.

    The three shape fields are read off the base model by
    [`CodeRotationAdapterConfig.from_base_model`], so a run only has to state the behavioural ones.

    Args:
        adapter_kind (`str`, *optional*, defaults to `"rotation"`):
            Which residual the adapter carries. `"rotation"` is the code space rotation the method rests
            on, bound with `parametrize`. `"tied"` is the temporal space residual attached with forward
            hooks, kept for the ablation that compares the two. The fields below marked as tied-only are
            ignored by the rotation kind.
        num_codebooks (`int`):
            Number of codebooks of one audio stream. The adapter holds one residual per stream codebook and
            the user heads of the depth decoder reuse the assistant residual.
        audio_vocab_size (`int`):
            Number of acoustic codes per codebook. The trailing special row of the embeddings is never touched.
        hidden_size (`int`):
            Dimension of the temporal transformer, shared by the audio input embeddings and the speaking skip term.
        gated (`bool`, *optional*, defaults to `True`):
            Whether the residual is factored as a zero initialized scalar gate times an orthogonally initialized
            matrix. When `False` the residual itself starts at zero and the orthogonal penalty acts on it directly.
            Tied kind only.
        tie_self_input (`bool`, *optional*, defaults to `True`):
            Whether the assistant stream tables carry the residual as well. The user stream tables always do.
            Both kinds read it, and both reach the same tables: the backbone audio embeddings of the assistant
            channel and, for the rotation kind, the depth decoder embeddings of the same codebooks. Turning it
            off is the ablation that leaves the model reading its own emitted codes under the frozen convention.
        ortho_lambda (`float`, *optional*, defaults to 0.01):
            Scale of the orthogonal penalty the training loop adds to its loss.
        nonneg_beta (`float`, *optional*, defaults to 10.0):
            Scale of the penalty on the negative entries of the rotation, which drives it to a
            permutation. Rotation kind only.
    """

    num_codebooks: int
    audio_vocab_size: int
    hidden_size: int
    adapter_kind: str = "rotation"
    gated: bool = True
    tie_self_input: bool = True
    ortho_lambda: float = 0.01
    nonneg_beta: float = 10.0

    def __post_init__(self):
        if self.adapter_kind not in ("rotation", "tied"):
            raise ValueError(f"`adapter_kind` must be 'tied' or 'rotation', got {self.adapter_kind}.")

    @classmethod
    def from_base_model(cls, model, **kwargs) -> "CodeRotationAdapterConfig":
        r"""
        Args:
            model (`PreTrainedModel`):
                The duplex model the adapter will be assembled onto.
            kwargs:
                Behavioural fields overriding the defaults.

        Returns:
            `CodeRotationAdapterConfig`: A configuration whose shapes match `model`.
        """
        depth_config = getattr(model.config, "depth_decoder_config", None)
        # The speaking skip term starts from the temporal hidden state, which the depth decoder knows as
        # its `input_size`. Both sides of the residual therefore live in the backbone dimension.
        hidden_size = getattr(depth_config, "input_size", None) or model.config.hidden_size
        return cls(
            num_codebooks=model.num_codebooks,
            audio_vocab_size=model.config.audio_vocab_size,
            hidden_size=hidden_size,
            **kwargs,
        )


__all__ = ["CodeRotationAdapterConfig"]
