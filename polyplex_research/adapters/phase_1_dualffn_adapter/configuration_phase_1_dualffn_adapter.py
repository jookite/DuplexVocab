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
"""Phase 1 DualFFN adapter configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Phase1DualFFNAdapterConfig:
    r"""
    This is the configuration class to store the configuration of a [`Phase1DualFFNAdapterModel`], the text
    adaptation of one language on a DualFFN model (SPEC S13.1).

    Args:
        language (`str`):
            Language whose reserved duplex FFN neurons are trained.
        duplex_slice_size (`int`, *optional*):
            Number of reserved neurons per layer handed to the language when it holds none yet. They are taken
            right after the slices other languages hold. Not needed for a language that already holds a slice.
        new_tokens (`list[int]`, *optional*):
            `[start, end)` ids of the language's own tokens, whose input embedding and LM head rows are trained.
            No row is trained when omitted.
        train_duplex_slice (`bool`, *optional*, defaults to `True`):
            Whether the language's duplex FFN neurons are trained.
        seed (`int`, *optional*, defaults to 0):
            Seed of the gate and up rows a newly handed slice starts from.
    """

    language: str
    duplex_slice_size: int | None = None
    new_tokens: list[int] | None = None
    train_duplex_slice: bool = True
    seed: int = 0


__all__ = ["Phase1DualFFNAdapterConfig"]
