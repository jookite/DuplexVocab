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
"""Processor class for Polyplex."""

from __future__ import annotations

from transformers.models.personaplex.processing_personaplex import PersonaPlexProcessor

from .tokenization_polyplex import PolyplexTokenizer


DEFAULT_NUM_CODEBOOKS = 8


class PolyplexProcessor(PersonaPlexProcessor):
    r"""
    Constructs a Polyplex processor, a PersonaPlex processor carrying the Polyplex text tokenizer.

    The released duplex repository ships a raw SentencePiece model rather than a `transformers` tokenizer, which
    the parent converts on the fly. A Polyplex checkpoint carries its own [`PolyplexTokenizer`], which this
    processor saves and loads.

    Args:
        feature_extractor (`FeatureExtractionMixin`, *optional*):
            Front end of the audio branch. Filled in from the default codec when omitted.
        tokenizer (`PolyplexTokenizer`, *optional*):
            Tokenizer of the text stream.
        audio_tokenizer (`MimiModel`, *optional*):
            Codec the audio branch encodes and decodes with.
        num_codebooks (`int`, *optional*, defaults to 8):
            Number of acoustic codebooks a frame carries.
    """

    tokenizer_class = "PolyplexTokenizer"

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        audio_tokenizer=None,
        num_codebooks: int = DEFAULT_NUM_CODEBOOKS,
        **kwargs,
    ):
        if tokenizer is not None and not isinstance(tokenizer, PolyplexTokenizer):
            # A tokenizer the parent converted from the raw SentencePiece model is wrapped in the Polyplex class.
            tokenizer = PolyplexTokenizer.from_backend(tokenizer)
        super().__init__(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            audio_tokenizer=audio_tokenizer,
            num_codebooks=num_codebooks,
            **kwargs,
        )


__all__ = ["DEFAULT_NUM_CODEBOOKS", "PolyplexProcessor"]
