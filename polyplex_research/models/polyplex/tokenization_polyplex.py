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
"""Tokenizer class for Polyplex."""

from __future__ import annotations

from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_file": "tokenizer.json",
}

# The byte-level mark Qwen3 puts on a piece that opens a word, the byte of a space read through the byte map.
WORD_START = "Ġ"


class PolyplexTokenizer(Qwen2Tokenizer):
    r"""
    Constructs a Polyplex tokenizer, the byte-level BPE tokenizer of Qwen3 with the vocabulary of the duplex
    checkpoint kept beside it.

    The text backbone reads Qwen3's pieces, so the tokenizer is Qwen3's: the same pre-tokenization regex, the same
    byte map and the same merges. Ids past Qwen3's vocabulary are grouped into blocks, and `blocks` records where
    each block starts and ends so the rows a block owns can be found from the tokenizer alone.

    Args:
        vocab (`str` or `dict[str, int]`, *optional*):
            Piece mapped to its id, or the path of a `vocab.json`.
        merges (`str` or `list[str]`, *optional*):
            The BPE merges in rank order, or the path of a `merges.txt`.
        blocks (`dict`, *optional*):
            Block name mapped to its `[start, end)` id range.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            Token standing for a piece the vocabulary does not hold. Byte-level BPE never needs it.
        bos_token (`str`, *optional*):
            Token opening a sequence. Qwen3 opens none.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            Token closing a sequence.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            Token filling a batch to its width.
        split_special_tokens (`bool`, *optional*, defaults to `True`):
            Whether a control token's string inside text is cut as text rather than read as the control token.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab: str | dict[str, int] | None = None,
        merges: str | list[str] | None = None,
        blocks: dict | None = None,
        unk_token: str = "<|endoftext|>",
        bos_token: str | None = None,
        eos_token: str = "<|endoftext|>",
        pad_token: str = "<|endoftext|>",
        split_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
        **kwargs,
    ):
        super().__init__(
            vocab=vocab,
            merges=merges,
            blocks=blocks,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            split_special_tokens=split_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )
        self.blocks = {name: list(span) for name, span in (blocks or {}).items()}

    @classmethod
    def from_backend(cls, tokenizer, **kwargs) -> "PolyplexTokenizer":
        r"""
        Wraps an already built tokenizer, keeping the special tokens it was built with.

        Args:
            tokenizer (`PreTrainedTokenizerFast`):
                The tokenizer whose backend is adopted.

        Returns:
            `PolyplexTokenizer`: The wrapped tokenizer.
        """
        defaults = {
            "unk_token": str(tokenizer.unk_token) if tokenizer.unk_token else "<|endoftext|>",
            "bos_token": str(tokenizer.bos_token) if tokenizer.bos_token else None,
            "eos_token": str(tokenizer.eos_token) if tokenizer.eos_token else "<|endoftext|>",
            "pad_token": str(tokenizer.pad_token) if tokenizer.pad_token else "<|endoftext|>",
            "split_special_tokens": False,
        }
        defaults.update(kwargs)
        return cls(tokenizer_object=tokenizer.backend_tokenizer, **defaults)


__all__ = ["PolyplexTokenizer", "WORD_START"]
