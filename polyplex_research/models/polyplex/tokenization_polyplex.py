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

import unicodedata

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import Unigram
from transformers.tokenization_utils_tokenizers import PreTrainedTokenizerFast


VOCAB_FILES_NAMES = {"tokenizer_file": "tokenizer.json"}

WORD_START = "▁"

# Scripts read off a character's Unicode name, in the order their blocks follow the base block.
NAMED_SCRIPTS = (
    "LATIN", "CYRILLIC", "GREEK", "ARMENIAN", "GEORGIAN", "HEBREW", "ARABIC", "DEVANAGARI", "BENGALI", "GURMUKHI",
    "GUJARATI", "ORIYA", "TAMIL", "TELUGU", "KANNADA", "MALAYALAM", "SINHALA", "THAI", "LAO", "KHMER", "MYANMAR",
)
BLOCK_ORDER = ("base", *(script.lower() for script in NAMED_SCRIPTS), "hangul", "han", "kana", "other", "mixed")

# How far below the lowest text piece a special token is scored. A special token's string is a piece of the Unigram
# model, and this keeps the segmentation from ever choosing it for text.
SPECIAL_SCORE_GAP = 1000.0


def script_of(character: str) -> str | None:
    r"""
    Args:
        character (`str`):
            One character.

    Returns:
        `str` or `None`: The script the character is written in, `None` for digits, punctuation, symbols, spaces
        and control characters.
    """
    code = ord(character)
    if 0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF or 0x3130 <= code <= 0x318F:
        return "hangul"
    if 0x3040 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF:
        return "kana"
    if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0xF900 <= code <= 0xFAFF or 0x20000 <= code <= 0x2FFFF:
        return "han"
    if character.isascii():
        return "latin" if character.isalpha() else None
    name = unicodedata.name(character, "")
    for script in NAMED_SCRIPTS:
        if name.startswith(script):
            return script.lower()
    if unicodedata.category(character)[0] in "PSZNC":
        return None
    return "other"


def block_of(piece: str) -> str:
    r"""
    Args:
        piece (`str`):
            A token string, its word-start mark included.

    Returns:
        `str`: The block the piece belongs to: `"base"` when it holds no letter of any script, the script when it
        holds one, `"han"` or `"kana"` for Han and kana together, and `"mixed"` for several scripts.
    """
    scripts = {script_of(character) for character in piece.replace(WORD_START, "")} - {None}
    if not scripts:
        return "base"
    if scripts <= {"han", "kana"}:
        return "kana" if "kana" in scripts else "han"
    if len(scripts) == 1:
        return next(iter(scripts))
    return "mixed"


def sentencepiece_pieces(model_file: str) -> list:
    r"""
    Reads a SentencePiece model in id order.

    Args:
        model_file (`str`):
            Path of the `.model` file.

    Returns:
        `list[tuple[str, float, str]]`: Each piece with its score and its kind, one of `"unknown"`, `"control"`,
        `"byte"` and `"normal"`.
    """
    import sentencepiece

    model = sentencepiece.SentencePieceProcessor(model_file=model_file)
    pieces = []
    for index in range(model.get_piece_size()):
        if model.is_unknown(index):
            kind = "unknown"
        elif model.is_control(index):
            kind = "control"
        elif model.is_byte(index):
            kind = "byte"
        else:
            kind = "normal"
        pieces.append((model.id_to_piece(index), model.get_score(index), kind))
    return pieces


class PolyplexTokenizer(PreTrainedTokenizerFast):
    r"""
    Constructs a Polyplex tokenizer, a SentencePiece Unigram tokenizer with byte fallback over the Helium, Qwen3 and
    added vocabulary (SPEC S12.1, S12.1a).

    Helium's tokens keep their ids, so `EPAD` is 0 and `PAD` is 3. The ids after them are grouped into blocks, the
    base block first and then one block per script; `blocks` records where each block starts and ends.

    Args:
        vocab (`list[tuple[str, float]]`, *optional*):
            Every piece in id order with its Unigram score. The tokenizer is built from it: NFC, the spaces turned
            into `▁` with one in front, Unigram segmentation with byte fallback.
        blocks (`dict`, *optional*):
            Block name mapped to its `[start, end)` id range.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            Token standing for a piece the vocabulary does not hold.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            Token opening a sequence.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            Token closing a sequence.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            Token filling a batch to its width.
        split_special_tokens (`bool`, *optional*, defaults to `True`):
            Whether a control token's string inside text is cut as text rather than read as the control token.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    model = Unigram

    def __init__(
        self,
        vocab: list | None = None,
        blocks: dict | None = None,
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: str = "<pad>",
        split_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
        **kwargs,
    ):
        if vocab is not None:
            vocab = [(piece, score) for piece, score in vocab]
            unk_id = next(index for index, (piece, _) in enumerate(vocab) if piece == str(unk_token))
            self._tokenizer = Tokenizer(Unigram(vocab, unk_id=unk_id, byte_fallback=True))
            # The layout `transformers` gives Helium's SentencePiece model, with NFC in front (SPEC S12.1).
            self._tokenizer.normalizer = normalizers.Sequence(
                [normalizers.NFC(), normalizers.Replace(" ", WORD_START)]
            )
            self._tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
                replacement=WORD_START, prepend_scheme="first", split=False
            )
            self._tokenizer.decoder = decoders.Sequence(
                [decoders.Replace(WORD_START, " "), decoders.ByteFallback(), decoders.Fuse(), decoders.Strip(" ", 1, 0)]
            )
        super().__init__(
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
            "unk_token": str(tokenizer.unk_token) if tokenizer.unk_token else "<unk>",
            "bos_token": str(tokenizer.bos_token) if tokenizer.bos_token else "<s>",
            "eos_token": str(tokenizer.eos_token) if tokenizer.eos_token else "</s>",
            "pad_token": str(tokenizer.pad_token) if tokenizer.pad_token else "<pad>",
            "split_special_tokens": False,
        }
        defaults.update(kwargs)
        return cls(tokenizer_object=tokenizer.backend_tokenizer, **defaults)

    @classmethod
    def from_vocabulary(cls, base_pieces: list, pieces: dict, control_tokens: list | tuple = ()) -> "PolyplexTokenizer":
        r"""
        Builds the tokenizer of the final vocabulary (SPEC S12.1a).

        The base vocabulary keeps its ids. The control tokens come right after it, then every other piece grouped
        into blocks: the base block (pieces holding no letter) first, then one block per script in
        `BLOCK_ORDER`, each ordered by score from the highest.

        Args:
            base_pieces (`list[tuple[str, float, str]]`):
                Helium's vocabulary in id order, each piece with its score and kind, as [`sentencepiece_pieces`]
                reads it.
            pieces (`dict`):
                Every text piece of the final vocabulary mapped to its Unigram score. A base piece found here
                takes this score.
            control_tokens (`list[str]`, *optional*):
                Control tokens added after the base vocabulary.

        Returns:
            `PolyplexTokenizer`: The tokenizer, with `blocks` set.
        """
        texts = [score for piece, score, kind in base_pieces if kind == "normal"] + list(pieces.values())
        special_score = min(texts) - SPECIAL_SCORE_GAP
        specials = [piece for piece, _, kind in base_pieces if kind in ("unknown", "control")]
        vocab = []
        for piece, score, kind in base_pieces:
            if kind == "normal":
                vocab.append((piece, pieces.get(piece, score)))
            else:
                vocab.append((piece, special_score if kind in ("unknown", "control") else 0.0))
        vocab += [(token, special_score) for token in control_tokens]
        held = {piece for piece, _ in vocab}
        grouped = {}
        for piece, score in pieces.items():
            if piece not in held:
                grouped.setdefault(block_of(piece), []).append((piece, score))

        blocks = {"base": [0, len(vocab) + len(grouped.get("base", []))]}
        for name in BLOCK_ORDER:
            members = sorted(grouped.get(name, []), key=lambda item: (-item[1], item[0]))
            if name != "base" and members:
                blocks[name] = [len(vocab), len(vocab) + len(members)]
            vocab += members

        names = dict(zip(("unk_token", "bos_token", "eos_token", "pad_token"), specials))
        return cls(vocab=vocab, blocks=blocks, additional_special_tokens=list(control_tokens), **names)


__all__ = ["BLOCK_ORDER", "PolyplexTokenizer", "block_of", "script_of", "sentencepiece_pieces"]
