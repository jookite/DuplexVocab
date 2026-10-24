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
"""The Polyplex tokenizer built from the final vocabulary (SPEC S12.1, S12.1a)."""

import pytest
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as model_pb2

from polyplex_research.models.polyplex import PolyplexTokenizer, sentencepiece_pieces


BASE_NORMAL = [("▁", -2.0), ("▁hello", -3.0), ("▁world", -4.0), ("o", -5.0), ("h", -6.0)]
CONTROL = ["<tool_call>", "</tool_call>", "<think>", "</think>"]


def base_pieces():
    # A Helium-shaped vocabulary: the four special tokens, the 256 byte tokens, then text pieces.
    pieces = [("<unk>", 0.0, "unknown"), ("<s>", 0.0, "control"), ("</s>", 0.0, "control"), ("<pad>", 0.0, "control")]
    pieces += [(f"<0x{byte:02X}>", 0.0, "byte") for byte in range(256)]
    pieces += [(piece, score, "normal") for piece, score in BASE_NORMAL]
    return pieces


def build(pieces=None, control=CONTROL):
    scores = dict(BASE_NORMAL)
    scores.update(pieces or {"▁hola": -3.5, "▁가": -4.5, "가": -5.0, "▁안녕": -3.2, ",": -6.0, "▁mundo": -3.8})
    return PolyplexTokenizer.from_vocabulary(base_pieces(), scores, control_tokens=control)


def test_the_base_tokens_keep_their_ids():
    tokenizer = build()
    base = base_pieces()
    assert tokenizer.convert_ids_to_tokens(list(range(len(base)))) == [piece for piece, _, _ in base]
    assert tokenizer.unk_token_id == 0 and tokenizer.pad_token_id == 3


def test_the_control_tokens_follow_the_base_and_never_come_from_text():
    # As in the real vocabulary, the characters of the marker are text pieces of their own.
    tokenizer = build({"▁<": -7.0, "<": -7.0, ">": -7.0, "think": -6.0})
    first = len(base_pieces())
    assert tokenizer.convert_tokens_to_ids(CONTROL) == list(range(first, first + len(CONTROL)))
    assert tokenizer.tokenize("<think>") == ["▁<", "think", ">"]


def test_the_blocks_are_contiguous_from_the_base_block():
    tokenizer = build()
    spans = sorted(tokenizer.blocks.values())
    assert spans[0] == tokenizer.blocks["base"] and spans[0][0] == 0
    assert all(left[1] == right[0] for left, right in zip(spans, spans[1:]))
    assert spans[-1][1] == len(tokenizer)


@pytest.mark.parametrize("piece,block", [("▁hola", "latin"), ("▁안녕", "hangul"), ("가", "hangul"), (",", "base")])
def test_a_new_piece_lands_in_the_block_of_its_script(piece, block):
    tokenizer = build()
    start, end = tokenizer.blocks[block]
    assert start <= tokenizer.convert_tokens_to_ids(piece) < end


def test_a_block_is_ordered_by_score():
    tokenizer = build()
    start, end = tokenizer.blocks["hangul"]
    assert tokenizer.convert_ids_to_tokens(list(range(start, end))) == ["▁안녕", "▁가", "가"]


def test_text_is_cut_after_nfc():
    tokenizer = build()
    assert tokenizer.tokenize("가") == tokenizer.tokenize("가") == ["▁가"]


def test_the_scores_decide_the_cut():
    assert build({"▁ab": -1.0, "▁a": -5.0, "b": -5.0}).tokenize("ab") == ["▁ab"]
    assert build({"▁ab": -20.0, "▁a": -1.0, "b": -1.0}).tokenize("ab") == ["▁a", "b"]


def test_a_character_outside_the_vocabulary_goes_to_bytes_and_decodes_back():
    tokenizer = build()
    assert tokenizer.tokenize("😀")[1:] == ["<0xF0>", "<0x9F>", "<0x98>", "<0x80>"]
    text = "hello 😀 가"
    assert tokenizer.decode(tokenizer(text, add_special_tokens=False).input_ids) == text


def test_saving_keeps_the_ids_and_the_blocks(tmp_path):
    tokenizer = build()
    tokenizer.save_pretrained(tmp_path)
    loaded = PolyplexTokenizer.from_pretrained(tmp_path)
    assert loaded.blocks == tokenizer.blocks
    assert loaded.get_vocab() == tokenizer.get_vocab()
    assert loaded("hello 가").input_ids == tokenizer("hello 가").input_ids


def test_sentencepiece_pieces_reads_ids_scores_and_kinds(tmp_path):
    model = model_pb2.ModelProto()
    model.trainer_spec.model_type = model_pb2.TrainerSpec.UNIGRAM
    model.trainer_spec.byte_fallback = True
    kinds = model_pb2.ModelProto.SentencePiece
    for piece, kind in (("<unk>", kinds.UNKNOWN), ("<s>", kinds.CONTROL)):
        model.pieces.add(piece=piece, score=0.0, type=kind)
    for byte in range(256):
        model.pieces.add(piece=f"<0x{byte:02X}>", score=0.0, type=kinds.BYTE)
    model.pieces.add(piece="▁a", score=-1.5, type=kinds.NORMAL)
    path = tmp_path / "base.model"
    path.write_bytes(model.SerializeToString())
    spm.SentencePieceProcessor(model_file=str(path))
    pieces = sentencepiece_pieces(str(path))
    assert pieces[:2] == [("<unk>", 0.0, "unknown"), ("<s>", 0.0, "control")]
    assert pieces[2] == ("<0x00>", 0.0, "byte") and pieces[-1] == ("▁a", -1.5, "normal")
