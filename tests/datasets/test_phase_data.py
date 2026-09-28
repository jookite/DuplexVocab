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
"""How a Phase 2 batch lays an utterance over its frames."""

import pytest
import torch

from conftest import AUDIO_VOCAB_SIZE, NUM_CODEBOOKS
from polyplex_research.datasets.base import SingleTurnTTSExample
from polyplex_research.datasets.phase_data import (
    delay_audio_codes,
    phase2_collate_fn,
    text_frame_positions,
)


TEXT_PAD_TOKEN_ID = 3


class StubTokenizer:
    r"""
    Constructs a tokenizer that returns one id per character, so a test can count frames.

    Args:
        offset (`int`, *optional*, defaults to 10):
            First id handed out, kept clear of the pad token.
    """

    def __init__(self, offset: int = 10):
        self.offset = offset

    def encode(self, text, add_special_tokens=False):
        return [self.offset + index for index in range(len(text))]


def build_examples(lengths, text="abcd"):
    return [
        SingleTurnTTSExample(
            text=text,
            codes=torch.randint(0, AUDIO_VOCAB_SIZE, (NUM_CODEBOOKS, frames)),
            audio_path=f"u{index}",
        )
        for index, frames in enumerate(lengths)
    ]


# ---------------------------------------------------------------------------
# Text placement
# ---------------------------------------------------------------------------


def test_spread_keeps_the_words_in_order_and_off_each_other():
    for num_tokens in range(1, 13):
        positions = text_frame_positions(num_tokens, 12, "spread")
        assert positions.tolist() == sorted(set(positions.tolist()))
        assert positions[0] == 0
        assert positions[-1] < 12


def test_spread_paces_the_words_over_the_whole_utterance():
    assert text_frame_positions(4, 12, "spread").tolist() == [0, 3, 6, 9]
    assert text_frame_positions(4, 4, "spread").tolist() == [0, 1, 2, 3]


def test_leading_packs_the_words_onto_the_opening_frames():
    assert text_frame_positions(4, 12, "leading").tolist() == [0, 1, 2, 3]


def test_an_unknown_alignment_is_refused():
    with pytest.raises(ValueError, match="text_alignment"):
        text_frame_positions(4, 12, "centred")


def test_the_collator_places_every_token_and_pads_the_rest():
    collate = phase2_collate_fn(
        StubTokenizer(),
        audio_vocab_size=AUDIO_VOCAB_SIZE,
        text_pad_token_id=TEXT_PAD_TOKEN_ID,
    )
    batch = collate(build_examples([12], text="abcd"))

    input_ids = batch["input_ids"][0]
    # The words are paced over the twelve acoustic frames, not over the frame the delay adds.
    assert input_ids[[0, 3, 6, 9]].tolist() == [10, 11, 12, 13]
    carried = torch.zeros(input_ids.shape[0], dtype=torch.bool)
    carried[[0, 3, 6, 9]] = True
    assert (input_ids[~carried] == TEXT_PAD_TOKEN_ID).all()


# ---------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------


def test_padding_frames_are_masked_and_never_scored():
    collate = phase2_collate_fn(StubTokenizer(), audio_vocab_size=AUDIO_VOCAB_SIZE)
    batch = collate(build_examples([5, 9]))

    # The delay costs a frame, so five codes occupy six frames of a batch ten wide.
    assert batch["attention_mask"][0].tolist() == [True] * 6 + [False] * 4
    assert (batch["audio_labels"][0, :, 6:] == -100).all()
    # The reserved row is what an unwritten frame carries on the conditioning side.
    assert (batch["assistant_audio_codes"][0, :, 6:] == AUDIO_VOCAB_SIZE).all()


def test_the_reserved_row_follows_the_configuration_it_is_given():
    collate = phase2_collate_fn(StubTokenizer(), audio_vocab_size=7)
    batch = collate(build_examples([4]))
    assert (batch["user_audio_codes"] == 7).all()


def test_max_frames_caps_the_batch():
    collate = phase2_collate_fn(StubTokenizer(), audio_vocab_size=AUDIO_VOCAB_SIZE, max_frames=6)
    batch = collate(build_examples([20, 3]))
    assert batch["input_ids"].shape[1] == 7
    assert batch["assistant_audio_codes"].shape == (2, NUM_CODEBOOKS, 7)


# ---------------------------------------------------------------------------
# Delay pattern
# ---------------------------------------------------------------------------


def test_delay_shifts_every_codebook_but_the_first():
    codes = torch.arange(1, NUM_CODEBOOKS * 4 + 1).reshape(NUM_CODEBOOKS, 4)
    delayed = delay_audio_codes(codes, AUDIO_VOCAB_SIZE, AUDIO_VOCAB_SIZE)

    assert delayed.shape == (NUM_CODEBOOKS, 5)
    assert delayed[0].tolist() == codes[0].tolist() + [AUDIO_VOCAB_SIZE]
    for codebook in range(1, NUM_CODEBOOKS):
        assert delayed[codebook].tolist() == [AUDIO_VOCAB_SIZE] + codes[codebook].tolist()


def test_delay_matches_the_layout_the_base_model_generates_in(base_model):
    r"""
    The frozen model has only ever read one layout, so ours is pinned to the one it builds itself
    rather than to a second reading of the same paragraph.
    """
    codes = torch.randint(0, AUDIO_VOCAB_SIZE, (NUM_CODEBOOKS, 7))
    upstream = base_model.build_delay_pattern_mask(
        codes.unsqueeze(0),
        bos_token_id=AUDIO_VOCAB_SIZE,
        pad_token_id=AUDIO_VOCAB_SIZE,
        max_length=codes.shape[1] + 1,
    )[1][0]

    ours = delay_audio_codes(codes, AUDIO_VOCAB_SIZE, AUDIO_VOCAB_SIZE)
    assert torch.equal(ours, upstream)


def test_the_collated_streams_carry_the_delay():
    collate = phase2_collate_fn(StubTokenizer(), audio_vocab_size=AUDIO_VOCAB_SIZE)
    examples = build_examples([6])
    codes = examples[0].codes
    batch = collate(examples)

    expected = delay_audio_codes(codes, AUDIO_VOCAB_SIZE, AUDIO_VOCAB_SIZE)
    assert torch.equal(batch["assistant_audio_codes"][0, :, :7], expected)
    # A stream nobody speaks on is the reserved row at every frame, which the delay leaves alone.
    assert (batch["user_audio_codes"] == AUDIO_VOCAB_SIZE).all()


def test_the_frames_the_delay_invents_are_not_scored():
    collate = phase2_collate_fn(StubTokenizer(), audio_vocab_size=AUDIO_VOCAB_SIZE)
    batch = collate(build_examples([6]))
    labels = batch["audio_labels"][0]

    # The opening frame of every shifted codebook and the closing frame of codebook 0 hold no code.
    assert (labels[1:, 0] == -100).all()
    assert labels[0, 6] == -100
    assert (labels[0, :6] != -100).all()
    assert (labels[1:, 1:7] != -100).all()


def test_an_unseen_mixture_drops_seen_and_repeated_sentences():
    from polyplex_research.datasets.phase_data import Phase1Example, build_unseen_dataset

    sources = {
        "conversation": [Phase1Example(target=t, origin="conversation") for t in ("가 나", "다 라", "가 나", "본 문장")],
        "translation": [Phase1Example(target=t, origin="translation") for t in ("마 바", "본 문장")],
    }
    dataset = build_unseen_dataset(sources, seen={"본 문장"}, seed=0)
    assert sorted(example.target for example in dataset.examples) == ["가 나", "다 라", "마 바"]
    assert dataset.export_config()["split"] == "train"


class CharPieces:
    r"""One piece per character, with the character code as its id."""

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]


def test_dense_rows_pack_documents_whole_and_split_one_longer_than_a_row(tmp_path):
    import json

    from polyplex_research.datasets.phase_data import DenseTextDataset, write_dense_cache

    documents = tmp_path / "docs.jsonl"
    documents.write_text(
        "\n".join(json.dumps({"text": text}) for text in ["ab", "cde", "f  g\nh", "ijklmnop"]) + "\n", encoding="utf-8"
    )
    rows = write_dense_cache(str(documents), CharPieces(), str(tmp_path / "dense"), max_tokens=6)
    dataset = DenseTextDataset(str(tmp_path / "dense"))

    def text(segment):
        return "".join(chr(int(piece)) for piece in segment)

    # A segment takes one frame more than its pieces for its opening frame.
    assert [[text(s) for s in dataset[i]["segments"]] for i in range(len(dataset))] == [
        ["ab"], ["cde"], ["f g h"], ["ijklm"], ["nop"],
    ]
    assert rows == len(dataset) == 5
    assert dataset.example_lengths() == [3, 4, 6, 6, 4]


def test_a_dense_cache_from_pieces_already_encoded_is_byte_identical(tmp_path):
    import json

    from polyplex_research.datasets.phase_data import write_dense_cache

    texts = ["ab", "", "cde", "f  g\nh", "ijklmnop", "q"]
    documents = tmp_path / "docs.jsonl"
    documents.write_text("\n".join(json.dumps({"text": text}) for text in texts) + "\n", encoding="utf-8")

    class Counting(CharPieces):
        calls = 0

        def encode(self, text, add_special_tokens=False):
            Counting.calls += 1
            return super().encode(text, add_special_tokens)

    pieces = [CharPieces().encode(" ".join(text.split())) for text in texts]
    encoded = write_dense_cache(str(documents), CharPieces(), str(tmp_path / "encoded"), max_tokens=6)
    reused = write_dense_cache(str(documents), Counting(), str(tmp_path / "reused"), max_tokens=6, pieces=pieces)
    assert encoded == reused and Counting.calls == 0
    for name in ("tokens.bin", "segments.npy", "rows.npy"):
        assert (tmp_path / "encoded" / name).read_bytes() == (tmp_path / "reused" / name).read_bytes()


def test_dense_rows_hold_several_short_documents(tmp_path):
    import json

    from polyplex_research.datasets.phase_data import DenseTextDataset, write_dense_cache

    documents = tmp_path / "docs.jsonl"
    documents.write_text("\n".join(json.dumps({"text": t}) for t in ["ab", "c", "de"]) + "\n", encoding="utf-8")
    write_dense_cache(str(documents), CharPieces(), str(tmp_path / "dense"), max_tokens=7)
    dataset = DenseTextDataset(str(tmp_path / "dense"))
    assert len(dataset) == 2
    assert [len(s) for s in dataset[0]["segments"]] == [2, 1]
