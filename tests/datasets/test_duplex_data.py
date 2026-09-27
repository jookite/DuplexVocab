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
r"""
Pins the layout a duplex batch has to arrive in.

None of what these check raises when it is wrong. A word on the wrong frame, a missing opening frame
or an undelayed codebook all produce a batch the model accepts and answers badly from, so a test is the
only place they surface.
"""

import numpy
import pytest
import torch

from conftest import AUDIO_VOCAB_SIZE, NUM_CODEBOOKS, build_model
from polyplex_research.datasets.duplex_data import (
    EPAD_TOKEN_ID,
    PAD_TOKEN_ID,
    duplex_collate_fn,
    split_trailing_punctuation,
    text_stream,
    word_spans,
)


class StubTokenizer:
    r"""Turns a word into one id per character, so a word's piece count is its length."""

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text.strip()]


def test_word_starts_on_the_frame_it_was_spoken_on():
    stream = text_stream(["ab", "c"], [0.0, 0.8], frames=20, tokenizer=StubTokenizer(), offset=4)

    # 0.8 s at 12.5 frames a second is frame 10, counted from the end of the prefix.
    assert stream[4:6].tolist() == [ord("a"), ord("b")]
    assert stream[14].item() == ord("c")


def test_a_word_at_the_very_start_is_kept():
    # Its EPAD lands on the prefix's closing frame, which carries padding.
    stream = text_stream(["ab"], [0.0], frames=12, tokenizer=StubTokenizer(), offset=4)

    assert stream[3].item() == EPAD_TOKEN_ID
    assert stream[4:6].tolist() == [ord("a"), ord("b")]


def test_a_word_starting_inside_the_previous_word_overwrites_its_tail():
    # "abcd" holds frames 4 to 7. "e" is spoken at frame 7, so it takes "d"'s frame and its EPAD, which would land on "c",
    # is left out.
    stream = text_stream(["abcd", "e"], [0.0, 0.24], frames=16, tokenizer=StubTokenizer(), offset=4)

    assert stream[3].item() == EPAD_TOKEN_ID
    assert stream[4:8].tolist() == [ord("a"), ord("b"), ord("c"), ord("e")]
    assert (stream[8:] == PAD_TOKEN_ID).all()


def test_the_overwritten_word_keeps_nothing_after_the_word_that_cut_it():
    # "abcdef" holds frames 2 to 7. "g" starts on frame 5, so "d" becomes "g" and "e" and "f" are cleared.
    stream = text_stream(["abcdef", "g"], [0.0, 0.24], frames=12, tokenizer=StubTokenizer(), offset=2)

    assert stream[1].item() == EPAD_TOKEN_ID
    assert stream[2:6].tolist() == [ord("a"), ord("b"), ord("c"), ord("g")]
    assert (stream[6:] == PAD_TOKEN_ID).all()


def test_a_word_right_after_the_previous_word_is_kept_without_its_epad():
    # "ab" holds frames 4 and 5. "c" is spoken at frame 6, so its EPAD would land on "b" and is left out.
    stream = text_stream(["ab", "c"], [0.0, 0.16], frames=12, tokenizer=StubTokenizer(), offset=4)

    assert stream[3].item() == EPAD_TOKEN_ID
    assert stream[4:7].tolist() == [ord("a"), ord("b"), ord("c")]
    assert (stream[7:] == PAD_TOKEN_ID).all()


def test_epad_sits_one_frame_ahead_of_every_word():
    stream = text_stream(["ab", "c"], [0.0, 0.8], frames=20, tokenizer=StubTokenizer(), offset=4)

    assert stream[3].item() == EPAD_TOKEN_ID
    assert stream[13].item() == EPAD_TOKEN_ID
    # Only the first piece of a word is announced, never the ones inside it.
    assert stream[5].item() != EPAD_TOKEN_ID


def test_everything_else_is_padded():
    stream = text_stream(["ab"], [0.4], frames=12, tokenizer=StubTokenizer(), offset=2)

    spoken = {7, 8}  # the word, five frames after the prefix
    announced = {6}  # its EPAD
    for frame in range(12):
        if frame in spoken or frame in announced:
            continue
        assert stream[frame].item() == PAD_TOKEN_ID


def test_a_word_that_would_run_past_the_stream_is_dropped():
    stream = text_stream(["abcdef"], [0.8], frames=12, tokenizer=StubTokenizer(), offset=0)

    assert (stream == PAD_TOKEN_ID).all()


def test_a_word_with_no_frame_for_its_epad_is_dropped():
    # At offset zero there is no frame in front of the first one to announce it on.
    stream = text_stream(["ab"], [0.0], frames=12, tokenizer=StubTokenizer(), offset=0)

    assert (stream == PAD_TOKEN_ID).all()


def test_pieces_of_one_word_stay_contiguous():
    stream = text_stream(["abcd"], [0.16], frames=16, tokenizer=StubTokenizer(), offset=3)

    assert stream[5:9].tolist() == [ord(c) for c in "abcd"]
    assert stream[4].item() == EPAD_TOKEN_ID


def test_the_prefix_is_written_before_the_words():
    prefix = torch.tensor([9, 9, 9], dtype=torch.long)
    stream = text_stream(["a"], [0.24], frames=10, tokenizer=StubTokenizer(), offset=3, prefix=prefix)

    assert stream[:3].tolist() == [9, 9, 9]
    assert stream[6].item() == ord("a")


def test_pad_and_epad_are_the_ids_the_checkpoint_uses():
    # `<pad>` and the row the model emits before a word. Named here so a tokenizer swap has to notice.
    assert (PAD_TOKEN_ID, EPAD_TOKEN_ID) == (3, 0)


def test_a_stream_with_no_words_is_all_padding():
    stream = text_stream([], [], frames=8, tokenizer=StubTokenizer(), offset=2)

    assert torch.equal(stream, torch.full((8,), PAD_TOKEN_ID, dtype=torch.long))


def test_trailing_punctuation_is_separated_from_the_word():
    assert split_trailing_punctuation("ab.") == ("ab", ".")
    assert split_trailing_punctuation("ab") == ("ab", "")
    assert split_trailing_punctuation("거야?!") == ("거야", "?!")
    # There is no word left to carry the mark, so the token stays whole.
    assert split_trailing_punctuation("...") == ("...", "")


def test_a_mark_follows_the_word_on_the_next_frame():
    stream = text_stream(["ab."], [0.0], frames=12, tokenizer=StubTokenizer(), offset=4)

    assert stream[3].item() == EPAD_TOKEN_ID
    assert stream[4:6].tolist() == [ord("a"), ord("b")]
    assert stream[6].item() == ord(".")


def test_a_mark_is_dropped_when_the_next_word_announces_itself_on_that_frame():
    # The second word starts on frame 7, so frame 6 is its EPAD and the mark has nowhere to go.
    stream = text_stream(["ab.", "c"], [0.0, 0.24], frames=12, tokenizer=StubTokenizer(), offset=4)

    assert stream[4:6].tolist() == [ord("a"), ord("b")]
    assert stream[6].item() == EPAD_TOKEN_ID
    assert stream[7].item() == ord("c")
    assert ord(".") not in stream.tolist()


def test_a_word_that_only_fits_without_its_mark_is_kept():
    # Three frames are left for a word of three pieces, which the mark would have pushed past the end.
    stream = text_stream(["abc."], [0.0], frames=5, tokenizer=StubTokenizer(), offset=2)

    assert stream[1].item() == EPAD_TOKEN_ID
    assert stream[2:5].tolist() == [ord(c) for c in "abc"]
    assert stream.shape[0] == 5


def test_a_mark_never_displaces_the_pieces_of_the_next_word():
    stream = text_stream(["a.", "bc"], [0.0, 0.16], frames=10, tokenizer=StubTokenizer(), offset=2)

    assert stream[2].item() == ord("a")
    assert stream[3].item() == EPAD_TOKEN_ID
    assert stream[4:6].tolist() == [ord("b"), ord("c")]


def test_a_word_that_cuts_the_previous_word_still_places_its_mark():
    # The first word runs to frame 7. "g." starts on frame 5, cuts it there, and its mark follows on frame 6.
    stream = text_stream(["abcdef", "g."], [0.0, 0.24], frames=12, tokenizer=StubTokenizer(), offset=2)

    assert stream[2:7].tolist() == [ord("a"), ord("b"), ord("c"), ord("g"), ord(".")]
    assert (stream[7:] == PAD_TOKEN_ID).all()


def test_a_word_of_punctuation_alone_is_written_as_it_is():
    stream = text_stream(["..."], [0.0], frames=10, tokenizer=StubTokenizer(), offset=2)

    assert stream[2:5].tolist() == [ord(".")] * 3


class StubProcessor:
    r"""Stands in for the processor with a prefix of known length and a silence of known value."""

    prefix_frames = 3

    def __init__(self):
        self.tokenizer = StubTokenizer()

    def __call__(self, text_prompt=None, return_tensors=None):
        frames = self.prefix_frames
        return {
            "input_ids": torch.full((1, frames), PAD_TOKEN_ID),
            "assistant_audio_codes": torch.zeros(1, NUM_CODEBOOKS, frames, dtype=torch.long),
            "user_audio_codes": torch.zeros(1, NUM_CODEBOOKS, frames, dtype=torch.long),
        }

    def get_silence_audio_codes(self, frames, batch_size=1):
        return torch.ones(batch_size, NUM_CODEBOOKS, frames, dtype=torch.long)


def test_audio_labels_mask_the_opening_and_the_prefix_under_the_delay():
    model = build_model()
    collate = duplex_collate_fn(StubProcessor(), model, audio_labels=True)
    codes = torch.full((NUM_CODEBOOKS, 6), 7, dtype=torch.long)
    batch = collate([{"codes": codes, "words": ["a"], "starts": [0.0]}])

    labels, codes_in = batch["audio_labels"][0], batch["assistant_audio_codes"][0]
    prefix = StubProcessor.prefix_frames
    # Column 0 is the opening frame and columns 1..prefix the prefix, for every codebook.
    assert (labels[:, : prefix + 1] == -100).all()
    # The delay puts every codebook after the first one column later, prefix included.
    assert (labels[1:, prefix + 1] == -100).all()
    assert labels[0, prefix + 1] == 7
    # The speech itself is the target, aligned with what the model is fed.
    assert torch.equal(labels[:, prefix + 2 :], codes_in[:, prefix + 2 :])


def test_audio_labels_are_off_unless_asked_for():
    collate = duplex_collate_fn(StubProcessor(), build_model())
    batch = collate([{"codes": torch.zeros(NUM_CODEBOOKS, 4, dtype=torch.long), "words": [], "starts": []}])
    assert "audio_labels" not in batch


def test_user_codes_fill_the_user_side_and_silence_fills_the_rest():
    model = build_model()
    collate = duplex_collate_fn(StubProcessor(), model)
    heard = torch.full((NUM_CODEBOOKS, 3), 5, dtype=torch.long)
    batch = collate(
        [{"codes": torch.zeros(NUM_CODEBOOKS, 5, dtype=torch.long), "words": [], "starts": [], "user_codes": heard}]
    )
    user = batch["user_audio_codes"][0]
    body = 1 + StubProcessor.prefix_frames
    # Codebook 0 is not delayed, so its body reads the user speech first and the silence after it.
    assert user[0, body : body + 3].tolist() == [5, 5, 5]
    # The delay pattern owns the final column, so the silence is checked where it is not in play.
    assert user[0, body + 3] == 1


def test_a_written_cache_reads_back_through_the_dataset(tmp_path):
    from polyplex_research.datasets.duplex_data import DuplexCacheWriter, DuplexUtteranceDataset

    first = numpy.arange(8 * 3, dtype=numpy.int16).reshape(8, 3)
    second = numpy.arange(8 * 2, dtype=numpy.int16).reshape(8, 2) + 100
    with DuplexCacheWriter(str(tmp_path / "cache")) as writer:
        writer.add("가 나", [0.0, 0.2], [0.2, 0.4], first)
        writer.add("다", [0.0], [0.16], second)

    dataset = DuplexUtteranceDataset(str(tmp_path / "cache"))
    assert len(dataset) == 2
    assert dataset[0]["words"] == ["가", "나"] and dataset[0]["starts"] == [0.0, 0.2]
    assert torch.equal(dataset[1]["codes"], torch.from_numpy(second.astype(numpy.int64)))
    assert dataset.example_lengths() == [3, 2]


def test_merged_caches_keep_every_utterance_and_its_codes(tmp_path):
    from polyplex_research.datasets.duplex_data import DuplexCacheWriter, DuplexUtteranceDataset

    for name, value in (("a", 1), ("b", 2)):
        with DuplexCacheWriter(str(tmp_path / name)) as writer:
            writer.add(name, [0.0], [0.1], numpy.full((8, 2), value, dtype=numpy.int16))
    DuplexCacheWriter.merge([str(tmp_path / "a"), str(tmp_path / "b")], str(tmp_path / "merged"))

    merged = DuplexUtteranceDataset(str(tmp_path / "merged"))
    assert [item["words"] for item in merged] == [["a"], ["b"]]
    assert merged[1]["codes"].unique().tolist() == [2]


def test_the_nominal_layout_spaces_words_by_their_piece_count():
    from polyplex_research.datasets.duplex_data import nominal_layout

    class Pieces:
        def encode(self, text, add_special_tokens=False):
            return list(range(len(text.strip())))

    starts, ends, frames = nominal_layout(["ab", "c"], Pieces(), frames_per_piece=2.5)
    assert starts == [0.0, 2 * 2.5 / 12.5]
    assert ends == [2 * 2.5 / 12.5, 3 * 2.5 / 12.5]
    assert frames == 10


def test_read_texts_collects_every_text_field(tmp_path):
    import json

    from polyplex_research.datasets.duplex_data import read_texts

    (tmp_path / "a.jsonl").write_text(
        json.dumps({"target": "가  나"}, ensure_ascii=False) + "\n" + json.dumps({"text": "다 라"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "b.json").write_text(json.dumps([{"ko": "마 바"}, "사 아"], ensure_ascii=False), encoding="utf-8")
    assert read_texts(str(tmp_path)) == {"가 나", "다 라"}
    assert read_texts(str(tmp_path / "b.json")) == {"마 바", "사 아"}


def test_a_cache_reads_back_what_was_written_without_python_records(tmp_path):
    import numpy as np

    from polyplex_research.datasets.duplex_data import DuplexCacheWriter, DuplexUtteranceDataset

    writer = DuplexCacheWriter(str(tmp_path))
    writer.add("가 나", [0.0, 0.5], [0.5, 1.0], np.full((8, 3), 7, dtype=np.int16))
    writer.add("다", [0.25], [0.75], np.full((8, 2), 9, dtype=np.int16))
    writer.close()

    dataset = DuplexUtteranceDataset(str(tmp_path))
    assert len(dataset) == 2
    assert dataset.example_lengths() == [3, 2]
    second = dataset[1]
    assert second["words"] == ["다"] and second["starts"] == [0.25]
    assert second["codes"].shape == (8, 2) and int(second["codes"].max()) == 9
    assert [entry["text"] for entry in dataset.entries] == ["가 나", "다"]
    assert isinstance(dataset.entries.offsets, np.ndarray)
    assert len(DuplexUtteranceDataset(str(tmp_path), limit=1)) == 1


class OnePiece:
    r"""Every word is one piece, so the nominal layout of `n` words spans `n` pieces."""

    def encode(self, text, add_special_tokens=False):
        return [1]


def test_a_dialogue_places_own_words_after_the_silence_of_the_turns_it_hears():
    from polyplex_research.datasets.duplex_data import dialogue_layout
    from polyplex_research.datasets.multisession import DialogueTurn

    turns = [DialogueTurn("가 나", False), DialogueTurn("다", True), DialogueTurn("라 마 바", False), DialogueTurn("사", True)]
    samples = dialogue_layout(turns, OnePiece(), max_frames=100, frames_per_piece=2.5)
    assert len(samples) == 1
    words, starts, ends, frames = samples[0]
    assert words == ["다", "사"]
    # The heard turn of two pieces takes ceil(3 * 2.5) = 8 frames, the own turn of one piece ceil(2 * 2.5) = 5.
    assert starts == [8 / 12.5, (8 + 5 + 10) / 12.5]
    assert ends == pytest.approx([(8 + 2.5) / 12.5, (8 + 5 + 10 + 2.5) / 12.5])
    assert frames == 8 + 5 + 10 + 5


def test_a_dialogue_longer_than_the_cap_continues_in_a_new_sample():
    from polyplex_research.datasets.duplex_data import dialogue_layout
    from polyplex_research.datasets.multisession import DialogueTurn

    turns = [DialogueTurn("가", True), DialogueTurn("나", False), DialogueTurn("다", True)]
    samples = dialogue_layout(turns, OnePiece(), max_frames=12, frames_per_piece=2.5)
    assert [sample[0] for sample in samples] == [["가"], ["다"]]
    # The second sample opens on the turn it heard before speaking.
    assert samples[1][1] == [5 / 12.5]


def test_dense_rows_open_every_document_on_its_own_frame_and_restart_its_positions():
    from polyplex_research.datasets.duplex_data import dense_collate_fn

    model = build_model()
    collate = dense_collate_fn(StubProcessor(), model)
    batch = collate([
        {"segments": [torch.tensor([5, 6, 7]), torch.tensor([8, 9])]},
        {"segments": [torch.tensor([10])]},
    ])
    opening = model.config.vocab_size
    assert "attention_mask" not in batch
    assert batch["input_ids"][0].tolist() == [opening, 5, 6, 7, opening, 8, 9]
    assert batch["position_ids"][0].tolist() == [0, 1, 2, 3, 0, 1, 2]
    assert batch["text_labels"][0].tolist() == [-100, 5, 6, 7, -100, 8, 9]
    # The short row is padded with a segment of its own that is never scored.
    assert batch["input_ids"][1].tolist()[:2] == [opening, 10]
    assert batch["text_labels"][1].tolist()[2:] == [-100] * 5
    assert batch["position_ids"][1].tolist()[2:] == [0, 1, 2, 3, 4]
    # Both acoustic streams are silent past each opening frame, which holds the reserved code.
    reserved = AUDIO_VOCAB_SIZE
    assert batch["assistant_audio_codes"][0, 0, 0] == reserved and batch["assistant_audio_codes"][0, 0, 4] == reserved
    assert batch["assistant_audio_codes"][0, 0, 1] == 1 and batch["user_audio_codes"][0, 0, 5] == 1


def test_text_only_dense_rows_carry_no_audio_codes_and_the_model_reads_them():
    from conftest import build_polyplex
    from polyplex_research.datasets.duplex_data import dense_collate_fn

    model = build_polyplex()
    silent = dense_collate_fn(StubProcessor(), model)
    text_only = dense_collate_fn(StubProcessor(), model, silent_audio=False)
    rows = [{"segments": [torch.tensor([5, 6, 7]), torch.tensor([8, 9])]}]
    batch, expected = text_only(rows), silent(rows)
    assert "assistant_audio_codes" not in batch and "user_audio_codes" not in batch
    for key in ("input_ids", "position_ids", "text_labels"):
        assert torch.equal(batch[key], expected[key])
    with torch.no_grad():
        logits = model(**{key: value for key, value in batch.items() if key != "text_labels"}, use_cache=False).logits
        embeds = model.model.embed_tokens(batch["input_ids"])
        direct = model.lm_head(model.model(inputs_embeds=embeds, position_ids=batch["position_ids"], use_cache=False).text_hidden_state)
    torch.testing.assert_close(logits, direct, rtol=0, atol=0)


def test_packed_documents_read_exactly_what_each_reads_alone():
    from polyplex_research.datasets.duplex_data import dense_collate_fn

    model = build_model()
    collate = dense_collate_fn(StubProcessor(), model)
    first, second = torch.tensor([5, 6, 7]), torch.tensor([8, 9])
    packed = collate([{"segments": [first, second]}])
    alone = collate([{"segments": [second]}])

    def states(batch):
        embeds = model.model.embed_tokens(batch["input_ids"]) + model._embed_audio_codes(
            torch.cat([batch["assistant_audio_codes"], batch["user_audio_codes"]], dim=1)
        )
        return model.model(inputs_embeds=embeds, position_ids=batch["position_ids"], use_cache=False).last_hidden_state

    with torch.no_grad():
        torch.testing.assert_close(states(packed)[0, 4:], states(alone)[0], rtol=1e-5, atol=1e-5)


def test_a_sample_cut_after_long_heard_turns_ends_with_its_last_own_turn():
    from polyplex_research.datasets.duplex_data import dialogue_layout
    from polyplex_research.datasets.multisession import DialogueTurn

    turns = [DialogueTurn("가", True), DialogueTurn("나 다 라 마", False), DialogueTurn("바", True)]
    samples = dialogue_layout(turns, OnePiece(), max_frames=12, frames_per_piece=2.5)
    assert [(sample[0], sample[3]) for sample in samples] == [(["가"], 5), (["바"], 5)]
    assert all(sample[3] <= 12 for sample in samples)


def test_a_text_cache_can_keep_one_origin_only(tmp_path, monkeypatch):
    import polyplex_research.datasets.duplex_data as duplex_data
    from polyplex_research.datasets.duplex_data import DuplexUtteranceDataset, build_text_cache
    from polyplex_research.datasets.phase_data import Phase1Example, build_unseen_dataset

    dataset = build_unseen_dataset({"all": [
        Phase1Example(target="가 나", origin="conversation"),
        Phase1Example(target="다 라", origin="translation"),
        Phase1Example(target="마", origin="translation"),
    ]}, set())
    dataset.examples.sort(key=lambda example: example.target)
    dataset.save_pretrained(str(tmp_path / "text"))

    class Processor:
        tokenizer = OnePiece()

    import polyplex_research.models.polyplex as polyplex

    monkeypatch.setattr(polyplex.PolyplexProcessor, "from_pretrained", staticmethod(lambda path: Processor()))
    monkeypatch.setattr(duplex_data, "codec_silent_frame", lambda processor, device: numpy.zeros((8, 1), dtype=numpy.int16))
    count = build_text_cache(str(tmp_path / "text"), "model", str(tmp_path / "cache"), limit=1, origin="translation")
    assert count == 1
    assert DuplexUtteranceDataset(str(tmp_path / "cache"))[0]["words"] == ["다", "라"]


def test_a_text_cache_can_skip_the_examples_an_earlier_cache_took(tmp_path, monkeypatch):
    import polyplex_research.datasets.duplex_data as duplex_data
    from polyplex_research.datasets.duplex_data import DuplexUtteranceDataset, build_text_cache
    from polyplex_research.datasets.phase_data import Phase1Example, build_unseen_dataset

    dataset = build_unseen_dataset({"all": [
        Phase1Example(target="가 나", origin="translation"),
        Phase1Example(target="다 라", origin="translation"),
        Phase1Example(target="마", origin="translation"),
    ]}, set())
    dataset.examples.sort(key=lambda example: example.target)
    dataset.save_pretrained(str(tmp_path / "text"))

    class Processor:
        tokenizer = OnePiece()

    import polyplex_research.models.polyplex as polyplex

    monkeypatch.setattr(polyplex.PolyplexProcessor, "from_pretrained", staticmethod(lambda path: Processor()))
    monkeypatch.setattr(duplex_data, "codec_silent_frame", lambda processor, device: numpy.zeros((8, 1), dtype=numpy.int16))
    count = build_text_cache(str(tmp_path / "text"), "model", str(tmp_path / "cache"), skip=1, origin="translation")
    assert count == 2
    assert [item["words"] for item in DuplexUtteranceDataset(str(tmp_path / "cache"))] == [["다", "라"], ["마"]]


def test_a_dialogue_cache_can_skip_the_samples_an_earlier_cache_took(tmp_path, monkeypatch):
    import polyplex_research.datasets.duplex_data as duplex_data
    from polyplex_research.datasets.duplex_data import DuplexUtteranceDataset, build_dialogue_cache
    from polyplex_research.datasets.multisession import DialogueTurn

    class Processor:
        tokenizer = OnePiece()

    import polyplex_research.models.polyplex as polyplex

    monkeypatch.setattr(polyplex.PolyplexProcessor, "from_pretrained", staticmethod(lambda path: Processor()))
    monkeypatch.setattr(duplex_data, "codec_silent_frame", lambda processor, device: numpy.zeros((8, 1), dtype=numpy.int16))
    dialogues = [[DialogueTurn(text, own=True)] for text in ["가 나", "다", "라 마"]]
    count = build_dialogue_cache(iter(dialogues), "model", str(tmp_path / "cache"), skip=2, limit=1)
    assert count == 1
    assert DuplexUtteranceDataset(str(tmp_path / "cache"))[0]["words"] == ["라", "마"]


def test_a_merge_leaves_out_utterances_whose_text_is_excluded(tmp_path):
    from polyplex_research.datasets.duplex_data import DuplexCacheWriter, DuplexUtteranceDataset

    for name, value in (("a", 1), ("b", 2), ("c", 3)):
        with DuplexCacheWriter(str(tmp_path / name)) as writer:
            writer.add(name, [0.0], [0.1], numpy.full((8, 2), value, dtype=numpy.int16))
    sources = [str(tmp_path / name) for name in ("a", "b", "c")]
    count = DuplexCacheWriter.merge(sources, str(tmp_path / "merged"), exclude={"b"})

    merged = DuplexUtteranceDataset(str(tmp_path / "merged"))
    assert count == 2
    assert [item["words"] for item in merged] == [["a"], ["c"]]
    assert merged[1]["codes"].unique().tolist() == [3]


def test_a_word_runs_from_its_first_token_to_the_next_words_first_token():
    starts, ends = word_spans([(4, 6), (6, 7), (7, 9)], [0, 0, 0, 0, 0.1, 0.3, 0.5, 0.9, 1.0, 1.4], 1.4)
    assert starts == [0.1, 0.5, 0.9]
    assert ends == [0.5, 0.9, 1.4]


def test_a_word_never_ends_before_it_starts():
    starts, ends = word_spans([(0, 1), (1, 2)], [0.5, 0.5], 0.4)
    assert ends == [0.5, 0.5]
