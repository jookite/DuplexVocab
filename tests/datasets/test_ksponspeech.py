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
"""KsponSpeech archive reading and the caches built from it."""

import json

import numpy
import torch

from polyplex_research.datasets.duplex_data import DuplexUtteranceDataset
from polyplex_research.datasets.ksponspeech import (
    KsponSpeechArchive,
    import_code_cache,
    read_alignment_index,
    subsample,
)


def test_the_transcript_is_grouped_by_shard(tmp_path):
    path = tmp_path / "train.trn"
    path.write_text(
        "KsponSpeech_01/KsponSpeech_0001/a.pcm :: 안녕\nKsponSpeech_02\\KsponSpeech_0101\\b.pcm :: 네\nbroken line\n",
        encoding="utf-8",
    )
    shards = KsponSpeechArchive.read_transcript(str(path))
    assert shards == {
        "KsponSpeech_01": [("KsponSpeech_01/KsponSpeech_0001/a.pcm", "안녕")],
        "KsponSpeech_02": [("KsponSpeech_02/KsponSpeech_0101/b.pcm", "네")],
    }


def test_pcm_is_scaled_to_unit_range_and_a_trailing_odd_byte_is_dropped():
    payload = numpy.array([0, 16384, -32768], dtype="<i2").tobytes() + b"\x01"
    waveform = KsponSpeechArchive.decode_pcm(payload)
    assert torch.equal(waveform, torch.tensor([0.0, 0.5, -1.0]))


def test_subsampling_keeps_a_stride_across_the_shard():
    entries = list(range(100))
    kept = subsample(entries, max_hours=10 * 5.55 / 3600)
    assert kept == list(range(0, 100, 10))
    assert subsample(entries, None) == entries


def test_the_alignment_index_keeps_usable_utterances(tmp_path):
    (tmp_path / "audio_paths.json").write_text(json.dumps({"u1": "S/a.pcm", "u2": "S/b.pcm"}))
    rows = [
        {"utterance_id": "u1", "subset": "train", "duration": 3.0, "text": "가", "start": [0], "end": [1]},
        {"utterance_id": "u2", "subset": "train", "duration": 30.0, "text": "나", "start": [0], "end": [1]},
        {"utterance_id": "u3", "subset": "train", "duration": 3.0, "text": "다", "start": [0], "end": [1]},
    ]
    (tmp_path / "alignment.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    kept = read_alignment_index(str(tmp_path), ("train",), 2.0, 12.0, 0)
    assert [item["utterance_id"] for item in kept] == ["u1"] and kept[0]["member"] == "S/a.pcm"


def test_an_imported_code_cache_drops_overlap_and_mismatched_words(tmp_path):
    source = tmp_path / "source"
    source.mkdir()

    def utterance(key, text, starts, other=False, subset="train"):
        numpy.savez(
            source / f"{key}.npz",
            codes=numpy.full((8, 5), len(key), dtype=numpy.int64),
            align_start=numpy.array(starts, dtype=float),
            align_end=numpy.array(starts, dtype=float) + 0.1,
        )
        meta = {"utterance_id": key, "text": text, "subset": subset, "duration": 3.0, "has_other_speaker": other}
        (source / f"{key}.json").write_text(json.dumps(meta, ensure_ascii=False))

    utterance("a", "가 나", [0.0, 0.5])
    utterance("bb", "다", [0.0], other=True)
    utterance("ccc", "라 마", [0.0])
    counts = import_code_cache(str(source), str(tmp_path / "out"), 2.0, 12.0)

    assert counts["train"] == 1
    dataset = DuplexUtteranceDataset(str(tmp_path / "out_train"))
    assert [item["words"] for item in dataset] == [["가", "나"]]
