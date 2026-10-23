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
"""Capacity measurement and the Phase 1 scoring tools."""

import math

import pytest
import torch

from polyplex_research.adapters.phase_1_efficient_adapter.modeling_phase_1_efficient_adapter import ConfinedCapacity
from polyplex_research.utils.evaluate import CAPACITY_GROUPS, GENERATION_OPENINGS, CapacityMeter, generation_metrics


def _moments(hidden: int, deploy_dims: int):
    deploy = torch.zeros(hidden, hidden)
    deploy[:deploy_dims, :deploy_dims] = torch.eye(deploy_dims)
    empty = torch.zeros(hidden, hidden)
    empty[deploy_dims:, deploy_dims:] = torch.eye(hidden - deploy_dims)
    return {
        "deploy_en_speech": deploy, "deploy_silent_pad": deploy, "empty_ko": empty,
        "empty_en": deploy, "speech_ko": empty,
    }


def test_the_analysis_finds_the_deploy_subspace_and_the_empty_energy_outside_it():
    moments = _moments(8, 3)
    report = CapacityMeter.analyze(moments, dict.fromkeys(CAPACITY_GROUPS, 1))
    assert report["protected_rank_99"] == 3
    assert abs(report["empty_outside_99"] - 1.0) < 1e-6
    assert report["allowed_rank"] == 5
    assert abs(report["wiring_reach"] - 1.0) < 1e-6


def test_confined_directions_avoid_the_deploy_subspace():
    directions = ConfinedCapacity.directions_from_moments(_moments(8, 3), rank=4)
    assert directions.shape == (8, 4)
    assert torch.allclose(directions.T @ directions, torch.eye(4), atol=1e-5)
    assert directions[:3].abs().max() < 1e-5


def test_frame_masks_select_groups_by_the_token_each_frame_reads():
    ids = torch.tensor([[5, 3, 7, 3]])
    labels = torch.tensor([[-100, 3, 7, 3]])
    masks = CapacityMeter.frame_masks({"input_ids": ids, "text_labels": labels}, "ko_silent")
    assert masks["deploy_silent_pad"].tolist() == [[False, True, False, True]]
    assert masks["empty_ko"].tolist() == [[False, False, True, False]]


def test_forced_text_pins_the_opening_frames_then_lets_go():
    from polyplex_research.utils.evaluate import ForceTextLogitsProcessor

    force = ForceTextLogitsProcessor([4, 2])
    scores = torch.zeros(1, 6)
    first = force(torch.zeros(1, 3, dtype=torch.long), scores.clone())
    second = force(torch.zeros(1, 4, dtype=torch.long), scores.clone())
    third = force(torch.zeros(1, 5, dtype=torch.long), scores.clone())
    assert first.argmax().item() == 4 and torch.isinf(first).sum() == 5
    assert second.argmax().item() == 2
    assert torch.equal(third, scores)


def test_silent_assistant_follows_the_silence_sequence_frame_by_frame():
    from polyplex_research.utils.evaluate import SilentAssistantLogitsProcessor

    silence = [[10, 11, 12], [20, 21, 22]]
    silent = SilentAssistantLogitsProcessor(silence)
    scores = torch.zeros(1, 32)
    picks = []
    for _ in range(4):
        for codebook in range(2):
            picks.append(silent(torch.zeros(1, codebook + 1, dtype=torch.long), scores.clone()).argmax().item())
    # The text token is position 0 of each depth step; frames past the sequence hold its last frame.
    assert picks == [10, 20, 11, 21, 12, 22, 12, 22]


def test_run_options_keep_only_what_assembly_reads(tmp_path):
    import json

    from polyplex_research.utils.assemble import run_options

    (tmp_path / "config.json").write_text(json.dumps({"base_row_mixture": 64, "session_expert_layers": "8-31", "learning_rate": 1}))
    assert run_options(str(tmp_path)) == {"base_row_mixture": 64, "session_expert_layers": "8-31"}


def test_a_paced_item_spaces_words_by_their_pieces():
    from polyplex_research.utils.evaluate import Phase1Scorer

    class Pieces:
        def encode(self, text, add_special_tokens=False):
            return list(range(len(text.strip())))

    item = Phase1Scorer.paced_item("ab c", Pieces(), frames_per_piece=2.5)
    assert item["words"] == ["ab", "c"]
    assert item["starts"] == [0.4, 2 * 2.5 / 12.5 + 0.4]
    assert item["codes"].shape == (8, int(5 * 2.5) + 8)


def test_generation_metrics_count_only_what_follows_the_opening():
    pairs = [
        ("안녕하세요,", "안녕하세요, 무엇을 도와드릴까요? 네."),
        ("그 부분은", "그 부분은 Hey 좋아요 좋아요"),
        ("네,", "네, 좋아요 정말 좋아요 정말 좋아요 정말 좋아요"),
        ("Hello", "Hello 1000000 원입니다."),
    ]
    metrics = generation_metrics(pairs)
    assert metrics == {"english": 0.25, "sentence_end": 0.5, "repeat": 0.25, "long_number": 0.25}


def test_generation_openings_are_sixteen_distinct_korean_starts():
    assert len(GENERATION_OPENINGS) == 16
    assert len(set(GENERATION_OPENINGS)) == 16


def test_dense_checks_find_no_leak_in_packed_rows():
    from conftest import BASE_VOCAB_SIZE, VOCAB_SIZE, build_model
    from polyplex_research.adapters.phase_1_efficient_adapter import EfficientAdapterP1Config, EfficientAdapterP1Model
    from polyplex_research.datasets.duplex_data import dense_collate_fn
    from polyplex_research.utils.evaluate import dense_checks

    class Silence:
        def get_silence_audio_codes(self, frames, batch_size=1):
            return torch.ones(batch_size, 4, frames, dtype=torch.long)

    base = build_model(seed=0)
    adapter = EfficientAdapterP1Model(base, EfficientAdapterP1Config.from_base_model(base, base_vocab_size=BASE_VOCAB_SIZE)).eval()
    rows = [
        {"segments": [torch.arange(1, 9) % VOCAB_SIZE, torch.arange(50, 56)]},
        {"segments": [torch.arange(10, 22)]},
    ]
    unigram = torch.full((VOCAB_SIZE,), -math.log(VOCAB_SIZE))
    report = dense_checks(adapter, dense_collate_fn(Silence(), base), rows, BASE_VOCAB_SIZE, unigram)
    assert report["future_leak"] == 0.0
    assert report["document_leak"] == 0.0
    assert report["unigram"] == pytest.approx(math.log(VOCAB_SIZE))
    assert set(report) >= {"loss", "loss_new", "loss_base", "shuffled", "unigram", "scored"}


def test_dense_generation_extends_the_prompt_and_is_repeatable_under_a_seed():
    from conftest import BASE_VOCAB_SIZE, build_model
    from polyplex_research.adapters.phase_1_efficient_adapter import EfficientAdapterP1Config, EfficientAdapterP1Model
    from polyplex_research.datasets.duplex_data import dense_collate_fn
    from polyplex_research.utils.evaluate import dense_generate

    class Silence:
        def get_silence_audio_codes(self, frames, batch_size=1):
            return torch.ones(batch_size, 4, frames, dtype=torch.long)

    base = build_model(seed=0)
    adapter = EfficientAdapterP1Model(base, EfficientAdapterP1Config.from_base_model(base, base_vocab_size=BASE_VOCAB_SIZE)).eval()
    collate = dense_collate_fn(Silence(), base)
    prompt = torch.tensor([5, 6, 7])
    first = dense_generate(adapter, collate, prompt, max_new_tokens=6, seed=3)
    second = dense_generate(adapter, collate, prompt, max_new_tokens=6, seed=3)
    greedy = dense_generate(adapter, collate, prompt, max_new_tokens=6, temperature=0.0)
    assert first.tolist() == second.tolist()
    assert len(first) == 6 and len(greedy) == 6
    assert int(first.max()) < adapter.config.vocab_size


def test_piece_kinds_split_a_word_into_its_first_piece_and_the_rest():
    from polyplex_research.utils.evaluate import piece_kind

    pieces = ["▁", "▁임원", "을", "닛", "<0xE6>", "<0x97>", "▁12", ",", "▁abc"]
    assert piece_kind("▁임원", "을") == "hangul_word_start"
    assert piece_kind("▁임원", "▁임원") == "hangul_word_start"
    assert piece_kind("을", "▁임원") == "hangul_inside_word"
    assert piece_kind("닛", "▁") == "hangul_word_start"
    assert piece_kind("<0x97>", "<0xE6>") == "byte"
    assert piece_kind("▁", "을") == "space"
    assert piece_kind("▁12", "을") == "digits"
    assert piece_kind(",", "을") == "punctuation"
    assert piece_kind("▁abc", "을") == "latin_word_start"
    assert piece_kind("ing", "▁walk") == "latin_inside_word"
    assert piece_kind("abc", "▁") == "latin_word_start"
    assert len(pieces) == 9


def test_base_only_scoring_reads_the_pretrained_vocabulary_alone():
    from conftest import BASE_VOCAB_SIZE, build_model
    from polyplex_research.adapters.phase_1_efficient_adapter import EfficientAdapterP1Config, EfficientAdapterP1Model
    from polyplex_research.datasets.duplex_data import dense_collate_fn
    from polyplex_research.utils.evaluate import piece_kind_scores

    class Silence:
        def get_silence_audio_codes(self, frames, batch_size=1):
            return torch.ones(batch_size, 4, frames, dtype=torch.long)

    class Names:
        def convert_ids_to_tokens(self, ids):
            return ["▁w" if i % 2 else "x" for i in ids]

    base = build_model(seed=0)
    adapter = EfficientAdapterP1Model(base, EfficientAdapterP1Config.from_base_model(base, base_vocab_size=BASE_VOCAB_SIZE)).eval()
    rows = [{"segments": [torch.tensor([5, 6, 7, 50, 8])]}]
    collate = dense_collate_fn(Silence(), base)
    every = piece_kind_scores(adapter, collate, rows, Names())
    pretrained = piece_kind_scores(adapter, collate, rows, Names(), base_only=True)
    assert every["all"]["share"] == 1.0
    # The appended target (50) is left out, and the pretrained targets lose the appended rows from the softmax.
    assert sum(1 for _ in rows[0]["segments"][0] if _ < BASE_VOCAB_SIZE) == 4
    assert pretrained["all"]["loss"] < every["all"]["loss"] + 1.0


def test_dense_generation_reads_the_next_piece_off_a_frame_the_delay_pattern_leaves_alone():
    from conftest import BASE_VOCAB_SIZE, build_model
    from polyplex_research.adapters.phase_1_efficient_adapter import EfficientAdapterP1Config, EfficientAdapterP1Model
    from polyplex_research.datasets.duplex_data import PAD_TOKEN_ID, dense_collate_fn
    from polyplex_research.utils.evaluate import dense_generate

    class Silence:
        def get_silence_audio_codes(self, frames, batch_size=1):
            return torch.ones(batch_size, 4, frames, dtype=torch.long)

    base = build_model(seed=0)
    adapter = EfficientAdapterP1Model(base, EfficientAdapterP1Config.from_base_model(base, base_vocab_size=BASE_VOCAB_SIZE)).eval()
    collate = dense_collate_fn(Silence(), base)
    generator = torch.Generator().manual_seed(0)
    expected, produced = [], []
    for _ in range(20):
        prompt = torch.randint(4, BASE_VOCAB_SIZE, (5,), generator=generator)
        batch = collate([{"segments": [torch.cat([prompt, torch.tensor([PAD_TOKEN_ID])])]}])
        with torch.no_grad():
            logits = adapter(**{k: v for k, v in batch.items() if k != "text_labels"}).logits
        expected.append(int(logits[0, len(prompt), : adapter.config.vocab_size].argmax()))
        produced += dense_generate(adapter, collate, prompt, max_new_tokens=1, temperature=0.0).tolist()
    assert produced == expected


def test_character_error_rate_counts_edits_over_the_reference_without_spaces_or_marks():
    from polyplex_research.utils.evaluate import character_error_rate

    assert character_error_rate("안녕 하세요.", "안녕하세요") == 0.0
    assert character_error_rate("가나다라", "가나라") == 0.25
    assert character_error_rate("가나", "가나다라") == 1.0
    assert character_error_rate("", "가") == 1.0


def test_a_speak_log_reads_back_as_one_record_per_sample(tmp_path):
    from polyplex_research.utils.evaluate import read_speak_log

    log = tmp_path / "speak.log"
    log.write_text(
        "noise\n"
        "[speak '안녕하세요, 오늘 무엇을' #0] cer 0.97\n    text : 안녕하세요, 오늘 무엇을 이케\n    heard: 아\n"
        "[speak '네, 말씀하신 영화는' #1] cer 1.00\n    text : 네, 말씀하신 영화는 로는\n    heard: 고원은\n"
        "[speak metrics] 2 outputs  mean cer 0.985\n",
        encoding="utf-8",
    )
    records = read_speak_log(str(log))
    assert records == [
        {"opening": "안녕하세요, 오늘 무엇을", "seed": 0, "cer": 0.97, "text": "안녕하세요, 오늘 무엇을 이케", "heard": "아"},
        {"opening": "네, 말씀하신 영화는", "seed": 1, "cer": 1.0, "text": "네, 말씀하신 영화는 로는", "heard": "고원은"},
    ]


class _Letters:
    r"""A tokenizer over single letters, so piece counts and text lengths are known exactly."""

    def decode(self, ids):
        return "".join(chr(ord("a") + index) for index in ids)


def test_the_unigram_baseline_skips_each_documents_first_piece_and_counts_bytes():
    from polyplex_research.utils.evaluate import dense_unigram_baseline

    log_probs = torch.log(torch.tensor([0.5, 0.25, 0.25]))
    rows = [{"segments": [torch.tensor([0, 1, 2]), torch.tensor([2, 0])]}]
    result = dense_unigram_baseline(rows, log_probs, _Letters())
    # Scored: 1, 2 of the first document and 0 of the second.
    expected = -(log_probs[1] + log_probs[2] + log_probs[0]).item() / 3
    assert result["loss"] == pytest.approx(expected)
    assert result["tokens"] == 3
    assert result["bytes_per_token"] == pytest.approx(1.0)
    assert result["characters_per_token"] == pytest.approx(1.0)


def test_frame_embedding_norms_compare_the_text_and_the_audio_parts():
    from polyplex_research.utils.evaluate import frame_embedding_norms

    text = torch.tensor([[[3.0, 4.0], [0.0, 1.0]]])
    audio = torch.tensor([[[6.0, 8.0], [0.0, 2.0]]])
    mask = torch.tensor([[True, False]])
    norms = frame_embedding_norms(text, audio, mask)
    assert norms == {"text": pytest.approx(5.0), "audio": pytest.approx(10.0), "audio_over_text": pytest.approx(2.0)}


def test_nvfp4_variant_losses_put_the_model_back_between_variants():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from polyplex_research.utils.evaluate import nvfp4_variant_losses

    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=16)
    model = Qwen3ForCausalLM(config).eval()
    rows = [[1, 5, 9, 3, 7, 2], [4, 8, 15, 16, 23, 42]]
    first = nvfp4_variant_losses(model, rows)
    assert set(first) == {"full precision", "W4A4", "W4 only", "W4A4, down_proj full", "W4A4, MLP full",
                          "W4A4, first and last 2 layers full", "W4A4, Hadamard 128"}
    assert all(not type(module).__name__.startswith("Emulated") for module in model.modules())
    assert nvfp4_variant_losses(model, rows, variants={})["full precision"] == first["full precision"]
    assert first["W4A4"] != first["full precision"]


class SilentProcessor:
    def get_silence_audio_codes(self, frames, batch_size=1):
        return torch.ones(batch_size, 4, frames, dtype=torch.long)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) - ord("a") + 5 for character in text]


def test_the_language_ffn_report_measures_each_layer_and_puts_the_ffns_back():
    from conftest import build_polyplex
    from polyplex_research.datasets.duplex_data import dense_collate_fn
    from polyplex_research.utils.evaluate import language_ffn_report

    model = build_polyplex()
    language = [layer.mlp.language_mlp for layer in model.model.layers]
    collate = dense_collate_fn(SilentProcessor(), model)
    report = language_ffn_report(model, collate, CharacterTokenizer(), ["abcdefgh"] * 2, skip=1, top=2)
    assert [entry["layer"] for entry in report] == list(range(len(model.model.layers)))
    for entry in report:
        assert 0 <= entry["mean_share"] <= 1 and 0 < entry["top_share"] <= 1 and len(entry["top_dims"]) == 2
        assert entry["language_rms"] > 0 and entry["duplex_rms"] > 0
    assert [layer.mlp.language_mlp for layer in model.model.layers] == language


def test_a_neuron_that_writes_a_large_constant_is_ranked_first_and_its_removal_leaves_little():
    from polyplex_research.utils.evaluate import ffn_neuron_energy

    torch.manual_seed(0)
    gate, up, down = torch.randn(32, 8) * 0.1, torch.randn(32, 8) * 0.1, torch.randn(8, 32) * 0.1
    inputs = torch.randn(100, 8)
    # Neuron 5 fires strongly on every input and writes to output dimension 2.
    gate[5], up[5] = 0, 0
    gate[5, 0], up[5, 0] = 0.0, 0.0
    inputs[:, 7] = 1.0
    gate[5, 7], up[5, 7] = 10.0, 10.0
    down[:, 5] = 0
    down[2, 5] = 5.0
    report = ffn_neuron_energy(inputs, gate, up, down, top=2)
    assert report["neurons"][0] == 5 and report["output_dims"][0][0] == 2
    assert report["energy_share"][0] > 0.99 and report["output_energy_left"] < 0.01


class SpeechTokenizer:
    r"""Turns a word into one id per character, inside the tiny vocabulary and clear of `PAD` and `EPAD`."""

    def encode(self, text, add_special_tokens=False):
        return [10 + ord(character) % 40 for character in text.strip()]


class SpeechProcessor:
    r"""Stands in for the processor: a three frame prefix, codec silence of ones, and a codec that counts samples."""

    def __init__(self, num_codebooks: int = 4):
        from types import SimpleNamespace

        self.tokenizer = SpeechTokenizer()
        self.num_codebooks = num_codebooks
        self.audio_tokenizer = SimpleNamespace(config=SimpleNamespace(sampling_rate=2400, frame_rate=12.5))
        self.encoded = []

    def __call__(self, text_prompt=None, return_tensors=None):
        from polyplex_research.datasets.duplex_data import PAD_TOKEN_ID

        return {
            "input_ids": torch.full((1, 3), PAD_TOKEN_ID),
            "assistant_audio_codes": torch.zeros(1, self.num_codebooks, 3, dtype=torch.long),
            "user_audio_codes": torch.zeros(1, self.num_codebooks, 3, dtype=torch.long),
        }

    def get_silence_audio_codes(self, frames, batch_size=1):
        return torch.ones(batch_size, self.num_codebooks, frames, dtype=torch.long)

    def _encode_waveform(self, waveform):
        self.encoded.append(len(waveform))
        frames = max(1, len(waveform) // 192)
        return torch.arange(self.num_codebooks * frames).remainder(16).view(1, self.num_codebooks, frames)


def test_held_out_speech_items_are_the_first_examples_with_their_own_codes(tmp_path):
    from polyplex_research.datasets.base import SingleTurnTTSExample
    from polyplex_research.utils.evaluate import heldout_speech_items

    examples = [
        SingleTurnTTSExample(text=text, codes=torch.full((8, 3), index))
        for index, text in enumerate(("All is said", "Second one", "Third"))
    ]
    path = tmp_path / "english_heldout_speech.pt"
    torch.save(examples, path)
    items = heldout_speech_items(str(path), num_codebooks=4, count=2)
    assert [item["words"] for item in items] == [["All", "is", "said"], ["Second", "one"]]
    assert items[0]["starts"] == [0.0, 0.0, 0.0]
    assert items[1]["codes"].shape == (4, 3) and items[1]["codes"].dtype == torch.long
    assert int(items[1]["codes"][0, 0]) == 1


def test_speech_losses_split_the_model_losses_by_stream_and_codebook():
    import torch.nn.functional as F

    from polyplex_research.datasets.duplex_data import duplex_collate_fn, packed_words
    from polyplex_research.utils.evaluate import speech_losses

    from conftest import AUDIO_VOCAB_SIZE, NUM_CODEBOOKS, VOCAB_SIZE, build_polyplex

    model = build_polyplex()
    processor = SpeechProcessor(NUM_CODEBOOKS)
    torch.manual_seed(1)
    item = {"codes": torch.randint(0, AUDIO_VOCAB_SIZE, (NUM_CODEBOOKS, 12)), "words": ["ab", "cde"], "starts": [0.0, 0.0]}
    collate = duplex_collate_fn(processor, model, audio_labels=True, layout=packed_words)
    batch = collate([item])
    audio_labels = batch.pop("audio_labels")
    with torch.no_grad():
        output = model(**batch)
    result = speech_losses(model, collate, [item], torch.full((VOCAB_SIZE,), 2))
    # Every scored token stands for 2 bytes here, so bits per byte is the model's text loss over 2 ln 2.
    assert result["text_bits_per_byte"] == pytest.approx(output.loss.item() / (2 * math.log(2)), rel=1e-4)
    assert result["text_tokens"] == 5
    # 12 frames of speech: the delay layout keeps 11 of them in the window for every codebook.
    assert result["audio_codes_per_codebook"] == [11, 11, 11, 11]

    # The depth decoder reads frame t + 1's text and lower codes, conditioned on the state of frame t.
    targets = audio_labels[0, :, 1:].T.masked_fill(audio_labels[0, :, 1:].T == AUDIO_VOCAB_SIZE, -100)
    kept = (targets != -100).any(dim=-1)
    text = batch["input_ids"][0, 1:][kept].unsqueeze(1)
    codes = batch["assistant_audio_codes"][0, :, 1:].T[kept]
    with torch.no_grad():
        states = output.last_hidden_state[0, :-1][kept].unsqueeze(1)
        logits = model.depth_decoder(last_hidden_state=states, input_ids=torch.cat([text, codes[:, :-1]], 1)).logits
    first = F.cross_entropy(logits[:, 0][targets[kept][:, 0] != -100], targets[kept][:, 0][targets[kept][:, 0] != -100])
    assert result["audio_semantic_loss"] == pytest.approx(first.item(), rel=1e-5)
    weights = result["audio_codes_per_codebook"]
    acoustic = sum(l * c for l, c in zip(result["audio_loss_per_codebook"][1:], weights[1:])) / sum(weights[1:])
    assert result["audio_acoustic_loss"] == pytest.approx(acoustic)


def test_a_language_model_continues_a_prompt_greedily_after_it():
    from transformers import LlamaConfig, LlamaForCausalLM

    from polyplex_research.utils.evaluate import language_model_continuation

    class Characters:
        def encode(self, text, add_special_tokens=False):
            return [ord(character) % 64 for character in text]

        def decode(self, ids):
            return ",".join(str(index) for index in ids)

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, eos_token_id=None, pad_token_id=0,
    )
    model = LlamaForCausalLM(config).eval()
    text = language_model_continuation(model, Characters(), "abc", max_new_tokens=5)
    assert text == language_model_continuation(model, Characters(), "abc", max_new_tokens=5)
    ids = torch.tensor([[ord(character) % 64 for character in "abc"]])
    expected = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=5, do_sample=False)[0, 3:]
    assert text == ",".join(str(index) for index in expected.tolist())
    sampled = language_model_continuation(model, Characters(), "abc", max_new_tokens=5, temperature=0.8, seed=3)
    assert sampled == language_model_continuation(model, Characters(), "abc", max_new_tokens=5, temperature=0.8, seed=3)
    torch.manual_seed(3)
    expected = model.generate(
        ids, attention_mask=torch.ones_like(ids), max_new_tokens=5, do_sample=True, temperature=0.8, top_k=50
    )[0, 3:]
    assert sampled == ",".join(str(index) for index in expected.tolist())
