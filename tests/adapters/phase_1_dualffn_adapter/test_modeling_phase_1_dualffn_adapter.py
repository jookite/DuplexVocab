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
"""Invariants of the Phase 1 DualFFN adapter."""

import pytest
import torch

from conftest import AUDIO_VOCAB_SIZE, NUM_CODEBOOKS, VOCAB_SIZE
from polyplex_research.adapters.phase_1_dualffn_adapter import (
    Phase1DualFFNAdapterConfig,
    Phase1DualFFNAdapterModel,
)
from polyplex_research.models.polyplex import PolyplexConfig, PolyplexForConditionalGeneration


NEW_TOKENS = [56, VOCAB_SIZE]


def build_model(seed: int = 0) -> PolyplexForConditionalGeneration:
    torch.manual_seed(seed)
    config = PolyplexConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        intermediate_size=48,
        duplex_pretrained_size=40,
        audio_vocab_size=AUDIO_VOCAB_SIZE,
        num_codebooks=NUM_CODEBOOKS,
        max_position_embeddings=64,
        sliding_window=64,
        depth_decoder_config={
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "ffn_dim": 32,
            "num_codebooks": NUM_CODEBOOKS,
        },
    )
    model = PolyplexForConditionalGeneration(config).eval()
    # The reserved neurons as an assembled checkpoint holds them.
    with torch.no_grad():
        for layer in model.model.layers:
            mlp = layer.mlp.duplex_mlp
            mlp.gate_proj.weight[40:] = 0
            mlp.up_proj.weight[40:] = 0
            mlp.down_proj.weight[:, 40:] = 0
    return model


def build_adapter(model=None, trained=True, **overrides):
    model = model if model is not None else build_model()
    values = dict(language="ko", duplex_slice_size=4, new_tokens=NEW_TOKENS, seed=1)
    values.update(overrides)
    adapter = Phase1DualFFNAdapterModel(model, Phase1DualFFNAdapterConfig(**values))
    if trained:
        # A trained slice, so that the tests see the slice's contribution and not only zeros.
        with torch.no_grad():
            for piece in adapter.slices:
                piece.down.normal_(0, 0.1)
    return adapter


@pytest.fixture
def batch():
    torch.manual_seed(2)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 6))
    input_ids[0, 3] = 58
    codes = torch.randint(0, AUDIO_VOCAB_SIZE, (2, NUM_CODEBOOKS, 6))
    return {
        "input_ids": input_ids,
        "assistant_audio_codes": codes,
        "user_audio_codes": torch.full_like(codes, AUDIO_VOCAB_SIZE),
        "text_labels": input_ids.clone(),
    }


def test_wrapping_leaves_the_output_unchanged(batch):
    model = build_model()
    with torch.no_grad():
        expected = model(**batch).logits.clone()
    adapter = build_adapter(model, trained=False)
    with torch.no_grad():
        torch.testing.assert_close(adapter(**batch).logits, expected, rtol=1e-5, atol=1e-6)


def test_allocation_records_the_slice_and_starts_from_a_silent_down_projection():
    adapter = build_adapter(trained=False)
    assert adapter.model.config.duplex_ffn_languages == {"ko": [40, 44]}
    for piece in adapter.slices:
        assert torch.count_nonzero(piece.gate) > 0 and torch.count_nonzero(piece.up) > 0
        assert torch.count_nonzero(piece.down) == 0


def test_the_next_language_takes_the_next_slice_and_the_reserve_is_bounded():
    model = build_adapter().fold_into_base()
    adapter = build_adapter(model, language="ja", duplex_slice_size=3, new_tokens=None)
    assert model.config.duplex_ffn_languages == {"ko": [40, 44], "ja": [44, 47]}
    assert adapter.neurons == (44, 47)
    with pytest.raises(ValueError, match="reserved"):
        build_adapter(adapter.fold_into_base(), language="zh", duplex_slice_size=2, new_tokens=None)


def test_a_language_that_holds_a_slice_keeps_it():
    model = build_adapter().fold_into_base()
    adapter = build_adapter(model, duplex_slice_size=None)
    assert adapter.neurons == (40, 44)


def test_only_the_language_slice_and_its_rows_train():
    adapter = build_adapter()
    trainable = {name for name, parameter in adapter.named_parameters() if parameter.requires_grad}
    assert all(not parameter.requires_grad for parameter in adapter.model.parameters())
    assert trainable == {
        *(f"slices.{i}.{part}" for i in range(2) for part in ("gate", "up", "down")),
        "input_rows",
        "output_rows",
    }
    assert adapter.slices[0].gate.shape == (4, 32) and adapter.slices[0].down.shape == (32, 4)
    assert adapter.input_rows.shape == adapter.output_rows.shape == (NEW_TOKENS[1] - NEW_TOKENS[0], 32)


def test_the_frozen_slice_is_zero_inside_the_model():
    adapter = build_adapter()
    for layer in adapter.model.model.layers:
        mlp = layer.mlp.duplex_mlp
        assert torch.count_nonzero(mlp.gate_proj.weight[40:44]) == 0
        assert torch.count_nonzero(mlp.down_proj.weight[:, 40:44]) == 0


def test_the_loss_reaches_every_trained_part(batch):
    adapter = build_adapter().train()
    adapter(**batch).loss.backward()
    for name, parameter in adapter.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0, name


def test_a_step_moves_nothing_of_the_base_model(batch):
    adapter = build_adapter().train()
    before = {name: tensor.clone() for name, tensor in adapter.model.state_dict().items()}
    optimizer = torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad], lr=1e-2, weight_decay=0.1)
    adapter(**batch).loss.backward()
    optimizer.step()
    for name, tensor in adapter.model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


def test_new_rows_replace_the_lookup_and_the_logits_of_their_tokens(batch):
    adapter = build_adapter()
    model = adapter.model
    start, end = NEW_TOKENS
    with torch.no_grad():
        adapter.input_rows.mul_(3.0)
        adapter.output_rows.mul_(3.0)
        logits = adapter(**batch).logits
        text = model.model.embed_tokens(batch["input_ids"])
        audio = model._embed_audio_codes(torch.cat([batch["assistant_audio_codes"], batch["user_audio_codes"]], 1))
        hidden = model.model(inputs_embeds=text + audio).text_hidden_state
        reference = model.lm_head(hidden)
    assert torch.equal(text[0, 3], adapter.input_rows[58 - start])
    torch.testing.assert_close(logits[..., start:end], hidden @ adapter.output_rows.T, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(logits[..., :start], reference[..., :start], rtol=0, atol=0)


def test_folding_reproduces_the_adapter_on_the_bare_model(batch):
    adapter = build_adapter().train()
    optimizer = torch.optim.SGD([p for p in adapter.parameters() if p.requires_grad], lr=0.5)
    adapter(**batch).loss.backward()
    optimizer.step()
    adapter.eval()
    with torch.no_grad():
        expected = adapter(**batch).logits.clone()
    model = adapter.fold_into_base()
    with torch.no_grad():
        torch.testing.assert_close(model(**batch).logits, expected, rtol=1e-5, atol=1e-5)


def test_a_saved_adapter_loads_onto_a_fresh_model(tmp_path, batch):
    adapter = build_adapter()
    with torch.no_grad():
        for parameter in adapter.parameters():
            if parameter.requires_grad:
                parameter.add_(0.05)
        expected = adapter(**batch).logits.clone()
    adapter.save_pretrained(tmp_path)
    reloaded = Phase1DualFFNAdapterModel.from_pretrained(tmp_path, model=build_model())
    with torch.no_grad():
        torch.testing.assert_close(reloaded(**batch).logits, expected, rtol=0, atol=0)


def test_a_new_language_needs_a_slice_size():
    model = build_model()
    with pytest.raises(ValueError, match="duplex_slice_size"):
        Phase1DualFFNAdapterModel(model, Phase1DualFFNAdapterConfig(language="ja"))
