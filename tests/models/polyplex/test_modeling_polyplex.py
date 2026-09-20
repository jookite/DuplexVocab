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
"""Invariants of the DualFFN Polyplex model."""

import pytest
import torch
from transformers.models.personaplex.modeling_personaplex import PersonaPlexRMSNorm
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3Model, Qwen3RMSNorm

from conftest import (
    AUDIO_VOCAB_SIZE,
    BASE_VOCAB_SIZE,
    INTERMEDIATE_SIZE,
    NUM_CODEBOOKS,
    PRETRAINED_SIZE,
    VOCAB_SIZE,
    build_polyplex,
    build_polyplex_config,
)
from polyplex_research.models.polyplex import (
    PolyplexDualFFN,
    PolyplexForConditionalGeneration,
    PolyplexModel,
)


def test_the_backbone_inherits_the_qwen3_decoder():
    model = build_polyplex()
    assert isinstance(model.model, Qwen3Model)
    assert all(isinstance(layer, Qwen3DecoderLayer) for layer in model.model.layers)
    assert isinstance(model.model, PolyplexModel)


def test_the_head_normalization_is_off_by_default():
    attention = build_polyplex().model.layers[0].self_attn
    assert isinstance(attention.q_norm, torch.nn.Identity) and isinstance(attention.k_norm, torch.nn.Identity)


def test_the_head_normalization_can_be_switched_on():
    attention = build_polyplex(use_qk_norm=True).model.layers[0].self_attn
    assert isinstance(attention.q_norm, Qwen3RMSNorm) and isinstance(attention.k_norm, Qwen3RMSNorm)


def test_every_layer_is_a_sliding_window_layer():
    config = build_polyplex_config()
    assert config.layer_types == ["sliding_attention"] * config.num_hidden_layers


def test_each_ffn_reads_the_stream_through_its_own_normalization():
    layer = build_polyplex().model.layers[0]
    assert isinstance(layer.mlp, PolyplexDualFFN)
    assert isinstance(layer.mlp.duplex_norm, PersonaPlexRMSNorm)
    assert isinstance(layer.mlp.language_norm, Qwen3RMSNorm)
    assert layer.mlp.duplex_norm.variance_epsilon == 1e-8
    assert layer.mlp.language_norm.variance_epsilon == 1e-6


def test_the_ffn_output_is_the_sum_of_the_two_ffns():
    ffn = build_polyplex().model.layers[0].mlp
    torch.manual_seed(3)
    hidden = torch.randn(2, 5, 32)
    expected = ffn.duplex_mlp(ffn.duplex_norm(hidden)) + ffn.language_mlp(ffn.language_norm(hidden))
    torch.testing.assert_close(ffn(hidden), expected, rtol=0, atol=0)


def test_the_language_scale_multiplies_the_language_ffn_output_per_dimension():
    ffn = build_polyplex(language_scale=True).model.layers[0].mlp
    torch.manual_seed(3)
    hidden = torch.randn(2, 5, 32)
    with torch.no_grad():
        ffn.language_scale.copy_(torch.linspace(-1, 1, 32))
    language = ffn.language_mlp(ffn.language_norm(hidden)) * torch.linspace(-1, 1, 32)
    expected = ffn.duplex_mlp(ffn.duplex_norm(hidden)) + language
    torch.testing.assert_close(ffn(hidden), expected, rtol=0, atol=0)


def _stream_model():
    model = build_polyplex(language_scale=True, language_stream=True)
    torch.manual_seed(5)
    with torch.no_grad():
        for layer in model.model.layers:
            rotation, _ = torch.linalg.qr(torch.randn(32, 32))
            layer.mlp.stream_rotation.copy_(rotation)
            layer.mlp.stream_scale.fill_(0.5)
            layer.mlp.language_scale.copy_(torch.linspace(-1, 1, 32))
    return model


def test_the_language_stream_layer_follows_the_two_stream_update():
    model = _stream_model()
    layer = model.model.layers[0]
    ffn = layer.mlp
    torch.manual_seed(3)
    h, q = torch.randn(1, 5, 32), torch.randn(1, 5, 32)
    position_ids = torch.arange(5)[None]
    position_embeddings = model.model.rotary_emb(h, position_ids)
    with torch.no_grad():
        out = layer(torch.cat([h, q], -1), position_embeddings=position_embeddings, position_ids=position_ids)
        attention, _ = layer.self_attn(layer.input_layernorm(h), position_embeddings=position_embeddings, attention_mask=None)
        h1 = h + attention
        q1 = q + attention @ ffn.stream_rotation.T / 0.5
        f = ffn.language_mlp(ffn.language_norm(q1))
        h2 = h1 + ffn.duplex_mlp(ffn.duplex_norm(h1)) + ffn.language_scale * (0.5 * f @ ffn.stream_rotation)
    torch.testing.assert_close(out, torch.cat([h2, q1 + f], -1), rtol=1e-5, atol=1e-5)


def test_a_zero_language_scale_on_the_stream_gives_the_model_with_the_language_ffn_off(text_batch):
    from polyplex_research.utils.phase import language_ffn_off

    model = _stream_model()
    plain = build_polyplex()
    plain.load_state_dict(model.state_dict(), strict=False)
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.language_scale.zero_()
        with language_ffn_off(plain):
            expected = plain(input_ids=text_batch["input_ids"]).logits
        streamed = model(input_ids=text_batch["input_ids"]).logits
    torch.testing.assert_close(streamed, expected, rtol=1e-5, atol=1e-5)


def test_the_language_stream_starts_from_the_text_embeddings_alone(audio_batch):
    model = _stream_model()
    ffn = model.model.layers[0].mlp
    into_stream = ffn.into_stream
    starts = []

    def recording(hidden_states):
        starts.append(hidden_states)
        return into_stream(hidden_states)

    ffn.into_stream = recording
    with torch.no_grad():
        model(input_ids=audio_batch["input_ids"], assistant_audio_codes=audio_batch["assistant_audio_codes"],
              user_audio_codes=audio_batch["user_audio_codes"])
        text = model.embed_text(audio_batch["input_ids"])
    torch.testing.assert_close(starts[0], text, rtol=0, atol=0)


def test_a_stream_started_from_the_whole_input_reads_the_audio_embeddings_too(audio_batch):
    model = _stream_model()
    model.config.language_stream_from_text = False
    ffn = model.model.layers[0].mlp
    into_stream = ffn.into_stream
    starts = []

    def recording(hidden_states):
        starts.append(hidden_states)
        return into_stream(hidden_states)

    ffn.into_stream = recording
    with torch.no_grad():
        model(input_ids=audio_batch["input_ids"], assistant_audio_codes=audio_batch["assistant_audio_codes"],
              user_audio_codes=audio_batch["user_audio_codes"])
        text = model.embed_text(audio_batch["input_ids"])
    assert not torch.equal(starts[0], text)


def test_without_audio_codes_the_stream_starts_from_the_same_embeddings_as_the_backbone(text_batch):
    model = _stream_model()
    with torch.no_grad():
        embeds = model.embed_text(text_batch["input_ids"])
        expected = model.lm_head(model.model(inputs_embeds=embeds).text_hidden_state)
        out = model(input_ids=text_batch["input_ids"]).logits
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_unscored_text_frames_still_condition_the_depth_decoder(audio_batch):
    model = build_polyplex()
    batch = dict(audio_batch)
    batch["text_labels"] = batch["input_ids"].clone()
    with torch.no_grad():
        scored = model(**batch).depth_loss
        batch["text_labels"][:, :2] = -100
        unscored = model(**batch).depth_loss
    torch.testing.assert_close(unscored, scored, rtol=0, atol=0)


def test_the_language_stream_needs_the_language_scale():
    with pytest.raises(ValueError, match="language_scale"):
        build_polyplex_config(language_stream=True)


def test_a_model_without_the_language_scale_has_no_such_parameter():
    assert not any("language_scale" in name for name, _ in build_polyplex().named_parameters())






def test_a_configured_slice_outside_the_reserve_is_refused():
    with pytest.raises(ValueError, match="reserved"):
        build_polyplex_config(duplex_ffn_languages={"ko": [30, 44]})


def test_text_logits_read_the_language_side_normalization(text_batch):
    model = build_polyplex()
    with torch.no_grad():
        model.model.text_norm.weight.mul_(2.0)
        embeds = model.model.embed_tokens(text_batch["input_ids"])
        out = model(inputs_embeds=embeds)
        backbone = model.model(inputs_embeds=embeds)
    torch.testing.assert_close(out.logits, model.lm_head(backbone.text_hidden_state), rtol=0, atol=0)
    assert not torch.equal(backbone.text_hidden_state, backbone.last_hidden_state)


def test_the_model_trains_on_text_and_audio_labels(audio_batch):
    model = build_polyplex().train()
    batch = dict(audio_batch)
    batch["text_labels"] = batch["input_ids"].clone()
    batch["user_audio_codes"] = batch["user_audio_codes"]
    out = model(**batch)
    assert out.loss is not None and out.depth_loss is not None
    torch.testing.assert_close(out.loss - out.depth_loss, model.loss_function(
        logits=out.logits, labels=batch["text_labels"], vocab_size=model.config.vocab_size
    ))
    out.loss.backward()
    assert model.lm_head.weight.grad is not None



def test_a_saved_model_reloads_with_its_languages(tmp_path):
    model = build_polyplex(duplex_ffn_languages={"ko": [40, 44]})
    model.save_pretrained(tmp_path)
    reloaded = PolyplexForConditionalGeneration.from_pretrained(tmp_path)
    assert reloaded.config.duplex_ffn_languages == {"ko": [40, 44]}
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[name], tensor, rtol=0, atol=0)


def test_a_document_packed_after_another_reads_only_itself():
    model = build_polyplex()
    torch.manual_seed(9)
    first = torch.randint(0, VOCAB_SIZE, (1, 4))
    second = torch.randint(0, VOCAB_SIZE, (1, 5))
    codes = torch.randint(0, AUDIO_VOCAB_SIZE, (1, NUM_CODEBOOKS, 9))
    packed_positions = torch.cat([torch.arange(4), torch.arange(5)])[None]
    with torch.no_grad():
        packed = model(
            input_ids=torch.cat([first, second], dim=1),
            position_ids=packed_positions,
            assistant_audio_codes=codes,
            user_audio_codes=codes,
            use_cache=False,
        ).logits[:, 4:]
        alone = model(
            input_ids=second,
            position_ids=torch.arange(5)[None],
            assistant_audio_codes=codes[..., 4:],
            user_audio_codes=codes[..., 4:],
            use_cache=False,
        ).logits
    torch.testing.assert_close(packed, alone, rtol=1e-4, atol=1e-5)


def build_composed(seed: int = 0):
    r"""
    Returns:
        [`PolyplexForConditionalGeneration`]: A model whose text tables share composition coefficients over the first
        `BASE_VOCAB_SIZE` tokens, with tokens 50 and 51 composed of base tokens and every stored row random.
    """
    model = build_polyplex(seed, text_composition_size=2, base_vocab_size=BASE_VOCAB_SIZE)
    with torch.no_grad():
        model.text_composition.indices[50] = torch.tensor([3, 17])
        model.text_composition.values[50] = torch.tensor([0.5, -1.5])
        model.text_composition.indices[51] = torch.tensor([4, 0])
        model.text_composition.values[51] = torch.tensor([2.0, 0.0])
    return model


def composed_row(model, table, token):
    composition = model.text_composition
    basis = table[:BASE_VOCAB_SIZE]
    return table[token] + (composition.values[token][:, None] * basis[composition.indices[token]]).sum(0)


def test_every_text_table_reads_its_stored_row_plus_the_shared_composition_of_its_own_base_rows():
    model = build_composed()
    ids = torch.tensor([[VOCAB_SIZE, 3, 50, 51]])
    embeds = model.embed_text(ids)
    table = model.model.embed_tokens.weight
    torch.testing.assert_close(embeds[0, 2], composed_row(model, table, 50))
    torch.testing.assert_close(embeds[0, 3], composed_row(model, table, 51))
    # A base token and the opening frame's reserved id have no coefficients and read their stored rows.
    torch.testing.assert_close(embeds[0, 1], table[3])
    torch.testing.assert_close(embeds[0, 0], table[VOCAB_SIZE])

    hidden = torch.randn(5, 32)
    head = model.lm_head.weight
    expected = torch.stack([composed_row(model, head, token) for token in range(VOCAB_SIZE)])
    torch.testing.assert_close(model.lm_head(hidden), hidden @ expected.T)

    depth = model.depth_decoder.model.text_embed_tokens
    torch.testing.assert_close(depth(torch.tensor([50]))[0], composed_row(model, depth.weight, 50))


def test_the_coefficients_start_at_zero():
    model = build_polyplex(text_composition_size=2, base_vocab_size=BASE_VOCAB_SIZE)
    assert model.text_composition.values.shape == (VOCAB_SIZE + 1, 2)
    assert not model.text_composition.values.any()


def test_a_token_seen_only_as_a_target_moves_its_coefficients_and_so_its_input_row():
    model = build_composed().train()
    ids = torch.tensor([[VOCAB_SIZE, 40, 41, 42, 43]])
    labels = torch.tensor([[-100, -100, -100, -100, 50]])
    codes = torch.zeros(1, NUM_CODEBOOKS, 5, dtype=torch.long)
    out = model(input_ids=ids, assistant_audio_codes=codes, user_audio_codes=codes, text_labels=labels, use_cache=False)
    out.loss.backward()
    # Token 50 never enters as input, so its coefficients are moved through the head alone.
    assert model.text_composition.values.grad[50].abs().sum() > 0


def test_a_saved_model_reloads_its_composition(tmp_path):
    model = build_composed()
    model.save_pretrained(tmp_path)
    reloaded = PolyplexForConditionalGeneration.from_pretrained(tmp_path)
    assert torch.equal(reloaded.text_composition.indices, model.text_composition.indices)
    ids = torch.tensor([[1, 50, 51]])
    torch.testing.assert_close(reloaded.embed_text(ids), model.embed_text(ids))


def test_a_bfloat16_composition_is_summed_in_float32_and_returned_in_the_tables_dtype():
    model = build_composed().to(torch.bfloat16)
    composition = model.text_composition
    basis = model.lm_head.weight[:BASE_VOCAB_SIZE]
    ids = torch.tensor([50, 51, 3])
    rows = composition(basis, ids)
    assert rows.dtype == torch.bfloat16
    expected = (composition.values[ids].float()[..., None] * basis.float()[composition.indices[ids]]).sum(1)
    torch.testing.assert_close(rows, expected.to(torch.bfloat16))
    rows.float().sum().backward()
    assert composition.values.grad is not None and composition.values.grad.dtype == torch.bfloat16


def test_the_composition_sends_no_gradient_to_the_fixed_base_rows():
    model = build_composed()
    model.embed_text(torch.tensor([[50]])).sum().backward()
    gradient = model.model.embed_tokens.weight.grad
    assert gradient[50].abs().sum() > 0
    assert not gradient[3].any() and not gradient[17].any()
    assert model.text_composition.values.grad[50].abs().sum() > 0

    # Every head row gets its own logit's gradient; the base rows get nothing more through the composition.
    hidden = torch.randn(2, 32)
    model.lm_head(hidden).sum().backward()
    head = model.lm_head.weight
    torch.testing.assert_close(head.grad, hidden.sum(0).expand_as(head))


def test_a_sparse_input_table_gets_gradient_rows_only_for_the_ids_read():
    model = build_composed()
    model.model.embed_tokens.sparse = True
    model.embed_text(torch.tensor([[50, 3]])).sum().backward()
    gradient = model.model.embed_tokens.weight.grad
    assert gradient.is_sparse
    assert set(gradient.coalesce().indices()[0].tolist()) == {3, 50}


def test_a_language_table_of_its_own_starts_the_stream_unrotated(audio_batch):
    from torch import nn

    model = _stream_model()
    model.config.language_embeddings = True
    model.language_embed_tokens = nn.Embedding(model.model.embed_tokens.num_embeddings, model.config.hidden_size)
    starts = []
    model.model.layers[0].register_forward_pre_hook(lambda module, args: starts.append(args[0]))
    with torch.no_grad():
        model(input_ids=audio_batch["input_ids"], assistant_audio_codes=audio_batch["assistant_audio_codes"],
              user_audio_codes=audio_batch["user_audio_codes"])
        expected = model.language_embed_tokens(audio_batch["input_ids"])
    torch.testing.assert_close(starts[0][..., model.config.hidden_size :], expected, rtol=0, atol=0)


def test_a_config_with_a_language_lora_rank_builds_the_corrections_at_zero():
    model = build_polyplex(language_lora_rank=3)
    mlp = model.model.layers[1].mlp.language_mlp
    assert mlp.gate_lora[0].weight.shape == (3, model.config.hidden_size)
    assert mlp.down_lora[1].weight.shape == (model.config.hidden_size, 3)
    assert not mlp.down_lora[1].weight.any() and not mlp.up_lora[1].weight.any()


def test_a_language_stream_whose_layers_share_one_rotation_folds_into_the_same_function(text_batch):
    from polyplex_research.utils.assemble import fold_language_stream

    model = _stream_model()
    torch.manual_seed(7)
    rotation, _ = torch.linalg.qr(torch.randn(32, 32))
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.stream_rotation.copy_(rotation)
            layer.mlp.language_norm.weight.copy_(torch.rand(32) + 0.5)
        expected = model(input_ids=text_batch["input_ids"]).logits
    fold_language_stream(model, rotation, 0.5)
    assert model.config.language_stream_folded
    assert not any("stream_rotation" in name or "stream_scale" in name for name, _ in model.named_buffers())
    assert all(torch.equal(layer.mlp.language_norm.weight, torch.ones(32)) for layer in model.model.layers)
    with torch.no_grad():
        folded = model(input_ids=text_batch["input_ids"]).logits
    torch.testing.assert_close(folded, expected, rtol=1e-4, atol=1e-4)


def test_a_folded_stream_cannot_be_folded_again():
    from polyplex_research.utils.assemble import fold_language_stream

    model = _stream_model()
    fold_language_stream(model, torch.eye(32), 1.0)
    with pytest.raises(ValueError, match="unfolded"):
        fold_language_stream(model, torch.eye(32), 1.0)


def test_a_folded_language_stream_config_builds_no_rotation_buffers():
    model = build_polyplex(language_scale=True, language_stream=True, language_stream_folded=True)
    assert not any("stream_rotation" in name for name, _ in model.named_buffers())
