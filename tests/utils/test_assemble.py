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
"""What a phase leaves behind has to become a checkpoint the base class loads."""

import os

import pytest
import torch
import torch.nn.utils.parametrize as parametrize

from conftest import AUDIO_VOCAB_SIZE, BASE_VOCAB_SIZE, NUM_CODEBOOKS, VOCAB_SIZE, build_config
from polyplex_research.adapters.phase_1_efficient_adapter import (
    EfficientAdapterP1Config,
    EfficientAdapterP1Model,
)
from polyplex_research.adapters.phase_2_code_rotation_adapter import (
    CodeRotationAdapterConfig,
    CodeRotationAdapterModel,
)
from polyplex_research.models.polyplex import PolyplexForConditionalGeneration
from polyplex_research.models.polyplex.configuration_polyplex import PolyplexConfig
from polyplex_research.utils.assemble import Phase1Assemble, Phase2Assemble


def build_polyplex(predicts_user_stream: bool = True, seed: int = 0):
    r"""
    Args:
        predicts_user_stream (`bool`, *optional*, defaults to `True`):
            Whether the depth decoder carries the user stream heads as well.
        seed (`int`, *optional*, defaults to 0):
            Seed of the weight initialization.

    Returns:
        `PolyplexForConditionalGeneration`: A model of the class the phases save and load.
    """
    torch.manual_seed(seed)
    config = PolyplexConfig(**build_config(predicts_user_stream).to_dict())
    return PolyplexForConditionalGeneration(config).eval()


@pytest.fixture
def audio_batch_both_streams():
    torch.manual_seed(3)
    codes = torch.randint(0, AUDIO_VOCAB_SIZE, (2, NUM_CODEBOOKS, 6))
    return {
        "input_ids": torch.randint(0, BASE_VOCAB_SIZE, (2, 6)),
        "attention_mask": torch.ones(2, 6, dtype=torch.bool),
        "assistant_audio_codes": codes,
        "user_audio_codes": torch.randint(0, AUDIO_VOCAB_SIZE, (2, NUM_CODEBOOKS, 6)),
        "audio_labels": codes.clone(),
    }


# ---------------------------------------------------------------------------
# Which tables the rotation reaches
# ---------------------------------------------------------------------------


def rotated_tables(model) -> dict:
    r"""
    Args:
        model (`PolyplexForConditionalGeneration`):
            A model the rotation was attached to.

    Returns:
        `dict`: Whether each audio table of the backbone and the depth decoder is parametrized, keyed by
        a readable name.
    """
    tables = {
        f"backbone.{index}": parametrize.is_parametrized(embedding, "weight")
        for index, embedding in enumerate(model.embed_tokens)
    }
    tables.update(
        {
            f"depth.{index}": parametrize.is_parametrized(embedding, "weight")
            for index, embedding in enumerate(model.depth_decoder.model.embed_tokens)
        }
    )
    return tables


def test_tie_self_input_rotates_every_table_that_reads_a_code_back():
    model = build_polyplex()
    CodeRotationAdapterModel(
        model, CodeRotationAdapterConfig.from_base_model(model, tie_self_input=True)
    )

    assert all(rotated_tables(model).values())
    assert parametrize.is_parametrized(model.depth_decoder.lm_heads, "weight")


def test_the_rotation_stops_at_what_is_not_an_acoustic_code():
    r"""
    The residual is a change of basis on the code index, so it reaches a table only if that table is
    indexed by a code. Inside the depth decoder two are not: the text stream is Phase 1's, and the
    temporal hidden state arrives as a vector rather than an index.
    """
    model = build_polyplex()
    CodeRotationAdapterModel(model, CodeRotationAdapterConfig.from_base_model(model))

    depth = model.depth_decoder.model
    assert not parametrize.is_parametrized(depth.text_embed_tokens, "weight")
    assert not parametrize.is_parametrized(depth.input_projections, "weight")
    assert not parametrize.is_parametrized(model.model.embed_tokens, "weight")


def test_untied_self_input_leaves_the_assistant_tables_on_the_frozen_convention():
    model = build_polyplex()
    CodeRotationAdapterModel(
        model, CodeRotationAdapterConfig.from_base_model(model, tie_self_input=False)
    )

    rotated = rotated_tables(model)
    backbone = [rotated[f"backbone.{index}"] for index in range(2 * NUM_CODEBOOKS)]
    assert backbone == [False] * NUM_CODEBOOKS + [True] * NUM_CODEBOOKS
    # The depth decoder holds one table per codebook that can be context, the last one excluded.
    depth = [rotated[f"depth.{index}"] for index in range(2 * NUM_CODEBOOKS - 1)]
    assert depth == [False] * NUM_CODEBOOKS + [True] * (NUM_CODEBOOKS - 1)


def test_reading_a_code_back_inverts_the_head_at_a_permutation():
    r"""
    The claim the rotation rests on: what the head calls `c` is what the tables read as `c`.
    """
    model = build_polyplex()
    adapter = CodeRotationAdapterModel(model, CodeRotationAdapterConfig.from_base_model(model))

    frozen = model.embed_tokens[0].parametrizations.weight.original.clone()
    permutation = torch.randperm(AUDIO_VOCAB_SIZE)
    matrix = torch.zeros(AUDIO_VOCAB_SIZE, AUDIO_VOCAB_SIZE)
    matrix[torch.arange(AUDIO_VOCAB_SIZE), permutation] = 1.0
    with torch.no_grad():
        adapter.adapter.delta_w.copy_(matrix - torch.eye(AUDIO_VOCAB_SIZE))

    rotated = model.embed_tokens[0].weight
    assert torch.allclose(rotated[: AUDIO_VOCAB_SIZE], frozen[permutation], atol=1e-5)
    # The reserved row past the vocabulary is never moved.
    assert torch.allclose(rotated[AUDIO_VOCAB_SIZE], frozen[AUDIO_VOCAB_SIZE])


def test_the_rotation_is_evaluated_against_the_weight_and_not_against_the_identity():
    r"""
    The rotation is applied as `W + dW W`, never as `(I + dW) W`.

    The two are the same matrix and not the same computation in a low precision dtype: forming
    `I + dW` first rounds every entry against 1, where bfloat16 spaces values 2^-8 apart, so a residual
    that has not yet grown past that is discarded before it reaches the weight at all. The form used
    here rounds against the weight instead, which is the scale the values actually live on.
    """
    model = build_polyplex()
    adapter = CodeRotationAdapterModel(model, CodeRotationAdapterConfig.from_base_model(model))

    heads = model.depth_decoder.lm_heads
    frozen = heads.parametrizations.weight.original.clone()
    with torch.no_grad():
        adapter.adapter.delta_w.normal_(0, 0.02)

    assert torch.allclose(
        heads.weight,
        frozen + torch.matmul(adapter.adapter.delta_w, frozen),
        atol=1e-6,
    )


def test_every_head_is_rotated_including_the_user_stream():
    r"""
    A head writes a code index and something later reads it back, so leaving one on the frozen
    convention would have the rotated embeddings misread what it emitted. The channels keep their roles
    through their own frozen weights, not through which of them the relabelling reaches.
    """
    model = build_polyplex()
    adapter = CodeRotationAdapterModel(model, CodeRotationAdapterConfig.from_base_model(model))

    heads = model.depth_decoder.lm_heads
    frozen = heads.parametrizations.weight.original.clone()
    with torch.no_grad():
        adapter.adapter.delta_w.normal_(0, 0.02)

    moved = (heads.weight - frozen).abs().amax(dim=(1, 2))
    assert (moved > 0).all(), "a head kept its frozen value"
    assert moved.shape[0] == 2 * NUM_CODEBOOKS


# ---------------------------------------------------------------------------
# Folding
# ---------------------------------------------------------------------------


def test_phase2_dessemble_reproduces_the_adapter_exactly(tmp_path, audio_batch_both_streams):
    base = build_polyplex()
    checkpoint = tmp_path / "base"
    base.save_pretrained(checkpoint)

    adapter = CodeRotationAdapterModel(base, CodeRotationAdapterConfig.from_base_model(base))
    adapter.adapter.float()
    with torch.no_grad():
        adapter.adapter.delta_w.normal_(0, 0.02)
    adapter_dir = tmp_path / "adapter"
    adapter.save_pretrained(adapter_dir)

    with torch.no_grad():
        expected = adapter(**audio_batch_both_streams).logits.clone()

    merged_dir = tmp_path / "merged"
    Phase2Assemble(str(checkpoint), torch_dtype="float32").dessemble(
        str(adapter_dir), output_dir=str(merged_dir)
    )

    reloaded = PolyplexForConditionalGeneration.from_pretrained(merged_dir).eval()
    assert not parametrize.is_parametrized(reloaded.depth_decoder.lm_heads, "weight")

    # A zero residual on the merged checkpoint is the same computation the trained adapter ran.
    probe = CodeRotationAdapterModel(
        reloaded, CodeRotationAdapterConfig.from_base_model(reloaded)
    )
    with torch.no_grad():
        probe.adapter.delta_w.zero_()
        assert torch.allclose(probe(**audio_batch_both_streams).logits, expected, atol=1e-5)


def test_phase2_dessemble_refuses_the_tied_kind(tmp_path):
    base = build_polyplex()
    adapter = CodeRotationAdapterModel(
        base, CodeRotationAdapterConfig.from_base_model(base, adapter_kind="tied")
    )
    with pytest.raises(ValueError, match="no weight space form"):
        adapter.fold_into_base()


def test_phase1_dessemble_folds_the_tie_into_the_input_table(tmp_path, text_batch):
    base = build_polyplex()
    base.config.base_vocab_size = BASE_VOCAB_SIZE
    checkpoint = tmp_path / "expanded"
    base.save_pretrained(checkpoint)

    config = EfficientAdapterP1Config.from_base_model(base, base_vocab_size=BASE_VOCAB_SIZE)
    adapter = EfficientAdapterP1Model(base, config)
    adapter.freeze_base()
    with torch.no_grad():
        head = base.get_output_embeddings().weight
        head[BASE_VOCAB_SIZE:] = torch.randn_like(head[BASE_VOCAB_SIZE:])
    adapter_dir = tmp_path / "adapter"
    adapter.save_pretrained(adapter_dir)

    merged_dir = tmp_path / "merged"
    merged = Phase1Assemble(str(checkpoint), torch_dtype="float32").dessemble(
        str(adapter_dir), output_dir=str(merged_dir)
    )

    # The plain lookup of an appended id now returns what `tied_input_embeddings` read off the head.
    scale = adapter._tied_input_scale
    rows = merged.get_input_embeddings().weight[BASE_VOCAB_SIZE : config.vocab_size]
    expected = merged.get_output_embeddings().weight[BASE_VOCAB_SIZE : config.vocab_size] * scale
    assert torch.allclose(rows, expected, atol=1e-5)
    # The pretrained rows and the reserved row are untouched by the fold.
    reloaded = PolyplexForConditionalGeneration.from_pretrained(merged_dir)
    assert torch.allclose(
        reloaded.get_input_embeddings().weight[:BASE_VOCAB_SIZE],
        merged.get_input_embeddings().weight[:BASE_VOCAB_SIZE],
    )


def test_phase1_fold_refuses_what_no_table_can_hold():
    base = build_polyplex()
    config = EfficientAdapterP1Config.from_base_model(
        base, base_vocab_size=BASE_VOCAB_SIZE, group_gated_softmax=True
    )
    adapter = EfficientAdapterP1Model(base, config)
    with pytest.raises(ValueError, match="group_gated_softmax"):
        adapter.fold_into_base()


# ---------------------------------------------------------------------------
# DualFFN grafting
# ---------------------------------------------------------------------------


def build_language_model(seed: int = 3):
    r"""
    Returns:
        `Qwen3ForCausalLM`: A text LLM of the backbone's width, with three layers so a pairing has to choose.
    """
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    config = Qwen3Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
    )
    return Qwen3ForCausalLM(config).eval()


def build_grafted(language_ffn_layers=None, appended_layers=0):
    from conftest import build_model

    from polyplex_research.utils.assemble import dualffn_config, graft_helium, graft_language_model

    helium = build_model(predicts_user_stream=True)
    language_model = build_language_model()
    config = dualffn_config(helium.config, language_model.config, language_ffn_layers, appended_layers=appended_layers)
    torch.manual_seed(5)
    model = PolyplexForConditionalGeneration(config).eval()
    graft_helium(model, helium)
    graft_language_model(model, language_model)
    return model, helium, language_model


def test_the_dualffn_config_takes_its_widths_from_both_models():
    model, helium, language_model = build_grafted([0, 2])
    config = model.config
    assert config.duplex_pretrained_size == helium.config.ffn_dim // 2 == 32
    assert config.intermediate_size == language_model.config.intermediate_size == 48
    assert config.vocab_size == language_model.config.vocab_size
    assert config.depth_decoder_config.vocab_size == config.vocab_size
    assert config.language_ffn_layers == [0, 2]
    assert config.language_rms_norm_eps == language_model.config.rms_norm_eps
    assert config.rms_norm_eps == helium.config.rms_norm_eps


def test_the_default_pairing_keeps_the_first_layers():
    model, _, _ = build_grafted()
    assert model.config.language_ffn_layers == [0, 1]


def test_a_pairing_outside_the_language_model_is_refused():
    from polyplex_research.utils.assemble import dualffn_config
    from conftest import build_model

    with pytest.raises(ValueError, match="layer"):
        dualffn_config(build_model().config, build_language_model().config, [0, 3])


def test_with_a_silent_language_ffn_the_graft_is_personaplex(audio_batch_both_streams):
    model, helium, _ = build_grafted()
    with torch.no_grad():
        model.model.embed_tokens.weight.copy_(helium.model.embed_tokens.weight)
        for layer in model.model.layers:
            layer.mlp.language_mlp.down_proj.weight.zero_()
        batch = dict(audio_batch_both_streams)
        batch["text_labels"] = batch["input_ids"].clone()
        ours = model(**batch)
        theirs = helium(**batch)
    torch.testing.assert_close(ours.depth_loss, theirs.depth_loss, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ours.last_hidden_state, theirs.last_hidden_state, rtol=1e-5, atol=1e-5)


def test_the_duplex_ffn_is_heliums_mlp_with_a_zero_reserve():
    model, helium, _ = build_grafted()
    torch.manual_seed(6)
    hidden = torch.randn(2, 5, 32)
    for ours, theirs in zip(model.model.layers, helium.model.layers):
        expected = theirs.mlp(theirs.post_attention_layernorm(hidden))
        torch.testing.assert_close(ours.mlp.duplex_mlp(ours.mlp.duplex_norm(hidden)), expected, rtol=1e-5, atol=1e-6)
        assert torch.count_nonzero(ours.mlp.duplex_mlp.gate_proj.weight[32:]) == 0


def test_a_reserved_neuron_gets_no_gradient_after_the_graft():
    model, _, _ = build_grafted()
    torch.manual_seed(9)
    ffn = model.model.layers[0].mlp
    ffn(torch.randn(2, 5, 32)).square().sum().backward()
    mlp = ffn.duplex_mlp
    assert torch.count_nonzero(mlp.gate_proj.weight.grad[32:]) == 0
    assert torch.count_nonzero(mlp.up_proj.weight.grad[32:]) == 0
    assert torch.count_nonzero(mlp.down_proj.weight.grad[:, 32:]) == 0


def test_the_language_ffn_is_the_paired_layer_of_the_language_model():
    model, _, language_model = build_grafted([2, 0])
    torch.manual_seed(7)
    hidden = torch.randn(2, 5, 32)
    for ours, index in zip(model.model.layers, [2, 0]):
        theirs = language_model.model.layers[index]
        expected = theirs.mlp(theirs.post_attention_layernorm(hidden))
        torch.testing.assert_close(ours.mlp.language_mlp(ours.mlp.language_norm(hidden)), expected, rtol=0, atol=0)


def test_the_text_head_is_the_language_models_head_behind_its_final_norm():
    model, _, language_model = build_grafted()
    torch.manual_seed(8)
    hidden = torch.randn(2, 5, 32)
    expected = language_model.lm_head(language_model.model.norm(hidden))
    torch.testing.assert_close(model.lm_head(model.model.text_norm(hidden)), expected, rtol=0, atol=0)
    vocab = model.config.vocab_size
    torch.testing.assert_close(
        model.model.embed_tokens.weight[:vocab], language_model.model.embed_tokens.weight, rtol=0, atol=0
    )


def test_the_opening_frame_keeps_heliums_reserved_text_row():
    model, helium, _ = build_grafted()
    torch.testing.assert_close(
        model.model.embed_tokens.weight[-1], helium.model.embed_tokens.weight[-1], rtol=0, atol=0
    )


def test_nvfp4_reaches_every_projection_of_both_ffns(cpu_nvfp4):
    from polyplex_research.utils.assemble import quantize_frozen_backbone
    from polyplex_research.utils.quantization import FrozenNVFP4Linear

    model, _, _ = build_grafted()
    quantize_frozen_backbone(model, "nvfp4")
    for layer in model.model.layers:
        for mlp in (layer.mlp.duplex_mlp, layer.mlp.language_mlp):
            for projection in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
                assert isinstance(projection, FrozenNVFP4Linear)
        assert isinstance(layer.self_attn.q_proj, FrozenNVFP4Linear)


def test_a_full_precision_layer_keeps_its_attention_and_both_its_ffns(cpu_nvfp4):
    from polyplex_research.utils.assemble import quantize_frozen_backbone
    from polyplex_research.utils.quantization import FrozenNVFP4Linear

    model, _, _ = build_grafted()
    quantize_frozen_backbone(model, "nvfp4", full_precision_layers=(1,))
    kept, quantized = model.model.layers[1], model.model.layers[0]
    for mlp in (kept.mlp.duplex_mlp, kept.mlp.language_mlp):
        assert all(type(getattr(mlp, name)) is torch.nn.Linear for name in ("gate_proj", "up_proj", "down_proj"))
    assert type(kept.self_attn.q_proj) is torch.nn.Linear
    for mlp in (quantized.mlp.duplex_mlp, quantized.mlp.language_mlp):
        assert isinstance(mlp.down_proj, FrozenNVFP4Linear)


def test_depth_text_rows_average_heliums_rows_over_each_tokens_pieces():
    from polyplex_research.utils.assemble import depth_text_rows

    class Surfaces:
        def decode(self, ids):
            return {0: "ab", 1: "c"}[ids[0]]

    class Pieces:
        eos_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return [ord(character) - ord("a") + 1 for character in text]

    table = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    rows = depth_text_rows(Surfaces(), 2, Pieces(), table)
    torch.testing.assert_close(rows, torch.stack([table[[1, 2]].mean(0), table[3]]))


# ---------------------------------------------------------------------------
# Vocabulary grafting (SPEC S12.1a, S12.2, S12.3a)
# ---------------------------------------------------------------------------


HELIUM_TEXT = ["▁"] + list("abcdefghijklmnopqrstuvwxyz") + [f"▁w{index}" for index in range(33)]
QWEN_ONLY = ["▁hola", "▁mundo", "▁안녕", "안", "녕"]
ADDED = {"▁holamundo": -0.4, "▁안녕안": -0.45}


COMPOSITION_SIZE = 8
# Each Qwen3-only piece's Qwen3 head row is this combination of shared pieces' rows, before the block scale.
QWEN_COMPOSITION = {
    "▁hola": {"▁w3": 0.8, "h": -0.4},
    "▁mundo": {"▁w7": 1.2, "m": 0.3},
    "▁안녕": {"▁w11": -0.6, "a": 0.9},
    "안": {"b": 0.5, "▁w20": 0.5},
    "녕": {"c": 1.5, "▁w30": -0.2},
}
# The Hangul block comes out three times longer, which the block scale undoes.
ROW_FACTOR = {"▁안녕": 3.0, "안": 3.0, "녕": 3.0}


def build_vocabulary_graft():
    r"""
    Returns:
        A DualFFN model with the vocabulary grafted, and what it was grafted from: the tokenizer, Helium, the Qwen3
        head and its piece ids. A shared piece's Qwen3 head row is its Helium head row times a random transform, and
        a Qwen3-only piece's row is the sparse combination `QWEN_COMPOSITION` of shared pieces' rows, scaled per
        script block.
    """
    from conftest import build_model
    from conftest import build_polyplex as build_dualffn

    from polyplex_research.models.polyplex import PolyplexTokenizer
    from polyplex_research.utils.assemble import graft_vocabulary

    helium = build_model()
    with torch.no_grad():
        helium.model.norm.weight.uniform_(0.5, 1.5)
    base = [("<unk>", 0.0, "unknown"), ("<s>", 0.0, "control"), ("</s>", 0.0, "control"), ("<pad>", 0.0, "control")]
    base += [(piece, -(index + 1) / 10, "normal") for index, piece in enumerate(HELIUM_TEXT)]
    assert len(base) == VOCAB_SIZE
    scores = {piece: score for piece, score, kind in base if kind == "normal"}
    scores.update({"▁hola": -0.5, "▁mundo": -0.6, "▁안녕": -0.7, "안": -0.8, "녕": -0.9, **ADDED})
    tokenizer = PolyplexTokenizer.from_vocabulary(base, scores, ["<think>"])

    torch.manual_seed(7)
    transform = torch.randn(32, 32)
    shared = HELIUM_TEXT[1:]
    shared_rows = helium.lm_head.weight.detach()[tokenizer.convert_tokens_to_ids(shared)] @ transform
    qwen_rows = [
        sum(value * shared_rows[shared.index(piece)] for piece, value in QWEN_COMPOSITION[name].items())
        * ROW_FACTOR.get(name, 1.0)
        for name in QWEN_ONLY
    ]
    qwen_head = torch.cat([shared_rows, torch.stack(qwen_rows)])
    qwen_ids = {piece: index for index, piece in enumerate(shared + QWEN_ONLY)}

    model = build_dualffn(
        vocab_size=-(-len(tokenizer) // 8) * 8 + 8, text_composition_size=COMPOSITION_SIZE, base_vocab_size=VOCAB_SIZE
    )
    graft_vocabulary(model, tokenizer, helium, qwen_head, qwen_ids)
    return model, tokenizer, helium, qwen_head, qwen_ids


def text_tables(model):
    return {
        "input": model.model.embed_tokens.weight,
        "head": model.lm_head.weight,
        "depth": model.depth_decoder.model.text_embed_tokens.weight,
    }


def helium_tables(helium):
    return {
        "input": helium.model.embed_tokens.weight.detach(),
        "head": helium.lm_head.weight.detach(),
        "depth": helium.depth_decoder.model.text_embed_tokens.weight.detach(),
    }


def read_rows(model, ids):
    ids = torch.tensor(ids)
    with torch.no_grad():
        return {
            "input": model.embed_text(ids),
            "head": model.lm_head.composed_weight()[ids],
            "depth": model.depth_decoder.model.text_embed_tokens(ids),
        }


def test_heliums_tokens_keep_their_rows_in_all_three_tables_with_no_coefficients():
    model, _, helium, *_ = build_vocabulary_graft()
    ours, theirs = text_tables(model), helium_tables(helium)
    for name in ours:
        torch.testing.assert_close(ours[name][:VOCAB_SIZE], theirs[name][:VOCAB_SIZE], rtol=0, atol=0)
    for name in ("input", "depth"):
        # The opening frame's reserved row is Helium's.
        torch.testing.assert_close(ours[name][-1], theirs[name][-1], rtol=0, atol=0)
    values = model.text_composition.values
    assert not values[:VOCAB_SIZE].any() and not values[model.config.vocab_size].any()
    rows = read_rows(model, list(range(VOCAB_SIZE)))
    for name in rows:
        torch.testing.assert_close(rows[name], theirs[name][:VOCAB_SIZE], rtol=0, atol=0)


def test_new_tokens_store_zero_residuals():
    model, tokenizer, *_ = build_vocabulary_graft()
    for name, table in text_tables(model).items():
        stop = model.config.vocab_size
        assert not table[VOCAB_SIZE:stop].any(), name


def raw_composition(tokenizer, table, token):
    r"""The combination `QWEN_COMPOSITION` gives a Qwen3-only token over `table`'s Helium rows, before any scale."""
    return ROW_FACTOR.get(token, 1.0) * sum(
        value * table[tokenizer.convert_tokens_to_ids(piece)] for piece, value in QWEN_COMPOSITION[token].items()
    )


def test_a_qwen3_token_is_composed_of_shared_tokens_in_the_direction_of_its_qwen3_head_row_in_every_table():
    model, tokenizer, helium, *_ = build_vocabulary_graft()
    theirs = helium_tables(helium)
    rows = read_rows(model, tokenizer.convert_tokens_to_ids(QWEN_ONLY))
    raw = {name: torch.stack([raw_composition(tokenizer, table, token) for token in QWEN_ONLY]) for name, table in theirs.items()}
    for name in theirs:
        cosine = torch.nn.functional.cosine_similarity(rows[name], raw[name], dim=-1)
        torch.testing.assert_close(cosine, torch.ones(len(QWEN_ONLY)), rtol=0, atol=1e-4)
    # One scale per token, shared by the three tables.
    scales = rows["head"].norm(dim=-1) / raw["head"].norm(dim=-1)
    for name in ("input", "depth"):
        torch.testing.assert_close(rows[name].norm(dim=-1) / raw[name].norm(dim=-1), scales, rtol=1e-4, atol=1e-5)


def test_each_blocks_new_head_rows_take_heliums_median_norm_and_none_is_longer_than_heliums_longest():
    model, tokenizer, helium, *_ = build_vocabulary_graft()
    helium_norms = helium.lm_head.weight.detach().norm(dim=-1)
    with torch.no_grad():
        norms = model.lm_head.composed_weight().norm(dim=-1)
    for start, end in tokenizer.blocks.values():
        members = norms[max(start, VOCAB_SIZE) : end]
        members = members[members > 0]
        if len(members):
            torch.testing.assert_close(members.median(), helium_norms.median(), rtol=1e-4, atol=1e-5)
    assert norms[VOCAB_SIZE : len(tokenizer)].max() <= helium_norms.max() * (1 + 1e-5)


def test_an_added_piece_points_along_the_mean_of_its_pieces_compositions():
    model, tokenizer, helium, *_ = build_vocabulary_graft()
    theirs = helium_tables(helium)
    for piece, cut in (("▁holamundo", ["▁hola", "m", "u", "n", "d", "o"]), ("▁안녕안", ["▁안녕", "안"])):
        added = read_rows(model, [tokenizer.convert_tokens_to_ids(piece)])
        for name, table in theirs.items():
            parts = [
                raw_composition(tokenizer, table, part)
                if part in QWEN_COMPOSITION
                else table[tokenizer.convert_tokens_to_ids(part)]
                for part in cut
            ]
            cosine = torch.nn.functional.cosine_similarity(added[name][0], torch.stack(parts).mean(0), dim=0)
            assert cosine > 0.9999, (piece, name)


def test_the_padded_head_rows_are_zero():
    model, tokenizer, *_ = build_vocabulary_graft()
    assert model.config.vocab_size > len(tokenizer)
    with torch.no_grad():
        assert not model.lm_head.composed_weight()[len(tokenizer):].any()


def test_the_boundary_of_the_frozen_rows_is_heliums_vocabulary():
    model, *_ = build_vocabulary_graft()
    assert model.config.base_vocab_size == VOCAB_SIZE


def test_the_head_reads_heliums_final_norm_once_the_vocabulary_is_grafted():
    model, _, helium, *_ = build_vocabulary_graft()
    torch.testing.assert_close(model.model.text_norm.weight, helium.model.norm.weight, rtol=0, atol=0)


def test_the_dualffn_config_takes_a_vocabulary_and_a_shared_composition_over_heliums_tokens():
    from conftest import build_model

    from polyplex_research.utils.assemble import dualffn_config

    config = dualffn_config(build_model().config, build_language_model().config, vocab_size=96, composition_size=8)
    assert config.vocab_size == 96 and config.text_composition_size == 8 and config.base_vocab_size == VOCAB_SIZE


def test_grafting_the_language_model_can_leave_the_vocabulary_alone():
    from conftest import build_model

    from polyplex_research.utils.assemble import dualffn_config, graft_language_model

    helium, language_model = build_model(), build_language_model()
    config = dualffn_config(helium.config, language_model.config, vocab_size=96, composition_size=8)
    torch.manual_seed(5)
    model = PolyplexForConditionalGeneration(config).eval()
    head = model.lm_head.weight.detach().clone()
    graft_language_model(model, language_model, vocabulary=False)
    torch.testing.assert_close(model.lm_head.weight, head, rtol=0, atol=0)
    layer = model.model.layers[0].mlp.language_mlp
    torch.testing.assert_close(layer.down_proj.weight, language_model.model.layers[0].mlp.down_proj.weight)


# ---------------------------------------------------------------------------
# Appended layers (SPEC S11.4a)
# ---------------------------------------------------------------------------


def test_appended_layers_pair_every_layer_of_the_language_model():
    model, helium, language_model = build_grafted(appended_layers=1)
    assert model.config.num_hidden_layers == 3 == language_model.config.num_hidden_layers
    assert model.config.appended_layers == [2]
    assert model.config.language_ffn_layers == [0, 1, 2]
    torch.testing.assert_close(
        model.model.layers[2].mlp.language_mlp.gate_proj.weight, language_model.model.layers[2].mlp.gate_proj.weight
    )


def test_an_appended_layer_copies_heliums_last_layer_and_starts_as_the_identity():
    model, helium, _ = build_grafted(appended_layers=1)
    appended, source = model.model.layers[2], helium.model.layers[1]
    q_proj = getattr(source.self_attn.q_proj, "linear", source.self_attn.q_proj)
    torch.testing.assert_close(appended.self_attn.q_proj.weight, q_proj.weight, rtol=0, atol=0)
    assert not appended.self_attn.o_proj.weight.any()
    assert not appended.mlp.duplex_mlp.down_proj.weight.any()
    assert not appended.mlp.language_mlp.down_proj.weight.any()


def test_an_appended_layer_keeps_the_text_llm_output_projection_when_asked():
    from polyplex_research.utils.assemble import graft_language_layer

    model, _, language_model = build_grafted(appended_layers=1)
    with torch.no_grad():
        graft_language_layer(model.model.layers[2], language_model.model.layers[2], True, keep_appended_output=True)
    torch.testing.assert_close(
        model.model.layers[2].mlp.language_mlp.down_proj.weight,
        language_model.model.layers[2].mlp.down_proj.weight,
        rtol=0,
        atol=0,
    )


def test_with_appended_layers_and_a_silent_language_ffn_the_graft_is_still_personaplex(audio_batch_both_streams):
    model, helium, _ = build_grafted(appended_layers=1)
    with torch.no_grad():
        model.model.embed_tokens.weight.copy_(helium.model.embed_tokens.weight)
        for layer in model.model.layers:
            layer.mlp.language_mlp.down_proj.weight.zero_()
        batch = dict(audio_batch_both_streams)
        batch["text_labels"] = batch["input_ids"].clone()
        ours = model(**batch)
        theirs = helium(**batch)
    torch.testing.assert_close(ours.depth_loss, theirs.depth_loss, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ours.last_hidden_state, theirs.last_hidden_state, rtol=1e-5, atol=1e-5)


def test_layer_precision_follows_the_spec_table(cpu_nvfp4):
    from polyplex_research.utils.assemble import quantize_frozen_backbone
    from polyplex_research.utils.quantization import FrozenNVFP4Linear, HadamardNVFP4Linear

    model, _, _ = build_grafted(appended_layers=1)
    quantize_frozen_backbone(model, "nvfp4", full_precision_layers=(0,))
    full, frozen, appended = model.model.layers
    assert type(full.self_attn.q_proj) is torch.nn.Linear and type(full.mlp.language_mlp.down_proj) is torch.nn.Linear
    assert isinstance(frozen.self_attn.q_proj, FrozenNVFP4Linear)
    assert isinstance(frozen.mlp.duplex_mlp.down_proj, FrozenNVFP4Linear)
    for projection in (appended.self_attn.o_proj, appended.mlp.duplex_mlp.down_proj, appended.mlp.language_mlp.up_proj):
        assert isinstance(projection, HadamardNVFP4Linear) and projection.weight.requires_grad


# ---------------------------------------------------------------------------
# Shared composition coefficients (SPEC S12.3a)
# ---------------------------------------------------------------------------


def test_omp_recovers_a_sparse_combination_of_the_basis():
    from polyplex_research.utils.assemble import omp_coefficients

    torch.manual_seed(0)
    basis = torch.randn(50, 64)
    indices = torch.tensor([[3, 17, 41], [0, 9, 22]])
    values = torch.tensor([[1.5, -0.7, 0.3], [0.2, 2.0, -1.1]])
    targets = (values[..., None] * basis[indices]).sum(1)
    found, coefficients = omp_coefficients(targets, basis, 3)
    for row in range(2):
        order = found[row].argsort()
        assert found[row][order].tolist() == indices[row].tolist()
        torch.testing.assert_close(coefficients[row][order], values[row], rtol=0, atol=1e-4)


def test_coefficients_found_in_one_table_carry_over_to_tables_built_from_the_same_composition():
    from polyplex_research.utils.assemble import composition_reconstruction

    torch.manual_seed(0)
    atoms, composed, width = 40, 20, 32
    weights = torch.zeros(composed, atoms)
    for row in range(composed):
        weights[row, torch.randperm(atoms)[:2]] = torch.rand(2) + 0.5
    tables = {}
    for name in ("donor_input", "donor_head", "input", "head"):
        base = torch.randn(atoms, width)
        tables[name] = torch.cat([base, weights @ base])
    result = composition_reconstruction(
        tables["donor_input"], tables["donor_head"], tables["input"], tables["head"],
        held_out=torch.arange(atoms, atoms + 20), k=2,
    )
    for donor in ("donor input", "donor head"):
        for table in ("input", "head"):
            assert result[f"composition from {donor}, {table}"] > 0.99
    assert set(result) >= {"linear head to input, input", "linear donor head to head, head"}
    # An exact composition rebuilds the rows at their own length.
    assert result["composition from donor head, head, norm ratio median"] == pytest.approx(1.0, abs=1e-3)
    assert result["composition from donor head, head, norm ratio p99"] == pytest.approx(1.0, abs=1e-3)


def test_a_ridge_shrinks_the_coefficients():
    from polyplex_research.utils.assemble import omp_coefficients

    torch.manual_seed(0)
    basis = torch.randn(50, 64)
    targets = torch.randn(4, 64)
    _, plain = omp_coefficients(targets, basis, 8)
    _, shrunk = omp_coefficients(targets, basis, 8, ridge=1.0)
    assert shrunk.norm(dim=-1).lt(plain.norm(dim=-1)).all()


def test_capping_the_head_norms_shrinks_only_the_coefficients_of_rows_above_the_limit():
    from polyplex_research.utils.assemble import cap_composed_head_norms

    model, tokenizer, *_ = build_vocabulary_graft()
    with torch.no_grad():
        before = model.lm_head.composed_weight().norm(dim=-1)
    new = before[VOCAB_SIZE : len(tokenizer)]
    # The grafted rows sit at Helium's median norm, so a limit well below it shrinks some and leaves the rest.
    limit = new.quantile(0.3).item()
    capped = cap_composed_head_norms(model, limit)
    with torch.no_grad():
        after = model.lm_head.composed_weight().norm(dim=-1)
    assert capped == int((new > limit).sum())
    torch.testing.assert_close(after[:VOCAB_SIZE], before[:VOCAB_SIZE], rtol=0, atol=0)
    long = torch.arange(VOCAB_SIZE, len(tokenizer))[new > limit]
    torch.testing.assert_close(after[long], torch.full((len(long),), limit), rtol=1e-4, atol=1e-5)
    short = torch.arange(VOCAB_SIZE, len(tokenizer))[new <= limit]
    torch.testing.assert_close(after[short], before[short], rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Context-independent logits of the new head rows (SPEC S12.3a)
# ---------------------------------------------------------------------------


class SilentProcessor:
    r"""Stands in for the processor where only the silence is read."""

    def get_silence_audio_codes(self, frames, batch_size=1):
        return torch.ones(batch_size, 4, frames, dtype=torch.long)


def test_setting_components_moves_rows_only_inside_the_subspace():
    from polyplex_research.utils.assemble import with_components

    torch.manual_seed(0)
    rows = torch.randn(7, 16) * 3
    directions = torch.linalg.qr(torch.randn(16, 3)).Q.T
    targets = torch.tensor([1.0, -2.0, 0.5])
    moved = with_components(rows, directions, targets)
    torch.testing.assert_close(moved @ directions.T, targets.expand(7, 3))
    rest = rows - (rows @ directions.T) @ directions
    torch.testing.assert_close(moved - (moved @ directions.T) @ directions, rest)


def test_the_prior_correction_sets_the_mean_state_and_main_directions_to_heliums_values():
    from polyplex_research.utils.assemble import head_prior_correction

    torch.manual_seed(0)
    states = torch.randn(50, 16) + 3.0
    helium_head = torch.randn(30, 16)
    rows = torch.randn(9, 16)
    corrected = head_prior_correction(rows, states, helium_head, count=4)
    direction = torch.nn.functional.normalize(states.mean(0), dim=0)
    torch.testing.assert_close(corrected @ direction, (helium_head @ direction).min().expand(9))
    centered = states - states.mean(0)
    centered = centered - (centered @ direction)[:, None] * direction
    principal = torch.linalg.svd(centered, full_matrices=False).Vh[:4]
    torch.testing.assert_close(
        (corrected @ principal.T).abs(), (helium_head.mean(0) @ principal.T).abs().expand(9, 4), rtol=1e-4, atol=1e-5
    )


def test_heliums_text_states_are_what_its_head_reads_behind_the_opening_frame():
    from conftest import build_model

    from polyplex_research.datasets.duplex_data import dense_collate_fn
    from polyplex_research.utils.assemble import helium_text_states

    helium = build_model()

    class Pieces:
        def encode(self, text, add_special_tokens=False):
            return [ord(character) - ord("a") + 5 for character in text]

    states = helium_text_states(helium, SilentProcessor(), Pieces(), ["abcde", "fgh"])
    collate = dense_collate_fn(SilentProcessor(), helium)
    expected = []
    for text in ("abcde", "fgh"):
        batch = collate([{"segments": [torch.tensor(Pieces().encode(text))]}])
        batch.pop("text_labels")
        with torch.no_grad():
            expected.append(helium(**batch, use_cache=False).logits[0, 1:-1])
    torch.testing.assert_close(states @ helium.lm_head.weight.T, torch.cat(expected), rtol=1e-4, atol=1e-5)


def test_with_calibration_states_the_new_head_rows_start_with_heliums_context_independent_logits():
    from conftest import build_model
    from conftest import build_polyplex as build_dualffn

    from polyplex_research.utils.assemble import graft_vocabulary, head_prior_correction

    plain, tokenizer, helium, qwen_head, qwen_ids = build_vocabulary_graft()
    torch.manual_seed(3)
    states = torch.randn(40, 32) + 1.0
    model = build_dualffn(
        vocab_size=plain.config.vocab_size, text_composition_size=COMPOSITION_SIZE, base_vocab_size=VOCAB_SIZE
    )
    graft_vocabulary(model, tokenizer, helium, qwen_head, qwen_ids, states=states, principal_directions=4)
    size = len(tokenizer)
    with torch.no_grad():
        composed = plain.lm_head.composed_weight()
        expected = head_prior_correction(composed[VOCAB_SIZE:size], states, helium.lm_head.weight.detach(), 4)
        torch.testing.assert_close(model.lm_head.composed_weight()[VOCAB_SIZE:size], expected, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(model.lm_head.composed_weight()[:VOCAB_SIZE], composed[:VOCAB_SIZE], rtol=0, atol=0)
        torch.testing.assert_close(model.model.embed_tokens.weight, plain.model.embed_tokens.weight, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Layer by layer assembly (SPEC S11.2)
# ---------------------------------------------------------------------------


def build_layer_by_layer(appended_layers=1, quantization=None, full_precision_layers=(), alignment=None, removed_neurons=None):
    r"""
    Returns:
        The DualFFN model built on the meta device and filled one layer at a time, and the two models it was
        grafted from, whose used layers have been dropped.
    """
    import copy

    from conftest import build_model

    from polyplex_research.utils.assemble import (
        dualffn_config,
        graft_helium_rest,
        graft_language_rest,
        graft_layers,
        materialize_outside_layers,
    )

    helium = build_model(predicts_user_stream=True)
    language_model = build_language_model()
    config = dualffn_config(helium.config, language_model.config, appended_layers=appended_layers)
    with torch.device("meta"):
        model = PolyplexForConditionalGeneration(config).eval()
    sources = copy.deepcopy(helium), copy.deepcopy(language_model)
    graft_layers(model, *sources, "cpu", quantization, full_precision_layers, alignment, removed_neurons)
    materialize_outside_layers(model, "cpu")
    graft_helium_rest(model, sources[0])
    graft_language_rest(model, sources[1])
    return model, sources, helium, language_model


def test_a_layer_by_layer_graft_is_the_whole_model_graft(audio_batch_both_streams):
    model, _, helium, language_model = build_layer_by_layer()
    reference, _, _ = build_grafted(appended_layers=1)
    ours, theirs = model.state_dict(), reference.state_dict()
    assert set(ours) == set(theirs)
    for name in theirs:
        torch.testing.assert_close(ours[name], theirs[name], rtol=0, atol=0, msg=name)
    with torch.no_grad():
        batch = dict(audio_batch_both_streams)
        batch["text_labels"] = batch["input_ids"].clone()
        torch.testing.assert_close(model(**batch).logits, reference(**batch).logits, rtol=0, atol=0)


def test_a_layer_by_layer_graft_drops_each_source_layer_once_it_is_used():
    model, (helium, language_model), *_ = build_layer_by_layer()
    assert all(isinstance(layer, torch.nn.Identity) for layer in helium.model.layers)
    used = set(model.config.language_ffn_layers)
    for index, layer in enumerate(language_model.model.layers):
        assert isinstance(layer, torch.nn.Identity) == (index in used), index


def test_a_layer_by_layer_graft_quantizes_each_layer_as_the_spec_table(cpu_nvfp4):
    from polyplex_research.utils.quantization import FrozenNVFP4Linear, HadamardNVFP4Linear

    model, *_ = build_layer_by_layer(quantization="nvfp4", full_precision_layers=(0,))
    full, frozen, appended = model.model.layers
    assert type(full.self_attn.q_proj) is torch.nn.Linear and type(full.mlp.language_mlp.down_proj) is torch.nn.Linear
    assert isinstance(frozen.self_attn.q_proj, FrozenNVFP4Linear)
    assert isinstance(frozen.mlp.duplex_mlp.down_proj, FrozenNVFP4Linear)
    for projection in (appended.self_attn.o_proj, appended.mlp.duplex_mlp.down_proj, appended.mlp.language_mlp.up_proj):
        assert isinstance(projection, HadamardNVFP4Linear) and projection.weight.requires_grad
    assert not any(parameter.is_meta for parameter in model.parameters())


def test_compiling_the_backbone_wraps_every_layer_and_nothing_else():
    from polyplex_research.utils.assemble import compile_backbone

    from conftest import build_polyplex

    model = build_polyplex()
    compile_backbone(model)
    assert all(layer._compiled_call_impl is not None for layer in model.model.layers)
    assert model._compiled_call_impl is None and model.depth_decoder._compiled_call_impl is None


# ---------------------------------------------------------------------------
# Residual basis alignment (SPEC S11.2a)
# ---------------------------------------------------------------------------


def test_procrustes_recovers_the_rotation_between_two_bases():
    from polyplex_research.utils.assemble import procrustes_rotation

    torch.manual_seed(0)
    rotation, _ = torch.linalg.qr(torch.randn(16, 16))
    sources = torch.randn(200, 16)
    targets = sources @ rotation.T + 0.01 * torch.randn(200, 16)
    found = procrustes_rotation(targets.T @ sources)
    torch.testing.assert_close(found, rotation, atol=1e-2, rtol=0)
    torch.testing.assert_close(found @ found.T, torch.eye(16), atol=1e-5, rtol=0)


def test_a_folded_language_ffn_reads_the_rotated_residual_and_writes_back_in_the_helium_basis():
    from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm

    from polyplex_research.utils.assemble import fold_residual_alignment

    torch.manual_seed(0)
    config = build_language_model().config
    mlp, norm = Qwen3MLP(config).double(), Qwen3RMSNorm(32, eps=1e-6).double()
    with torch.no_grad():
        norm.weight.copy_(torch.rand(32) + 0.5)
    rotation, _ = torch.linalg.qr(torch.randn(32, 32, dtype=torch.float64))
    hidden = torch.randn(5, 32, dtype=torch.float64)
    expected = 0.7 * (mlp(norm(hidden @ rotation.T)) @ rotation)
    fold_residual_alignment(mlp, norm, rotation, 0.7)
    mlp.double()
    assert torch.equal(norm.weight, torch.ones(32, dtype=torch.float64))
    torch.testing.assert_close(mlp(norm(hidden)), expected, atol=1e-5, rtol=1e-5)


def test_a_given_gain_is_folded_in_place_of_the_normalization_weight():
    from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm

    from polyplex_research.utils.assemble import fold_residual_alignment

    torch.manual_seed(1)
    config = build_language_model().config
    mlp, norm = Qwen3MLP(config).double(), Qwen3RMSNorm(32, eps=1e-6).double()
    gain = torch.rand(32, dtype=torch.float64) + 0.5
    hidden = torch.randn(4, 32, dtype=torch.float64)
    with torch.no_grad():
        norm.weight.copy_(gain)
    expected = mlp(norm(hidden))
    with torch.no_grad():
        norm.weight.fill_(3.0)
    fold_residual_alignment(mlp, norm, torch.eye(32, dtype=torch.float64), 1.0, gain)
    torch.testing.assert_close(mlp(norm(hidden)), expected, atol=1e-6, rtol=1e-6)


def test_tokens_are_paired_where_both_cuts_end_at_the_same_character():
    from polyplex_research.utils.assemble import aligned_positions

    class Cut:
        def __init__(self, ends):
            self.ends = ends

        def __call__(self, text, add_special_tokens, return_offsets_mapping):
            starts = [0] + self.ends[:-1]
            return {"input_ids": list(range(len(self.ends))), "offset_mapping": list(zip(starts, self.ends))}

    first, second, first_index, second_index = aligned_positions("abcdefgh", Cut([2, 4, 6, 8]), Cut([3, 4, 8]))
    assert first == [0, 1, 2, 3] and second == [0, 1, 2]
    assert first_index == [1, 3] and second_index == [1, 2]


def test_an_identity_alignment_leaves_the_layer_by_layer_graft_as_it_was(tmp_path, audio_batch_both_streams):
    from safetensors.torch import save_file

    from polyplex_research.utils.assemble import load_residual_alignment

    reference, *_ = build_layer_by_layer()
    layers = len(reference.model.layers)
    path = str(tmp_path / "identity.safetensors")
    save_file({"rotations": torch.eye(32).expand(layers, 32, 32).contiguous(), "scales": torch.ones(layers)}, path)
    aligned, *_ = build_layer_by_layer(alignment=load_residual_alignment(path))
    with torch.no_grad():
        batch = dict(audio_batch_both_streams)
        batch["text_labels"] = batch["input_ids"].clone()
        torch.testing.assert_close(aligned(**batch).logits, reference(**batch).logits, atol=1e-5, rtol=1e-5)
    assert all(torch.equal(layer.mlp.language_norm.weight, torch.ones(32)) for layer in aligned.model.layers)


def test_a_shared_alignment_file_gives_every_layer_its_one_rotation_and_scale(tmp_path):
    from safetensors.torch import save_file

    from polyplex_research.utils.assemble import load_residual_alignment

    rotation, _ = torch.linalg.qr(torch.randn(8, 8))
    path = str(tmp_path / "shared.safetensors")
    save_file({"rotations": rotation[None].contiguous(), "scales": torch.tensor([0.25])}, path)
    layer = load_residual_alignment(path)
    for index in (0, 5, 35):
        read, scale, gain = layer(index)
        assert torch.equal(read, rotation) and scale == 0.25 and gain is None


def test_a_frozen_nvfp4_language_ffn_is_folded_from_its_original_weight_and_quantized_again(monkeypatch):
    from conftest import build_polyplex

    from polyplex_research.utils import quantization
    from polyplex_research.utils.assemble import fold_language_stream, fold_language_weights
    from polyplex_research.utils.quantization import FrozenNVFP4Linear, _to_nvfp4_stochastic, dequantize_nvfp4

    def quantize(tensor, amax=None):
        torch.manual_seed(0)
        return _to_nvfp4_stochastic(tensor)

    monkeypatch.setattr(quantization, "_to_nvfp4", quantize)
    model = build_polyplex(language_scale=True, language_stream=True)
    mlp = model.model.layers[0].mlp.language_mlp
    names = ("gate", "up", "down")
    originals = {name: getattr(mlp, f"{name}_proj").weight.detach().clone() for name in names}
    for name in names:
        setattr(mlp, f"{name}_proj", FrozenNVFP4Linear.from_linear(getattr(mlp, f"{name}_proj")))
    gain = model.model.layers[0].mlp.language_norm.weight.detach().clone()
    rotation, _ = torch.linalg.qr(torch.randn(32, 32))

    with pytest.raises(ValueError, match="original weight given"):
        fold_language_stream(model, rotation, 0.5, original=lambda index, name: originals[name] + 1.0)
    fold_language_stream(model, rotation, 0.5, original=lambda index, name: originals[name])
    expected = fold_language_weights(*(originals[name] for name in names), rotation, 0.5, gain)
    for name, weight in zip(names, expected):
        projection = getattr(mlp, f"{name}_proj")
        assert isinstance(projection, FrozenNVFP4Linear)
        reference = quantize(quantization._rotate(weight, projection.forward_transform.to(weight.dtype)))
        torch.testing.assert_close(
            dequantize_nvfp4(projection.quantized_weight, torch.float32), dequantize_nvfp4(reference, torch.float32)
        )


def test_an_nvfp4_dualffn_is_saved_once_and_loaded_with_its_frozen_layers_still_four_bit(monkeypatch, tmp_path):
    from polyplex_research.utils import quantization
    from polyplex_research.utils.assemble import load_assembled_dualffn, save_assembled_dualffn
    from polyplex_research.utils.quantization import (
        FrozenNVFP4Linear,
        HadamardNVFP4Linear,
        _to_nvfp4_stochastic,
        dequantize_nvfp4,
    )

    monkeypatch.setattr(quantization, "_to_nvfp4", lambda tensor, amax=None: _to_nvfp4_stochastic(tensor))
    torch.manual_seed(0)
    model, *_ = build_layer_by_layer(quantization="nvfp4", full_precision_layers=(0,))
    model.to(torch.bfloat16)
    # Shards this small split projections across files.
    save_assembled_dualffn(model, str(tmp_path), full_precision_layers=(0,), max_shard_size="20KB")
    assert len([name for name in os.listdir(tmp_path) if name.endswith(".safetensors")]) > 2
    loaded = load_assembled_dualffn(str(tmp_path), "cpu")

    full, frozen, appended = loaded.model.layers
    assert type(full.self_attn.q_proj) is torch.nn.Linear
    assert isinstance(frozen.mlp.language_mlp.gate_proj, FrozenNVFP4Linear)
    assert isinstance(appended.mlp.language_mlp.up_proj, HadamardNVFP4Linear)
    ours = dict(loaded.named_modules())
    for name, module in model.named_modules():
        if isinstance(module, FrozenNVFP4Linear):
            torch.testing.assert_close(
                dequantize_nvfp4(ours[name].quantized_weight, torch.float32),
                dequantize_nvfp4(module.quantized_weight, torch.float32),
                rtol=0,
                atol=0,
                msg=name,
            )
    expected = model.state_dict()
    for name, tensor in loaded.state_dict().items():
        assert torch.equal(tensor, expected[name]), name
    assert not any(tensor.is_meta for tensor in [*loaded.parameters(), *loaded.buffers()])
    torch.testing.assert_close(loaded.model.rotary_emb.inv_freq, model.model.rotary_emb.inv_freq.float(), rtol=1e-2, atol=0)


def test_a_dense_release_holds_every_nvfp4_projection_as_frozen_codes_the_trained_forward_used(monkeypatch, tmp_path):
    from polyplex_research.utils import quantization
    from polyplex_research.utils.assemble import load_assembled_dualffn, save_dense_release
    from polyplex_research.utils.quantization import FrozenNVFP4Linear, _to_nvfp4_stochastic, dequantize_nvfp4

    # A seeded stand-in for the triton quantization, so the same weight gives the same codes twice.
    def quantize(tensor, amax=None):
        torch.manual_seed(0)
        return _to_nvfp4_stochastic(tensor)

    monkeypatch.setattr(quantization, "_to_nvfp4", quantize)
    torch.manual_seed(0)
    model, *_ = build_layer_by_layer(quantization="nvfp4", full_precision_layers=(0,))
    model.to(torch.bfloat16)
    trained = model.model.layers[2].mlp.language_mlp.up_proj
    expected = dequantize_nvfp4(FrozenNVFP4Linear.from_linear(trained).quantized_weight, torch.float32)
    auto_map = {"AutoConfig": "configuration_polyplex.PolyplexConfig"}
    save_dense_release(model, str(tmp_path), auto_map=auto_map, max_shard_size="20KB")

    loaded = load_assembled_dualffn(str(tmp_path), "cpu")
    full, frozen, appended = loaded.model.layers
    assert type(full.self_attn.q_proj) is torch.nn.Linear
    assert isinstance(frozen.mlp.language_mlp.gate_proj, FrozenNVFP4Linear)
    assert isinstance(appended.mlp.language_mlp.up_proj, FrozenNVFP4Linear)
    torch.testing.assert_close(
        dequantize_nvfp4(appended.mlp.language_mlp.up_proj.quantized_weight, torch.float32), expected, rtol=0, atol=0
    )
    assert loaded.config.auto_map == auto_map
    assert not any(name.endswith("up_proj.weight") for name, _ in loaded.named_parameters() if ".2." in name)


def test_removed_language_ffn_neurons_add_nothing_and_the_rest_is_unchanged():
    from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

    from polyplex_research.utils.assemble import remove_language_neurons

    torch.manual_seed(0)
    mlp = Qwen3MLP(build_language_model().config)
    hidden = torch.randn(5, 32)
    removed = [3, 17]
    activation = torch.nn.functional.silu(mlp.gate_proj(hidden)) * mlp.up_proj(hidden)
    activation[:, removed] = 0
    expected = mlp.down_proj(activation)
    remove_language_neurons(mlp, removed)
    torch.testing.assert_close(mlp(hidden), expected)
    assert not mlp.gate_proj.weight[removed].any() and not mlp.down_proj.weight[:, removed].any()


def test_assembly_removes_the_named_neurons_of_each_layer_before_quantizing():
    model, *_ = build_layer_by_layer(removed_neurons={1: [0, 5]})
    mlp = model.model.layers[1].mlp.language_mlp
    assert not mlp.up_proj.weight[[0, 5]].any() and not mlp.down_proj.weight[:, [0, 5]].any()
    assert model.model.layers[0].mlp.language_mlp.up_proj.weight[[0, 5]].any()


def fake_drive_store(monkeypatch, store):
    """Replaces `letify.stash` and `letify.unstash` with copies into and out of `store`, keeping directories."""
    import shutil

    import letify

    def stash(name, path):
        target = store / name
        shutil.rmtree(target, ignore_errors=True)
        (shutil.copytree if os.path.isdir(path) else shutil.copy)(path, target)

    def unstash(name, path):
        source = store / name
        if not source.exists():
            return None
        (shutil.copytree if source.is_dir() else shutil.copy)(source, path)
        return path

    monkeypatch.setattr(letify, "stash", stash, raising=False)
    monkeypatch.setattr(letify, "unstash", unstash, raising=False)


def test_an_assembly_on_an_ephemeral_disk_goes_through_the_drive_store(tmp_path, monkeypatch):
    import shutil

    import letify

    import polyplex_research.utils.assemble as assemble

    store = tmp_path / "store"
    store.mkdir()
    fake_drive_store(monkeypatch, store)
    monkeypatch.setenv("POLYPLEX_EPHEMERAL_DISK", "1")
    assembled, loaded = [], []

    def save(model, path, full_precision_layers=()):
        os.makedirs(path, exist_ok=True)
        for file in ("config.json", "model.safetensors"):
            open(os.path.join(path, file), "w").write(file)

    monkeypatch.setattr(assemble, "assemble_dualffn", lambda device, **settings: assembled.append(settings) or "model")
    monkeypatch.setattr(assemble, "save_assembled_dualffn", save)
    monkeypatch.setattr(assemble, "load_assembled_dualffn", lambda path, device, dtype: loaded.append(path) or "loaded")
    monkeypatch.setattr("polyplex_research.utils.quantization.set_nvfp4_backward", lambda model, dtype: 0)

    freed = []
    monkeypatch.setattr(
        "polyplex_research.utils.phase.free_downloads",
        lambda repo_ids: freed.append((list(repo_ids), sorted(path.name for path in store.iterdir()))),
    )
    monkeypatch.setenv("HF_HOME", str(tmp_path / "first"))
    assert assemble.load_or_assemble_dualffn(
        "cpu", quantization="nvfp4", appended_layers=4, personaplex="p/plex", language_model="q/wen"
    ) == "model"
    assert len(assembled) == 1
    # The sources are deleted before the handoff, which writes a second copy of the assembly to the disk.
    assert freed == [(["p/plex", "q/wen"], [])]
    assert not any((tmp_path / "first" / "assembled").iterdir())
    names = sorted(path.name for path in store.iterdir())
    # The whole directory is one handoff.
    assert len(names) == 1 and names[0].startswith("dualffn-") and (store / names[0] / "config.json").exists()

    monkeypatch.setenv("HF_HOME", str(tmp_path / "second"))
    assert assemble.load_or_assemble_dualffn(
        "cpu", quantization="nvfp4", appended_layers=4, personaplex="p/plex", language_model="q/wen"
    ) == "loaded"
    assert len(assembled) == 1
    # The fetched model is read, then removed from the session disk; the Drive store keeps it.
    assert loaded == [str(tmp_path / "second" / "assembled" / names[0])] and not os.path.exists(loaded[0])


def test_an_assembly_name_hashes_settings_without_files_as_json_and_a_file_by_its_content(tmp_path):
    import hashlib
    import json

    from polyplex_research.utils.assemble import assembly_name

    settings = {
        "personaplex": "nvidia/personaplex-7b-v1",
        "language_model": "Qwen/Qwen3-8B",
        "quantization": "nvfp4",
        "appended_layers": 4,
        "alignment": None,
        "full_precision_layers": (0, 31),
        "tokenizer": str(tmp_path),
    }
    before = hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()[:16]
    assert assembly_name(settings) == f"dualffn-{before}"

    first, second, other = tmp_path / "a.pt", tmp_path / "b.pt", tmp_path / "c.pt"
    first.write_bytes(b"rotation")
    second.write_bytes(b"rotation")
    other.write_bytes(b"another rotation")
    named = assembly_name({**settings, "alignment": str(first)})
    assert named == assembly_name({**settings, "alignment": str(second)})
    assert named != assembly_name({**settings, "alignment": str(other)})
    assert named != assembly_name(settings)


@pytest.mark.parametrize("value,expected", [(None, False), ("0", False), ("1", True), ("true", False)])
def test_a_run_is_on_an_ephemeral_disk_only_when_the_launcher_set_it_to_one(monkeypatch, value, expected):
    from polyplex_research.utils.assemble import runs_on_ephemeral_disk

    if value is None:
        monkeypatch.delenv("POLYPLEX_EPHEMERAL_DISK", raising=False)
    else:
        monkeypatch.setenv("POLYPLEX_EPHEMERAL_DISK", value)
    assert runs_on_ephemeral_disk() is expected


def test_an_assembly_the_drive_store_refuses_still_trains(tmp_path, monkeypatch, caplog):
    import letify

    import polyplex_research.utils.assemble as assemble

    def refuse(name, path):
        raise OSError("Network is unreachable")

    monkeypatch.setattr(letify, "stash", refuse, raising=False)
    monkeypatch.setattr(letify, "unstash", lambda name, path: None, raising=False)
    monkeypatch.setattr("polyplex_research.utils.phase.free_downloads", lambda repo_ids: None)
    monkeypatch.setenv("POLYPLEX_EPHEMERAL_DISK", "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    def save(model, path, full_precision_layers=()):
        os.makedirs(path, exist_ok=True)
        open(os.path.join(path, "config.json"), "w").write("{}")

    monkeypatch.setattr(assemble, "assemble_dualffn", lambda device, **settings: "model")
    monkeypatch.setattr(assemble, "save_assembled_dualffn", save)
    monkeypatch.setattr("polyplex_research.utils.quantization.set_nvfp4_backward", lambda model, dtype: 0)
    assert assemble.load_or_assemble_dualffn("cpu", quantization="nvfp4", appended_layers=4) == "model"
    assert "not stored" in caplog.text


def test_a_stored_assembly_listed_file_by_file_is_fetched_file_by_file(tmp_path, monkeypatch):
    import json

    from polyplex_research.utils.assemble import fetch_stored_assembly

    store = tmp_path / "store"
    store.mkdir()
    fake_drive_store(monkeypatch, store)
    (store / "dualffn-0").write_text(json.dumps(["config.json", "model.safetensors"]))
    for file in ("config.json", "model.safetensors"):
        (store / f"dualffn-0-{file}").write_text(file)
    target = tmp_path / "a" / "dualffn-0"
    assert fetch_stored_assembly("dualffn-0", str(target)) is True
    assert sorted(os.listdir(target)) == ["config.json", "model.safetensors"]
    assert (target / "model.safetensors").read_text() == "model.safetensors"


def test_a_missing_stored_assembly_is_reported_as_missing(tmp_path, monkeypatch):
    import letify

    from polyplex_research.utils.assemble import fetch_stored_assembly

    monkeypatch.setattr(letify, "unstash", lambda name, path: None, raising=False)
    assert fetch_stored_assembly("dualffn-0", str(tmp_path / "a")) is False


def test_each_language_row_is_the_mean_of_the_text_llm_rows_of_its_text():
    from polyplex_research.utils.assemble import language_embedding_rows

    class Pieces:
        def __init__(self, pieces):
            self.pieces = pieces

        def __len__(self):
            return len(self.pieces)

        def convert_ids_to_tokens(self, ids):
            return [self.pieces[index] for index in ids]

        def convert_tokens_to_string(self, tokens):
            return "".join(tokens)

    class Characters:
        def encode(self, text, add_special_tokens=False):
            return [ord(character) - ord("a") for character in text]

    table = torch.arange(12, dtype=torch.float32).view(4, 3)
    rows = language_embedding_rows(Pieces(["a", "bd", ""]), Characters(), table, 5)
    torch.testing.assert_close(rows[0], table[0])
    torch.testing.assert_close(rows[1], (table[1] + table[3]) / 2)
    assert rows[2:].abs().sum() == 0 and rows.shape == (5, 3)


def test_the_residual_alignment_command_writes_its_file_into_the_output_directory(monkeypatch, tmp_path):
    import transformers

    import polyplex_research.utils.assemble as assemble
    import polyplex_research.utils.phase as phase

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda name: None)
    monkeypatch.setattr(phase, "dense_documents", lambda *args, **kwargs: ["text"] * 400)
    seen = {}

    def fit(personaplex, language_model, texts, appended_layers, device, shared):
        seen["shared"] = shared
        return {"rotations": torch.eye(2)[None], "scales": torch.ones(1), "report": [], "fitted": 1, "measured": 1}

    monkeypatch.setattr(assemble, "residual_alignment", fit)
    assemble.main(["residual-alignment", "--output", "elsewhere/shared.safetensors", "--shared", "--output-dir", tmp_path])
    assert seen["shared"] is True
    assert (tmp_path / "shared.safetensors").exists() and (tmp_path / "shared.json").exists()
