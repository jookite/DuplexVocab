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
"""Invariants of the text injection adapter."""

import pytest
import torch
import torch.nn.functional as F

from conftest import AUDIO_VOCAB_SIZE, BASE_VOCAB_SIZE, VOCAB_SIZE, build_model
from polyplex_research.adapters.phase_1_efficient_adapter import (
    EfficientAdapterP1Config,
    EfficientAdapterP1Model,
)
from polyplex_research.adapters.phase_1_efficient_adapter.modeling_phase_1_efficient_adapter import (
    TEXT_ADAPTER_WEIGHTS_NAME,
    text_tables,
)


def build_adapter(model, **kwargs):
    config = EfficientAdapterP1Config.from_base_model(
        model, base_vocab_size=BASE_VOCAB_SIZE, **kwargs
    )
    adapter = EfficientAdapterP1Model(model, config)
    adapter.freeze_base()
    return adapter


# ---------------------------------------------------------------------------
# Table selection
# ---------------------------------------------------------------------------


def test_text_tables_are_the_three_upstream_tables(base_model):
    found = {name: module for _, name, module, _ in text_tables(base_model)}
    assert set(found) == {"__input__", "__output__", "text_embed_tokens"}
    assert found["__input__"] is base_model.model.embed_tokens
    assert found["__output__"] is base_model.lm_head
    assert found["text_embed_tokens"] is base_model.depth_decoder.model.text_embed_tokens


def test_reserved_row_is_reported_only_where_it_exists(base_model):
    reserved = {name: flag for _, name, _, flag in text_tables(base_model)}
    # The two embeddings carry the ungenerated-frame row, the head does not.
    assert reserved == {"__input__": True, "__output__": False, "text_embed_tokens": True}


def test_only_requested_tables_are_selected(base_model):
    config = EfficientAdapterP1Config.from_base_model(
        base_model, base_vocab_size=BASE_VOCAB_SIZE,
        train_lm_head=True, train_input_embeddings=False, train_depth_text_embeddings=True,
    )
    adapter = EfficientAdapterP1Model(base_model, config)
    assert [name for name, _ in adapter._select_tables()] == ["__output__", "text_embed_tokens"]


def test_freeze_base_rejects_a_table_no_gradient_reaches(base_model):
    # The input lookup runs inside the `no_grad` block and is a different tensor from the head, so
    # selecting it on its own would train nothing.
    with pytest.raises(RuntimeError, match="__input__"):
        build_adapter(base_model, train_input_embeddings=True, train_lm_head=False)


def test_the_depth_text_table_is_reachable_through_the_audio_objective(base_model):
    adapter = build_adapter(base_model, train_depth_text_embeddings=True)
    assert [name for name, _ in adapter._tables] == ["__output__", "text_embed_tokens"]


# ---------------------------------------------------------------------------
# What is allowed to move
# ---------------------------------------------------------------------------


def test_freeze_base_leaves_only_the_selected_tables_trainable(base_model):
    adapter = build_adapter(base_model)
    trainable = {name for name, parameter in adapter.named_parameters() if parameter.requires_grad}
    assert trainable == {"model.lm_head.weight"}
    assert [id(p) for p in adapter.adapter_parameters()] == [id(base_model.lm_head.weight)]


def test_num_trainable_rows_counts_the_appended_rows_only(base_model):
    adapter = build_adapter(base_model)
    assert adapter.num_trainable_rows() == VOCAB_SIZE - BASE_VOCAB_SIZE


def test_gradient_hook_zeroes_the_pretrained_rows(base_model, text_batch):
    adapter = build_adapter(base_model)
    # Every row has to be reachable for the assertion on the appended rows to mean anything.
    text_batch["text_labels"] = torch.arange(12).reshape(2, 6) % VOCAB_SIZE
    adapter(**text_batch).loss.backward()

    grad = base_model.lm_head.weight.grad
    assert grad is not None
    assert torch.count_nonzero(grad[:BASE_VOCAB_SIZE]) == 0
    assert torch.count_nonzero(grad[BASE_VOCAB_SIZE:]) > 0


def test_backbone_receives_no_gradient(base_model, text_batch):
    adapter = build_adapter(base_model)
    adapter(**text_batch).loss.backward()

    for name, parameter in base_model.named_parameters():
        if parameter is base_model.lm_head.weight:
            continue
        assert parameter.grad is None, f"{name} was differentiated"


def test_backbone_stays_detached_under_enabled_grad(base_model, text_batch):
    adapter = build_adapter(base_model)
    with torch.enable_grad():
        outputs = adapter(**text_batch)
    outputs.loss.backward()
    assert base_model.model.embed_tokens.weight.grad is None


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def test_loss_is_the_shifted_masked_cross_entropy(base_model, text_batch):
    adapter = build_adapter(base_model)
    outputs = adapter(**text_batch)

    expected = F.cross_entropy(
        outputs.logits[:, :-1].float().reshape(-1, VOCAB_SIZE),
        text_batch["text_labels"][:, 1:].reshape(-1),
        ignore_index=-100,
    )
    assert torch.allclose(outputs.loss, expected, atol=1e-5)


def test_fully_masked_labels_give_zero_rather_than_nan(base_model, text_batch):
    adapter = build_adapter(base_model)
    text_batch["text_labels"] = torch.full_like(text_batch["text_labels"], -100)
    loss = adapter(**text_batch).loss
    assert torch.isfinite(loss) and loss.item() == 0.0


def test_penalty_is_zero(base_model):
    assert build_adapter(base_model).penalty().item() == 0.0


def test_forward_without_inputs_raises(base_model):
    with pytest.raises(ValueError):
        build_adapter(base_model)(input_ids=None, inputs_embeds=None)


# ---------------------------------------------------------------------------
# Replication
# ---------------------------------------------------------------------------


def test_save_pretrained_writes_only_the_appended_rows(base_model, tmp_path):
    from safetensors.torch import load_file

    adapter = build_adapter(base_model)
    adapter.save_pretrained(tmp_path)

    stored = load_file(tmp_path / TEXT_ADAPTER_WEIGHTS_NAME)
    assert set(stored) == {"__output__"}
    assert stored["__output__"].shape == (VOCAB_SIZE - BASE_VOCAB_SIZE, base_model.config.hidden_size)
    assert torch.equal(stored["__output__"], base_model.lm_head.weight.data[BASE_VOCAB_SIZE:])


def test_from_pretrained_restores_the_rows(base_model, tmp_path):
    from conftest import build_model

    adapter = build_adapter(base_model)
    with torch.no_grad():
        base_model.lm_head.weight[BASE_VOCAB_SIZE:] += 1.0
    adapter.save_pretrained(tmp_path)

    fresh = build_model(seed=1)
    restored = EfficientAdapterP1Model.from_pretrained(tmp_path, model=fresh)
    assert torch.equal(
        fresh.lm_head.weight.data[BASE_VOCAB_SIZE:],
        base_model.lm_head.weight.data[BASE_VOCAB_SIZE:],
    )
    assert restored.config.base_vocab_size == BASE_VOCAB_SIZE


# ---------------------------------------------------------------------------
# The mixture of pretrained rows
# ---------------------------------------------------------------------------

MIXTURE = 4


def span_fit(support, row):
    r"""
    Args:
        support (`torch.FloatTensor` of shape `(mixture, hidden_size)`):
            The pretrained rows an appended row may be built from.
        row (`torch.FloatTensor` of shape `(hidden_size,)`):
            The row being fitted.

    Returns:
        `torch.FloatTensor`: The closest point of the span of `support` to `row`.
    """
    return support.T @ torch.linalg.lstsq(support.T, row).solution


def test_row_mixture_makes_the_coefficients_the_parameters(base_model):
    adapter = build_adapter(base_model, base_row_mixture=MIXTURE)

    assert len(adapter.row_mixture) == 1
    assert adapter.row_mixture[0].shape == (VOCAB_SIZE - BASE_VOCAB_SIZE, MIXTURE)
    assert [id(p) for p in adapter.adapter_parameters()] == [id(adapter.row_mixture[0])]
    # The table is a function of the coefficients now, so it holds no parameter of its own.
    assert not base_model.lm_head.weight.requires_grad


def test_row_mixture_is_off_by_default(base_model):
    adapter = build_adapter(base_model)
    assert adapter.row_mixture is None
    assert [id(p) for p in adapter.adapter_parameters()] == [id(base_model.lm_head.weight)]


def test_row_mixture_starts_at_the_fit_of_the_expanded_rows(base_model):
    original = base_model.lm_head.weight.data[BASE_VOCAB_SIZE:].clone()
    adapter = build_adapter(base_model, base_row_mixture=MIXTURE)

    rows = base_model.lm_head.weight.data[BASE_VOCAB_SIZE:]
    pretrained = base_model.lm_head.weight.data[:BASE_VOCAB_SIZE]
    for index in range(rows.shape[0]):
        closest = span_fit(pretrained[adapter._row_support[index]], original[index])
        assert (rows[index] - original[index]).norm() <= (
            closest - original[index]
        ).norm() * 1.05 + 1e-4


def test_a_step_moves_the_appended_rows_and_leaves_the_pretrained_ones_bit_identical(
    base_model, text_batch
):
    adapter = build_adapter(base_model, base_row_mixture=MIXTURE)
    frozen = base_model.lm_head.weight.data[:BASE_VOCAB_SIZE].clone()
    appended = base_model.lm_head.weight.data[BASE_VOCAB_SIZE:].clone()

    optimizer = torch.optim.AdamW(adapter.adapter_parameters(), lr=0.1)
    adapter(**text_batch).loss.backward()
    assert torch.count_nonzero(adapter.row_mixture[0].grad) > 0
    assert base_model.lm_head.weight.grad is None
    optimizer.step()
    adapter.materialize_rows()

    assert torch.equal(base_model.lm_head.weight.data[:BASE_VOCAB_SIZE], frozen)
    assert not torch.equal(base_model.lm_head.weight.data[BASE_VOCAB_SIZE:], appended)


def test_every_materialised_row_lies_in_the_span_of_its_support(base_model):
    adapter = build_adapter(base_model, base_row_mixture=MIXTURE)
    with torch.no_grad():
        adapter.row_mixture[0].normal_()
    adapter.materialize_rows()

    rows = base_model.lm_head.weight.data[BASE_VOCAB_SIZE:]
    pretrained = base_model.lm_head.weight.data[:BASE_VOCAB_SIZE]
    for index in range(rows.shape[0]):
        support = pretrained[adapter._row_support[index]]
        assert torch.allclose(span_fit(support, rows[index]), rows[index], atol=1e-4)


def test_the_saved_artifact_carries_no_parameterisation(base_model, text_batch, tmp_path):
    from safetensors.torch import load_file

    from conftest import build_model

    adapter = build_adapter(base_model, base_row_mixture=MIXTURE)
    with torch.no_grad():
        adapter.row_mixture[0].normal_()
    expected = adapter(**text_batch).logits
    adapter.save_pretrained(tmp_path)

    stored = load_file(tmp_path / TEXT_ADAPTER_WEIGHTS_NAME)
    assert set(stored) == {"__output__"}
    assert stored["__output__"].shape == (VOCAB_SIZE - BASE_VOCAB_SIZE, base_model.config.hidden_size)

    # The same base model, so any difference in the logits is the parameterisation and nothing else.
    plain = build_adapter(build_model(seed=0), base_row_mixture=0)
    with torch.no_grad():
        plain.model.lm_head.weight[BASE_VOCAB_SIZE:] = stored["__output__"]
    assert torch.allclose(plain(**text_batch).logits, expected, atol=1e-5)


def test_partition_direction_needs_a_mixture(base_model):
    with pytest.raises(ValueError, match="partition_direction"):
        build_adapter(base_model, partition_direction=True)


def test_partition_direction_starts_at_the_mixture_alone(base_model, text_batch):
    expected = build_adapter(build_model(seed=0), base_row_mixture=MIXTURE)(**text_batch).logits
    adapter = build_adapter(base_model, base_row_mixture=MIXTURE, partition_direction=True)
    assert torch.equal(adapter(**text_batch).logits, expected)


def test_partition_direction_shifts_every_appended_logit_by_its_score(base_model):
    adapter = build_adapter(base_model, base_row_mixture=MIXTURE, partition_direction=True)
    # Read on fixed states: through the tie the rows also move what an appended input token embeds to.
    hidden = torch.randn(2, 5, base_model.config.hidden_size)
    before = adapter.text_logits(hidden)
    with torch.no_grad():
        adapter.partition_direction.normal_()
    after = adapter.text_logits(hidden)

    assert torch.equal(after[..., :BASE_VOCAB_SIZE], before[..., :BASE_VOCAB_SIZE])
    shift = after[..., BASE_VOCAB_SIZE:] - before[..., BASE_VOCAB_SIZE:]
    assert torch.allclose(shift, shift[..., :1].expand_as(shift), atol=1e-4)
    assert shift.abs().max() > 0


def test_partition_direction_is_trained_with_the_coefficients(base_model, text_batch):
    adapter = build_adapter(base_model, base_row_mixture=MIXTURE, partition_direction=True)
    assert any(p is adapter.partition_direction for p in adapter.adapter_parameters())
    adapter(**text_batch).loss.backward()
    assert torch.count_nonzero(adapter.partition_direction.grad) > 0


def test_partition_direction_round_trips_through_a_save(tmp_path):
    from polyplex_research.utils.phase import load_adapter_slices

    def round_trip(direction, loader):
        adapter = build_adapter(build_model(seed=0), base_row_mixture=MIXTURE, partition_direction=direction)
        torch.manual_seed(0)
        with torch.no_grad():
            adapter.row_mixture[0].normal_()
            if direction:
                adapter.partition_direction.normal_()
        hidden = torch.randn(2, 5, adapter.model.config.hidden_size)
        directory = tmp_path / f"{direction}_{loader}"
        adapter.save_pretrained(directory)
        if loader == "hub":
            restored = EfficientAdapterP1Model.from_pretrained(directory, model=build_model(seed=0))
        else:
            restored = build_adapter(
                build_model(seed=0), base_row_mixture=MIXTURE, partition_direction=direction
            )
            load_adapter_slices(restored, directory)
        if direction:
            assert torch.equal(restored.partition_direction.data, adapter.partition_direction.data)
        return (restored.text_logits(hidden) - adapter.text_logits(hidden)).abs().max()

    # The refit of the coefficients is a ridge solve and not exact, so the direction is held to the
    # error the mixture alone already comes back with.
    for loader in ("hub", "slices"):
        assert round_trip(True, loader) <= round_trip(False, loader) + 1e-4


# ---------------------------------------------------------------------------
# Speaking through the shared mixture
# ---------------------------------------------------------------------------


def speaking_adapter(model):
    return build_adapter(
        model, base_row_mixture=MIXTURE, share_row_mixture=True, train_depth_text_embeddings=True
    )


def test_shared_mixture_needs_a_mixture(base_model):
    with pytest.raises(ValueError, match="share_row_mixture"):
        build_adapter(base_model, share_row_mixture=True)


def test_shared_mixture_builds_every_table_from_one_set_of_coefficients(base_model):
    adapter = speaking_adapter(base_model)
    assert len(adapter.row_mixture) == 1
    assert len(adapter._tables) == 2

    coefficients = adapter.row_mixture[0].detach()
    for _, module in adapter._tables:
        pretrained = module.weight.data[:BASE_VOCAB_SIZE].float()
        expected = (coefficients.unsqueeze(-1) * pretrained[adapter._row_support]).sum(1)
        torch.testing.assert_close(
            module.weight.data[BASE_VOCAB_SIZE:VOCAB_SIZE].float(), expected, rtol=1e-4, atol=1e-5
        )


def test_audio_loss_matches_the_depth_convention_on_pretrained_tokens(audio_batch):
    # With no appended token on any frame the adapter must reduce to the model's own depth path,
    # conditioned on frame t and scored on frame t + 1.
    audio_batch["input_ids"][:] = 3
    ours = speaking_adapter(build_model())(**audio_batch).loss

    model = build_model()
    input_ids, codes = audio_batch["input_ids"], audio_batch["assistant_audio_codes"]
    labels = audio_batch["audio_labels"]
    with torch.no_grad():
        embeds = model._embed_audio_codes(torch.cat([codes, audio_batch["user_audio_codes"]], dim=1))
        embeds = embeds + model.model.embed_tokens(input_ids)
        hidden = model.model(inputs_embeds=embeds, attention_mask=audio_batch["attention_mask"])
        hidden = hidden.last_hidden_state
        count = labels.shape[1]
        depth_ids = torch.cat(
            [input_ids[:, 1:].reshape(-1, 1), codes[:, :, 1:].transpose(1, 2).reshape(-1, count)], dim=1
        )[:, :count]
        logits = model.depth_decoder(
            last_hidden_state=hidden[:, :-1].reshape(-1, 1, hidden.shape[-1]), input_ids=depth_ids
        ).logits
        expected = F.cross_entropy(
            logits.float().reshape(-1, AUDIO_VOCAB_SIZE),
            labels[:, :, 1:].transpose(1, 2).reshape(-1),
        )
    torch.testing.assert_close(ours.detach(), expected, rtol=1e-5, atol=1e-5)


def test_the_einsum_path_matches_every_per_codebook_linear(base_model):
    from polyplex_research.adapters.phase_1_efficient_adapter.modeling_phase_1_efficient_adapter import (
        per_codebook_einsum,
    )

    depth = base_model.depth_decoder
    layers = [m for m in depth.modules() if type(m).__name__ == "PersonaPlexFlexibleLinear"]
    assert layers
    for layer in layers:
        num_layers, _, width = layer.weight.shape
        index = torch.arange(num_layers).flip(0)
        for x, layer_idx in (
            (torch.randn(3, num_layers, width), None),
            (torch.randn(3, num_layers, width), index),
            (torch.randn(3, 1, width), index),
        ):
            expected = layer(x, layer_idx)
            with per_codebook_einsum(depth):
                torch.testing.assert_close(layer(x, layer_idx), expected, rtol=1e-5, atol=1e-5)
        assert "forward" not in vars(layer)


def test_audio_loss_reaches_the_shared_coefficients_and_nothing_else(base_model, audio_batch):
    adapter = speaking_adapter(base_model)
    # An appended token on a conditioned frame is what connects the codes to the rows.
    audio_batch["input_ids"][:, 1:] = BASE_VOCAB_SIZE + 1
    adapter(**audio_batch).loss.backward()

    grad = adapter.row_mixture[0].grad
    assert grad is not None and grad.abs().sum() > 0
    assert [name for name, p in base_model.named_parameters() if p.grad is not None] == []


def test_audio_loss_is_blind_to_rows_no_frame_reads(base_model, audio_batch):
    # Without an appended token on a conditioned frame the loss has no path to the coefficients.
    audio_batch["input_ids"][:] = 3
    assert not speaking_adapter(base_model)(**audio_batch).loss.requires_grad


def test_audio_only_batch_skips_the_text_head(base_model, audio_batch):
    outputs = speaking_adapter(base_model)(**audio_batch)
    assert outputs.logits is None
    assert torch.isfinite(outputs.loss)


def test_audio_loss_of_a_fully_masked_batch_is_zero(base_model, audio_batch):
    audio_batch["audio_labels"][:] = -100
    assert speaking_adapter(base_model)(**audio_batch).loss.item() == 0.0


# ---------------------------------------------------------------------------
# Capacity confined to fixed input directions
# ---------------------------------------------------------------------------


def confined_adapter(model, tmp_path, rank=2, layers=(0,)):
    from safetensors.torch import save_file

    hidden = model.config.hidden_size
    basis = {
        f"layer_{index}": torch.linalg.qr(torch.randn(hidden, rank, generator=torch.Generator().manual_seed(index)))[0].contiguous()
        for index in layers
    }
    path = tmp_path / "basis.safetensors"
    save_file(basis, path)
    return build_adapter(model, base_row_mixture=MIXTURE, confined_basis=str(path)), basis


def test_confined_capacity_starts_at_the_model_alone(text_batch, tmp_path):
    expected = build_adapter(build_model(seed=0), base_row_mixture=MIXTURE)(**text_batch).logits
    adapter, _ = confined_adapter(build_model(seed=0), tmp_path)
    assert torch.equal(adapter(**text_batch).logits, expected)


def test_confined_capacity_is_the_only_backbone_parameter_that_trains(base_model, text_batch, tmp_path):
    adapter, _ = confined_adapter(base_model, tmp_path)
    trainable = [name for name, parameter in adapter.named_parameters() if parameter.requires_grad]
    assert any(name.endswith("mlp.up") for name in trainable)
    assert all(name.endswith("mlp.up") or "row_mixture" in name for name in trainable)
    adapter(**text_batch).loss.backward()
    assert torch.count_nonzero(adapter.confined[0].up.grad) > 0


def test_confined_capacity_ignores_inputs_outside_its_directions(base_model, tmp_path):
    adapter, basis = confined_adapter(base_model, tmp_path)
    module = adapter.confined[0]
    with torch.no_grad():
        module.up.normal_()
    hidden = torch.randn(3, 5, base_model.config.hidden_size)
    outside = hidden - (hidden @ basis["layer_0"]) @ basis["layer_0"].T
    torch.testing.assert_close(module(outside), module.wrapped(outside), rtol=1e-4, atol=1e-4)
    assert not torch.allclose(module(hidden), module.wrapped(hidden))


def test_confined_capacity_round_trips_through_a_save(text_batch, tmp_path):
    from polyplex_research.utils.phase import load_adapter_slices

    adapter, _ = confined_adapter(build_model(seed=0), tmp_path)
    with torch.no_grad():
        adapter.confined[0].up.normal_()
    adapter.save_pretrained(tmp_path / "run")

    # The rows come back through the ridge refit of the mixture, which is not exact; the correction
    # is stored as it is.
    restored, _ = confined_adapter(build_model(seed=0), tmp_path)
    load_adapter_slices(restored, tmp_path / "run")
    assert torch.equal(restored.confined[0].up.data, adapter.confined[0].up.data)


def test_confined_attention_capacity_starts_silent_trains_and_round_trips(text_batch, tmp_path):
    from safetensors.torch import save_file

    from polyplex_research.utils.phase import load_adapter_slices

    model = build_model(seed=0)
    expected = build_adapter(build_model(seed=0), base_row_mixture=MIXTURE)(**text_batch).logits
    hidden = model.config.hidden_size
    basis = {"layer_0": torch.linalg.qr(torch.randn(hidden, 2, generator=torch.Generator().manual_seed(0)))[0]}
    path = tmp_path / "attention.safetensors"
    save_file({k: v.contiguous() for k, v in basis.items()}, path)

    adapter = build_adapter(model, base_row_mixture=MIXTURE, confined_attention_basis=str(path))
    module = adapter.confined["v0"]
    assert module.up.shape[0] == module.wrapped.out_features
    assert torch.equal(adapter(**text_batch).logits, expected)
    adapter(**text_batch).loss.backward()
    assert torch.count_nonzero(module.up.grad) > 0

    with torch.no_grad():
        module.up.normal_()
    adapter.save_pretrained(tmp_path / "run")
    restored = build_adapter(build_model(seed=0), base_row_mixture=MIXTURE, confined_attention_basis=str(path))
    load_adapter_slices(restored, tmp_path / "run")
    assert torch.equal(restored.confined["v0"].up.data, module.up.data)


def test_session_expert_starts_silent_trains_both_sides_and_round_trips(text_batch, tmp_path):
    from polyplex_research.utils.phase import load_adapter_slices

    expected = build_adapter(build_model(seed=0), base_row_mixture=MIXTURE)(**text_batch).logits
    adapter = build_adapter(
        build_model(seed=0), base_row_mixture=MIXTURE, session_expert_layers="0-0", session_expert_rank=2
    )
    module = adapter.confined[0]
    assert isinstance(module.basis, torch.nn.Parameter)
    assert torch.equal(adapter(**text_batch).logits, expected)
    adapter(**text_batch).loss.backward()
    assert torch.count_nonzero(module.up.grad) > 0

    with torch.no_grad():
        module.up.normal_()
        module.basis.normal_()
    adapter.zero_grad()
    adapter(**text_batch).loss.backward()
    assert torch.count_nonzero(module.basis.grad) > 0
    with adapter.corrections_disabled():
        assert torch.equal(adapter(**text_batch).logits, expected)

    adapter.save_pretrained(tmp_path / "run")
    restored = build_adapter(
        build_model(seed=0), base_row_mixture=MIXTURE, session_expert_layers="0-0", session_expert_rank=2
    )
    load_adapter_slices(restored, tmp_path / "run")
    assert torch.equal(restored.confined[0].basis.data, module.basis.data)
    assert torch.equal(restored.confined[0].up.data, module.up.data)


def test_fixed_rows_leave_the_session_expert_as_the_only_trainable_part(text_batch):
    adapter = build_adapter(
        build_model(seed=0),
        base_row_mixture=MIXTURE,
        session_expert_layers="0-0",
        session_expert_rank=2,
        train_rows=False,
    )
    # Assembly and the trainer each freeze the model once.
    adapter.freeze_base()
    module = adapter.confined[0]
    trainable = {id(parameter) for parameter in adapter.parameters() if parameter.requires_grad}
    assert trainable == {id(module.up), id(module.basis)}

    rows = adapter.model.get_output_embeddings().weight.detach().clone()
    with torch.no_grad():
        module.up.normal_()
    adapter(**text_batch).loss.backward()
    assert all(parameter.grad is None for parameter in adapter.row_mixture)
    assert torch.count_nonzero(module.up.grad) > 0
    assert torch.equal(adapter.model.get_output_embeddings().weight, rows)


def test_fixed_corrections_leave_the_session_expert_out_of_training(text_batch):
    adapter = build_adapter(
        build_model(seed=0),
        base_row_mixture=MIXTURE,
        session_expert_layers="0-0",
        session_expert_rank=2,
        train_corrections=False,
    )
    adapter.freeze_base()
    module = adapter.confined[0]
    trainable = {id(parameter) for parameter in adapter.parameters() if parameter.requires_grad}
    assert trainable == {id(parameter) for parameter in adapter.row_mixture}

    with torch.no_grad():
        module.up.normal_()
    adapter(**text_batch).loss.backward()
    assert module.up.grad is None and module.basis.grad is None
    assert all(parameter.grad is not None for parameter in adapter.row_mixture)


def test_fixing_both_the_rows_and_the_corrections_is_rejected():
    with pytest.raises(RuntimeError, match="train_rows"):
        build_adapter(
            build_model(seed=0),
            base_row_mixture=MIXTURE,
            session_expert_layers="0-0",
            session_expert_rank=2,
            train_rows=False,
            train_corrections=False,
        )


def test_fixed_rows_without_a_correction_train_nothing_and_are_rejected():
    with pytest.raises(RuntimeError, match="train_rows"):
        build_adapter(build_model(seed=0), base_row_mixture=MIXTURE, train_rows=False)


def confined_adapter_with_moments(model, tmp_path, rank=2):
    hidden = model.config.hidden_size
    adapter, basis = confined_adapter(model, tmp_path, rank=rank)
    moments_dir = tmp_path / "moments"
    moments_dir.mkdir()
    generator = torch.Generator().manual_seed(1)
    torch.save(
        {
            group: (lambda x: x @ x.T)(torch.randn(hidden, hidden, generator=generator)).half()
            for group in ("deploy_en_speech", "deploy_silent_pad", "empty_ko")
        },
        moments_dir / "moments_00_ffn.pt",
    )
    config = EfficientAdapterP1Config.from_base_model(
        build_model(seed=0),
        base_vocab_size=BASE_VOCAB_SIZE,
        base_row_mixture=MIXTURE,
        confined_basis=str(tmp_path / "basis.safetensors"),
        confined_moments=str(moments_dir),
    )
    adapter = EfficientAdapterP1Model(build_model(seed=0), config)
    adapter.freeze_base()
    return adapter


def test_confined_leak_is_the_deploy_share_and_trains_only_the_numerator(tmp_path):
    adapter = confined_adapter_with_moments(build_model(seed=0), tmp_path)
    module = adapter.confined[0]
    with torch.no_grad():
        module.up.normal_()
    leak = adapter.confined_leak()
    expected = module.output_energy(module.deploy_moment) / module.output_energy(module.empty_moment)
    torch.testing.assert_close(leak, expected)
    leak.backward()
    deploy_only = torch.autograd.grad(module.output_energy(module.deploy_moment), module.up)[0]
    torch.testing.assert_close(module.up.grad, deploy_only / module.output_energy(module.empty_moment).detach())


def test_corrections_disabled_answers_as_the_backbone_and_restores(text_batch, tmp_path):
    adapter, _ = confined_adapter(build_model(seed=0), tmp_path)
    with torch.no_grad():
        adapter.confined[0].up.normal_()
    before = adapter.confined[0].up.data.clone()
    plain = build_adapter(build_model(seed=0), base_row_mixture=MIXTURE)(**text_batch).logits
    with adapter.corrections_disabled():
        torch.testing.assert_close(adapter(**text_batch).logits, plain)
    assert torch.equal(adapter.confined[0].up.data, before)


def test_timing_kl_is_zero_with_the_corrections_off_and_positive_with_them_on(text_batch, tmp_path):
    from types import SimpleNamespace

    from polyplex_research.datasets.duplex_data import PAD_TOKEN_ID
    from polyplex_research.utils.phase import Phase1Trainer

    adapter, _ = confined_adapter(build_model(seed=0), tmp_path)
    batch = {**text_batch, "input_ids": text_batch["input_ids"].clone()}
    batch["input_ids"][:, 3] = PAD_TOKEN_ID
    stub = SimpleNamespace(_prefix_frames=0)
    quiet = Phase1Trainer._timing_kl(stub, adapter, batch, adapter(**batch).logits)
    torch.testing.assert_close(quiet, torch.zeros(()), atol=1e-6, rtol=0)
    with torch.no_grad():
        adapter.confined[0].up.normal_()
    loud = Phase1Trainer._timing_kl(stub, adapter, batch, adapter(**batch).logits)
    assert loud > 0


def test_parameters_replaced_answers_with_the_given_state_and_restores(text_batch):
    adapter = build_adapter(
        build_model(seed=0), base_row_mixture=MIXTURE, session_expert_layers="0-0", session_expert_rank=2
    )
    with torch.no_grad():
        adapter.confined[0].up.normal_()
    start = adapter.trainable_state()
    expected = adapter(**text_batch).logits
    with torch.no_grad():
        adapter.row_mixture[0].add_(0.5)
        adapter.confined[0].up.normal_()
    moved = adapter(**text_batch).logits
    with adapter.parameters_replaced(start):
        torch.testing.assert_close(adapter(**text_batch).logits, expected)
    torch.testing.assert_close(adapter(**text_batch).logits, moved)


def test_timing_kl_against_the_starting_rows_is_zero_until_the_rows_move(text_batch):
    from types import SimpleNamespace

    from polyplex_research.datasets.duplex_data import PAD_TOKEN_ID
    from polyplex_research.utils.phase import Phase1Trainer

    adapter = build_adapter(
        build_model(seed=0), base_row_mixture=MIXTURE, session_expert_layers="0-0", session_expert_rank=2,
        train_corrections=False,
    )
    with torch.no_grad():
        adapter.confined[0].up.normal_()
    batch = {**text_batch, "input_ids": text_batch["input_ids"].clone()}
    batch["input_ids"][:, 3] = PAD_TOKEN_ID
    stub = SimpleNamespace(_prefix_frames=0, timing_reference="start", _start_state=adapter.trainable_state())
    quiet = Phase1Trainer._timing_kl(stub, adapter, batch, adapter(**batch).logits)
    torch.testing.assert_close(quiet, torch.zeros(()), atol=1e-6, rtol=0)
    with torch.no_grad():
        adapter.row_mixture[0].normal_()
    loud = Phase1Trainer._timing_kl(stub, adapter, batch, adapter(**batch).logits)
    assert loud > 0


def test_config_to_dict_lists_every_field():
    from dataclasses import fields

    from polyplex_research.adapters.phase_1_efficient_adapter import EfficientAdapterP1Config

    config = EfficientAdapterP1Config(base_vocab_size=4, vocab_size=6)
    exported = config.to_dict()
    assert set(exported) == {field.name for field in fields(config)}
    assert exported["vocab_size"] == 6


@pytest.mark.parametrize("kwargs", [{}, {"session_expert_layers": "0-0", "session_expert_rank": 2}], ids=["rows", "expert"])
def test_packed_documents_given_position_ids_do_not_attend_to_each_other(kwargs):
    adapter = build_adapter(build_model(seed=0), base_row_mixture=MIXTURE, **kwargs)
    opening = VOCAB_SIZE

    def batch(ids, positions):
        codes = torch.ones(1, 8, len(ids), dtype=torch.long)
        return {
            "input_ids": torch.tensor([ids]),
            "position_ids": torch.tensor([positions]),
            "assistant_audio_codes": codes[:, :4],
            "user_audio_codes": codes[:, 4:],
        }

    with torch.no_grad():
        packed = adapter(**batch([opening, 5, 6, 7, opening, 8, 9], [0, 1, 2, 3, 0, 1, 2])).logits
        alone = adapter(**batch([opening, 8, 9], [0, 1, 2])).logits
    torch.testing.assert_close(packed[0, 4:], alone[0], rtol=1e-5, atol=1e-5)
