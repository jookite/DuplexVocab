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
"""Invariants of the phase optimizers."""

import torch

import pytest

from polyplex_research.utils.optimization import FactoredAdam, MuonWithAdamW


def build():
    torch.manual_seed(0)
    matrix = torch.nn.Parameter(torch.randn(8, 4))
    table = torch.nn.Parameter(torch.randn(6, 4))
    vector = torch.nn.Parameter(torch.randn(4))
    optimizer = MuonWithAdamW(
        [matrix],
        [{"params": [table], "weight_decay": 0.0}, {"params": [vector], "weight_decay": 0.0}],
        lr=1e-2,
    )
    return optimizer, matrix, table, vector


def test_both_inner_optimizers_step_their_own_parameters():
    optimizer, matrix, table, vector = build()
    before = [p.detach().clone() for p in (matrix, table, vector)]
    (matrix.sum() ** 2 + table.square().sum() + vector.sum()).backward()
    optimizer.step()
    for old, new in zip(before, (matrix, table, vector)):
        assert not torch.equal(old, new.detach())
    assert matrix in optimizer.muon.param_groups[0]["params"]
    assert table in optimizer.adamw.param_groups[0]["params"]


def test_a_scheduler_reaches_both_inner_optimizers():
    optimizer, *_ = build()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5)
    assert scheduler.get_last_lr() == [5e-3] * 3
    assert optimizer.muon.param_groups[0]["lr"] == 5e-3
    assert all(group["lr"] == 5e-3 for group in optimizer.adamw.param_groups)


def test_the_state_survives_a_round_trip():
    optimizer, matrix, table, vector = build()
    (matrix.sum() ** 2 + table.square().sum() + vector.sum()).backward()
    optimizer.step()
    restored, *_ = build()
    restored.load_state_dict(optimizer.state_dict())
    assert restored.muon.param_groups[0] is restored.param_groups[0]
    assert len(restored.state_dict()["adamw"]["state"]) == 2


def test_a_run_with_no_weight_matrix_steps_on_adamw_alone():
    table = torch.nn.Parameter(torch.randn(6, 4))
    optimizer = MuonWithAdamW([], [{"params": [table], "weight_decay": 0.0}, {"params": [], "weight_decay": 0.0}], lr=1e-2)
    assert optimizer.muon is None
    before = table.detach().clone()
    table.square().sum().backward()
    optimizer.step()
    assert not torch.equal(before, table.detach())


# ---------------------------------------------------------------------------
# FactoredAdam (SPEC S7.2a)
# ---------------------------------------------------------------------------


def test_a_factored_step_is_adams_step_when_the_squared_gradient_has_rank_one():
    torch.manual_seed(0)
    start = torch.randn(6, 4)
    shape = torch.outer(torch.rand(6) + 0.1, torch.rand(4) + 0.1)
    gradients = [shape * scale for scale in (1.0, -1.0, 2.0)]
    factored = torch.nn.Parameter(start.clone())
    adam = torch.nn.Parameter(start.clone())
    ours = FactoredAdam([factored], lr=1e-2)
    reference = torch.optim.Adam([adam], lr=1e-2)
    for gradient in gradients:
        factored.grad, adam.grad = gradient.clone(), gradient.clone()
        ours.step()
        reference.step()
    torch.testing.assert_close(factored.detach(), adam.detach())
    state = ours.state[factored]
    assert state["exp_avg_sq_row"].shape == (6,) and state["exp_avg_sq_col"].shape == (4,)
    assert "exp_avg_sq" not in state


def test_a_sparse_gradient_moves_only_its_rows_and_their_moments():
    torch.manual_seed(0)
    table = torch.nn.Parameter(torch.randn(8, 4))
    optimizer = FactoredAdam([table], lr=1e-2)
    table.grad = torch.randn(8, 4)
    optimizer.step()
    before = table.detach().clone()
    moments = {key: value.clone() for key, value in optimizer.state[table].items()}
    rows = torch.tensor([1, 5])
    table.grad = torch.sparse_coo_tensor(rows[None], torch.randn(2, 4), (8, 4))
    optimizer.step()
    untouched = torch.tensor([0, 2, 3, 4, 6, 7])
    assert torch.equal(table.detach()[untouched], before[untouched])
    assert not torch.equal(table.detach()[rows], before[rows])
    state = optimizer.state[table]
    assert torch.equal(state["exp_avg"][untouched], moments["exp_avg"][untouched])
    assert torch.equal(state["exp_avg_sq_row"][untouched], moments["exp_avg_sq_row"][untouched])
    assert not torch.equal(state["exp_avg_sq_row"][rows], moments["exp_avg_sq_row"][rows])


def test_a_sparse_step_matches_the_dense_step_of_the_same_rows_from_a_fresh_state():
    torch.manual_seed(0)
    values = torch.randn(3, 4)
    rows = torch.tensor([0, 2, 5])
    sparse = torch.nn.Parameter(torch.randn(6, 4))
    dense = torch.nn.Parameter(sparse.detach()[rows].clone())
    sparse_optimizer, dense_optimizer = FactoredAdam([sparse], lr=1e-2), FactoredAdam([dense], lr=1e-2)
    sparse.grad = torch.sparse_coo_tensor(rows[None], values, (6, 4))
    dense.grad = values.clone()
    sparse_optimizer.step()
    dense_optimizer.step()
    torch.testing.assert_close(sparse.detach()[rows], dense.detach())


def test_an_adam_state_becomes_a_factored_one():
    torch.manual_seed(0)
    exp_avg_sq = torch.outer(torch.rand(6), torch.rand(4))
    state = {"step": torch.tensor(3.0), "exp_avg": torch.randn(6, 4), "exp_avg_sq": exp_avg_sq}
    factored = FactoredAdam.from_adam_state(state)
    assert torch.equal(factored["exp_avg"], state["exp_avg"]) and float(factored["step"]) == 3.0
    rebuilt = torch.outer(factored["exp_avg_sq_row"], factored["exp_avg_sq_col"]) / factored["exp_avg_sq_row"].mean()
    torch.testing.assert_close(rebuilt, exp_avg_sq)


def test_the_moments_keep_the_parameter_dtype_and_the_factors_stay_float32():
    table = torch.nn.Parameter(torch.randn(6, 4).to(torch.bfloat16))
    optimizer = FactoredAdam([table], lr=1e-2)
    table.grad = torch.randn(6, 4).to(torch.bfloat16)
    optimizer.step()
    state = optimizer.state[table]
    assert state["exp_avg"].dtype == torch.bfloat16
    assert state["exp_avg_sq_row"].dtype == torch.float32 and state["exp_avg_sq_col"].dtype == torch.float32


def test_tables_get_their_own_inner_optimizer_and_survive_a_round_trip():
    torch.manual_seed(0)
    matrix, table, vector = (torch.nn.Parameter(torch.randn(*shape)) for shape in ((8, 4), (6, 4), (4,)))
    build_one = lambda: MuonWithAdamW([matrix], [{"params": [vector], "weight_decay": 0.0}], lr=1e-2, table_params=[table])
    optimizer = build_one()
    assert table in optimizer.tables.param_groups[0]["params"]
    (matrix.sum() ** 2 + table.square().sum() + vector.sum()).backward()
    optimizer.step()
    restored = build_one()
    restored.load_state_dict(optimizer.state_dict())
    assert set(optimizer.state_dict()) == {"muon", "adamw", "tables"}
    scheduler = torch.optim.lr_scheduler.LambdaLR(restored, lambda step: 0.5)
    assert restored.tables.param_groups[0]["lr"] == 5e-3 and len(scheduler.get_last_lr()) == 3


def test_an_unknown_gradient_shape_is_refused():
    vector = torch.nn.Parameter(torch.randn(4))
    with pytest.raises(ValueError):
        FactoredAdam([vector], lr=1e-2)


def test_a_loaded_state_keeps_the_factors_in_float32_beside_a_bfloat16_table():
    table = torch.nn.Parameter(torch.randn(6, 4).to(torch.bfloat16))
    optimizer = FactoredAdam([table], lr=1e-2)
    table.grad = torch.randn(6, 4).to(torch.bfloat16)
    optimizer.step()
    restored = FactoredAdam([table], lr=1e-2)
    restored.load_state_dict(optimizer.state_dict())
    state = restored.state[table]
    assert state["exp_avg_sq_row"].dtype == torch.float32 and state["exp_avg_sq_col"].dtype == torch.float32
    assert state["exp_avg"].dtype == torch.bfloat16
    rows = torch.tensor([1, 3])
    table.grad = torch.sparse_coo_tensor(rows[None], torch.randn(2, 4).to(torch.bfloat16), (6, 4))
    restored.step()
