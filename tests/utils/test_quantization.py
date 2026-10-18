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
"""Invariants of the NVFP4 projections."""

import pytest
import torch
from torch import nn

from conftest import StandInNVFP4
from polyplex_research.utils import quantization
from polyplex_research.utils.quantization import (
    FrozenNVFP4Linear,
    HadamardNVFP4Linear,
    _hadamard_matrix,
    _to_nvfp4_stochastic,
    dequantize_nvfp4,
    quantize_linears_nvfp4,
    set_nvfp4_backward,
)


def test_hadamard_matrix_is_orthogonal():
    matrix = _hadamard_matrix(16, torch.float32, torch.device("cpu"))
    torch.testing.assert_close(matrix @ matrix.T, torch.eye(16))


def test_every_linear_under_the_module_is_replaced_and_keeps_its_weight():
    block = nn.Sequential(nn.Linear(32, 48, bias=False), nn.ReLU(), nn.Sequential(nn.Linear(48, 16)))
    weight = block[0].weight
    replaced = quantize_linears_nvfp4(block)
    assert replaced == ["0", "2.0"]
    assert isinstance(block[0], HadamardNVFP4Linear) and isinstance(block[2][0], HadamardNVFP4Linear)
    assert block[0].weight is weight


def test_a_linear_off_the_rotation_group_is_left_alone():
    block = nn.Sequential(nn.Linear(20, 16))
    assert quantize_linears_nvfp4(block) == []
    assert type(block[0]) is nn.Linear


def test_the_projection_stays_close_to_the_full_precision_one(cpu_nvfp4):
    torch.manual_seed(0)
    linear = nn.Linear(256, 128, bias=False)
    hidden = torch.randn(64, 256)
    result = HadamardNVFP4Linear.from_linear(linear)(hidden)
    cosine = torch.nn.functional.cosine_similarity(result.flatten(), linear(hidden).flatten(), dim=0)
    assert cosine > 0.98


@pytest.mark.parametrize("outlier", [0.0, 1e3, 1e5])
def test_a_token_is_quantized_from_its_own_activations_only(cpu_nvfp4, outlier):
    torch.manual_seed(0)
    linear = nn.Linear(256, 64)
    linear.weight.requires_grad_(False)
    quantized = HadamardNVFP4Linear.from_linear(linear)
    hidden = torch.randn(2, 8, 256)
    hidden[1, 7, 123] = outlier
    whole = quantized(hidden)[0, :4]
    prefix = quantized(hidden[:1, :4])[0]
    torch.testing.assert_close(whole, prefix, rtol=0, atol=0)


def test_a_zero_token_gives_the_bias(cpu_nvfp4):
    linear = nn.Linear(256, 128)
    out = HadamardNVFP4Linear.from_linear(linear)(torch.zeros(3, 256))
    torch.testing.assert_close(out, linear.bias.detach().expand_as(out), rtol=0, atol=0)


def test_a_frozen_weight_passes_a_gradient_to_its_input_only(cpu_nvfp4):
    torch.manual_seed(0)
    linear = nn.Linear(256, 128, bias=False)
    linear.weight.requires_grad_(False)
    quantized = HadamardNVFP4Linear.from_linear(linear)
    hidden = torch.randn(64, 256, requires_grad=True)
    quantized(hidden).square().sum().backward()
    assert quantized.weight.grad is None

    reference = hidden.detach().clone().requires_grad_(True)
    linear(reference).square().sum().backward()
    cosine = torch.nn.functional.cosine_similarity(hidden.grad.flatten(), reference.grad.flatten(), dim=0)
    assert cosine > 0.95


def test_a_trained_weight_gets_a_gradient_close_to_full_precision(cpu_nvfp4):
    torch.manual_seed(0)
    linear = nn.Linear(256, 128, bias=False)
    quantized = HadamardNVFP4Linear.from_linear(nn.Linear(256, 128, bias=False))
    quantized.weight.data.copy_(linear.weight.data)
    hidden = torch.randn(64, 256)
    quantized(hidden).square().sum().backward()
    linear(hidden).square().sum().backward()
    cosine = torch.nn.functional.cosine_similarity(quantized.weight.grad.flatten(), linear.weight.grad.flatten(), dim=0)
    assert cosine > 0.95


# ---------------------------------------------------------------------------
# CPU emulation of the NVFP4 projection, for measuring its error
# ---------------------------------------------------------------------------


def test_the_round_trip_keeps_values_on_the_fp4_grid():
    from polyplex_research.utils.quantization import nvfp4_round_trip

    # Each block of 16 holds 6, the block maximum, so its scale is 1 and every FP4 value is exact.
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] * 2).repeat(4, 1)
    grid[:, 0] = 6.0
    torch.testing.assert_close(nvfp4_round_trip(grid, amax=6.0), grid)


def test_the_emulated_projection_quantizes_each_token_on_its_own():
    from polyplex_research.utils.quantization import EmulatedNVFP4Linear

    torch.manual_seed(0)
    linear = EmulatedNVFP4Linear.from_linear(nn.Linear(32, 8))
    hidden = torch.randn(5, 32)
    hidden[3] *= 50
    together = linear(hidden)
    alone = torch.cat([linear(row[None]) for row in hidden])
    # Only the float32 summation order of the matmul differs between a batch and a single row.
    torch.testing.assert_close(together, alone, rtol=0, atol=1e-5)


def test_weight_only_emulation_leaves_the_activations_alone():
    from polyplex_research.utils.quantization import EmulatedNVFP4Linear, _rotate, nvfp4_round_trip

    torch.manual_seed(1)
    source = nn.Linear(32, 8, bias=False)
    linear = EmulatedNVFP4Linear.from_linear(source, quantize_activations=False)
    hidden = torch.randn(3, 32)
    transform = _hadamard_matrix(16, torch.float32, torch.device("cpu"))
    expected = _rotate(hidden, transform) @ nvfp4_round_trip(_rotate(source.weight.detach(), transform)).T
    torch.testing.assert_close(linear(hidden), expected)


def test_emulated_layers_can_be_skipped_and_put_back():
    from polyplex_research.utils.quantization import EmulatedNVFP4Linear, emulate_linears_nvfp4

    block = nn.ModuleDict({"up": nn.Linear(32, 16), "down": nn.Linear(16, 32)})
    originals = dict(block.items())
    restore = emulate_linears_nvfp4(block, skip=lambda name: name == "down")
    assert isinstance(block["up"], EmulatedNVFP4Linear) and block["down"] is originals["down"]
    restore()
    assert block["up"] is originals["up"]


# ---------------------------------------------------------------------------
# Frozen NVFP4 projections that hold no full precision weight (SPEC S7.1)
# ---------------------------------------------------------------------------


def test_a_frozen_projection_keeps_no_full_precision_weight(cpu_nvfp4):
    from polyplex_research.utils.quantization import FrozenNVFP4Linear

    linear = nn.Linear(256, 128)
    frozen = FrozenNVFP4Linear.from_linear(linear)
    assert set(dict(frozen.named_parameters())) == {"bias"}
    assert not hasattr(frozen, "weight")
    assert isinstance(frozen.quantized_weight, StandInNVFP4)


def test_a_frozen_projection_matches_the_quantized_one_and_passes_a_gradient_to_its_input(cpu_nvfp4):
    from polyplex_research.utils.quantization import FrozenNVFP4Linear

    torch.manual_seed(0)
    linear = nn.Linear(256, 128)
    linear.weight.requires_grad_(False)
    reference = HadamardNVFP4Linear.from_linear(linear)
    frozen = FrozenNVFP4Linear.from_linear(linear)
    hidden = torch.randn(4, 9, 256, requires_grad=True)
    out = frozen(hidden)
    torch.testing.assert_close(out, reference(hidden), rtol=0, atol=0)
    out.square().sum().backward()
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0


def test_freezing_the_backbone_can_keep_chosen_layers_in_full_precision(cpu_nvfp4):
    from polyplex_research.utils.quantization import FrozenNVFP4Linear

    block = nn.ModuleDict({"up": nn.Linear(32, 16), "down": nn.Linear(16, 32)})
    down = block["down"]
    replaced = quantize_linears_nvfp4(block, frozen=True, keep=lambda name: name == "down")
    assert replaced == ["up"]
    assert isinstance(block["up"], FrozenNVFP4Linear) and block["down"] is down


# ---------------------------------------------------------------------------
# Restoring NVFP4 operands for the backward (compile safe)
# ---------------------------------------------------------------------------


def test_the_e2m1_table_holds_every_fp4_code():
    from torchao.prototype.mx_formats.kernels import f4_unpacked_to_f32

    from polyplex_research.utils.quantization import E2M1_VALUES

    codes = torch.arange(16, dtype=torch.uint8)
    torch.testing.assert_close(E2M1_VALUES, f4_unpacked_to_f32(codes), rtol=0, atol=0)


@pytest.mark.parametrize("swizzled", [False, True])
def test_the_table_restore_is_torchaos_dequantize(swizzled):
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor, per_tensor_amax_to_scale

    from polyplex_research.utils.quantization import dequantize_nvfp4

    torch.manual_seed(0)
    tensor = torch.randn(256, 128) * 3
    quantized = NVFP4Tensor.to_nvfp4(
        tensor,
        block_size=16,
        per_tensor_scale=per_tensor_amax_to_scale(tensor.abs().max()),
        is_swizzled_scales=swizzled,
        use_triton_kernel=False,
    )
    for dtype in (torch.float32, torch.bfloat16):
        torch.testing.assert_close(dequantize_nvfp4(quantized, dtype), quantized.dequantize(dtype), rtol=0, atol=0)
    transposed = quantized.t()
    torch.testing.assert_close(dequantize_nvfp4(transposed, torch.float32), transposed.dequantize(torch.float32))


def test_the_compile_gradient_check_reports_no_difference_for_the_same_layer(cpu_nvfp4):
    from polyplex_research.utils.quantization import compiled_gradient_error

    error = compiled_gradient_error(rows=8, size=256, device="cpu", dtype=torch.float32, compiler=lambda module: module)
    assert error == {"trained_grad_input": 0.0, "trained_grad_weight": 0.0, "frozen_grad_input": 0.0}


@pytest.mark.parametrize("swizzled", [False, True])
def test_the_table_restore_reads_the_block_scales_in_float32(swizzled):
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor, per_tensor_amax_to_scale
    from torchao.prototype.mx_formats.utils import from_blocked

    from polyplex_research.utils.quantization import E2M1_VALUES, dequantize_nvfp4

    torch.manual_seed(0)
    tensor = (torch.randn(256, 128) * 3).to(torch.bfloat16)
    quantized = NVFP4Tensor.to_nvfp4(
        tensor,
        block_size=16,
        per_tensor_scale=per_tensor_amax_to_scale(tensor.float().abs().max()),
        is_swizzled_scales=swizzled,
        use_triton_kernel=False,
    )
    scales = from_blocked(quantized.scale, 256, 8) if swizzled else quantized.scale
    packed = quantized.qdata.contiguous().view(torch.uint8)
    codes = torch.stack([packed & 0b1111, packed >> 4], dim=-1).flatten(-2)
    exact = E2M1_VALUES[codes.long()].view(256, 8, 16) * (scales.float() * quantized.per_tensor_scale.float())[..., None]
    torch.testing.assert_close(dequantize_nvfp4(quantized, torch.float32), exact.view(256, 128), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Backward precision (SPEC S7.1a)
# ---------------------------------------------------------------------------


def test_an_unknown_backward_dtype_is_refused():
    with pytest.raises(ValueError):
        HadamardNVFP4Linear(64, 32, backward_dtype="fp16")


@pytest.mark.parametrize("backward_dtype", ["mxfp8", "nvfp4"])
@pytest.mark.parametrize("frozen", [False, True])
def test_a_low_precision_backward_gives_gradients_close_to_full_precision(cpu_nvfp4, backward_dtype, frozen):
    torch.manual_seed(0)
    linear = nn.Linear(256, 128, bias=False)
    linear.weight.requires_grad_(not frozen)
    kind = FrozenNVFP4Linear if frozen else HadamardNVFP4Linear
    quantized = kind.from_linear(linear, backward_dtype=backward_dtype)
    hidden = torch.randn(64, 256, requires_grad=True)
    quantized(hidden).square().sum().backward()
    reference = hidden.detach().clone().requires_grad_(True)
    linear(reference).square().sum().backward()

    cosine = torch.nn.functional.cosine_similarity(hidden.grad.flatten(), reference.grad.flatten(), dim=0)
    assert cosine > 0.95
    if not frozen:
        cosine = torch.nn.functional.cosine_similarity(
            quantized.weight.grad.flatten(), linear.weight.grad.flatten(), dim=0
        )
        assert cosine > 0.95


@pytest.mark.parametrize("frozen", [False, True])
def test_the_backward_dtype_picks_the_matmul_each_gradient_runs_in(cpu_nvfp4, monkeypatch, frozen):
    calls = []
    for name in ("_mxfp8_mm", "_nvfp4_mm"):
        original = getattr(quantization, name)
        monkeypatch.setattr(
            quantization, name, lambda *args, _name=name, _original=original, **kwargs: calls.append(_name) or _original(*args, **kwargs)
        )
    linear = nn.Linear(64, 32, bias=False)
    linear.weight.requires_grad_(not frozen)
    kind = FrozenNVFP4Linear if frozen else HadamardNVFP4Linear
    gradients = 1 if frozen else 2
    for backward_dtype, expected in [("bf16", []), ("mxfp8", ["_mxfp8_mm"]), ("nvfp4", ["_nvfp4_mm"])]:
        calls.clear()
        layer = kind.from_linear(linear, backward_dtype=backward_dtype)
        layer(torch.randn(32, 64, requires_grad=True)).sum().backward()
        assert calls == expected * gradients


def test_the_backward_can_be_switched_on_every_nvfp4_layer(cpu_nvfp4):
    model = nn.Sequential(nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 64)), nn.Sequential(nn.Linear(64, 64)))
    quantize_linears_nvfp4(model[0], backward_dtype="mxfp8")
    quantize_linears_nvfp4(model[1], frozen=True)
    layers = [model[0][0], model[0][1], model[1][0]]
    assert [layer.backward_dtype for layer in layers] == ["mxfp8", "mxfp8", "bf16"]
    assert set_nvfp4_backward(model, "nvfp4") == 3
    assert [layer.backward_dtype for layer in layers] == ["nvfp4"] * 3
    with pytest.raises(ValueError):
        set_nvfp4_backward(model, "fp16")


def test_stochastic_rounding_is_unbiased():
    torch.manual_seed(0)
    tensor = torch.randn(4, 64)
    draws = torch.stack([dequantize_nvfp4(_to_nvfp4_stochastic(tensor), torch.float32) for _ in range(2000)])
    assert not torch.equal(draws[0], draws[1])
    # One draw is off by up to a grid step; the mean of 2000 draws comes back to the tensor.
    single = (draws[0] - tensor).abs().max()
    assert (draws.mean(0) - tensor).abs().max() < single / 5


# ---------------------------------------------------------------------------
# Saving a frozen NVFP4 projection (SPEC S11.2)
# ---------------------------------------------------------------------------


@pytest.fixture
def cpu_real_nvfp4(monkeypatch):
    # Stochastic rounding builds a real NVFP4 tensor without the GPU kernel.
    monkeypatch.setattr(quantization, "_to_nvfp4", lambda tensor, amax=None: _to_nvfp4_stochastic(tensor))


def test_a_frozen_projection_saves_its_four_bit_weight_and_loads_it_back(cpu_real_nvfp4):
    torch.manual_seed(0)
    linear = nn.Linear(64, 32, bias=True).to(torch.bfloat16)
    frozen = FrozenNVFP4Linear.from_linear(linear, backward_dtype="mxfp8")
    state = frozen.state_dict()
    assert set(state) == {"bias", "codes", "block_scales", "global_scale"}
    assert state["codes"].dtype == torch.uint8 and state["codes"].shape == (32, 32)
    assert state["block_scales"].dtype == torch.float8_e4m3fn and state["block_scales"].shape == (32, 4)

    empty = FrozenNVFP4Linear.empty(64, 32, bias=True, backward_dtype="mxfp8")
    assert empty.quantized_weight is None
    empty.load_state_dict(state)
    torch.testing.assert_close(
        dequantize_nvfp4(empty.quantized_weight, torch.float32),
        dequantize_nvfp4(frozen.quantized_weight, torch.float32),
        rtol=0,
        atol=0,
    )
    assert torch.equal(empty.bias, frozen.bias) and empty.backward_dtype == "mxfp8"
    assert torch.equal(empty.forward_transform, frozen.forward_transform)
    assert empty.backward_weight is None and frozen.backward_weight is None


def test_a_cached_backward_weight_is_quantized_once_and_gives_the_same_gradient(cpu_nvfp4, monkeypatch):
    torch.manual_seed(0)
    linear = nn.Linear(64, 32, bias=False)
    linear.weight.requires_grad_(False)
    layer = FrozenNVFP4Linear.from_linear(linear, backward_dtype="mxfp8")
    hidden = torch.randn(32, 64)

    def gradient():
        inputs = hidden.clone().requires_grad_(True)
        layer(inputs).square().sum().backward()
        return inputs.grad

    expected = gradient()
    set_nvfp4_backward(layer, "mxfp8", cache_weights=True)
    quantized = []
    original = quantization._to_mxfp8
    monkeypatch.setattr(quantization, "_to_mxfp8", lambda tensor: quantized.append(tensor.shape) or original(tensor))
    first, second = gradient(), gradient()
    # Only the gradient side is quantized in each backward; the weight was quantized when cached.
    assert quantized == [(32, 32), (32, 32)]
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    torch.testing.assert_close(second, expected, rtol=0, atol=0)
    set_nvfp4_backward(layer, "mxfp8")
    assert layer.backward_weight is None
