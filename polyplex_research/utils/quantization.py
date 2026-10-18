# Copyright 2026 LatentForge and the HuggingFace Inc. team. All rights reserved.
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
"""NVFP4 linear projections for a frozen backbone."""

from __future__ import annotations

import logging

import torch
from torch import nn
from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.constants import F4_E2M1_MAX, F8E4M3_MAX
from torchao.prototype.mx_formats.kernels import f32_to_f4_unpacked, pack_uint4
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor,
    _addmm_nvfp4_dispatch,
    per_tensor_amax_to_scale,
)
from torchao.prototype.mx_formats.utils import from_blocked, hp_data_dims_to_swizzled_scale_dims_nvfp4, to_blocked
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference


logger = logging.getLogger(__name__)

NVFP4_BLOCK_SIZE = 16
"""Elements per NVFP4 block scale. The packed layout admits no other value."""

MXFP8_BLOCK_SIZE = 32
"""Elements per MXFP8 block scale."""

BACKWARD_DTYPES = ("bf16", "mxfp8", "nvfp4")
"""Precisions the two backward matmuls of an NVFP4 projection can run in (SPEC S7.1a)."""


def _hadamard_matrix(size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Builds a normalized Hadamard matrix by Sylvester's construction.

    Args:
        size (`int`):
            Order of the matrix. Must be a power of two.
        dtype (`torch.dtype`):
            Precision the matrix is returned in.
        device (`torch.device`):
            Device the matrix is built on.

    Returns:
        `torch.Tensor`: An orthogonal `(size, size)` matrix, scaled so that `H @ H.T` is the
        identity rather than `size` times it.

    Raises:
        ValueError: If `size` is not a positive power of two.
    """
    if size <= 0 or size & (size - 1):
        raise ValueError(f"Hadamard order must be a positive power of two, got {size}")
    matrix = torch.ones(1, 1, dtype=torch.float32, device=device)
    while matrix.shape[0] < size:
        matrix = torch.cat(
            [torch.cat([matrix, matrix], dim=1), torch.cat([matrix, -matrix], dim=1)], dim=0
        )
    return (matrix * size**-0.5).to(dtype)


def _rotate(tensor: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Applies a block diagonal rotation along the last dimension.

    Args:
        tensor (`torch.Tensor`):
            The tensor to rotate, whose last dimension is a multiple of the transform's order.
        transform (`torch.Tensor`):
            The `(group, group)` rotation applied to each group of the last dimension.

    Returns:
        `torch.Tensor`: The rotated tensor, in the shape it came in.
    """
    group = transform.shape[0]
    shape = tensor.shape
    return (tensor.reshape(-1, group) @ transform).reshape(shape)


def _to_nvfp4(tensor: torch.Tensor, amax: float | None = None):
    """Quantizes one tensor to NVFP4 under a per-tensor scale.

    Args:
        tensor (`torch.Tensor`):
            A two dimensional tensor whose last dimension is a multiple of [`NVFP4_BLOCK_SIZE`].
        amax (`float`, *optional*):
            The absolute maximum the global scale is set from. Read off `tensor` when omitted.

    Returns:
        `NVFP4Tensor`: The quantized tensor, carrying its block scales and its global scale. The block
        scales come out already in the layout the FP4 matmul reads, so no call has to rearrange them.
    """
    if amax is None:
        amax = torch.max(torch.abs(tensor))
    else:
        amax = torch.tensor(amax, dtype=torch.float32, device=tensor.device)
    scale = per_tensor_amax_to_scale(amax)
    return NVFP4Tensor.to_nvfp4(
        tensor,
        block_size=NVFP4_BLOCK_SIZE,
        per_tensor_scale=scale,
        is_swizzled_scales=True,
        # The fused quantization kernel of MSLK, one launch instead of a chain of elementwise ops.
        use_triton_kernel=True,
    )


E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
"""The value of each 4 bit E2M1 code, the sign in the high bit."""


def dequantize_nvfp4(quantized, dtype: torch.dtype) -> torch.Tensor:
    r"""
    Restores an NVFP4 tensor to `dtype` by looking each 4 bit code up in [`E2M1_VALUES`] and multiplying it by its block
    scale and the global scale in float32, the values the FP4 matmul computes with. torchao decodes the codes bit by bit, and under `torch.compile` that decoding fused into the matmul
    after it gives wrong values (backward gradients 0.70 off, VoiceStudio ce52cac6); a table lookup compiles correctly.
    A tensor that is not an `NVFP4Tensor`, such as a test stand-in, is restored by its own `dequantize`.

    Args:
        quantized (`NVFP4Tensor`):
            A two dimensional NVFP4 tensor, possibly transposed.
        dtype (`torch.dtype`):
            The dtype restored to.

    Returns:
        `torch.Tensor`: The restored tensor.
    """
    if not isinstance(quantized, NVFP4Tensor):
        return quantized.dequantize(dtype)
    transposed = quantized.qdata.stride(-2) < quantized.qdata.stride(-1)
    data = quantized.qdata.transpose(-2, -1) if transposed else quantized.qdata
    packed = data.contiguous().view(torch.uint8)
    codes = torch.stack([packed & 0b1111, packed >> 4], dim=-1).flatten(-2)
    rows, columns = codes.shape
    values = E2M1_VALUES.to(codes.device)[codes.long()]
    # torchao 0.17's `get_hp_scales` rounds the block scales to the original dtype, 0.6% off for a bfloat16 weight;
    # they are read from the E4M3 scales and the global scale in float32 instead.
    scales = quantized.scale.transpose(-2, -1) if transposed else quantized.scale
    if quantized.is_swizzled_scales:
        scales = from_blocked(scales, rows, columns // quantized.block_size)
    scales = scales.to(torch.float32)
    if quantized.per_tensor_scale is not None:
        scales = scales * quantized.per_tensor_scale.to(torch.float32)
    scales = scales.reshape(rows, columns // quantized.block_size, 1)
    restored = (values.view(rows, columns // quantized.block_size, quantized.block_size) * scales).view(rows, columns)
    restored = restored.to(dtype)
    return restored.transpose(-2, -1) if transposed else restored


def _to_mxfp8(tensor: torch.Tensor) -> MXTensor:
    """Quantizes a `(m, k)` tensor to MXFP8 along its last axis, padded with zeros to whole blocks.

    Args:
        tensor (`torch.Tensor`):
            The tensor.

    Returns:
        `MXTensor`: The quantized tensor.
    """
    pad = -tensor.shape[-1] % MXFP8_BLOCK_SIZE
    if pad:
        # Zeros along the reduction axis add nothing to the product.
        tensor = torch.nn.functional.pad(tensor, (0, pad))
    return MXTensor.to_mx(
        tensor.contiguous(),
        torch.float8_e4m3fn,
        MXFP8_BLOCK_SIZE,
        # Rounding the shared exponent up keeps the block maximum representable; rounding it
        # down, the OCP default, saturates it and is reported to diverge in training.
        scaling_mode=ScaleCalculationMode.RCEIL,
        kernel_preference=KernelPreference.AUTO,
    )


def _mxfp8_mm(left: torch.Tensor, right: torch.Tensor, quantized_right: MXTensor | None = None) -> torch.Tensor:
    """Computes `left @ right.T` with both operands quantized to MXFP8 along the reduction axis.

    Args:
        left (`torch.Tensor`):
            A `(m, k)` operand.
        right (`torch.Tensor`):
            A `(n, k)` operand. Not read when `quantized_right` is given.
        quantized_right (`MXTensor`, *optional*):
            `right` already quantized by [`_to_mxfp8`].

    Returns:
        `torch.Tensor`: The `(m, n)` product, in `left`'s dtype.
    """
    if quantized_right is None:
        quantized_right = _to_mxfp8(right)
    return torch.mm(_to_mxfp8(left), quantized_right.t()).to(left.dtype)


def _to_nvfp4_stochastic(tensor: torch.Tensor) -> NVFP4Tensor:
    """Quantizes one tensor to NVFP4 with stochastic rounding of its elements.

    The global and block scales are computed as torchao's round to nearest quantizer computes them; each
    scaled element then rounds up to the next E2M1 value with a probability equal to its distance from the
    one below, so the quantized tensor is unbiased.

    Args:
        tensor (`torch.Tensor`):
            A two dimensional tensor whose last dimension is a multiple of [`NVFP4_BLOCK_SIZE`].

    Returns:
        `NVFP4Tensor`: The quantized tensor, with block scales in the layout the FP4 matmul reads.
    """
    rows, columns = tensor.shape
    per_tensor_scale = per_tensor_amax_to_scale(tensor.abs().max())
    data = tensor.float().reshape(rows, -1, NVFP4_BLOCK_SIZE)
    block_scale = data.abs().amax(dim=-1) / F4_E2M1_MAX
    block_scale_fp8 = torch.clamp(
        block_scale / per_tensor_scale, min=torch.finfo(torch.float8_e4m3fn).tiny, max=F8E4M3_MAX
    ).to(torch.float8_e4m3fn)
    scaled = data * ((1.0 / per_tensor_scale) / block_scale_fp8.float()).unsqueeze(-1)
    scaled = torch.clamp(scaled, -F4_E2M1_MAX, F4_E2M1_MAX).reshape(rows, columns)

    # The E2M1 magnitudes are spaced 0.5 apart below 2, 1 apart up to 4 and 2 apart up to 6.
    magnitude = scaled.abs()
    step = torch.where(magnitude >= 4.0, 2.0, torch.where(magnitude >= 2.0, 1.0, 0.5))
    lower = torch.floor(magnitude / step) * step
    upper = torch.clamp(lower + step, max=F4_E2M1_MAX)
    round_up = torch.rand_like(magnitude) * step < magnitude - lower
    snapped = torch.where(round_up, upper, lower).copysign(scaled)

    scale_rows, scale_columns = hp_data_dims_to_swizzled_scale_dims_nvfp4(rows, columns)
    swizzled_scale = to_blocked(block_scale_fp8.view(rows, columns // NVFP4_BLOCK_SIZE)).flatten()
    return NVFP4Tensor(
        pack_uint4(f32_to_f4_unpacked(snapped)),
        swizzled_scale.view(scale_rows, scale_columns),
        NVFP4_BLOCK_SIZE,
        tensor.dtype,
        per_tensor_scale,
        None,
        True,
    )


def _random_hadamard(size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Builds a Hadamard matrix with its rows multiplied by a fixed pseudo random sign vector.

    Args:
        size (`int`):
            Order of the matrix. Must be a power of two.
        dtype (`torch.dtype`):
            Precision the matrix is returned in.
        device (`torch.device`):
            Device the matrix is built on.

    Returns:
        `torch.Tensor`: An orthogonal `(size, size)` matrix. The signs come from a fixed seed, so every layer
        and every call uses the same one.
    """
    signs = torch.randint(0, 2, (size,), generator=torch.Generator().manual_seed(0)) * 2 - 1
    matrix = signs.to(torch.float32)[:, None] * _hadamard_matrix(size, torch.float32, torch.device("cpu"))
    return matrix.to(device=device, dtype=dtype)


def _nvfp4_mm(left: torch.Tensor, right: torch.Tensor, stochastic_left: bool) -> torch.Tensor:
    """Computes `left @ right.T` with both operands quantized to NVFP4 along the reduction axis.

    Args:
        left (`torch.Tensor`):
            A `(m, k)` operand, the gradient in either backward matmul.
        right (`torch.Tensor`):
            A `(n, k)` operand.
        stochastic_left (`bool`):
            Whether `left` is rounded stochastically. `right` is always rounded to nearest.

    Returns:
        `torch.Tensor`: The `(m, n)` product, in `left`'s dtype.
    """
    pad = -left.shape[-1] % (2 * NVFP4_BLOCK_SIZE)
    if pad:
        # Zeros along the reduction axis add nothing to the product.
        left = torch.nn.functional.pad(left, (0, pad))
        right = torch.nn.functional.pad(right, (0, pad))
    left = left.contiguous()
    quantized_left = _to_nvfp4_stochastic(left) if stochastic_left else _to_nvfp4(left)
    return _addmm_nvfp4_dispatch(quantized_left, _to_nvfp4(right.contiguous()).t(), None, None).to(left.dtype)


def _check_backward_dtype(backward_dtype: str) -> None:
    if backward_dtype not in BACKWARD_DTYPES:
        raise ValueError(f"backward_dtype must be one of {BACKWARD_DTYPES}, got {backward_dtype!r}")


class _HadamardNVFP4Matmul(torch.autograd.Function):
    r"""NVFP4 matmul on rotated operands, differentiated at the precision it was rotated in.

    A Hadamard rotation is orthogonal, so rotating both operands along the reduction axis leaves
    the product alone: `(x H) (W H)^T` is `x W^T`. What it does change is the distribution each
    operand is quantized from, since the rotation spreads an outlier across its whole group
    instead of forcing the group's scale to cover it alone.

    The forward runs on the FP4 tensor cores. The backward is the straight through estimator: it reads
    the same operands back dequantized and runs its two matmuls in bf16, or with `backward_dtype="mxfp8"`
    on the FP8 tensor cores with every operand quantized to MXFP8 along that matmul's reduction axis. With
    `backward_dtype="nvfp4"` both run on the FP4 tensor cores: the gradient is rounded stochastically and the
    other operand to nearest, and the weight gradient's operands are first mixed by a random Hadamard along
    the token axis. Only the gradients that are asked for are computed: a frozen weight needs no gradient of
    its own, so its activations are not kept for one either.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        transform: torch.Tensor,
        quantized_weight=None,
        backward_dtype: str = "bf16",
        wgrad_transform: torch.Tensor | None = None,
        backward_weight=None,
    ) -> torch.Tensor:
        ctx.backward_dtype = backward_dtype
        ctx.backward_weight = backward_weight
        ctx.wgrad_transform = wgrad_transform
        # A scale shared across rows would let other positions and other samples, the future ones of
        # a teacher forced batch included, change how a token is rounded. Each row is brought to unit
        # amax and quantized under a fixed global scale, and the row's amax multiplies its output back.
        rotated = _rotate(hidden, transform)
        row_scale = rotated.abs().amax(dim=-1, keepdim=True).float()
        row_scale = torch.where(row_scale > 0, row_scale, torch.ones_like(row_scale))
        quantized_hidden = _to_nvfp4((rotated.float() / row_scale).to(rotated.dtype), amax=1.0)
        if quantized_weight is None:
            quantized_weight = _to_nvfp4(_rotate(weight, transform))
        ctx.save_for_backward(
            quantized_hidden if ctx.needs_input_grad[1] else None, row_scale, quantized_weight, transform
        )
        out = _addmm_nvfp4_dispatch(quantized_hidden, quantized_weight.t(), None, None)
        # The bias is added after the rows are rescaled, so it is not multiplied by their scale.
        out = (out.float() * row_scale).to(hidden.dtype)
        return out if bias is None else out + bias.to(out.dtype)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: torch.Tensor):
        quantized_hidden, row_scale, quantized_weight, transform = ctx.saved_tensors

        # The rotation sits between the parameter and the matmul, so each gradient comes back in
        # rotated coordinates and is carried out of them by the transpose. The row scales are
        # constants: under the straight through estimator the output is `x W^T` whatever they are.
        grad_hidden = grad_weight = grad_bias = None
        backward_dtype = ctx.backward_dtype
        if ctx.needs_input_grad[0]:
            cached = backward_dtype == "mxfp8" and ctx.backward_weight is not None
            if not cached:
                rotated_weight = dequantize_nvfp4(quantized_weight, quantized_weight.orig_dtype).to(grad_output.dtype)
            if cached:
                grad_hidden = _mxfp8_mm(grad_output, None, ctx.backward_weight)
            elif backward_dtype == "mxfp8":
                grad_hidden = _mxfp8_mm(grad_output, rotated_weight.t())
            elif backward_dtype == "nvfp4":
                grad_hidden = _nvfp4_mm(grad_output, rotated_weight.t(), stochastic_left=True)
            else:
                grad_hidden = grad_output @ rotated_weight
            grad_hidden = _rotate(grad_hidden, transform.t())
        if ctx.needs_input_grad[1]:
            rotated_hidden = (dequantize_nvfp4(quantized_hidden, torch.float32) * row_scale).to(grad_output.dtype)
            if backward_dtype == "mxfp8":
                grad_weight = _mxfp8_mm(grad_output.t(), rotated_hidden.t())
            elif backward_dtype == "nvfp4":
                # A random Hadamard along the token axis, the reduction axis of this matmul, spreads the
                # gradient's outliers before the four bit cast; applied to both operands it cancels in the product.
                pad = -grad_output.shape[0] % NVFP4_BLOCK_SIZE
                grad_t = torch.nn.functional.pad(grad_output.t(), (0, pad))
                hidden_t = torch.nn.functional.pad(rotated_hidden.t(), (0, pad))
                mix = ctx.wgrad_transform.to(grad_t.dtype)
                grad_weight = _nvfp4_mm(_rotate(grad_t, mix), _rotate(hidden_t, mix), stochastic_left=True)
            else:
                grad_weight = grad_output.t() @ rotated_hidden
            grad_weight = _rotate(grad_weight, transform.t())
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_hidden, grad_weight, grad_bias, None, None, None, None, None


class HadamardNVFP4Linear(nn.Linear):
    r"""
    Constructs a linear whose forward runs in NVFP4 on Hadamard rotated operands.

    The weight stays a full precision `Parameter`. A trained weight is quantized on every forward; a
    frozen one is quantized once and reused, and the layer only passes a gradient back to its input.

    Args:
        in_features (`int`):
            Size of the reduction axis. Must be a multiple of `hadamard_group_size`.
        out_features (`int`):
            Size of the output axis.
        bias (`bool`, *optional*, defaults to `True`):
            Whether the layer carries a bias. The bias is added outside the quantized matmul.
        hadamard_group_size (`int`, *optional*, defaults to 16):
            Order of the rotation applied to each group of the reduction axis. Matching the NVFP4
            block size keeps a rotation group and a scale group the same set of elements.
        device (`torch.device`, *optional*):
            Device the layer is built on.
        dtype (`torch.dtype`, *optional*):
            Precision of the weight.
        backward_dtype (`str`, *optional*, defaults to `"bf16"`):
            Precision of the two backward matmuls, one of [`BACKWARD_DTYPES`] (SPEC S7.1a).

    Raises:
        ValueError: If `in_features` is not a multiple of `hadamard_group_size`, or `backward_dtype` is not one
            of [`BACKWARD_DTYPES`].
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        hadamard_group_size: int = NVFP4_BLOCK_SIZE,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        backward_dtype: str = "bf16",
    ):
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)
        _check_backward_dtype(backward_dtype)
        if in_features % hadamard_group_size:
            raise ValueError(
                f"in_features {in_features} is not a multiple of the rotation group "
                f"{hadamard_group_size}"
            )
        self.hadamard_group_size = hadamard_group_size
        self.backward_dtype = backward_dtype
        # Not persistent: Sylvester's construction is deterministic, so a rebuilt basis is the one
        # the layer always used.
        self.register_buffer(
            "forward_transform",
            _hadamard_matrix(hadamard_group_size, self.weight.dtype, self.weight.device),
            persistent=False,
        )
        # The random Hadamard the NVFP4 backward mixes the weight gradient's token axis with.
        self.register_buffer(
            "wgrad_transform",
            _random_hadamard(NVFP4_BLOCK_SIZE, self.weight.dtype, self.weight.device),
            persistent=False,
        )

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, hadamard_group_size: int = NVFP4_BLOCK_SIZE, backward_dtype: str = "bf16"
    ) -> "HadamardNVFP4Linear":
        r"""
        Wraps an existing linear, keeping its weights.

        Args:
            linear (`nn.Linear`):
                The layer to replace.
            hadamard_group_size (`int`, *optional*, defaults to 16):
                Order of the rotation.
            backward_dtype (`str`, *optional*, defaults to `"bf16"`):
                Precision of the two backward matmuls.

        Returns:
            `HadamardNVFP4Linear`: A layer holding `linear`'s weight and bias.
        """
        replacement = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            hadamard_group_size=hadamard_group_size,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            backward_dtype=backward_dtype,
        )
        replacement.weight = linear.weight
        replacement.bias = linear.bias
        return replacement

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        r"""
        Runs the rotated NVFP4 matmul.

        Args:
            hidden (`torch.Tensor`):
                Activations, of any rank whose last dimension is `in_features`.

        Returns:
            `torch.Tensor`: The result, in `hidden`'s shape and dtype.
        """
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        transform = self.forward_transform.to(flat.dtype)
        weight = self.weight.to(flat.dtype)
        quantized_weight = None
        if not self.weight.requires_grad:
            # A frozen weight is the same tensor on every call, so it is quantized once.
            key = (weight.dtype, weight.device)
            if getattr(self, "_quantized_key", None) != key:
                self._quantized_weight = _to_nvfp4(_rotate(weight, transform))
                self._quantized_key = key
            quantized_weight = self._quantized_weight
        out = _HadamardNVFP4Matmul.apply(
            flat, weight, self.bias, transform, quantized_weight, self.backward_dtype, self.wgrad_transform
        )
        return out.reshape(*shape[:-1], out.shape[-1])


class FrozenNVFP4Linear(nn.Module):
    r"""
    A frozen projection held in NVFP4 alone (SPEC S7.1). Its weight is rotated and quantized once when the
    projection is built and the full precision weight is dropped, so the module carries the quantized weight, the
    rotation and the bias only. It computes what [`HadamardNVFP4Linear`] computes for a frozen weight, and passes a
    gradient to its input.

    Args:
        quantized_weight (`NVFP4Tensor`):
            The rotated weight, quantized.
        bias (`nn.Parameter`, *optional*):
            The bias, kept as it is.
        hadamard_group_size (`int`, *optional*, defaults to 16):
            Order of the rotation the weight was quantized under.
        backward_dtype (`str`, *optional*, defaults to `"bf16"`):
            Precision of the input gradient's matmul, one of [`BACKWARD_DTYPES`] (SPEC S7.1a).

    Raises:
        ValueError: If `backward_dtype` is not one of [`BACKWARD_DTYPES`].
    """

    def __init__(
        self, quantized_weight, bias=None, hadamard_group_size: int = NVFP4_BLOCK_SIZE, backward_dtype: str = "bf16"
    ):
        super().__init__()
        _check_backward_dtype(backward_dtype)
        self.backward_dtype = backward_dtype
        self.out_features, self.in_features = quantized_weight.shape
        # Saved as its codes and scales (`_save_to_state_dict`), not as the tensor subclass itself.
        self.register_buffer("quantized_weight", quantized_weight, persistent=False)
        self.bias = bias
        self.register_buffer(
            "forward_transform",
            _hadamard_matrix(hadamard_group_size, quantized_weight.dtype, quantized_weight.device),
            persistent=False,
        )
        self.backward_weight = None

    def cache_backward_weight(self, cache: bool = True) -> None:
        r"""
        Keeps the weight the MXFP8 input gradient reads, quantized once, so a backward does not quantize it again
        (SPEC S7.1a). It costs one byte per weight on the device.

        Args:
            cache (`bool`, *optional*, defaults to `True`):
                Whether to keep it. `False` drops it.
        """
        self.backward_weight = None
        if cache and self.backward_dtype == "mxfp8":
            weight = dequantize_nvfp4(self.quantized_weight, self.quantized_weight.orig_dtype)
            self.backward_weight = _to_mxfp8(weight.t())

    @classmethod
    def empty(
        cls,
        in_features: int,
        out_features: int,
        bias: bool = True,
        hadamard_group_size: int = NVFP4_BLOCK_SIZE,
        backward_dtype: str = "bf16",
        device=None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "FrozenNVFP4Linear":
        r"""
        A frozen projection with no weight yet, for a saved one to be loaded into ([`~torch.nn.Module.load_state_dict`]).

        Args:
            in_features (`int`):
                Size of the reduction axis.
            out_features (`int`):
                Size of the output axis.
            bias (`bool`, *optional*, defaults to `True`):
                Whether the projection carries a bias.
            hadamard_group_size (`int`, *optional*, defaults to 16):
                Order of the rotation the weight was quantized under.
            backward_dtype (`str`, *optional*, defaults to `"bf16"`):
                Precision of the input gradient's matmul.
            device (`torch.device`, *optional*):
                Device of the bias and the rotation.
            dtype (`torch.dtype`, *optional*, defaults to `torch.bfloat16`):
                Precision the weight was quantized from and the bias is held in.

        Returns:
            `FrozenNVFP4Linear`: The projection, whose `quantized_weight` is `None` until loaded.
        """
        module = cls.__new__(cls)
        nn.Module.__init__(module)
        _check_backward_dtype(backward_dtype)
        module.backward_dtype = backward_dtype
        module.in_features, module.out_features = in_features, out_features
        module.weight_dtype = dtype
        module.register_buffer("quantized_weight", None, persistent=False)
        module.bias = (
            nn.Parameter(torch.empty(out_features, device=device, dtype=dtype), requires_grad=False) if bias else None
        )
        module.register_buffer(
            "forward_transform", _hadamard_matrix(hadamard_group_size, dtype, device or torch.device("cpu")), persistent=False
        )
        module.backward_weight = None
        return module

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if self.quantized_weight is None:
            return
        weight = self.quantized_weight
        rows, columns = weight.shape
        block_scales = weight.scale
        if weight.is_swizzled_scales:
            block_scales = from_blocked(block_scales, rows, columns // NVFP4_BLOCK_SIZE)
        destination[prefix + "codes"] = weight.qdata.contiguous().view(torch.uint8)
        destination[prefix + "block_scales"] = block_scales.reshape(rows, columns // NVFP4_BLOCK_SIZE).contiguous()
        destination[prefix + "global_scale"] = weight.per_tensor_scale.reshape(()).float()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        keys = [prefix + name for name in ("codes", "block_scales", "global_scale")]
        if all(key in state_dict for key in keys):
            codes, block_scales, global_scale = (state_dict.pop(key) for key in keys)
            rows, columns = self.out_features, self.in_features
            scale_rows, scale_columns = hp_data_dims_to_swizzled_scale_dims_nvfp4(rows, columns)
            swizzled = to_blocked(block_scales.view(rows, columns // NVFP4_BLOCK_SIZE)).flatten()
            dtype = getattr(self, "weight_dtype", None) or self.forward_transform.dtype
            self.quantized_weight = NVFP4Tensor(
                codes.view(torch.uint8),
                swizzled.view(scale_rows, scale_columns),
                NVFP4_BLOCK_SIZE,
                dtype,
                global_scale.float(),
                None,
                True,
            )
            if self.forward_transform.device != codes.device:
                self.forward_transform = self.forward_transform.to(codes.device)
        elif strict:
            missing_keys.extend(key for key in keys if key not in state_dict)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, hadamard_group_size: int = NVFP4_BLOCK_SIZE, backward_dtype: str = "bf16"
    ) -> "FrozenNVFP4Linear":
        r"""
        Args:
            linear (`nn.Linear`):
                The projection to freeze. Its weight is quantized and not kept.
            hadamard_group_size (`int`, *optional*, defaults to 16):
                Order of the rotation.
            backward_dtype (`str`, *optional*, defaults to `"bf16"`):
                Precision of the input gradient's matmul.

        Returns:
            `FrozenNVFP4Linear`: The frozen projection.
        """
        weight = linear.weight.detach()
        transform = _hadamard_matrix(hadamard_group_size, weight.dtype, weight.device)
        bias = linear.bias
        if bias is not None:
            bias.requires_grad_(False)
        return cls(_to_nvfp4(_rotate(weight, transform)), bias, hadamard_group_size, backward_dtype)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            hidden (`torch.Tensor`):
                Activations, of any rank whose last dimension is `in_features`.

        Returns:
            `torch.Tensor`: The result, in `hidden`'s shape and dtype.
        """
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        transform = self.forward_transform.to(flat.dtype)
        out = _HadamardNVFP4Matmul.apply(
            flat, None, self.bias, transform, self.quantized_weight, self.backward_dtype, None, self.backward_weight
        )
        return out.reshape(*shape[:-1], out.shape[-1])


def quantize_linears_nvfp4(
    module: nn.Module,
    hadamard_group_size: int = NVFP4_BLOCK_SIZE,
    frozen: bool = False,
    keep=None,
    backward_dtype: str = "bf16",
) -> list[str]:
    r"""
    Replaces every `nn.Linear` under `module` with its rotated NVFP4 form, in place.

    Args:
        module (`nn.Module`):
            The subtree to convert.
        hadamard_group_size (`int`, *optional*, defaults to 16):
            Order of the rotation applied to each group of the reduction axis.
        frozen (`bool`, *optional*, defaults to `False`):
            Whether the layers become [`FrozenNVFP4Linear`], dropping their full precision weights, rather than
            [`HadamardNVFP4Linear`], which keeps them trainable.
        keep (`Callable[[str], bool]`, *optional*):
            Returns `True` for the qualified names of layers left in full precision.
        backward_dtype (`str`, *optional*, defaults to `"bf16"`):
            Precision of the backward matmuls of the replaced layers, one of [`BACKWARD_DTYPES`].

    Returns:
        `list[str]`: Qualified names of the replaced layers, in traversal order. A layer whose
        `in_features` is not a multiple of the rotation group is left as it is.
    """
    replaced: list[str] = []

    def walk(parent: nn.Module, prefix: str) -> None:
        for name, child in parent.named_children():
            qualified = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and not isinstance(child, (HadamardNVFP4Linear, EmulatedNVFP4Linear)):
                if keep is not None and keep(qualified):
                    continue
                if child.in_features % hadamard_group_size:
                    logger.warning(
                        "%s has in_features %d, not a multiple of %d, so it stays unquantized",
                        qualified,
                        child.in_features,
                        hadamard_group_size,
                    )
                    continue
                kind = FrozenNVFP4Linear if frozen else HadamardNVFP4Linear
                setattr(parent, name, kind.from_linear(child, hadamard_group_size, backward_dtype))
                replaced.append(qualified)
            else:
                walk(child, qualified)

    _check_backward_dtype(backward_dtype)
    walk(module, "")
    logger.info("%d linear projections now run in NVFP4", len(replaced))
    return replaced


def set_nvfp4_backward(model: nn.Module, backward_dtype: str, cache_weights: bool = False) -> int:
    r"""
    Switches the backward precision of every NVFP4 projection in a model (SPEC S7.1a). The forward is untouched.

    Args:
        model (`nn.Module`):
            The model whose projections are switched.
        backward_dtype (`str`):
            One of [`BACKWARD_DTYPES`].
        cache_weights (`bool`, *optional*, defaults to `False`):
            Whether each frozen projection keeps the weight its MXFP8 input gradient reads
            ([`FrozenNVFP4Linear.cache_backward_weight`]), about one byte per weight on the device.

    Returns:
        `int`: How many projections were switched.

    Raises:
        ValueError: If `backward_dtype` is not one of [`BACKWARD_DTYPES`].
    """
    _check_backward_dtype(backward_dtype)
    layers = [module for module in model.modules() if isinstance(module, (HadamardNVFP4Linear, FrozenNVFP4Linear))]
    for layer in layers:
        layer.backward_dtype = backward_dtype
        if isinstance(layer, FrozenNVFP4Linear):
            layer.cache_backward_weight(cache_weights)
    return len(layers)


def compiled_gradient_error(
    rows: int = 4096, size: int = 4096, device: str = "cuda", dtype: torch.dtype = torch.bfloat16, compiler=None
) -> dict:
    r"""
    Measures how far the backward of the NVFP4 projections under `torch.compile` is from the eager one (AGENTS H31):
    the relative difference of the input and weight gradients of a trained [`HadamardNVFP4Linear`], and of the input
    gradient of a [`FrozenNVFP4Linear`], on the same random input and output gradient.

    Args:
        rows (`int`, *optional*, defaults to 4096):
            Tokens of the input.
        size (`int`, *optional*, defaults to 4096):
            Input and output width of the projection.
        device (`str`, *optional*, defaults to `"cuda"`):
            Where it runs.
        dtype (`torch.dtype`, *optional*, defaults to `torch.bfloat16`):
            Dtype of the projection and the input.
        compiler (`Callable`, *optional*):
            Turns a module into its compiled version, `torch.compile` when omitted.

    Returns:
        `dict`: `trained_grad_input`, `trained_grad_weight` and `frozen_grad_input`, each `|compiled - eager| /
        |eager|` in the Frobenius norm.
    """
    compiler = compiler or torch.compile
    torch.manual_seed(0)
    source = nn.Linear(size, size, bias=False, device=device, dtype=dtype)
    hidden = torch.randn(rows, size, device=device, dtype=dtype)
    grad = torch.randn(rows, size, device=device, dtype=dtype)

    def gradients(layer, trained: bool):
        given = hidden.clone().requires_grad_(True)
        layer(given).backward(grad)
        weight = layer.weight.grad.float() if trained else None
        if trained:
            layer.weight.grad = None
        return given.grad.float(), weight

    def relative(compiled, eager):
        return ((compiled - eager).norm() / eager.norm().clamp_min(1e-30)).item()

    trained = HadamardNVFP4Linear.from_linear(source)
    eager_input, eager_weight = gradients(trained, True)
    compiled_input, compiled_weight = gradients(compiler(trained), True)
    frozen_source = nn.Linear(size, size, bias=False, device=device, dtype=dtype)
    frozen_source.weight.data.copy_(source.weight.data)
    frozen_source.weight.requires_grad_(False)
    frozen = FrozenNVFP4Linear.from_linear(frozen_source)
    frozen_eager, _ = gradients(frozen, False)
    frozen_compiled, _ = gradients(compiler(frozen), False)
    return {
        "trained_grad_input": relative(compiled_input, eager_input),
        "trained_grad_weight": relative(compiled_weight, eager_weight),
        "frozen_grad_input": relative(frozen_compiled, frozen_eager),
    }


__all__ = ["HadamardNVFP4Linear", "NVFP4_BLOCK_SIZE", "quantize_linears_nvfp4"]


def nvfp4_round_trip(tensor: torch.Tensor, amax: float | None = None) -> torch.Tensor:
    r"""
    The values a tensor takes after NVFP4 quantization, computed without the FP4 kernels so it runs on any device.

    Args:
        tensor (`torch.Tensor`):
            A two dimensional tensor whose last dimension is a multiple of [`NVFP4_BLOCK_SIZE`].
        amax (`float`, *optional*):
            The absolute maximum the global scale is set from. Read off `tensor` when omitted.

    Returns:
        `torch.Tensor`: The dequantized tensor, in float32.
    """
    tensor = tensor.float()
    amax = tensor.abs().max() if amax is None else torch.tensor(amax, dtype=torch.float32, device=tensor.device)
    quantized = NVFP4Tensor.to_nvfp4(
        tensor,
        block_size=NVFP4_BLOCK_SIZE,
        per_tensor_scale=per_tensor_amax_to_scale(amax),
        is_swizzled_scales=False,
        use_triton_kernel=False,
    )
    return quantized.dequantize(torch.float32)


class EmulatedNVFP4Linear(nn.Linear):
    r"""
    Computes what [`HadamardNVFP4Linear`] computes, from dequantized operands in float32, so its error can be
    measured on any device.

    Args:
        in_features (`int`):
            Size of the reduction axis. Must be a multiple of `hadamard_group_size`.
        out_features (`int`):
            Size of the output.
        bias (`bool`, *optional*, defaults to `True`):
            Whether the projection carries a bias.
        hadamard_group_size (`int`, *optional*, defaults to 16):
            Order of the rotation applied to each group of the reduction axis.
        quantize_activations (`bool`, *optional*, defaults to `True`):
            Whether the activations are quantized too, each token under its own scale. Without it only the weight is.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        hadamard_group_size: int = NVFP4_BLOCK_SIZE,
        quantize_activations: bool = True,
        **kwargs,
    ):
        super().__init__(in_features, out_features, bias=bias, **kwargs)
        self.quantize_activations = quantize_activations
        self.register_buffer(
            "forward_transform", _hadamard_matrix(hadamard_group_size, torch.float32, self.weight.device), persistent=False
        )
        self._rounded_weight = None

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, hadamard_group_size: int = NVFP4_BLOCK_SIZE, quantize_activations: bool = True
    ) -> "EmulatedNVFP4Linear":
        r"""
        Args:
            linear (`nn.Linear`):
                The projection to emulate. Its weight and bias are shared, not copied.
            hadamard_group_size (`int`, *optional*, defaults to 16):
                Order of the rotation.
            quantize_activations (`bool`, *optional*, defaults to `True`):
                Whether the activations are quantized too.

        Returns:
            `EmulatedNVFP4Linear`: The emulated projection.
        """
        replacement = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            hadamard_group_size=hadamard_group_size,
            quantize_activations=quantize_activations,
            device="meta",
        )
        replacement.weight = linear.weight
        replacement.bias = linear.bias
        replacement.forward_transform = _hadamard_matrix(hadamard_group_size, torch.float32, linear.weight.device)
        return replacement

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            hidden (`torch.Tensor`):
                Activations, of any rank whose last dimension is `in_features`.

        Returns:
            `torch.Tensor`: The result, in `hidden`'s shape and dtype.
        """
        shape = hidden.shape
        rotated = _rotate(hidden.reshape(-1, shape[-1]).float(), self.forward_transform)
        if self.quantize_activations:
            row_scale = rotated.abs().amax(dim=-1, keepdim=True)
            row_scale = torch.where(row_scale > 0, row_scale, torch.ones_like(row_scale))
            rotated = nvfp4_round_trip(rotated / row_scale, amax=1.0) * row_scale
        if self._rounded_weight is None:
            # The weight is frozen while its error is measured, so it is rounded once.
            self._rounded_weight = nvfp4_round_trip(_rotate(self.weight.detach().float(), self.forward_transform))
        out = rotated @ self._rounded_weight.T
        if self.bias is not None:
            out = out + self.bias.float()
        return out.to(hidden.dtype).reshape(*shape[:-1], out.shape[-1])


def emulate_linears_nvfp4(
    module: nn.Module,
    hadamard_group_size: int = NVFP4_BLOCK_SIZE,
    quantize_activations: bool = True,
    skip=None,
):
    r"""
    Replaces every `nn.Linear` under `module` with [`EmulatedNVFP4Linear`], in place.

    Args:
        module (`nn.Module`):
            The subtree to convert.
        hadamard_group_size (`int`, *optional*, defaults to 16):
            Order of the rotation.
        quantize_activations (`bool`, *optional*, defaults to `True`):
            Whether the activations are quantized too.
        skip (`Callable[[str], bool]`, *optional*):
            Returns `True` for the qualified names of layers to leave in full precision.

    Returns:
        `Callable[[], None]`: Puts the original layers back.
    """
    swapped = []
    for qualified, child in list(module.named_modules()):
        if not isinstance(child, nn.Linear) or isinstance(child, EmulatedNVFP4Linear):
            continue
        if child.in_features % hadamard_group_size or (skip is not None and skip(qualified)):
            continue
        parent_name, _, name = qualified.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, name, EmulatedNVFP4Linear.from_linear(child, hadamard_group_size, quantize_activations))
        swapped.append((parent, name, child))

    def restore():
        for parent, name, child in swapped:
            setattr(parent, name, child)

    return restore
