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

"""Elements per MXFP8 block scale."""

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


__all__ = ["HadamardNVFP4Linear", "NVFP4_BLOCK_SIZE", "quantize_linears_nvfp4"]
