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
"""Objective terms of the transfer phases."""

from __future__ import annotations

import torch


def orthogonal_penalty(delta_w: torch.Tensor, nonneg_beta: float = 0.0) -> torch.Tensor:
    r"""
    Measures how far $I + \Delta W$ is from being a permutation matrix.

    The orthogonal term is evaluated on the full rotation rather than on $\Delta W$ alone.
    A zero initialized $\Delta W$ then starts on the identity, which already
    satisfies the constraint, so the regularizer never has to pull the model away from
    the frozen English weights before the speech loss moves them.

    Speaking emits one discrete code per frame and listening receives only that code, so the
    transpose acts as an exact inverse only when the rotation is a permutation. Orthogonality
    alone is far weaker than that. An orthogonal matrix whose entries are all non negative is a
    permutation matrix, hence the second term, which is also zero at the identity.

    Both terms are averaged over the codes rather than summed over the entries. The summed form grows
    with the square of the vocabulary, so the scale a run has to pick for it would depend on the codec
    rather than on the objective, and it would arrive at a per-token cross entropy three orders of
    magnitude too large. Per code, the penalty reads as the squared deviation one code's row carries,
    which is the unit the speech loss is already in.

    Args:
        delta_w (`torch.Tensor` of shape `(audio_vocab_size, audio_vocab_size)`):
            The residual weight.
        nonneg_beta (`float`, *optional*, defaults to 0.0):
            Scale of the penalty on the negative entries of the rotation.

    Returns:
        `torch.Tensor`: The scalar penalty.
    """
    rotation = torch.eye(delta_w.shape[0], dtype=torch.float32, device=delta_w.device)
    rotation = rotation + delta_w.float()
    gram = torch.matmul(rotation.t(), rotation)
    identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    penalty = (gram - identity).pow(2).sum()
    if nonneg_beta:
        penalty = penalty + nonneg_beta * torch.relu(-rotation).pow(2).sum()
    return penalty / rotation.shape[0]


def partial_isometry_penalty(matrices: torch.Tensor) -> torch.Tensor:
    r"""
    Measures how far each residual of the tied acoustic adapter is from a partial isometry.

    The temporal space residual is not square, one matrix per codebook maps the acoustic vocabulary into
    the backbone dimension, so it cannot be asked to be orthogonal. Asking `M M^T` to be the identity is
    the closest constraint that still makes the transpose act as an inverse on the codes the model emits.

    It is averaged over the codes of every codebook, for the reason [`orthogonal_penalty`] gives, so that
    one `ortho_lambda` means the same thing whichever residual a run drives.

    Args:
        matrices (`torch.Tensor` of shape `(num_codebooks, audio_vocab_size, hidden_size)`):
            The per-codebook residuals.

    Returns:
        `torch.Tensor`: `sum_k ||M_k M_k^T - I||_F^2`, per code.
    """
    matrices = matrices.float()
    gram = torch.matmul(matrices, matrices.transpose(1, 2))
    identity = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
    return (gram - identity).pow(2).sum() / (gram.shape[0] * gram.shape[1])


__all__ = ["orthogonal_penalty", "partial_isometry_penalty"]
