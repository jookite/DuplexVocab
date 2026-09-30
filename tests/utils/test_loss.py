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
"""What the objective terms of the transfer phases are worth at the points that matter."""

import pytest
import torch

from polyplex_research.utils.loss import orthogonal_penalty, partial_isometry_penalty


VOCAB = 24


def test_the_penalty_is_zero_at_the_identity():
    r"""
    A zero residual starts on the identity, so the constraint never pulls the model off the frozen
    English weights before the speech loss has moved them.
    """
    zero = torch.zeros(VOCAB, VOCAB)
    assert orthogonal_penalty(zero, nonneg_beta=10.0).item() == pytest.approx(0.0, abs=1e-9)


def test_the_penalty_is_zero_at_every_permutation():
    r"""
    A permutation is the point the two terms agree on, and the only rotation whose transpose inverts a
    discrete code exactly.
    """
    torch.manual_seed(0)
    permutation = torch.zeros(VOCAB, VOCAB)
    permutation[torch.arange(VOCAB), torch.randperm(VOCAB)] = 1.0

    delta_w = permutation - torch.eye(VOCAB)
    assert orthogonal_penalty(delta_w, nonneg_beta=10.0).item() == pytest.approx(0.0, abs=1e-9)


def test_an_orthogonal_rotation_with_negative_entries_is_still_charged():
    r"""
    Orthogonality alone is weaker than a permutation, which is what the second term is for.
    """
    torch.manual_seed(0)
    rotation = torch.linalg.qr(torch.randn(VOCAB, VOCAB))[0]
    delta_w = rotation - torch.eye(VOCAB)

    assert orthogonal_penalty(delta_w, nonneg_beta=0.0).item() == pytest.approx(0.0, abs=1e-8)
    assert orthogonal_penalty(delta_w, nonneg_beta=10.0).item() > 1e-3


def test_the_orthogonal_penalty_reads_per_code():
    r"""
    Summed over entries the term grows with the square of the vocabulary, so `ortho_lambda` would carry
    the codec size rather than the objective. Per code it is the deviation one row is charged for.
    """
    torch.manual_seed(0)
    delta_w = torch.randn(VOCAB, VOCAB) * 0.05
    rotation = torch.eye(VOCAB) + delta_w
    gram = rotation.t() @ rotation
    summed = (gram - torch.eye(VOCAB)).pow(2).sum() + 10.0 * torch.relu(-rotation).pow(2).sum()

    assert orthogonal_penalty(delta_w, nonneg_beta=10.0).item() == pytest.approx(
        (summed / VOCAB).item(), rel=1e-5
    )


def test_the_isometry_penalty_reads_per_code_of_every_codebook():
    r"""
    Both residual kinds report through the same `ortho_lambda`, so both are charged in the same unit.
    """
    torch.manual_seed(0)
    matrices = torch.randn(3, VOCAB, 8) * 0.05
    gram = torch.matmul(matrices, matrices.transpose(1, 2))
    summed = (gram - torch.eye(VOCAB)).pow(2).sum()

    assert partial_isometry_penalty(matrices).item() == pytest.approx(
        (summed / (3 * VOCAB)).item(), rel=1e-5
    )
