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
"""Optimizers of the training phases."""

from __future__ import annotations

import torch


class FactoredAdam(torch.optim.Optimizer):
    r"""
    Constructs Adam for large two dimensional tables whose second moment is stored as row and column means (SPEC S7.2a).

    The first moment is kept whole, in the parameter's dtype. The second moment keeps only the mean over columns of
    each row, $r$, and the mean over rows of each column, $c$, in float32, and is read back as $\hat v = r c^\top / \bar r$,
    the factorization of Adafactor (Shazeer and Stern, 2018). The step is Adam's, $m / (\sqrt{\hat v} + \epsilon)$ with
    bias correction, so a squared gradient of rank one gives exactly Adam's step.

    A sparse gradient, as an embedding with `sparse=True` gives, steps only the rows it holds and leaves the others
    and their moments where they are, as [`torch.optim.SparseAdam`] does. Its column means are updated from those
    rows alone, and $\bar r$ is the mean over those rows.

    Args:
        params (`list` of `torch.nn.Parameter`):
            Two dimensional tables.
        lr (`float`):
            Learning rate.
        betas (`tuple[float, float]`, *optional*, defaults to `(0.9, 0.999)`):
            Decay of the first moment and of the second moment's factors.
        eps (`float`, *optional*, defaults to 1e-8):
            Added to the square root of the second moment.
        chunk (`int`, *optional*, defaults to 16384):
            Rows stepped at a time, so no float32 copy of a whole table is held.

    Raises:
        ValueError: If a parameter is not two dimensional.
    """

    def __init__(self, params, lr: float, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, chunk: int = 16384):
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps})
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.ndim != 2:
                    raise ValueError(f"FactoredAdam steps two dimensional tables, got shape {tuple(parameter.shape)}")
        self.chunk = chunk

    @staticmethod
    def from_adam_state(state: dict) -> dict:
        r"""
        Converts the state Adam or AdamW kept for one table into this optimizer's, keeping the first moment and the
        step and reducing the second moment to its row and column means.

        Args:
            state (`dict`):
                `step`, `exp_avg` and `exp_avg_sq` of one table.

        Returns:
            `dict`: `step`, `exp_avg`, `exp_avg_sq_row` and `exp_avg_sq_col`.
        """
        second = state["exp_avg_sq"]
        rows, columns = second.shape
        row = torch.empty(rows, dtype=torch.float32, device=second.device)
        column = torch.zeros(columns, dtype=torch.float32, device=second.device)
        # In chunks of rows, so no float32 copy of a whole table is held.
        for start in range(0, rows, 16384):
            block = second[start : start + 16384].float()
            row[start : start + 16384] = block.mean(dim=1)
            column += block.sum(dim=0)
        return {
            "step": torch.as_tensor(state["step"], dtype=torch.float32).clone(),
            "exp_avg": state["exp_avg"],
            "exp_avg_sq_row": row,
            "exp_avg_sq_col": column / rows,
        }

    def load_state_dict(self, state_dict):
        r"""
        Args:
            state_dict (`dict`):
                What [`~torch.optim.Optimizer.state_dict`] returned.
        """
        super().load_state_dict(state_dict)
        # Loading casts every floating state to its parameter's dtype; the factors stay float32.
        for state in self.state.values():
            for key in ("exp_avg_sq_row", "exp_avg_sq_col"):
                if key in state:
                    state[key] = state[key].float()

    def _init_state(self, parameter: torch.Tensor) -> dict:
        state = self.state[parameter]
        if not state:
            rows, columns = parameter.shape
            state["step"] = torch.zeros((), dtype=torch.float32)
            state["exp_avg"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
            state["exp_avg_sq_row"] = torch.zeros(rows, dtype=torch.float32, device=parameter.device)
            state["exp_avg_sq_col"] = torch.zeros(columns, dtype=torch.float32, device=parameter.device)
        return state

    @torch.no_grad()
    def step(self, closure=None):
        r"""
        Args:
            closure (`Callable`, *optional*):
                Re-evaluates the model and returns the loss.

        Returns:
            The loss the closure returned, or `None`.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self._init_state(parameter)
                state["step"] += 1
                step = float(state["step"])
                scale = group["lr"] / (1 - beta1**step)
                correction = (1 - beta2**step) ** 0.5
                if parameter.grad.is_sparse:
                    self._sparse_step(parameter, state, group, beta1, beta2, scale, correction)
                else:
                    self._dense_step(parameter, state, group, beta1, beta2, scale, correction)
        return loss

    def _dense_step(self, parameter, state, group, beta1, beta2, scale, correction):
        gradient = parameter.grad
        rows, columns = parameter.shape
        row_means = torch.empty(rows, dtype=torch.float32, device=parameter.device)
        column_sums = torch.zeros(columns, dtype=torch.float32, device=parameter.device)
        for start in range(0, rows, self.chunk):
            squared = gradient[start : start + self.chunk].float().square()
            row_means[start : start + self.chunk] = squared.mean(dim=1)
            column_sums += squared.sum(dim=0)
        state["exp_avg"].mul_(beta1).add_(gradient, alpha=1 - beta1)
        state["exp_avg_sq_row"].mul_(beta2).add_(row_means, alpha=1 - beta2)
        state["exp_avg_sq_col"].mul_(beta2).add_(column_sums / rows, alpha=1 - beta2)
        self._apply(parameter, state, torch.arange(rows, device=parameter.device), group["eps"], scale, correction)

    def _sparse_step(self, parameter, state, group, beta1, beta2, scale, correction):
        gradient = parameter.grad.coalesce()
        index = gradient.indices()[0]
        values = gradient.values()
        squared = values.float().square()
        exp_avg = state["exp_avg"]
        exp_avg[index] = (exp_avg[index] * beta1 + values * (1 - beta1)).to(exp_avg.dtype)
        row = state["exp_avg_sq_row"]
        row[index] = row[index] * beta2 + squared.mean(dim=1) * (1 - beta2)
        state["exp_avg_sq_col"].mul_(beta2).add_(squared.mean(dim=0), alpha=1 - beta2)
        self._apply(parameter, state, index, group["eps"], scale, correction)

    def _apply(self, parameter, state, index, eps, scale, correction):
        row, column = state["exp_avg_sq_row"], state["exp_avg_sq_col"]
        # The column means come from the stepped rows, so the row means are normalized over the same rows. Rows that
        # have all been zero so far have no scale yet; their first moment is zero as well.
        mean = row[index].mean().clamp(min=torch.finfo(torch.float32).tiny)
        for start in range(0, index.numel(), self.chunk):
            rows = index[start : start + self.chunk]
            second = row[rows, None] * column[None] / mean
            denominator = second.sqrt() / correction + eps
            update = state["exp_avg"][rows].float() / denominator
            parameter[rows] = (parameter[rows].float() - scale * update).to(parameter.dtype)


class MuonWithAdamW(torch.optim.Optimizer):
    r"""
    Constructs an optimizer that steps weight matrices with Muon, large text tables with [`FactoredAdam`] and
    everything else with AdamW.

    Muon orthogonalizes the update of a whole matrix, which suits a matrix whose rows act together, such as
    the output side of a correction. A table whose rows are separate tokens is the opposite case: an
    orthogonalized step would tie unrelated tokens together, so such tensors stay on AdamW.

    The parameter groups of the inner optimizers are this optimizer's own groups, the same dictionaries, so
    a learning rate scheduler that writes `lr` into `param_groups` reaches all of them.

    Args:
        muon_params (`list` of `torch.nn.Parameter`):
            Two dimensional parameters stepped by Muon. May be empty.
        adamw_groups (`list` of `dict`):
            Parameter groups stepped by AdamW, each with its own `weight_decay`. Empty groups are dropped.
        lr (`float`):
            Learning rate of both. Muon's is matched to AdamW's update size (`adjust_lr_fn="match_rms_adamw"`).
        betas (`tuple[float, float]`, *optional*, defaults to `(0.9, 0.999)`):
            AdamW moments.
        eps (`float`, *optional*, defaults to 1e-8):
            AdamW epsilon.
        muon_weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay of the Muon parameters.
        momentum (`float`, *optional*, defaults to 0.95):
            Muon momentum.
        table_params (`list` of `torch.nn.Parameter`, *optional*):
            Two dimensional text tables stepped by [`FactoredAdam`] with the AdamW `betas` and `eps` (SPEC S7.2a).
    """

    def __init__(
        self,
        muon_params: list,
        adamw_groups: list,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        muon_weight_decay: float = 0.0,
        momentum: float = 0.95,
        table_params: list | None = None,
    ):
        adamw_groups = [group for group in adamw_groups if group["params"]]
        # Either side may have nothing to step, and the inner optimizers refuse an empty parameter list.
        self.muon = (
            torch.optim.Muon(
                muon_params, lr=lr, weight_decay=muon_weight_decay, momentum=momentum, adjust_lr_fn="match_rms_adamw"
            )
            if muon_params
            else None
        )
        self.adamw = torch.optim.AdamW(adamw_groups, lr=lr, betas=betas, eps=eps) if adamw_groups else None
        self.tables = FactoredAdam(table_params, lr=lr, betas=betas, eps=eps) if table_params else None
        inner = [optimizer for optimizer in self._inner() if optimizer is not None]
        super().__init__([dict(group) for optimizer in inner for group in optimizer.param_groups], {})
        self._relink()

    def _inner(self):
        return (self.muon, self.adamw, self.tables)

    def _relink(self):
        r"""
        Makes the inner optimizers read their groups from `self.param_groups`.
        """
        start = 0
        for optimizer in self._inner():
            if optimizer is not None:
                stop = start + len(optimizer.param_groups)
                optimizer.param_groups = self.param_groups[start:stop]
                start = stop

    @torch.no_grad()
    def step(self, closure=None):
        r"""
        Args:
            closure (`Callable`, *optional*):
                Re-evaluates the model and returns the loss.

        Returns:
            The loss the closure returned, or `None`.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for optimizer in self._inner():
            if optimizer is not None:
                optimizer.step()
        return loss

    def state_dict(self):
        r"""
        Returns:
            `dict`: The state of each inner optimizer.
        """
        return {
            name: optimizer.state_dict()
            for name, optimizer in zip(("muon", "adamw", "tables"), self._inner())
            if optimizer is not None
        }

    def load_state_dict(self, state_dict):
        r"""
        Args:
            state_dict (`dict`):
                What [`state_dict`] returned.
        """
        groups = []
        for name, optimizer in zip(("muon", "adamw", "tables"), self._inner()):
            if optimizer is not None:
                optimizer.load_state_dict(state_dict[name])
                groups.extend(optimizer.param_groups)
        self.param_groups[:] = groups


__all__ = ["FactoredAdam", "MuonWithAdamW"]
