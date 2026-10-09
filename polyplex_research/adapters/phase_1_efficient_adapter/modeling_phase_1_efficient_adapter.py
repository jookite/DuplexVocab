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
"""PyTorch Phase 1 text injection adapter, assembled onto an expanded duplex model."""

import math
import os
import types
from contextlib import contextmanager
from dataclasses import asdict, fields
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import ModelHubMixin, hf_hub_download
from safetensors.torch import load_file, save_file
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_phase_1_efficient_adapter import EfficientAdapterP1Config


TEXT_ADAPTER_WEIGHTS_NAME = "text_rows.safetensors"

_CONFIG_FIELDS = {field.name for field in fields(EfficientAdapterP1Config)}


def text_tables(model) -> list[tuple[object, str, nn.Module, bool]]:
    r"""
    Collects every table indexed by the text vocabulary.

    Moshi derived models hold three: the backbone input embedding, the language modelling head, and the
    depth decoder's own text embedding. The two embeddings carry one row more than the vocabulary, an id
    the model reserves for the frame it has not generated yet, which is why the reserved row is reported
    separately: it has to stay at the end of the table when the vocabulary grows, and it is never trained.

    Args:
        model (`PreTrainedModel`):
            The model whose text tables are collected.

    Returns:
        `list`: Entries of `(owner, attribute_name, module, has_reserved_row)`. The attribute name is
        `"__input__"` or `"__output__"` for the two tables the model exposes through accessors.
    """
    vocab_size = model.config.vocab_size
    tables = []

    input_embed = model.get_input_embeddings()
    if input_embed is not None:
        tables.append((model, "__input__", input_embed, input_embed.num_embeddings == vocab_size + 1))

    output_embed = model.get_output_embeddings()
    if output_embed is not None:
        tables.append((model, "__output__", output_embed, output_embed.weight.shape[0] == vocab_size + 1))

    depth = getattr(model, "depth_decoder", None)
    depth_model = getattr(depth, "model", None)
    depth_embed = getattr(depth_model, "text_embed_tokens", None)
    if depth_embed is not None:
        tables.append(
            (depth_model, "text_embed_tokens", depth_embed, depth_embed.num_embeddings == vocab_size + 1)
        )

    return tables


def protect_leading_rows(weight: nn.Parameter, num_protected: int, gain: float = 1.0):
    r"""
    Zeroes the gradient of the leading rows of a table on every backward pass.

    Phase 1 trains rows inside tables it does not otherwise own, so freezing cannot be expressed by
    `requires_grad`, which is per tensor and not per row. A gradient hook is the granularity that matches:
    the optimizer still steps the whole tensor, but the protected rows receive exactly zero and therefore
    keep the values the pretrained model learned English on.

    Weight decay and momentum would still move a row whose gradient is zero, so a Phase 1 run has to leave
    weight decay off for these tables. [`EfficientAdapterP1Model.adapter_parameters`] is what an optimizer
    should be built from.

    Args:
        weight (`torch.nn.Parameter` of shape `(rows, dim)`):
            The table to protect, modified in place through its hook.
        num_protected (`int`):
            Number of leading rows whose gradient is zeroed.
        gain (`float`, *optional*, defaults to 1.0):
            Factor the remaining rows are scaled by. A tied row serves the head and the input table at
            once, so one update has to satisfy both uses, and this is what sets how far it travels. It
            rides the same hook as the protection because both act per row.

    Returns:
        `torch.utils.hooks.RemovableHandle`: Handle removing the protection.
    """

    def hook(grad: torch.Tensor) -> torch.Tensor:
        guarded = grad.clone()
        guarded[:num_protected] = 0
        if gain != 1.0:
            guarded[num_protected:] *= gain
        return guarded

    return weight.register_hook(hook)


def _einsum_flexible_linear(self, x, layer_idx=None):
    r"""
    The per codebook linear of the depth decoder, computed without materialising its weights per frame.

    Args:
        x (`torch.FloatTensor` of shape `(batch_size, sequence_length, input_size)`):
            Input, where `sequence_length` is 1 when one state is read by every codebook.
        layer_idx (`torch.LongTensor` of shape `(num_layers,)`, *optional*):
            Which codebook's weights each position uses.

    Returns:
        `torch.FloatTensor` of shape `(batch_size, num_layers, output_size)`: The same values as the
        module's own forward.
    """
    weight = torch.index_select(self.weight, 0, layer_idx) if layer_idx is not None else self.weight
    x = x.expand(x.shape[0], weight.shape[0], x.shape[-1])
    return torch.einsum("nsi,soi->nso", x, weight)


@contextmanager
def per_codebook_einsum(depth: nn.Module):
    r"""
    Runs every per codebook linear of the depth decoder through [`_einsum_flexible_linear`].

    The module's own forward broadcasts a `matmul` over the frames, which copies the stacked weights once
    per frame and keeps that copy for backward whenever its input carries a gradient. Decoding thousands
    of frames with a graph, as the audio objective does, needs gigabytes per frame that way.

    Args:
        depth (`nn.Module`):
            The depth decoder.
    """
    patched = [
        module for module in depth.modules() if type(module).__name__ == "PersonaPlexFlexibleLinear"
    ]
    for module in patched:
        module.forward = types.MethodType(_einsum_flexible_linear, module)
    try:
        yield
    finally:
        for module in patched:
            del module.forward


class MixtureRows(torch.autograd.Function):
    r"""
    Builds each appended row as a weighted sum of the pretrained rows of its support.

    $$w_i = \sum_{j=1}^{k} c_{ij}\, w_{s_{ij}}$$

    The gather $w_{s}$ has `appended_rows * k * hidden` entries, which is far larger than either the
    table it builds or the coefficients it is differentiated against, so it is never held whole: the
    forward walks the vocabulary in chunks and the backward recomputes each chunk instead of saving it.
    Autograd on the plain expression would keep every chunk alive until backward, which is what this
    exists to avoid. Only the coefficients receive a gradient, since the pretrained rows are the fixed
    basis the phase is constrained to.

    Args:
        weights (`torch.FloatTensor` of shape `(appended_rows, k)`):
            The mixture coefficients, the only differentiable input.
        pretrained (`torch.FloatTensor` of shape `(base_vocab_size, hidden_size)`):
            The frozen rows the mixture is taken over.
        support (`torch.LongTensor` of shape `(appended_rows, k)`):
            Which pretrained rows each appended row mixes.
        chunk (`int`):
            How many appended rows are built at a time.

    Returns:
        `torch.FloatTensor` of shape `(appended_rows, hidden_size)`: The appended rows.
    """

    @staticmethod
    def forward(ctx, weights, pretrained, support, chunk):
        ctx.save_for_backward(pretrained, support)
        ctx.chunk = chunk
        rows = weights.new_empty((weights.shape[0], pretrained.shape[1]))
        for start in range(0, weights.shape[0], chunk):
            stop = start + chunk
            gathered = pretrained[support[start:stop]].to(weights.dtype)
            rows[start:stop] = torch.bmm(weights[start:stop].unsqueeze(1), gathered).squeeze(1)
        return rows

    @staticmethod
    def backward(ctx, grad_rows):
        pretrained, support = ctx.saved_tensors
        grad_rows = grad_rows.contiguous()
        grad_weights = grad_rows.new_empty(support.shape)
        for start in range(0, support.shape[0], ctx.chunk):
            stop = start + ctx.chunk
            gathered = pretrained[support[start:stop]].to(grad_rows.dtype)
            grad_weights[start:stop] = torch.bmm(
                gathered, grad_rows[start:stop].unsqueeze(-1)
            ).squeeze(-1)
        return grad_weights, None, None, None


class LoraLinear(nn.Module):
    r"""
    A frozen projection plus a low rank correction of it.

    $$W_{\text{eff}} x = Wx + \frac{\alpha}{r} B A x$$

    $B$ starts at zero, so the wrapped module answers exactly as it did before the first step.

    Args:
        wrapped (`nn.Linear`):
            The pretrained projection, kept frozen.
        rank (`int`):
            Rank of the correction.
        alpha (`float`):
            Scale of the correction.
    """

    def __init__(self, wrapped: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.wrapped = wrapped
        self.scaling = alpha / rank
        self.down = nn.Parameter(torch.zeros(rank, wrapped.in_features, dtype=torch.float32))
        self.up = nn.Parameter(torch.zeros(wrapped.out_features, rank, dtype=torch.float32))
        nn.init.normal_(self.down, std=1 / rank)

    def forward(self, hidden):
        correction = (hidden.float() @ self.down.T) @ self.up.T
        return self.wrapped(hidden) + correction.to(hidden.dtype) * self.scaling


class ConfinedCapacity(nn.Module):
    r"""
    A frozen module plus a linear correction that reads only fixed input directions.

    $$f(x) + U V^\top x$$

    As built, $V$ is fixed: its columns are the directions of the module's input that deployment barely uses
    ([`~ConfinedCapacity.directions_from_moments`]), so the correction answers almost nothing there however $U$ is
    trained. A session expert replaces it with a trained `Parameter`, so the correction reads the whole
    input. $U$ starts at zero, so the wrapped module answers exactly as it did before the first step.

    Args:
        wrapped (`nn.Module`):
            The pretrained module, kept frozen: an FFN, or the value projection of an attention block.
        basis (`torch.FloatTensor` of shape `(hidden_size, rank)`):
            The directions the correction reads.
        deploy_moment (`torch.FloatTensor` of shape `(rank, rank)`, *optional*):
            Second moment of deployment inputs seen through `basis`.
        empty_moment (`torch.FloatTensor` of shape `(rank, rank)`, *optional*):
            Second moment of empty space inputs seen through `basis`.
        out_features (`int`, *optional*):
            Width of the wrapped module's output, `hidden_size` when omitted.
    """

    def __init__(
        self, wrapped: nn.Module, basis: torch.Tensor, deploy_moment=None, empty_moment=None, out_features=None
    ):
        super().__init__()
        self.wrapped = wrapped
        self.register_buffer("basis", basis.float().contiguous())
        self.up = nn.Parameter(
            torch.zeros(out_features or basis.shape[0], basis.shape[1], dtype=torch.float32)
        )
        # Second moments of the FFN input seen through `basis`, `(rank, rank)`: the deployment inputs the
        # correction must stay quiet on, and the empty space inputs it is meant to answer.
        self.register_buffer("deploy_moment", None if deploy_moment is None else deploy_moment.float())
        self.register_buffer("empty_moment", None if empty_moment is None else empty_moment.float())

    @staticmethod
    def directions_from_moments(moments: dict, rank: int) -> torch.Tensor:
        r"""
        Picks the input directions a confined correction reads, from the second moments of one layer.

        Args:
            moments (`dict`):
                Normalized second moments of one layer and module, keyed by frame group, as
                [`~polyplex_research.utils.evaluate.CapacityMeter`] writes them.
            rank (`int`):
                How many directions to return.

        Returns:
            `torch.FloatTensor` of shape `(hidden_size, rank)`: Directions outside the subspace holding 99% of the
            deploy energy, in order of the Korean silent-condition energy they carry.
        """
        deploy = 0.5 * (moments["deploy_en_speech"].double() + moments["deploy_silent_pad"].double())
        values, vectors = torch.linalg.eigh(deploy)
        values, vectors = values.flip(0).clamp(min=0), vectors.flip(1)
        protected = int(((values.cumsum(0) / values.sum()) < 0.99).sum()) + 1
        basis = vectors[:, :protected]
        outside = torch.eye(basis.shape[0], dtype=basis.dtype, device=basis.device) - basis @ basis.T
        _, residual = torch.linalg.eigh(outside @ moments["empty_ko"].double().to(basis.device) @ outside)
        return residual.flip(1)[:, :rank].float()

    def output_energy(self, moment: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            moment (`torch.FloatTensor` of shape `(rank, rank)`):
                Second moment of the inputs, read through `basis`.

        Returns:
            `torch.Tensor`: The mean squared norm of the correction over those inputs, $\operatorname{tr}(U M U^\top)$.
        """
        return torch.einsum("hr,rs,hs->", self.up, moment, self.up)

    def forward(self, hidden, *args, **kwargs):
        output = self.wrapped(hidden, *args, **kwargs)
        correction = (hidden.to(self.up.dtype) @ self.basis.to(self.up.dtype)) @ self.up.T
        return output + correction.to(output.dtype)


class EfficientAdapterP1Model(
    nn.Module,
    ModelHubMixin,
    library_name="polyplex-research",
    tags=["polyplex", "phase-1", "text-injection"],
    coders={
        EfficientAdapterP1Config: (
            asdict,
            lambda stored: EfficientAdapterP1Config(
                **{name: value for name, value in stored.items() if name in _CONFIG_FIELDS}
            ),
        )
    },
):
    r"""
    Assembles the Phase 1 text injection adapter onto an expanded duplex model.

    Phase 1 introduces no new parameter. The vocabulary expansion has already appended rows to the text
    tables, and this class is what makes those rows, and only those rows, the trainable surface:

    - Every pretrained weight is frozen, so the temporal transformer keeps the English turn taking
      behaviour and the audio path is untouched. The backbone is run under `torch.no_grad` and only the
      text head is differentiated, which is what the phase means by concentrating on the LM head.
    - The selected text tables are unfrozen as whole tensors, and a gradient hook zeroes their leading
      `base_vocab_size` rows. The English text space is therefore held fixed while the appended rows move
      to align with it, which is what replaces continued pretraining.
    - The reserved row past the vocabulary sits after the new rows and is protected by the same slice
      bound, since the trainable window stops at `vocab_size`.

    The saved artifact is the new row slices alone, not the expanded 7B checkpoint.

    Args:
        model (`PreTrainedModel`):
            The expanded duplex model to adapt, modified in place.
        config (`EfficientAdapterP1Config`):
            Which tables are trained and where the pretrained vocabulary ends.
    """

    def __init__(self, model, config: EfficientAdapterP1Config):
        super().__init__()
        self.config = config
        self.model = model
        self._handles = []
        self._tables = self._select_tables()
        self._tied_input_scale = self._measure_tied_input_scale() if config.tie_new_input_rows else 1.0

        self.lora = {}
        if config.lora_layers:
            layers = model.model.layers
            if config.lora_layers > len(layers):
                raise ValueError(
                    f"`lora_layers` is {config.lora_layers}, more than the {len(layers)} the backbone "
                    "has."
                )
            for depth in range(len(layers) - config.lora_layers, len(layers)):
                attention = layers[depth].self_attn
                for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    projection = getattr(attention, name)
                    inner = getattr(projection, "linear", projection)
                    wrapper = LoraLinear(inner, config.lora_rank, config.lora_alpha)
                    if inner is projection:
                        setattr(attention, name, wrapper)
                    else:
                        projection.linear = wrapper
                    self.lora[f"{depth}_{name}"] = wrapper

        # Keyed by layer index for an FFN and by `v<index>` for a value projection, the names they are saved under.
        self.confined = {}
        layers = model.model.layers
        if config.confined_basis:
            for index, basis, deploy_moment, empty_moment in self._confined_bases(config.confined_basis, "ffn"):
                wrapper = ConfinedCapacity(layers[index].mlp, basis, deploy_moment, empty_moment)
                wrapper.to(next(layers[index].mlp.parameters()).device)
                layers[index].mlp = wrapper
                self.confined[index] = wrapper
        if config.session_expert_layers:
            first, last = (int(value) for value in config.session_expert_layers.split("-"))
            generator = torch.Generator().manual_seed(0)
            for index in range(first, last + 1):
                mlp = layers[index].mlp
                width = model.config.hidden_size
                # The read side starts as random orthonormal directions over the whole FFN input and is
                # trained with the write side, so nothing about the input space is ruled out.
                basis = torch.linalg.qr(
                    torch.randn(width, config.session_expert_rank, generator=generator)
                )[0]
                wrapper = ConfinedCapacity(mlp, basis)
                wrapper.basis = nn.Parameter(wrapper.basis.data)
                wrapper.to(next(mlp.parameters()).device)
                layers[index].mlp = wrapper
                self.confined[index] = wrapper
        if config.confined_attention_basis:
            bases = self._confined_bases(config.confined_attention_basis, "attention")
            for index, basis, deploy_moment, empty_moment in bases:
                attention = layers[index].self_attn
                projection = attention.v_proj
                inner = getattr(projection, "linear", projection)
                # The value is what attention mixes across frames, so a correction there reaches later
                # frames and not only the one it reads.
                wrapper = ConfinedCapacity(inner, basis, deploy_moment, empty_moment, inner.out_features)
                wrapper.to(inner.weight.device)
                if inner is projection:
                    attention.v_proj = wrapper
                else:
                    projection.linear = wrapper
                self.confined[f"v{index}"] = wrapper

        self.head_body = None
        if config.head_hidden_size:
            width = model.get_output_embeddings().weight.shape[1]
            self.head_body = nn.Sequential(
                nn.LayerNorm(width, dtype=torch.float32),
                nn.Linear(width, config.head_hidden_size, dtype=torch.float32),
                nn.GELU(),
                nn.Linear(config.head_hidden_size, width, dtype=torch.float32),
            )
            # Zero on the way out, so the run starts from the linear head it is compared against.
            nn.init.zeros_(self.head_body[-1].weight)
            nn.init.zeros_(self.head_body[-1].bias)
            self.head_body.to(model.get_output_embeddings().weight.device)

        self.shared_residual = None
        if config.shared_token_residual:
            stored = load_file(config.shared_token_residual)
            reference = model.get_output_embeddings().weight
            self.register_buffer("_english_mean", stored["mean"].float(), persistent=False)
            self.register_buffer("_english_basis", stored["basis"].float(), persistent=False)
            self.register_buffer("_shared_ids", stored["shared_ids"].long(), persistent=False)
            corrected = (
                config.base_vocab_size if config.residual_all_frozen_rows else len(stored["shared_ids"])
            )
            self.shared_residual = nn.Parameter(
                torch.zeros(corrected, reference.shape[1], dtype=torch.float32)
            )
            self.to(reference.device)

        if config.group_gated_softmax:
            reference = model.get_output_embeddings().weight
            self.group_gate = nn.Linear(reference.shape[1], 1, dtype=torch.float32)
            nn.init.zeros_(self.group_gate.weight)
            # Starting at the corpus proportion means the gate begins where a constant predictor would.
            prior = min(max(config.group_prior, 1e-4), 1 - 1e-4)
            nn.init.constant_(self.group_gate.bias, math.log(prior / (1 - prior)))
            self.group_gate.to(reference.device)
        else:
            self.group_gate = None

        # Which pretrained rows each appended row may be built from, with the coefficients over them as
        # the parameter, so that every point of the trajectory satisfies the constraint by construction.
        self.row_mixture = None
        self.register_buffer("_row_support", None, persistent=False)
        if config.share_row_mixture and not config.base_row_mixture:
            raise ValueError(
                "`share_row_mixture` shares the coefficients of `base_row_mixture`, which is 0, so there "
                "are no coefficients to share."
            )
        self.partition_direction = None
        if config.partition_direction:
            if not config.base_row_mixture:
                raise ValueError(
                    "`partition_direction` is added on top of the mixture of `base_row_mixture`, which "
                    "is 0, so there is no mixture to add it to."
                )
            reference = model.get_output_embeddings().weight
            # Zero, so the run starts from the rows the mixture alone gives.
            self.partition_direction = nn.Parameter(
                torch.zeros(reference.shape[1], dtype=torch.float32, device=reference.device)
            )

        if config.base_row_mixture:
            reference = model.get_output_embeddings().weight
            self._row_support = self._build_row_support(config.base_row_mixture)
            # Sized so one chunk of the gather stays near 64MB whatever the mixture width is.
            self._mixture_chunk = max(
                1, 2**24 // (config.base_row_mixture * reference.shape[1])
            )
            if config.share_row_mixture:
                # Fitted on the head, whose rows the support was chosen from. Every other table then
                # holds the same blend of its own pretrained rows.
                self.row_mixture = nn.ParameterList(
                    [nn.Parameter(self._fit_mixture(self._mixture_part(reference.data)))]
                )
            else:
                self.row_mixture = nn.ParameterList(
                    [
                        nn.Parameter(self._fit_mixture(self._mixture_part(module.weight.data)))
                        for _, module in self._tables
                    ]
                )
            self.materialize_rows()

    # -----------------------------------------------------------------
    # Assembly
    # -----------------------------------------------------------------

    def _select_tables(self) -> list[tuple[str, nn.Module]]:
        r"""
        Picks the text tables this run trains, skipping any the configuration switched off.

        A model with tied input and output embeddings exposes the same tensor twice. It is kept once, so
        that the protection hook is registered a single time and the row count is not double reported.

        Returns:
            `list`: Entries of `(name, module)` in the order [`text_tables`] reports them.
        """
        wanted = {
            "__input__": self.config.train_input_embeddings,
            "__output__": self.config.train_lm_head,
            "text_embed_tokens": self.config.train_depth_text_embeddings,
        }

        selected = []
        seen = set()
        for _, name, module, _ in text_tables(self.model):
            if not wanted.get(name, False) or id(module.weight) in seen:
                continue
            seen.add(id(module.weight))
            selected.append((name, module))
        return selected

    def _reaches_gradient(self, module: nn.Module) -> bool:
        r"""
        Reports whether a table is on the differentiated side of the Phase 1 forward.

        The test is tensor identity against the output head, not the name the table was collected under.
        A model that ties its text input and output embeddings holds one tensor, so training the head
        trains the lookup as well and the input table is reachable; a model that keeps them apart, which
        PersonaPlex does since the two differ by the reserved row, leaves the lookup inside the `no_grad`
        block with nothing flowing back into it.

        The depth decoder's text table is the other reachable one. [`audio_loss`] runs the depth decoder
        outside that block and reads the table through [`depth_text_embeddings`], which is how the
        speaking objective reaches the rows at all.

        Args:
            module (`torch.nn.Module`):
                A table reported by [`text_tables`].

        Returns:
            `bool`: Whether a gradient reaches the table.
        """
        head = self.model.get_output_embeddings()
        if head is not None and module.weight is head.weight:
            return True
        depth = getattr(getattr(self.model, "depth_decoder", None), "model", None)
        depth_text = getattr(depth, "text_embed_tokens", None)
        return depth_text is not None and module.weight is depth_text.weight

    def freeze_base(self):
        r"""
        Freezes every pretrained weight and leaves only the appended text rows trainable.

        Raises:
            RuntimeError: If no table was selected, so that the run would train nothing.
        """
        if not self._tables:
            raise RuntimeError(
                "No text table is selected, so Phase 1 would train nothing. Enable at least one of "
                "`train_input_embeddings`, `train_lm_head` or `train_depth_text_embeddings`."
            )

        unreachable = (
            []
            if self.keeps_graph
            else [name for name, module in self._tables if not self._reaches_gradient(module)]
        )
        if unreachable:
            raise RuntimeError(
                "Phase 1 runs the backbone under `torch.no_grad`, so a table trains only if it is the "
                f"tensor the output head reads. These selected tables would silently train nothing: "
                f"{unreachable}. Switch them off, or train them in a phase whose forward keeps a graph "
                "through the backbone."
            )

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for module in self.lora.values():
            module.down.requires_grad_(True)
            module.up.requires_grad_(True)
        for module in self.confined.values():
            module.up.requires_grad_(True)
            if isinstance(module.basis, nn.Parameter):
                module.basis.requires_grad_(True)

        self.detach_adapter()
        for _, module in self._tables:
            if self.config.float32_rows and module.weight.dtype != torch.float32:
                # The rows are the only thing that moves, and in bfloat16 a step of the size a head
                # takes rounds away: the format keeps about three significant digits, so an update of
                # 1e-4 on a row whose entries are around 1e-1 is lost. The backbone stays bfloat16, this
                # is one table.
                module.weight.data = module.weight.data.float()
            if self.row_mixture is not None:
                # The table holds no parameter of its own here. Its appended rows are a function of the
                # coefficients and its pretrained rows are never an argument of anything differentiated,
                # so leaving the whole tensor frozen is what keeps the English rows exactly fixed.
                module.weight.requires_grad_(False)
                continue
            module.weight.requires_grad_(True)
            self._handles.append(
                protect_leading_rows(
                    module.weight, self.config.base_vocab_size, self.config.tied_gradient_gain
                )
            )

        if self.row_mixture is not None and self.config.tie_new_input_rows and self.config.train_rows:
            gain = self.config.tied_gradient_gain
            if gain != 1.0:
                # The gain rides the coefficients now that no row carries a gradient of its own. It acts
                # on every appended row at once either way, so this is the same reweighting.
                for weights in self.row_mixture:
                    self._handles.append(weights.register_hook(lambda grad: grad * gain))

        if not self.config.train_rows:
            if not (self.lora or self.confined) or not self.config.train_corrections:
                raise RuntimeError(
                    "`train_rows=False` fixes the appended rows, and no backbone correction trains, so the run "
                    "would train nothing."
                )
            for parameter in self.adapter_parameters():
                parameter.requires_grad_(False)
            for _, module in self._tables:
                module.weight.requires_grad_(False)

        if not self.config.train_corrections:
            for module in list(self.lora.values()) + list(self.confined.values()):
                for parameter in module.parameters(recurse=False):
                    parameter.requires_grad_(False)

    def tied_input_embeddings(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        r"""
        Looks a batch up, taking the new rows from the head rather than from the input table.

        The pretrained ids keep reading the frozen input table. The appended ids read the head, so the
        two directions of the new vocabulary are one tensor and the gradient of the objective reaches
        the reading side as well as the writing side.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Token ids of the batch.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`: The embeddings.
        """
        table = self.model.get_input_embeddings()
        head = self.model.get_output_embeddings()
        start, stop = self.config.base_vocab_size, self.config.vocab_size

        # Only the appended vocabulary is read off the head. The duplex layout opens every batch on the
        # reserved id `vocab_size`, which is a row of the input table and has no counterpart in the head:
        # taken as appended it indexes one row past the end, and clamped it silently becomes another
        # token. Either way the frame that every absolute position is measured from is wrong.
        new = (input_ids >= start) & (input_ids < stop)
        with torch.no_grad():
            embeds = table(input_ids.masked_fill(new, 0))

        if not bool(new.any()):
            return embeds

        rows = self.appended_weight() * self._tied_input_scale
        # The scatter is what puts the head rows into the graph, so grad flows only through these.
        return embeds.masked_scatter(new.unsqueeze(-1), rows[input_ids[new] - start].to(embeds.dtype))

    def _measure_tied_input_scale(self) -> float:
        r"""
        Returns:
            `float`: The ratio between the row norms of the pretrained parts of the two tables, which is
            the factor that puts a head row into the range of the input table.
        """
        if self.config.tied_input_scale is not None:
            return float(self.config.tied_input_scale)

        head = self.model.get_output_embeddings()
        table = self.model.get_input_embeddings()
        if head is None or table is None:
            return 1.0

        base = self.config.base_vocab_size
        with torch.no_grad():
            head_norm = head.weight[:base].float().norm(dim=1).mean()
            table_norm = table.weight[:base].float().norm(dim=1).mean()
        return float(table_norm / head_norm.clamp(min=1e-6))

    def detach_adapter(self):
        r"""
        Removes every protection hook, so the tables train as whole tensors again.
        """
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def fold_into_base(self):
        r"""
        Writes what the phase learned into the tables of the base model and returns it.

        The trained rows already live in the tables, since Phase 1 moves slices of tensors the model
        owns. What does not is the tie: [`tied_input_embeddings`] reads the appended ids off the head at
        every forward, and a plain checkpoint looks them up in the input table instead. Copying the
        scaled head rows into that table is what makes the two agree.

        Returns:
            `PolyplexForConditionalGeneration`: The base model, modified in place.

        Raises:
            ValueError: If the run carried a term no table can hold.
        """
        unsupported = [
            name
            for name, carried in (
                ("shared_token_residual", self.shared_residual is not None),
                ("project_new_rows", self.config.project_new_rows),
                ("group_gated_softmax", self.group_gate is not None),
            )
            if carried
        ]
        if unsupported:
            raise ValueError(
                f"{unsupported} change how the logits are computed, not what the tables hold: the "
                "correction reads the part of the state the English subspace does not span, and the gate "
                "splits the softmax. Neither survives into a checkpoint the base class loads. Evaluate "
                "such a run through `EfficientAdapterP1Model`."
            )

        self.detach_adapter()
        self.materialize_rows()

        table = self.model.get_input_embeddings()
        head = self.model.get_output_embeddings()
        if not self.config.tie_new_input_rows or table is None or head is None:
            return self.model
        if table.weight is head.weight:
            return self.model

        start, stop = self.config.base_vocab_size, self.config.vocab_size
        with torch.no_grad():
            rows = head.weight[start:stop] * self._tied_input_scale
            table.weight[start:stop] = rows.to(table.weight.dtype)
        return self.model

    # -----------------------------------------------------------------
    # Training surface
    # -----------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.BoolTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        text_labels: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        r"""
        Runs the frozen backbone without a graph and the text head with one.

        Phase 1 moves rows of the output head only, so the backbone is pure inference here: it is called
        under `torch.no_grad` and its hidden states are detached before the head sees them. Nothing about
        the computation changes, only what is recorded for backward. The backbone is where essentially all
        of the activation memory of a 7B duplex model sits, and none of it is needed to differentiate a
        single linear layer, so this is what lets the phase run at a batch size worth training at.

        The audio path is not entered at all. Phase 1 is a text objective, and the depth decoder would only
        add frozen compute.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Text tokens of every frame.
            attention_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask of the temporal sequence.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Positions of the frames. A row that restarts them from zero holds several documents, and with
                no `attention_mask` the backbone keeps each document from attending to the others.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Precomputed input embeddings, used instead of looking `input_ids` up.
            text_labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Text tokens aligned with `input_ids`. The shift by one frame happens here, so passing
                `input_ids` with `-100` on the frames to skip is the expected usage.
            past_key_values (`Cache`, *optional*):
                Cached keys and values of the backbone.
            use_cache (`bool`, *optional*):
                Whether the backbone returns its cache.
            output_attentions (`bool`, *optional*):
                Whether the backbone returns its attentions.
            output_hidden_states (`bool`, *optional*):
                Whether the backbone returns every hidden state.

        Returns:
            `CausalLMOutputWithPast`: The text logits and, when `text_labels` is given, their cross entropy.

        Raises:
            ValueError: If neither `input_ids` nor `inputs_embeds` is given.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Pass one of `input_ids` and `inputs_embeds`.")

        backbone = self.model.model
        tied = self.config.tie_new_input_rows
        if position_ids is not None and past_key_values is None:
            # The backbone reads documents out of restarting positions only when it holds no cache, and it
            # creates one whenever `use_cache` resolves to the config default.
            use_cache = False
        # The duplex layout hands the two acoustic streams alongside the text one, and the backbone reads
        # the sum of all three. Dropping them leaves the state in a regime the
        # checkpoint never produces, which costs over two nats and does not raise.
        assistant_audio_codes = kwargs.pop("assistant_audio_codes", None)
        user_audio_codes = kwargs.pop("user_audio_codes", None)
        audio_labels = kwargs.pop("audio_labels", None)
        audio_embeds = None
        if assistant_audio_codes is not None and user_audio_codes is not None:
            with torch.no_grad():
                audio_embeds = self.model._embed_audio_codes(
                    torch.cat([assistant_audio_codes, user_audio_codes], dim=1)
                )

        if self.keeps_graph:
            # The tie already updates these rows through the head. Keeping the graph adds the reading
            # direction on top, which is why it is opt in rather than part of the tie.
            if inputs_embeds is None:
                # Detached unless the tie was asked to train through the backbone: a lookup that
                # carries a gradient makes every layer below the lowest adapted one store its
                # activations, which is the cost the depth of the correction was chosen to avoid.
                with torch.set_grad_enabled(tied and self.config.tied_backbone_grad):
                    inputs_embeds = (
                        self.tied_input_embeddings(input_ids)
                        if tied
                        else backbone.embed_tokens(input_ids)
                    )
                if audio_embeds is not None:
                    inputs_embeds = inputs_embeds + audio_embeds.to(inputs_embeds.dtype)
            decoder_outputs = backbone(
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            hidden = decoder_outputs.last_hidden_state
            if audio_labels is not None:
                return self._speak(hidden, input_ids, assistant_audio_codes, audio_labels, text_labels, decoder_outputs)
            return self._finish(self.text_logits(hidden), text_labels, decoder_outputs, hidden)

        with torch.no_grad():
            if inputs_embeds is None:
                inputs_embeds = (
                    self.tied_input_embeddings(input_ids) if tied else backbone.embed_tokens(input_ids)
                )
                if audio_embeds is not None:
                    inputs_embeds = inputs_embeds + audio_embeds.to(inputs_embeds.dtype)
            decoder_outputs = backbone(
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                **kwargs,
            )

        # Detached explicitly rather than relying on the `no_grad` block alone, so that the boundary
        # survives a caller that re-enables grad around this method.
        hidden = decoder_outputs.last_hidden_state.detach()
        if audio_labels is not None:
            return self._speak(hidden, input_ids, assistant_audio_codes, audio_labels, text_labels, decoder_outputs)
        return self._finish(self.text_logits(hidden), text_labels, decoder_outputs, hidden)

    def _speak(self, hidden, input_ids, assistant_audio_codes, audio_labels, text_labels, decoder_outputs):
        r"""
        Scores the assistant's codes and, when text targets are given too, adds their cross entropy.

        Args:
            hidden (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Backbone states.
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Text stream the depth decoder is conditioned on.
            assistant_audio_codes (`torch.LongTensor` of shape `(batch_size, num_codebooks, sequence_length)`):
                Assistant codes in the delay layout.
            audio_labels (`torch.LongTensor` of shape `(batch_size, num_codebooks, sequence_length)`):
                Targets aligned with `assistant_audio_codes`.
            text_labels (`torch.LongTensor` or `None`):
                Text targets, or `None` when only the audio is scored.
            decoder_outputs (`ModelOutput`):
                What the backbone returned.

        Returns:
            `CausalLMOutputWithPast`: The loss, and the text logits when text targets were given. The text
            head is not run for an audio only batch, since its logits span the whole vocabulary on every
            frame and nothing would read them.
        """
        audio_loss = self.audio_loss(hidden, input_ids, assistant_audio_codes, audio_labels)
        if text_labels is None:
            return CausalLMOutputWithPast(loss=audio_loss, past_key_values=decoder_outputs.past_key_values)
        outputs = self._finish(self.text_logits(hidden), text_labels, decoder_outputs, hidden)
        outputs.loss = audio_loss if outputs.loss is None else outputs.loss + audio_loss
        return outputs

    def _table_position(self, name: str) -> int | None:
        r"""
        Args:
            name (`str`):
                Name [`text_tables`] reports the table under.

        Returns:
            `int` or `None`: Its index among the trained tables, `None` when it is not trained.
        """
        for position, (table_name, _) in enumerate(self._tables):
            if table_name == name:
                return position
        return None

    def depth_text_embeddings(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        r"""
        Looks the depth decoder's text position up, carrying a graph to the appended rows.

        The depth decoder reads the frame's text token through its own table. Its rows are written back
        into that table after every step, but only as a detached copy, so a plain lookup would score the
        audio against rows the objective cannot move. The appended ids are read off the mixture instead,
        in the same way [`tied_input_embeddings`] reads the head.

        Args:
            input_ids (`torch.LongTensor` of shape `(num_frames, 1)`):
                Text token of every scored frame.

        Returns:
            `torch.FloatTensor` of shape `(num_frames, 1, depth_hidden_size)`: The embeddings.
        """
        table = self.model.depth_decoder.model.text_embed_tokens
        start, stop = self.config.base_vocab_size, self.config.vocab_size
        new = (input_ids >= start) & (input_ids < stop)
        with torch.no_grad():
            embeds = table(input_ids.masked_fill(new, 0))

        position = self._table_position("text_embed_tokens")
        if position is None or not bool(new.any()):
            return embeds
        rows = self.appended_rows(position) if self.row_mixture is not None else table.weight[start:stop]
        return embeds.masked_scatter(new.unsqueeze(-1), rows[input_ids[new] - start].to(embeds.dtype))

    def audio_loss(self, hidden, input_ids, assistant_audio_codes, audio_labels) -> torch.Tensor:
        r"""
        Cross entropy of the assistant's codes, taken through the depth decoder with a graph.

        The frame convention is the one [`CodeRotationAdapterModel.forward`] fixes: the depth decoder is
        conditioned on frame `t` and scored on frame `t + 1`. Only the frames that carry a target are
        decoded, since the prefix and the padding would otherwise cost depth compute for nothing.

        Args:
            hidden (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Backbone states, already detached.
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Text stream.
            assistant_audio_codes (`torch.LongTensor` of shape `(batch_size, num_codebooks, sequence_length)`):
                Assistant codes in the delay layout.
            audio_labels (`torch.LongTensor` of shape `(batch_size, num_codebooks, sequence_length)`):
                Targets aligned with `assistant_audio_codes`. Values of `-100` and of the reserved code are
                ignored.

        Returns:
            `torch.Tensor`: The mean cross entropy over the scored codes.

        Raises:
            ValueError: If the conditioning the depth decoder needs is missing.
        """
        if input_ids is None or assistant_audio_codes is None:
            raise ValueError("The audio loss conditions on `input_ids` and `assistant_audio_codes`.")

        reserved = self.model.config.audio_vocab_size
        num_codebooks = audio_labels.shape[1]
        targets = audio_labels[:, :, 1:].transpose(1, 2).reshape(-1, num_codebooks)
        targets = targets.masked_fill(targets == reserved, -100)
        scored = (targets != -100).any(dim=-1)
        if not bool(scored.any()):
            return hidden.new_zeros((), dtype=torch.float32)

        text = input_ids[:, 1:].reshape(-1, 1)[scored]
        codes = assistant_audio_codes[:, :, 1:].transpose(1, 2).reshape(-1, num_codebooks)[scored]
        codes = codes.masked_fill(codes == -100, reserved)
        states = hidden[:, :-1].reshape(-1, hidden.shape[-1])[scored].unsqueeze(1)

        depth = self.model.depth_decoder
        with torch.no_grad():
            code_embeds = [depth.model.embed_tokens[k](codes[:, [k]]) for k in range(num_codebooks - 1)]
        text_embeds = self.depth_text_embeddings(text)
        inputs_embeds = torch.cat([text_embeds.to(code_embeds[0].dtype), *code_embeds], dim=1)

        with per_codebook_einsum(depth):
            logits = depth(last_hidden_state=states, inputs_embeds=inputs_embeds).logits
        flat = targets[scored].to(logits.device)
        return F.cross_entropy(
            logits.float().reshape(-1, reserved), flat.reshape(-1), ignore_index=-100, reduction="sum"
        ) / (flat != -100).sum().clamp(min=1)

    def _confined_bases(self, path: str, module: str) -> list:
        r"""
        Reads one basis file and, when `confined_moments` is set, the moments each basis is scored on.

        Args:
            path (`str`):
                Safetensors file keyed `layer_<index>`.
            module (`str`):
                `"ffn"` or `"attention"`, which moments of [`~polyplex_research.utils.evaluate.CapacityMeter`] the basis reads.

        Returns:
            `list`: `(index, basis, deploy_moment, empty_moment)` per layer, the moments `None` without
            `confined_moments`.
        """
        entries = []
        for key, basis in load_file(path).items():
            index = int(key.removeprefix("layer_"))
            deploy_moment = empty_moment = None
            if self.config.confined_moments:
                moments = torch.load(
                    os.path.join(self.config.confined_moments, f"moments_{index:02d}_{module}.pt"),
                    map_location="cpu",
                )
                deploy = 0.5 * (moments["deploy_en_speech"].float() + moments["deploy_silent_pad"].float())
                deploy_moment = basis.T.float() @ deploy @ basis.float()
                empty_moment = basis.T.float() @ moments["empty_ko"].float() @ basis.float()
            entries.append((index, basis, deploy_moment, empty_moment))
        return entries

    def confined_leak(self) -> torch.Tensor:
        r"""
        Share of the confined corrections' output that lands on deployment inputs.

        $$\frac{\sum_\ell \operatorname{tr}(U_\ell M^{\text{deploy}}_\ell U_\ell^\top)}{\operatorname{sg}\big(\sum_\ell \operatorname{tr}(U_\ell M^{\text{empty}}_\ell U_\ell^\top)\big)}$$

        The moments are the ones [`~polyplex_research.utils.evaluate.CapacityMeter`] wrote, so the term costs no forward. The
        denominator carries no gradient: the term asks the corrections to fall silent where deployment
        reads, not to shrink where the empty space reads.

        Returns:
            `torch.Tensor`: The ratio, zero when no correction carries moments.

        Raises:
            ValueError: If the corrections were built without `confined_moments`.
        """
        modules = list(self.confined.values())
        if not modules or modules[0].deploy_moment is None:
            raise ValueError("`confined_leak` reads the moments of `confined_moments`, which were not given.")
        deploy = sum(module.output_energy(module.deploy_moment) for module in modules)
        empty = sum(module.output_energy(module.empty_moment) for module in modules)
        return deploy / empty.detach().clamp(min=1e-8)

    def trainable_state(self) -> dict:
        r"""
        Returns:
            `dict`: A detached copy of every parameter that currently trains, by name, such as the state a run
            started from.
        """
        return {name: parameter.detach().clone() for name, parameter in self.named_parameters() if parameter.requires_grad}

    @contextmanager
    def parameters_replaced(self, state: dict):
        r"""
        Runs the block with the named parameters set to `state`, restoring them afterwards.

        Args:
            state (`dict`):
                Tensors by parameter name, such as [`trainable_state`] returned when a run started.
        """
        parameters = dict(self.named_parameters())
        saved = {name: parameters[name].data.clone() for name in state}
        for name, value in state.items():
            parameters[name].data.copy_(value)
        try:
            yield
        finally:
            for name, value in saved.items():
                parameters[name].data.copy_(value)

    @contextmanager
    def corrections_disabled(self):
        r"""
        Runs the block with every confined correction switched off, so the model answers as the backbone.
        """
        saved = {index: module.up.data.clone() for index, module in self.confined.items()}
        for module in self.confined.values():
            module.up.data.zero_()
        try:
            yield
        finally:
            for index, module in self.confined.items():
                module.up.data.copy_(saved[index])

    @property
    def keeps_graph(self) -> bool:
        r"""
        Returns:
            `bool`: Whether the backbone runs with a graph, which it must whenever something inside it
            is trained. The weights stay frozen either way; what changes is that the gradient can reach
            a correction that sits between them.
        """
        confined = bool(
            self.config.confined_basis or self.config.confined_attention_basis or self.config.session_expert_layers
        )
        return bool(self.config.lora_layers) or confined or (
            self.config.tie_new_input_rows and self.config.tied_backbone_grad
        )

    def _build_row_support(self, mixture: int) -> torch.Tensor:
        r"""
        Picks, for every appended row, the pretrained rows it is allowed to be built from.

        The choice is made once, off the initialisation the vocabulary expansion produced, so a new row
        keeps the pretrained rows it already resembles.

        Args:
            mixture (`int`):
                How many pretrained rows an appended row mixes.

        Returns:
            `torch.LongTensor` of shape `(appended_rows, mixture)`: Indices into the pretrained rows.
        """
        weight = self.model.get_output_embeddings().weight.data
        base = nn.functional.normalize(weight[: self.config.base_vocab_size].float(), dim=-1)
        appended = nn.functional.normalize(weight[self.config.base_vocab_size :].float(), dim=-1)
        return torch.cat(
            [
                (appended[start : start + 512] @ base.T).topk(mixture, dim=-1).indices
                for start in range(0, appended.shape[0], 512)
            ]
        )

    @torch.no_grad()
    def _fit_mixture(self, weight: torch.Tensor) -> torch.Tensor:
        r"""
        Fits every appended row of a table onto the span of the pretrained rows it mixes.

        This is where the coefficients start, so the run begins at the row the vocabulary expansion
        produced rather than at noise: the fit is the closest point of the allowed span to that row.

        Args:
            weight (`torch.Tensor` of shape `(rows, hidden_size)`):
                The table whose appended rows are fitted.

        Returns:
            `torch.FloatTensor` of shape `(appended_rows, base_row_mixture)`: The coefficients.
        """
        base_vocab = self.config.base_vocab_size
        pretrained = weight[:base_vocab].float()
        eye = torch.eye(self.config.base_row_mixture, device=weight.device)
        fitted = torch.empty(
            self._row_support.shape, dtype=torch.float32, device=weight.device
        )
        for start in range(0, self._row_support.shape[0], self._mixture_chunk):
            stop = start + self._mixture_chunk
            support = pretrained[self._row_support[start:stop]]  # (chunk, mixture, hidden)
            rows = weight[base_vocab + start : base_vocab + stop].float().unsqueeze(-1)
            # Normal equations of the least squares fit of each row onto its own support.
            fitted[start:stop] = torch.linalg.solve(
                support @ support.transpose(1, 2) + 1e-3 * eye, support @ rows
            ).squeeze(-1)
        return fitted

    def appended_rows(self, position: int) -> torch.Tensor:
        r"""
        Builds the appended rows of one table out of its coefficients.

        The value is also written back into the table itself, so that anything reading the rows as
        ordinary values, a save or a fold into the base model, sees what the coefficients currently say
        without having to know that a parameterisation is in the way. That copy is detached; the tensor
        returned here is the one carrying the graph.

        Args:
            position (`int`):
                Index into the tables this adapter trains.

        Returns:
            `torch.FloatTensor` of shape `(appended_rows, hidden_size)`: The rows.
        """
        _, module = self._tables[position]
        start, stop = self.config.base_vocab_size, self.config.vocab_size
        rows = MixtureRows.apply(
            self.row_mixture[0 if self.config.share_row_mixture else position],
            module.weight.data[:start],
            self._row_support,
            self._mixture_chunk,
        )
        if self.partition_direction is not None and module.weight is self.model.get_output_embeddings().weight:
            rows = rows + self.partition_direction
        module.weight.data[start:stop] = rows.detach().to(module.weight.dtype)
        return rows

    def _mixture_part(self, weight: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            weight (`torch.Tensor` of shape `(rows, hidden_size)`):
                A table whose appended rows the mixture is fitted on.

        Returns:
            `torch.Tensor`: The table with the partition direction taken off the head's appended rows,
            which is the part the mixture coefficients hold. Any other table comes back as it is.
        """
        head = self.model.get_output_embeddings().weight
        if self.partition_direction is None or weight.data_ptr() != head.data_ptr():
            return weight
        weight = weight.clone()
        weight[self.config.base_vocab_size : self.config.vocab_size] -= self.partition_direction.to(weight.dtype)
        return weight

    def appended_weight(self) -> torch.Tensor:
        r"""
        Returns:
            `torch.FloatTensor` of shape `(appended_rows, hidden_size)`: The appended rows of the head as
            every read of them has to see them, which under `base_row_mixture` is the mixture rather than
            the slice of the table, since the slice carries no gradient back to the coefficients.
        """
        head = self.model.get_output_embeddings().weight
        if self.row_mixture is not None:
            for position, (_, module) in enumerate(self._tables):
                if module.weight is head:
                    return self.appended_rows(position)
        return head[self.config.base_vocab_size : self.config.vocab_size]

    @torch.no_grad()
    def materialize_rows(self):
        r"""
        Collapses the coefficients of every table into plain row values inside that table.
        """
        if self.row_mixture is None:
            return
        for position in range(len(self._tables)):
            self.appended_rows(position)

    @torch.no_grad()
    def refit_mixture(self):
        r"""
        Re-reads the coefficients off rows that were written into the tables from outside.

        Loading a finished run copies row values in, and under the parameterisation those values are not
        what the model reads: the coefficients are. Fitting them again is what makes a continued run
        start from the rows the checkpoint holds instead of overwriting them at the first forward.
        """
        if self.row_mixture is None:
            return
        if self.config.share_row_mixture:
            # The head is the table the shared coefficients were fitted on, and the others follow it.
            head = self.model.get_output_embeddings().weight.data
            self.row_mixture[0].data.copy_(self._fit_mixture(self._mixture_part(head)))
        else:
            for position, (_, module) in enumerate(self._tables):
                self.row_mixture[position].data.copy_(
                    self._fit_mixture(self._mixture_part(module.weight.data))
                )
        self.materialize_rows()

    def appended_view(self, hidden):
        r"""
        Args:
            hidden (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Backbone states.

        Returns:
            `torch.FloatTensor`: The state the appended rows are scored on, the state itself when the
            head is linear and a residual correction of it when `head_hidden_size` is set. Whatever
            this returns is projected afterwards when `project_new_rows` is set, so that the last thing
            the appended rows read is the part of the state English does not span.
        """
        if self.head_body is None:
            return hidden
        return hidden + self.head_body(hidden.float()).to(hidden.dtype)

    def text_logits(self, hidden):
        r"""
        Args:
            hidden (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Backbone states.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_size)`: The logits, with
            the pretrained rows always reading the state as pretraining left it.
        """
        if self.config.project_new_rows:
            return self.projected_logits(hidden)
        if self.head_body is None and self.row_mixture is None:
            return self.model.lm_head(hidden)

        base = self.config.base_vocab_size
        weight = self.model.get_output_embeddings().weight
        rows = self.appended_weight()
        frozen = hidden.to(weight.dtype) @ weight[:base].T
        appended = self.appended_view(hidden).to(rows.dtype) @ rows.T
        return torch.cat([frozen, appended.to(frozen.dtype)], dim=-1)

    def english_orthogonal(self, hidden):
        r"""
        Args:
            hidden (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Backbone states.

        Returns:
            `torch.FloatTensor`: The part of each state the English subspace does not span.
        """
        centred = hidden.float() - self._english_mean
        return centred - (centred @ self._english_basis) @ self._english_basis.T

    def projected_logits(self, hidden):
        r"""
        Scores the pretrained rows on the whole state and the appended rows on the orthogonal part.

        $$z_i = w_i \cdot h \quad (i \in A), \qquad z_j = w_j \cdot (I - P)h \quad (j \in B)$$

        An English state has almost nothing outside the English subspace, so every appended row scores
        near zero on it and the pretrained rows keep the text to themselves. A Korean state carries a
        real orthogonal part, which is what the appended rows read.

        Args:
            hidden (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Backbone states.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_size)`: The logits.
        """
        base = self.config.base_vocab_size
        weight = self.model.get_output_embeddings().weight
        # The correction is taken before the projection, never after it. A normalising layer handed a
        # vector the projection has already emptied rescales it back to full size, which hands the
        # appended rows an English state again and undoes the whole guarantee.
        outside = self.english_orthogonal(self.appended_view(hidden))

        frozen = hidden.to(weight.dtype) @ weight[:base].T
        if self.shared_residual is not None:
            # The shared rows are corrected on the state itself, not on the head's view of it.
            correction = self.english_orthogonal(hidden) @ self.shared_residual.T
            if self.config.residual_all_frozen_rows:
                frozen = frozen + correction.to(frozen.dtype)
            else:
                frozen = frozen.index_add(-1, self._shared_ids, correction.to(frozen.dtype))

        rows = self.appended_weight()
        appended = outside.to(rows.dtype) @ rows.T
        return torch.cat([frozen, appended.to(frozen.dtype)], dim=-1)

    def apply_shared_residual(self, logits, hidden):
        r"""
        Corrects the rows of the shared tokens on the part of the state English does not span.

        $$w_i^{\text{eff}} = w_i + \delta_i (I - P), \qquad P = UU^\top$$

        $U$ spans the English states, so $(I - P)h$ vanishes on them and an English context keeps the
        logits it was pretrained with. A Korean state keeps the part of itself that lies outside that
        span, which is what the correction reads.

        Args:
            logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_size)`):
                Text logits of the batch.
            hidden (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Backbone states behind those logits.

        Returns:
            `torch.FloatTensor`: The logits with the shared rows corrected.
        """
        centred = hidden.float() - self._english_mean
        outside = centred - (centred @ self._english_basis) @ self._english_basis.T
        correction = (outside @ self.shared_residual.T).to(logits.dtype)
        if self.config.residual_all_frozen_rows:
            corrected = logits.clone()
            corrected[..., : self.config.base_vocab_size] += correction
            return corrected
        return logits.index_add(-1, self._shared_ids, correction)

    def group_log_likelihood(self, logits, hidden, labels) -> torch.Tensor:
        r"""
        Scores a batch under the two group factorization of the vocabulary.

        The pretrained rows cannot move, so a single softmax over the whole vocabulary leaves the model
        no way to say how much probability the new language should hold: raising a new row raises its
        share of the total, and the frozen rows lose theirs whatever the target is. Splitting the
        distribution gives that share its own parameter.

        $$P(i \mid h) = P(g \mid h)\, \frac{\exp z_i}{\sum_{j \in g} \exp z_j}, \qquad
          P(B \mid h) = \sigma(w^\top h + b)$$

        Normalizing inside each group is what makes the pretrained half exact: the relative
        probabilities of the frozen rows no longer depend on anything that is being trained.

        Args:
            logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_size)`):
                Text logits of the batch.
            hidden (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Backbone states the gate reads.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Targets, with `-100` on the positions that are not scored.

        Returns:
            `torch.Tensor`: The mean negative log likelihood over the scored positions.
        """
        base = self.config.base_vocab_size
        keep = labels != -100
        logits, labels, hidden = logits[keep], labels[keep], hidden[keep]

        gate = self.group_gate(hidden.to(self.group_gate.weight.dtype)).squeeze(-1)
        in_new = labels >= base
        # `logsigmoid` on either sign keeps both branches stable at the tails.
        log_group = torch.where(in_new, F.logsigmoid(gate), F.logsigmoid(-gate))

        within = torch.where(
            in_new.unsqueeze(-1),
            F.log_softmax(logits[:, base:].float(), dim=-1).gather(
                1, (labels - base).clamp(min=0).unsqueeze(-1)
            ),
            F.log_softmax(logits[:, :base].float(), dim=-1).gather(
                1, labels.clamp(max=base - 1).unsqueeze(-1)
            ),
        ).squeeze(-1)

        return -(log_group + within).mean()

    def _finish(self, logits, text_labels, decoder_outputs, hidden=None) -> CausalLMOutputWithPast:
        r"""
        Args:
            logits (`torch.FloatTensor`):
                Text logits of the batch.
            text_labels (`torch.LongTensor` or `None`):
                Targets, with `-100` on the positions that are not scored.
            decoder_outputs (`ModelOutput`):
                What the backbone returned.

        Returns:
            `CausalLMOutputWithPast`: The logits and, when targets were given, their cross entropy.
        """
        if self.shared_residual is not None and not self.config.project_new_rows:
            logits = self.apply_shared_residual(logits.float(), hidden)

        loss = None
        if text_labels is not None:
            shifted_logits = logits[:, :-1].float()
            shifted_labels = text_labels[:, 1:].to(shifted_logits.device)
            if self.config.group_gated_softmax:
                loss = self.group_log_likelihood(shifted_logits, hidden[:, :-1], shifted_labels)
            else:
                loss = F.cross_entropy(
                    shifted_logits.reshape(-1, shifted_logits.shape[-1]),
                    shifted_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                ) / (shifted_labels != -100).sum().clamp(min=1)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )

    def penalty(self) -> torch.Tensor:
        r"""
        Returns:
            `torch.Tensor`: Zero. Phase 1 declares no residual, so there is no orthogonal constraint to
            impose. The method exists so that a training loop can read a penalty from either phase.
        """
        reference = self._tables[0][1].weight if self._tables else next(self.model.parameters())
        return torch.zeros((), dtype=torch.float32, device=reference.device)

    def table_parameters(self) -> list:
        r"""
        Returns:
            `list` of `torch.nn.Parameter`: The text tables themselves, trained or not. Weight decay
            reads no gradient and would shrink the pretrained rows these hold, so an optimizer has to
            exempt them whatever the phase is training.
        """
        return [module.weight for _, module in self._tables]

    def adapter_parameters(self) -> list:
        r"""
        Returns:
            `list` of `torch.nn.Parameter`: What the phase actually steps. Under `base_row_mixture` that
            is the mixture coefficients, since the rows are a function of them and hold no parameter of
            their own; otherwise it is the tables carrying the trainable rows.
        """
        tables = (
            list(self.row_mixture)
            if self.row_mixture is not None
            else [module.weight for _, module in self._tables]
        )
        if self.group_gate is not None:
            tables.extend(self.group_gate.parameters())
        if self.partition_direction is not None:
            tables.append(self.partition_direction)
        if self.shared_residual is not None:
            tables.append(self.shared_residual)
        return tables

    def num_trainable_rows(self) -> int:
        r"""
        Returns:
            `int`: Rows the run actually moves, which is the new rows times the number of tables. This is
            what a freeze policy test compares against.
        """
        return self.config.num_new_rows * len(self._tables)

    def gradient_checkpointing_enable(self, **kwargs):
        r"""
        Forwards gradient checkpointing to the base model.
        """
        self.model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self):
        r"""
        Forwards the input gradient hook of the base model, needed with gradient checkpointing.
        """
        self.model.enable_input_require_grads()

    # -----------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------

    def _save_pretrained(self, save_directory: Path):
        r"""
        Writes the appended row slices and whatever else the phase trained. The frozen base is not
        duplicated. A run that trains a correction the tables do not hold, the shared token residual or
        the group gate, is only reproducible if that correction travels with the rows.

        This is the hook `ModelHubMixin.save_pretrained` calls once it has created the directory and
        written `config.json`, so the configuration round trip and the upload are not repeated here.

        Args:
            save_directory (`Path`):
                Directory the weights are written to.
        """
        # Whatever parameterisation the run used, what is written is the row values it produced, so a
        # checkpoint of this phase is indistinguishable from one taken without it.
        self.materialize_rows()
        start, stop = self.config.base_vocab_size, self.config.vocab_size
        tensors = {
            name: module.weight.data[start:stop].detach().cpu().contiguous()
            for name, module in self._tables
        }
        if self.partition_direction is not None:
            tensors["__partition_direction__"] = self.partition_direction.data.detach().cpu().contiguous()
        if self.shared_residual is not None:
            tensors["__shared_residual__"] = self.shared_residual.data.detach().cpu().contiguous()
        for index, module in self.confined.items():
            tensors[f"__confined__{index}__up"] = module.up.data.detach().cpu().contiguous()
            if isinstance(module.basis, nn.Parameter):
                tensors[f"__confined__{index}__basis"] = module.basis.data.detach().cpu().contiguous()
        for key, module in self.lora.items():
            tensors[f"__lora__{key}__down"] = module.down.data.detach().cpu().contiguous()
            tensors[f"__lora__{key}__up"] = module.up.data.detach().cpu().contiguous()
        if self.head_body is not None:
            for name, tensor in self.head_body.state_dict().items():
                tensors[f"__head_body__{name}"] = tensor.detach().cpu().contiguous()
        if self.group_gate is not None:
            tensors["__group_gate_weight__"] = self.group_gate.weight.data.detach().cpu().contiguous()
            tensors["__group_gate_bias__"] = self.group_gate.bias.data.detach().cpu().contiguous()
        save_file(tensors, os.path.join(save_directory, TEXT_ADAPTER_WEIGHTS_NAME))

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str | None = None,
        cache_dir: str | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        config: EfficientAdapterP1Config | None = None,
        model=None,
        **kwargs,
    ) -> "EfficientAdapterP1Model":
        r"""
        Writes stored row slices back into the tables of an expanded base model.

        This is the hook `ModelHubMixin.from_pretrained` calls once it has read `config.json` and decoded
        it back into an [`EfficientAdapterP1Config`], so a local directory and a Hub repository reach the
        same code.

        Args:
            model_id (`str`):
                Local directory, or the `owner/name` of a Hub repository.
            revision (`str`, *optional*):
                Branch, tag or commit to download.
            cache_dir (`str`, *optional*):
                Directory the download is cached in.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether to download again over a cached file.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to refuse to reach the network.
            token (`str` or `bool`, *optional*):
                Hub token, defaults to the locally stored login.
            config (`EfficientAdapterP1Config`, *optional*):
                Decoded configuration, supplied by the mixin.
            model (`PreTrainedModel`):
                The expanded duplex model the rows are written into, modified in place.

        Returns:
            `EfficientAdapterP1Model`: The assembled model.

        Raises:
            ValueError: If `model` is missing, or a stored slice does not match the table it goes into.
        """
        if model is None:
            raise ValueError(
                "Phase 1 stores row slices, not a whole model, so loading needs the expanded base model "
                "to write them into. Pass `model=...`."
            )

        if os.path.isdir(model_id):
            weights_path = os.path.join(model_id, TEXT_ADAPTER_WEIGHTS_NAME)
        else:
            weights_path = hf_hub_download(
                repo_id=model_id,
                filename=TEXT_ADAPTER_WEIGHTS_NAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
            )

        assembled = cls(model, config)
        tensors = load_file(weights_path)

        start, stop = config.base_vocab_size, config.vocab_size
        for name, module in assembled._tables:
            rows = tensors[name]
            target = module.weight.data[start:stop]
            if rows.shape != target.shape:
                raise ValueError(
                    f"Stored rows for '{name}' have shape {tuple(rows.shape)} but the table expects "
                    f"{tuple(target.shape)}. The checkpoint and the expansion disagree."
                )
            target.copy_(rows.to(target.device, target.dtype))
        if assembled.partition_direction is not None and "__partition_direction__" in tensors:
            # Read before the refit, which takes it off the stored rows to recover the mixture.
            assembled.partition_direction.data.copy_(
                tensors["__partition_direction__"].to(assembled.partition_direction.device)
            )
        assembled.refit_mixture()

        for index, module in assembled.confined.items():
            if f"__confined__{index}__up" in tensors:
                module.up.data.copy_(tensors[f"__confined__{index}__up"].to(module.up.device))
            if f"__confined__{index}__basis" in tensors:
                module.basis.data.copy_(tensors[f"__confined__{index}__basis"].to(module.basis.device))
        if assembled.shared_residual is not None and "__shared_residual__" in tensors:
            assembled.shared_residual.data.copy_(
                tensors["__shared_residual__"].to(assembled.shared_residual.device)
            )
        if assembled.group_gate is not None and "__group_gate_weight__" in tensors:
            assembled.group_gate.weight.data.copy_(
                tensors["__group_gate_weight__"].to(assembled.group_gate.weight.device)
            )
            assembled.group_gate.bias.data.copy_(
                tensors["__group_gate_bias__"].to(assembled.group_gate.bias.device)
            )
        return assembled


__all__ = [
    "TEXT_ADAPTER_WEIGHTS_NAME",
    "EfficientAdapterP1Model",
    "MixtureRows",
    "protect_leading_rows",
    "text_tables",
]
