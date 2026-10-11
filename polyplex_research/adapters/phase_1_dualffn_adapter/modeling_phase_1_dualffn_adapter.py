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
"""PyTorch Phase 1 DualFFN adapter, the text adaptation of one language on a DualFFN model."""

import os
from dataclasses import asdict, fields
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import ModelHubMixin, hf_hub_download
from safetensors.torch import load_file, save_file

from .configuration_phase_1_dualffn_adapter import Phase1DualFFNAdapterConfig


DUALFFN_ADAPTER_WEIGHTS_NAME = "dualffn_slices.safetensors"

_CONFIG_FIELDS = {field.name for field in fields(Phase1DualFFNAdapterConfig)}


class DuplexSlice(nn.Module):
    r"""
    Constructs the trainable copy of one language's duplex FFN neurons in one layer.

    The neurons compute what they would inside the duplex FFN, $\mathrm{down}(\mathrm{SiLU}(\mathrm{gate}\,x)
    \odot \mathrm{up}\,x)$, on the same normalized input. Holding them apart from the frozen projections keeps
    the optimizer, weight decay included, off every other neuron.

    Args:
        gate (`torch.Tensor` of shape `(size, hidden_size)`):
            Initial gate rows.
        up (`torch.Tensor` of shape `(size, hidden_size)`):
            Initial up rows.
        down (`torch.Tensor` of shape `(hidden_size, size)`):
            Initial down columns.
    """

    def __init__(self, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor):
        super().__init__()
        self.gate = nn.Parameter(gate.detach().float().clone())
        self.up = nn.Parameter(up.detach().float().clone())
        self.down = nn.Parameter(down.detach().float().clone())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden = hidden_states.float()
        neurons = F.silu(hidden @ self.gate.T) * (hidden @ self.up.T)
        return (neurons @ self.down.T).to(hidden_states.dtype)


class Phase1DualFFNAdapterModel(
    nn.Module,
    ModelHubMixin,
    library_name="polyplex-research",
    tags=["polyplex", "phase-1", "dual-ffn"],
    coders={
        Phase1DualFFNAdapterConfig: (
            asdict,
            lambda stored: Phase1DualFFNAdapterConfig(
                **{name: value for name, value in stored.items() if name in _CONFIG_FIELDS}
            ),
        )
    },
):
    r"""
    Constructs the Phase 1 adapter of one language on a [`PolyplexForConditionalGeneration`].

    Every weight of the model is frozen. A language that holds no reserved duplex FFN neurons yet is first handed the
    next `duplex_slice_size` of them: their gate and up rows are drawn with the spread of the pretrained rows of the
    same projection, and their down columns start at zero, so the model's output does not move and the down columns
    receive a gradient at once. The slice is recorded in the model's `duplex_ffn_languages`. What trains is the
    language's slice in every layer and, when `new_tokens` is set, the language's rows of the input embedding and
    of the LM head. Each of these is moved out of the model into a parameter of its own, and the place it held in the model is
    set to zero, so the frozen projections can be quantized once and the trained part is computed beside them
    by forward hooks. The model computes exactly what it did before wrapping; [`~fold_into_base`] writes the
    trained values back.

    Args:
        model ([`PolyplexForConditionalGeneration`]):
            The DualFFN model, modified in place.
        config ([`Phase1DualFFNAdapterConfig`]):
            What to train.

    Raises:
        ValueError: If a new language comes without `duplex_slice_size`, or the reserved neurons left are too few.
    """

    def __init__(self, model, config: Phase1DualFFNAdapterConfig):
        super().__init__()
        languages = model.config.duplex_ffn_languages
        handed = config.language not in languages
        if handed:
            if config.duplex_slice_size is None:
                raise ValueError(f"Language {config.language!r} holds no reserved neurons; set `duplex_slice_size`.")
            start = model.config.reserved_end
            end = start + config.duplex_slice_size
            if end > model.config.intermediate_size:
                raise ValueError(
                    f"Language {config.language!r} asks for {config.duplex_slice_size} neurons, but only "
                    f"{model.config.intermediate_size - start} reserved neurons are left."
                )
            languages[config.language] = [start, end]
        self.model = model
        self.config = config
        self.neurons = tuple(languages[config.language])
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        self._handles = []
        self.slices = nn.ModuleList()
        if config.train_duplex_slice:
            start, end = self.neurons
            generator = torch.Generator().manual_seed(config.seed)
            pretrained_size = model.config.duplex_pretrained_size
            for layer in model.model.layers:
                mlp = layer.mlp.duplex_mlp
                gate, up = mlp.gate_proj.weight[start:end], mlp.up_proj.weight[start:end]
                if handed:
                    gate_std = mlp.gate_proj.weight[:pretrained_size].float().std()
                    up_std = mlp.up_proj.weight[:pretrained_size].float().std()
                    gate = torch.randn(gate.shape, generator=generator) * gate_std
                    up = torch.randn(up.shape, generator=generator) * up_std
                piece = DuplexSlice(gate, up, mlp.down_proj.weight[:, start:end]).to(mlp.gate_proj.weight.device)
                with torch.no_grad():
                    mlp.gate_proj.weight[start:end] = 0
                    mlp.up_proj.weight[start:end] = 0
                    mlp.down_proj.weight[:, start:end] = 0
                self.slices.append(piece)
                self._handles.append(mlp.register_forward_hook(self._slice_hook(piece)))

        self.input_rows = self.output_rows = None
        if config.new_tokens is not None:
            start, end = config.new_tokens
            embedding = model.model.embed_tokens
            head = model.lm_head
            self.input_rows = nn.Parameter(embedding.weight[start:end].detach().float().clone())
            self.output_rows = nn.Parameter(head.weight[start:end].detach().float().clone())
            self._handles.append(embedding.register_forward_hook(self._lookup_hook))
            self._handles.append(head.register_forward_hook(self._logits_hook))

    @staticmethod
    def _slice_hook(piece: DuplexSlice):
        def hook(module, inputs, output):
            return output + piece(inputs[0])

        return hook

    def _lookup_hook(self, module, inputs, output):
        start, end = self.config.new_tokens
        ids = inputs[0]
        inside = (ids >= start) & (ids < end)
        rows = self.input_rows[(ids - start).clamp(0, end - start - 1)].to(output.dtype)
        return torch.where(inside.unsqueeze(-1), rows, output)

    def _logits_hook(self, module, inputs, output):
        start, end = self.config.new_tokens
        logits = inputs[0].float() @ self.output_rows.T
        return torch.cat([output[..., :start], logits.to(output.dtype), output[..., end:]], dim=-1)

    def forward(self, **inputs):
        r"""
        Runs the wrapped model.

        Args:
            **inputs:
                The inputs [`PolyplexForConditionalGeneration.forward`] takes, `text_labels` and `audio_labels`
                included.

        Returns:
            [`PersonaPlexConditionalGenerationOutputWithPast`]: The model's output, its loss computed with the
            trained slice and rows.
        """
        return self.model(**inputs)

    @torch.no_grad()
    def fold_into_base(self):
        r"""
        Writes the trained slice and rows back into the model and removes the hooks.

        Returns:
            [`PolyplexForConditionalGeneration`]: The model, which now computes on its own what the adapter
            computed.
        """
        for handle in self._handles:
            handle.remove()
        self._handles = []
        start, end = self.neurons
        for layer, piece in zip(self.model.model.layers, self.slices):
            mlp = layer.mlp.duplex_mlp
            mlp.gate_proj.weight[start:end] = piece.gate.to(mlp.gate_proj.weight.dtype)
            mlp.up_proj.weight[start:end] = piece.up.to(mlp.up_proj.weight.dtype)
            mlp.down_proj.weight[:, start:end] = piece.down.to(mlp.down_proj.weight.dtype)
        if self.config.new_tokens is not None:
            start, end = self.config.new_tokens
            embedding = self.model.model.embed_tokens.weight
            head = self.model.lm_head.weight
            embedding[start:end] = self.input_rows.to(embedding.dtype)
            head[start:end] = self.output_rows.to(head.dtype)
        return self.model

    def _save_pretrained(self, save_directory: Path):
        r"""
        Writes the trained slices and rows. The frozen model is not duplicated.

        Args:
            save_directory (`Path`):
                Directory the weights are written to, after the mixin has written `config.json`.
        """
        tensors = {}
        for index, piece in enumerate(self.slices):
            for name in ("gate", "up", "down"):
                tensors[f"slices.{index}.{name}"] = getattr(piece, name).detach().cpu().contiguous()
        if self.input_rows is not None:
            tensors["input_rows"] = self.input_rows.detach().cpu().contiguous()
            tensors["output_rows"] = self.output_rows.detach().cpu().contiguous()
        save_file(tensors, os.path.join(save_directory, DUALFFN_ADAPTER_WEIGHTS_NAME))

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
        config: Phase1DualFFNAdapterConfig | None = None,
        model=None,
        **kwargs,
    ) -> "Phase1DualFFNAdapterModel":
        r"""
        Wraps a DualFFN model and loads the stored slices and rows into the adapter.

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
            config (`Phase1DualFFNAdapterConfig`, *optional*):
                Decoded configuration, supplied by the mixin.
            model ([`PolyplexForConditionalGeneration`]):
                The DualFFN model the adapter was trained on, modified in place.

        Returns:
            `Phase1DualFFNAdapterModel`: The adapter with its trained values.

        Raises:
            ValueError: If `model` is missing, or a stored tensor does not match the adapter.
        """
        if model is None:
            raise ValueError("The adapter stores slices, not a model, so loading needs `model=...`.")
        if os.path.isdir(model_id):
            weights_path = os.path.join(model_id, DUALFFN_ADAPTER_WEIGHTS_NAME)
        else:
            weights_path = hf_hub_download(
                repo_id=model_id,
                filename=DUALFFN_ADAPTER_WEIGHTS_NAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
            )
        adapter = cls(model, config)
        tensors = load_file(weights_path)
        own = {name: parameter for name, parameter in adapter.named_parameters() if parameter.requires_grad}
        if set(tensors) != set(own):
            raise ValueError(f"Stored tensors {sorted(tensors)} do not match the adapter's {sorted(own)}.")
        with torch.no_grad():
            for name, parameter in own.items():
                if tensors[name].shape != parameter.shape:
                    raise ValueError(
                        f"Stored '{name}' has shape {tuple(tensors[name].shape)}, the adapter expects "
                        f"{tuple(parameter.shape)}."
                    )
                parameter.copy_(tensors[name].to(parameter.device, parameter.dtype))
        return adapter


__all__ = ["DUALFFN_ADAPTER_WEIGHTS_NAME", "DuplexSlice", "Phase1DualFFNAdapterModel"]
