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
"""The trainer the phases are built on. This module is a library, it has no command line."""

from __future__ import annotations

from transformers import Trainer as HuggingFaceTrainer
from transformers import TrainingArguments


class Trainer(HuggingFaceTrainer):
    r"""
    Base of the phase trainers, holding what makes a phase a phase.

    A phase is the pairing of one adapter with one dataset, and getting either wrong is silent: a bare
    model trains every weight instead of the intended slice, and the other phase's dataset yields fields
    the model never reads, so the run looks healthy and learns nothing. Both are therefore checked here
    rather than left to each subclass.

    A subclass states its pairing through `adapter_class` and `dataset_class`, its supervision through
    `label_names`, and carries its own assembly defaults so that a run names a checkpoint rather than
    building an adapter first.

    Args:
        model (`torch.nn.Module`):
            The assembled adapter model of the phase.
        args (`TrainingArguments`, *optional*):
            Standard training configuration.
        train_dataset (`torch.utils.data.Dataset`, *optional*):
            The dataset of the phase.
    """

    adapter_class = None
    dataset_class = None
    label_names = []

    def _check_model(self, model):
        r"""
        Args:
            model (`torch.nn.Module`):
                The model handed to the trainer.

        Raises:
            TypeError: If it is not the adapter of this phase.
        """
        if not isinstance(model, self.adapter_class):
            raise TypeError(
                f"`{type(self).__name__}` drives a `{self.adapter_class.__name__}`, got "
                f"{type(model).__name__}. Training anything else here would move the wrong weights."
            )

    def _check_dataset(self, dataset):
        r"""
        Args:
            dataset (`torch.utils.data.Dataset` or `None`):
                The dataset handed to the trainer.

        Raises:
            TypeError: If it is not the dataset of this phase.
        """
        if dataset is not None and not isinstance(dataset, self.dataset_class):
            raise TypeError(
                f"`{type(self).__name__}` reads a `{self.dataset_class.__name__}`, got "
                f"{type(dataset).__name__}. Build it with `{self.dataset_class.__name__}(...)` or load a "
                f"published one with `{self.dataset_class.__name__}.from_pretrained(...)`."
            )

    def _prepare(self, model, args: TrainingArguments | None):
        r"""
        Applies the training arguments that have to reach the model before the trainer sees it.

        Args:
            model (`torch.nn.Module`):
                The assembled adapter model.
            args (`TrainingArguments` or `None`):
                Standard training configuration.
        """
        if args is not None:
            args.label_names = list(self.label_names)
            if args.gradient_checkpointing:
                model.gradient_checkpointing_enable()
                model.enable_input_require_grads()


__all__ = ["Trainer"]
