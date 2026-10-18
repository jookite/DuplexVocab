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

import logging
import math
import os
import signal

import torch
import torch.nn.functional as F
from transformers import Trainer as HuggingFaceTrainer
from transformers import TrainerCallback, TrainingArguments
from transformers.trainer_pt_utils import LengthGroupedSampler


logger = logging.getLogger(__name__)

SHUTDOWN_SIGNALS = (signal.SIGINT, signal.SIGTERM)


class GracefulShutdownCallback(TrainerCallback):
    r"""
    Turns a termination signal into the stop flag the training loop already honours.

    A signal handler cannot save from where it runs: it interrupts the process between two bytecodes,
    which may be inside the backward pass or halfway through a checkpoint write. It therefore only
    records the request, and this callback converts it at the next optimizer step, the one point where
    the weights, the optimizer state and the step counter agree with each other.

    `should_save` is raised together with the stop flag so the interrupted run leaves a checkpoint
    behind rather than only the slice the caller writes after `train` returns.
    """

    def __init__(self):
        self.requested = False

    def on_step_end(self, args, state, control, **kwargs):
        r"""
        Args:
            args (`TrainingArguments`):
                Configuration of the run.
            state (`TrainerState`):
                State of the loop at this step.
            control (`TrainerControl`):
                Flags the loop reads after the callback returns.

        Returns:
            `TrainerControl`: The flags, with the stop and save requests raised when a signal arrived.
        """
        if self.requested:
            control.should_training_stop = True
            control.should_save = True
        return control


class ResearchTreeCallback(TrainerCallback):
    r"""
    Constructs a callback that writes a finished run's result into its ResearchTree experiment pull request.

    The pull request of the `experiment/*` branch the run was started from is the lab note. At the end of
    training, the metrics of the last evaluation go into its `metrics` field and the W&B run link into its
    `wandb` field. Curves stay on W&B. Only the main process writes, and a run outside an experiment branch,
    without a token or without the `researchtree` package writes nothing and does not fail.

    Args:
        log (`Callable`, *optional*):
            Receives the final metrics as keyword arguments. Defaults to `researchtree.log`.
        set_fields (`Callable`, *optional*):
            Receives pull request fields as keyword arguments. Defaults to `researchtree.set`.
        wandb_url (`Callable`, *optional*):
            Returns the URL of the active W&B run, or `None`. Defaults to reading `wandb.run`.
    """

    def __init__(self, log=None, set_fields=None, wandb_url=None):
        self.log = log
        self.set_fields = set_fields
        self.wandb_url = wandb_url or _active_wandb_url
        self.metrics = {}

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        r"""
        Args:
            args (`TrainingArguments`):
                Configuration of the run.
            state (`TrainerState`):
                State of the loop.
            control (`TrainerControl`):
                Flags the loop reads after the callback returns.
            metrics (`dict`, *optional*):
                Metrics of the evaluation that just finished.
        """
        if state.is_world_process_zero and metrics:
            self.metrics = {
                key: float(value)
                for key, value in metrics.items()
                if key.startswith("eval_") and isinstance(value, (int, float))
            }

    def on_train_end(self, args, state, control, **kwargs):
        r"""
        Args:
            args (`TrainingArguments`):
                Configuration of the run.
            state (`TrainerState`):
                State of the loop.
            control (`TrainerControl`):
                Flags the loop reads after the callback returns.
        """
        if not state.is_world_process_zero:
            return
        log, set_fields = self.log, self.set_fields
        if log is None or set_fields is None:
            try:
                import researchtree
            except ImportError:
                logger.warning("researchtree is not installed; the experiment pull request is not updated.")
                return
            log = log or researchtree.log
            set_fields = set_fields or researchtree.set
        if self.metrics:
            log(**self.metrics)
        url = self.wandb_url()
        if url:
            set_fields(wandb=url)


class RunRecordCallback(TrainerCallback):
    r"""
    Constructs a callback that adds what the trainer does not know to the W&B run's configuration.

    The trainer records its own arguments. The starting checkpoint, the data, the phase options and the
    commit the run was started from are this project's, and are written once training begins, after the
    W&B callback has opened the run. A run without W&B is left alone.

    Args:
        record (`dict`):
            Values to add, keyed as they should appear in the configuration.
        active_run (`Callable`, *optional*):
            Returns the active W&B run, or `None`. Defaults to reading `wandb.run`.
    """

    def __init__(self, record: dict, active_run=None):
        self.record = dict(record)
        self.active_run = active_run or _active_wandb_run

    def on_train_begin(self, args, state, control, **kwargs):
        r"""
        Args:
            args (`TrainingArguments`):
                Configuration of the run.
            state (`TrainerState`):
                State of the loop.
            control (`TrainerControl`):
                Flags the loop reads after the callback returns.
        """
        run = self.active_run()
        if state.is_world_process_zero and run is not None:
            run.config.update(self.record, allow_val_change=True)


def git_record(path: str | None = None) -> dict:
    r"""
    Args:
        path (`str`, *optional*):
            A directory inside the checkout, the current directory by default.

    Returns:
        `dict`: `record/git_commit`, `record/git_branch` and `record/git_dirty` (uncommitted changes to tracked
        files), or an empty dict outside a git checkout.
    """
    import subprocess

    def git(*arguments):
        return subprocess.run(["git", *arguments], cwd=path, capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {
            "record/git_commit": git("rev-parse", "HEAD"),
            "record/git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "record/git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {}


def _active_wandb_run():
    r"""
    Returns:
        `wandb.sdk.wandb_run.Run` or `None`: The W&B run of this process, `None` when W&B is not in use.
    """
    try:
        import wandb
    except ImportError:
        return None
    return wandb.run


def _active_wandb_url():
    r"""
    Returns:
        `str` or `None`: The URL of the W&B run of this process, `None` when W&B is not in use.
    """
    try:
        import wandb
    except ImportError:
        return None
    return wandb.run.url if wandb.run is not None else None


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

    def _save(self, output_dir: str | None = None, state_dict=None):
        r"""
        Writes the slice the phase trains instead of the whole model.

        The base implementation serializes every weight, which for a phase is the frozen backbone
        written again at each checkpoint, and which fails outright once that backbone is loaded under
        weight-only quantization, since the packed tensors hold no storage safetensors can point at.
        The adapter already knows which slice moved and writes only that.

        Args:
            output_dir (`str`, *optional*):
                Directory the checkpoint is written to, defaulting to the run's output directory.
            state_dict (`dict`, *optional*):
                Ignored. The adapter reads its own slice rather than a full state dict.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    @staticmethod
    def token_group_metrics(base_vocab_size: int):
        r"""
        Builds the pair of callables that report the loss of each half of the vocabulary.

        A single number over the whole vocabulary is dominated by whichever half is more frequent, and
        in this corpus that is the punctuation the pretrained rows already own. Splitting it is what
        shows whether the new language is being learned or whether the spaces are just being counted.

        The logits are reduced to one loss per position before anything is gathered, since a vocabulary
        this wide would otherwise be held in memory for the whole split.

        Args:
            base_vocab_size (`int`):
                First row of the appended vocabulary. A target below it belongs to the pretrained half.

        Returns:
            `tuple`: The `preprocess_logits_for_metrics` and `compute_metrics` callables.
        """

        def reduce_logits(logits, labels):
            if isinstance(logits, tuple):
                logits = logits[0]
            shifted = logits[:, :-1].float()
            targets = labels[:, 1:]
            losses = F.cross_entropy(
                shifted.reshape(-1, shifted.shape[-1]),
                targets.reshape(-1).clamp(min=0),
                reduction="none",
            ).view(targets.shape)
            return torch.stack([losses, targets.float()], dim=-1)

        def compute(prediction):
            stacked = torch.as_tensor(prediction.predictions)
            losses, targets = stacked[..., 0].flatten(), stacked[..., 1].flatten()
            labels = torch.as_tensor(prediction.label_ids)[:, 1:].flatten()
            scored = labels != -100
            losses, targets = losses[scored], targets[scored]

            appended = targets >= base_vocab_size
            metrics = {}
            for name, mask in (("all", torch.ones_like(appended)), ("new", appended), ("base", ~appended)):
                if bool(mask.any()):
                    value = losses[mask].mean().item()
                    metrics[f"{name}_loss"] = value
                    metrics[f"{name}_ppl"] = math.exp(min(value, 30.0))
            return metrics

        return reduce_logits, compute

    def evaluate(self, *args, **kwargs):
        r"""
        Runs the standard evaluation and reports the perplexity next to the loss.

        A phase reports a cross entropy in nats, which says little on its own about how many tokens the
        model is really choosing between. The exponential is the number a held out split is read by, and
        it is what tells a run whether more epochs are still buying anything.

        Args:
            *args:
                Forwarded to `transformers.Trainer.evaluate`.
            **kwargs:
                Forwarded to `transformers.Trainer.evaluate`.

        Returns:
            `dict`: The standard metrics, with the perplexity of each reported loss added.
        """
        metrics = super().evaluate(*args, **kwargs)
        for key in [k for k in metrics if k.endswith("_loss")]:
            # A diverged run reports a loss the exponential cannot represent, so it is capped.
            metrics[key[: -len("_loss")] + "_perplexity"] = math.exp(min(metrics[key], 30.0))
        return metrics

    def train(self, *args, **kwargs):
        r"""
        Runs the training loop, stopping at the next step boundary when a termination signal arrives.

        A phase trains a slice that is only written once the loop returns, so a killed run loses
        everything since the last checkpoint. Both signals are caught for the duration of the loop and
        handed to a [`GracefulShutdownCallback`]. Any further signal restores the handlers that were
        installed before and is raised again against them, which is what keeps a hung run killable.

        Args:
            *args:
                Forwarded to `transformers.Trainer.train`.
            **kwargs:
                Forwarded to `transformers.Trainer.train`.

        Returns:
            `TrainOutput`: What the standard loop returns.
        """
        callback = GracefulShutdownCallback()
        previous = {}

        def handle(signum, frame):
            if callback.requested:
                restore()
                signal.raise_signal(signum)
                return
            callback.requested = True
            logger.warning(
                "Received %s, stopping after the current step and saving. Signal again to abort now.",
                signal.Signals(signum).name,
            )

        def restore():
            while previous:
                signum, handler = previous.popitem()
                signal.signal(signum, handler)

        for signum in SHUTDOWN_SIGNALS:
            try:
                previous[signum] = signal.signal(signum, handle)
            except ValueError:
                # Only the main thread may install handlers, a worker just runs without them.
                previous.pop(signum, None)

        self.add_callback(callback)
        try:
            return super().train(*args, **kwargs)
        finally:
            restore()
            self.remove_callback(callback)

    def _get_train_sampler(self, train_dataset=None):
        r"""
        Groups a batch by length when asked to, reading the lengths off the phase dataset.

        The standard sampler infers lengths by indexing the dataset and reading `input_ids`, which a
        phase dataset does not hold: it yields the example a collator later turns into tokens or
        frames. The dataset reports its own lengths instead, so the strategy works here at all.

        Args:
            train_dataset (`torch.utils.data.Dataset`, *optional*):
                The dataset to sample from, defaulting to the one the trainer holds.

        Returns:
            `torch.utils.data.Sampler`: The length grouped sampler, or whatever the standard
            selection returns for the other strategies.
        """
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        strategy = getattr(self.args, "train_sampling_strategy", None)
        if strategy != "group_by_length" or not hasattr(dataset, "example_lengths"):
            return super()._get_train_sampler(train_dataset)

        # A text dataset measures an example by tokenizing it, a duplex one already knows its frame
        # count, so the tokenizer is offered and not imposed.
        try:
            lengths = dataset.example_lengths(self.processing_class)
        except TypeError:
            lengths = dataset.example_lengths()
        logger.info(
            "Length grouped batches over %d examples, %d to %d long.",
            len(lengths),
            min(lengths),
            max(lengths),
        )
        return LengthGroupedSampler(
            self.args.train_batch_size * self.args.gradient_accumulation_steps,
            lengths=lengths,
        )

    def _get_eval_sampler(self, eval_dataset=None):
        r"""
        Groups the held out split by length as well, for the same reason the training split is.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                The split being evaluated.

        Returns:
            `torch.utils.data.Sampler`: The length grouped sampler, or whatever the standard selection
            returns for the other strategies.
        """
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        strategy = getattr(self.args, "train_sampling_strategy", None)
        if strategy != "group_by_length" or not hasattr(dataset, "example_lengths"):
            return super()._get_eval_sampler(eval_dataset)
        try:
            lengths = dataset.example_lengths(self.processing_class)
        except TypeError:
            lengths = dataset.example_lengths()
        return LengthGroupedSampler(self.args.eval_batch_size, lengths=lengths)

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
        Freezes the base and applies the training arguments that have to reach the model before the
        trainer sees it.

        A model handed in already assembled has not necessarily been frozen, and an unfrozen base trains
        the whole backbone silently. Freezing here covers both paths, and is idempotent.

        Args:
            model (`torch.nn.Module`):
                The assembled adapter model.
            args (`TrainingArguments` or `None`):
                Standard training configuration.
        """
        model.freeze_base()
        if args is not None:
            args.label_names = list(self.label_names)
            if args.gradient_checkpointing:
                model.gradient_checkpointing_enable()
                model.enable_input_require_grads()


__all__ = ["GracefulShutdownCallback", "ResearchTreeCallback", "RunRecordCallback", "Trainer", "git_record"]
