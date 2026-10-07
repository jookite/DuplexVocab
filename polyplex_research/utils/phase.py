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
"""One `Trainer` per training phase, each assembling its own model and accepting only its own dataset."""

from __future__ import annotations

import contextlib
import inspect
import logging
import json
import math
import os
import re
import shutil
import sys
from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from safetensors.torch import load_file

from transformers import (
    EarlyStoppingCallback,
    HfArgumentParser,
    TrainingArguments,
)

from ..adapters.phase_1_efficient_adapter import TEXT_ADAPTER_WEIGHTS_NAME, EfficientAdapterP1Model
from ..adapters.phase_2_code_rotation_adapter import CodeRotationAdapterModel
from ..datasets.duplex_data import EPAD_TOKEN_ID, PAD_TOKEN_ID, dense_collate_fn, duplex_collate_fn
from ..datasets.webtext import (
    DENSE_EVALUATION,
    MultilingualEduCorpus,
    dense_repositories,
    fineweb2_bytes,
    language_key,
    text_key,
)
from ..datasets.phase_data import (
    Phase1Dataset,
    Phase2Dataset,
    phase1_collate_fn,
    phase2_collate_fn,
)
from .assemble import Phase1Assemble, Phase2Assemble, runs_on_ephemeral_disk
from .optimization import MuonWithAdamW
from .quantization import set_nvfp4_backward
from .train import ResearchTreeCallback, RunRecordCallback, Trainer, git_record


logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "nvidia/personaplex-7b-v1"


def load_english_states(path: str) -> tuple:
    r"""
    Reads the cached English states the preservation term is scored on.

    The states are cached rather than recomputed because the backbone is frozen for the whole phase, so
    the state a given English frame produces is the same tensor at every step. Running the backbone
    again each step would buy nothing and would cost more than the Korean batch it is added to. With the
    states fixed and the pretrained rows fixed, what the term charges is a function of the appended rows
    alone, which is exactly what the phase trains.

    Args:
        path (`str`):
            A safetensors file holding `states` of shape `(num_frames, hidden_size)`, the backbone state
            at each scored English frame, and `labels` of shape `(num_frames,)`, the text id read at that
            frame. Written by [`write_english_states`].

    Returns:
        `tuple`: The states and their labels.

    Raises:
        ValueError: If either tensor is missing, carries the wrong rank, or the two disagree in length.
    """
    stored = load_file(path)
    missing = [name for name in ("states", "labels") if name not in stored]
    if missing:
        raise ValueError(
            f"{path} holds {sorted(stored)} and is missing {missing}. Build the cache with "
            "`python -m polyplex_research.utils.evaluate english-states`."
        )

    states, labels = stored["states"], stored["labels"]
    if states.ndim != 2 or labels.ndim != 1:
        raise ValueError(
            f"{path} holds states of rank {states.ndim} and labels of rank {labels.ndim}, where one row "
            "per frame and one label per frame are expected."
        )
    if states.shape[0] != labels.shape[0]:
        raise ValueError(
            f"{path} holds {states.shape[0]} states and {labels.shape[0]} labels. Every state is the "
            "frame its own label is read at, so a cache whose halves disagree pairs states with labels "
            "from other frames and scores a distribution nothing was measured on."
        )
    return states, labels.long()


@torch.no_grad()
def collect_english_states(model, collate, items: list, batch_size: int, device: str) -> tuple:
    r"""
    Runs the backbone over English utterances and keeps the state behind every scored frame.

    The state kept for a frame is the one the head reads to predict it, so the batch is shifted the way the
    loss shifts it: position `t` carries the label of position `t + 1`. The backbone reads the sum of the text
    stream and both acoustic ones, which is the layout the checkpoint is read in; a state measured on text
    alone is one deployment never produces.

    Args:
        model (`PolyplexForConditionalGeneration`):
            The expanded checkpoint, in evaluation mode on `device`.
        collate (`Callable`):
            The duplex collator the states are laid out by.
        items (`list`):
            Utterances in the form the collator reads.
        batch_size (`int`):
            How many utterances are run at a time.
        device (`str`):
            Device the backbone runs on.

    Returns:
        `tuple`: The states of shape `(num_frames, hidden_size)` in float16 and their labels.
    """
    states, labels = [], []
    for start in range(0, len(items), batch_size):
        batch = {key: value.to(device) for key, value in collate(items[start : start + batch_size]).items()}
        hidden = model.model(
            inputs_embeds=model.model.embed_tokens(batch["input_ids"])
            + model._embed_audio_codes(torch.cat([batch["assistant_audio_codes"], batch["user_audio_codes"]], dim=1)),
            attention_mask=batch["attention_mask"],
        ).last_hidden_state
        target = batch["text_labels"][:, 1:]
        keep = target != -100
        states.append(hidden[:, :-1][keep].half().cpu())
        labels.append(target[keep].cpu())
    return torch.cat(states), torch.cat(labels)


def write_english_states(
    model_dir: str,
    corpus_path: str,
    output: str,
    batch_size: int = 8,
    limit: int = 0,
    device: str = "cuda",
    persona: str = "You are a helpful assistant.",
    score_all_frames: bool = False,
    drop_audio: bool = False,
) -> int:
    r"""
    Writes the English state cache the preservation term reads (`english_states_path`).

    Args:
        model_dir (`str`):
            Expanded checkpoint the states come from.
        corpus_path (`str`):
            A `torch.save` of English utterances carrying `text` and their own Mimi `codes`. The corpus has no
            word times, so the words are packed at the front of the stream ([`packed_words`]).
        output (`str`):
            Safetensors file written, holding `states` (float16 `(num_frames, hidden_size)`) and `labels`.
        batch_size (`int`, *optional*, defaults to 8):
            Utterances run at a time.
        limit (`int`, *optional*, defaults to 0):
            Cap on utterances, `0` for all.
        device (`str`, *optional*, defaults to `"cuda"`):
            Device the backbone runs on.
        persona (`str`, *optional*, defaults to `"You are a helpful assistant."`):
            Instruction the prefix carries, which has to match the one the run trains under.
        score_all_frames (`bool`, *optional*, defaults to `False`):
            Keep the `PAD` and `EPAD` frames too, rather than only the word frames.
        drop_audio (`bool`, *optional*, defaults to `False`):
            Put codec silence in the assistant stream instead of the corpus's speech.

    Returns:
        `int`: Number of frames written.
    """
    from safetensors.torch import save_file

    from ..datasets.duplex_data import packed_words
    from ..models.polyplex import PolyplexProcessor

    processor = PolyplexProcessor.from_pretrained(model_dir)
    processor.audio_tokenizer.to(device)
    adapter, _ = Phase1Assemble(model_dir).assemble()
    model = adapter.model.to(device).eval()

    corpus = torch.load(corpus_path, weights_only=False)
    corpus = corpus[:limit] if limit else corpus
    items = [
        {
            "codes": torch.as_tensor(example.codes)[: model.num_codebooks].long(),
            "words": example.text.split(),
            "starts": [0.0] * len(example.text.split()),
        }
        for example in corpus
    ]
    collate = duplex_collate_fn(
        processor, model, persona=persona, score_words_only=not score_all_frames, drop_audio=drop_audio,
        layout=packed_words,
    )
    states, labels = collect_english_states(model, collate, items, batch_size, device)
    save_file({"states": states.contiguous(), "labels": labels.contiguous()}, output)
    return int(states.shape[0])


def load_adapter_slices(model, adapter_path: str):
    r"""
    Writes a finished run's rows and corrections into a freshly assembled model.

    Args:
        model (`EfficientAdapterP1Model`):
            The assembled model, modified in place.
        adapter_path (`str`):
            Directory holding the weights an earlier run wrote.

    Raises:
        FileNotFoundError: If the directory holds no adapter weights.
    """
    path = os.path.join(adapter_path, TEXT_ADAPTER_WEIGHTS_NAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No adapter weights at {path}.")

    stored = load_file(path)
    start, stop = model.config.base_vocab_size, model.config.vocab_size
    with torch.no_grad():
        for name, module in model._tables:
            if name in stored:
                module.weight.data[start:stop] = stored[name].to(
                    module.weight.device, module.weight.dtype
                )
        if getattr(model, "shared_residual", None) is not None and "__shared_residual__" in stored:
            model.shared_residual.data.copy_(
                stored["__shared_residual__"].to(model.shared_residual.device)
            )
        for key, module in getattr(model, "lora", {}).items():
            if f"__lora__{key}__down" in stored:
                module.down.data.copy_(stored[f"__lora__{key}__down"].to(module.down.device))
                module.up.data.copy_(stored[f"__lora__{key}__up"].to(module.up.device))
        for index, module in getattr(model, "confined", {}).items():
            if f"__confined__{index}__up" in stored:
                module.up.data.copy_(stored[f"__confined__{index}__up"].to(module.up.device))
            if f"__confined__{index}__basis" in stored:
                module.basis.data.copy_(stored[f"__confined__{index}__basis"].to(module.basis.device))
        if getattr(model, "partition_direction", None) is not None and "__partition_direction__" in stored:
            # Read before the refit, which takes it off the stored rows to recover the mixture.
            model.partition_direction.data.copy_(
                stored["__partition_direction__"].to(model.partition_direction.device)
            )
        if getattr(model, "head_body", None) is not None:
            prefix = "__head_body__"
            body = {k[len(prefix):]: v for k, v in stored.items() if k.startswith(prefix)}
            if body:
                model.head_body.load_state_dict(body)
    model.refit_mixture()
    logger.info("Continuing from %s.", adapter_path)


class Phase1Trainer(Trainer):
    r"""
    Trainer of the text injection phase.

    Phase 1 declares no new parameter and adds no term to the objective, so the loss is the text cross
    entropy the adapter reports. What makes the phase is which rows are allowed to move, and that is
    decided by the [`EfficientAdapterP1Model`] this trainer assembles or insists on.

    Args:
        model (`EfficientAdapterP1Model`, *optional*):
            An already assembled Phase 1 model. Assembled from `model_name_or_path` when omitted.
        args (`TrainingArguments`, *optional*):
            Standard training configuration.
        train_dataset (`Phase1Dataset`, *optional*):
            The mixed translation and instruction dataset of this phase.
        model_name_or_path (`str`, *optional*, defaults to `"nvidia/personaplex-7b-v1"`):
            Expanded checkpoint the phase starts from, as produced by `polyplex_research.utils.assemble`.
        base_vocab_size (`int`, *optional*):
            Vocabulary size before the expansion, read from the checkpoint config when omitted.
        torch_dtype (`str`, *optional*, defaults to `"bfloat16"`):
            Dtype the frozen weights are loaded in.
        quantization (`str`, *optional*):
            Quantization of the frozen backbone, one of `"nvfp4"` (weights and activations),
            `"int8"` and `"int4"` (weight-only). The tables the adapter binds to are kept in full precision.
        train_input_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the new rows of the backbone text input embedding are trained. Only possible on a
            model that ties that table to the output head.
        train_lm_head (`bool`, *optional*, defaults to `True`):
            Whether the new rows of the text output head are trained.
        train_depth_text_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the new rows of the depth decoder's text embedding are trained.
        adapter_path (`str`, *optional*):
            Directory of a finished Phase 1 run this one continues from. Without it a run starts at the
            expansion and discards whatever an earlier epoch learned.
        duplex (`bool`, *optional*, defaults to `False`):
            Whether the batches are laid out the way the checkpoint reads one, which needs the model
            that is assembled here and so cannot be decided by the caller's collator. See
            [`duplex_collate_fn`].
        dense (`bool`, *optional*, defaults to `False`):
            Whether the batches are packed text rows with no PAD frames, the dense stage of Phase 1. See
            [`dense_collate_fn`]. Excludes `duplex` and every term that reads duplex frames.
        processor (`PolyplexProcessor`, *optional*):
            Processor the duplex layout is built from. Required when `duplex` is set.
        score_words_only (`bool`, *optional*, defaults to `True`):
            Whether the duplex layout leaves `PAD` and `EPAD` frames out of the loss.
        english_lambda (`float`, *optional*, defaults to 0.0):
            Scale of the English preservation term. A row that learns Korean grows to the scale of the
            pretrained rows and takes probability mass from English inside the same softmax, which costs
            the English word loss 6.8 nats and the duplex timing with it. Blocking the directions
            English uses or shortening the rows afterwards repairs that only by spending over two nats
            of Korean, because neither is anything the rows were fitted against. Asking for it in the
            objective instead lets the rows choose a way around English themselves, and both axes come
            out ahead of every read time repair. `0` leaves the objective
            the Korean cross entropy alone and is exactly the run without the option.
        english_states_path (`str`, *optional*):
            Cache of backbone states on English frames, as written by
            [`write_english_states`]. Required whenever `english_lambda` is nonzero.
        english_frames (`int`, *optional*, defaults to 2048):
            How many cached frames the term is scored on each step. The whole cache is far larger than a
            step needs, and the measurements were taken at 2048 English frames against 2048 Korean ones.
        timing_lambda (`float`, *optional*, defaults to 0.0):
            Scale of a separate term on the cached English frames that hold `PAD` or `EPAD`, the ones that
            decide when the model speaks. A row that learns takes probability from those frames as well
            as from the words, and a single term cannot weigh the two differently: easing the word term
            to let Korean grow would ease the timing one with it. When nonzero the cache is split, the
            word frames are scored under `english_lambda` and these under this, each on
            `english_frames` of its own. Needs a cache written with `--score-all-frames`.
        share_row_mixture (`bool`, *optional*, defaults to `False`):
            Whether every trained table builds its appended rows from one set of coefficients, which is
            what lets the audio objective move the head and the backbone's input as well as the depth
            decoder's table. See [`EfficientAdapterP1Config`].
        train_rows (`bool`, *optional*, defaults to `True`):
            Whether the appended rows train. With `False` they stay as `adapter_path` holds them and only the
            backbone corrections, such as the session expert, are stepped.
        train_corrections (`bool`, *optional*, defaults to `True`):
            Whether the backbone corrections train. With `False` they stay as `adapter_path` holds them and
            only the appended rows are stepped.
        partition_direction (`bool`, *optional*, defaults to `False`):
            Whether one free vector shared by every appended row of the head is added to their mixture.
            Korean and English session states are linearly separable, but not along any direction the
            mixture of a row spans, so without it the rows cannot raise Korean on one and hold it down
            on the other. See [`EfficientAdapterP1Config`].
        audio_labels (`bool`, *optional*, defaults to `False`):
            Whether the training batches are scored on the assistant's speech through the depth decoder
            rather than on their transcript. This is the speaking objective of Phase 2: the transcript is
            the conditioning, and its register is the speech corpus's rather than the assistant's, so it
            is not scored. The evaluation loss is then the audio cross entropy alone.
        text_dataset_path (`str`, *optional*):
            Duplex cache of the text corpus Phase 1 was fitted on. When given, every step also scores one
            batch of it with silence in the acoustic slots, which is what keeps the rows' text ability
            while the audio objective moves them.
        text_weight (`float`, *optional*, defaults to 1.0):
            Scale of that text term.
        text_batch_size (`int`, *optional*):
            Batch size of that text term, defaulting to the training batch size. Its logits span the whole
            vocabulary on every frame, so it is the larger of the two in memory.
        session_expert_layers (`str`, *optional*):
            Inclusive layer range such as `"8-31"` whose FFNs gain a low rank correction reading the whole
            input, see [`EfficientAdapterP1Config`].
        session_expert_rank (`int`, *optional*, defaults to 256):
            Rank of that correction.
        confined_attention_basis (`str`, *optional*):
            Fixed attention input directions per layer for a value projection correction, see
            [`EfficientAdapterP1Config`].
        confined_moments (`str`, *optional*):
            Directory of second moments the confined corrections are scored on, see
            [`EfficientAdapterP1Config`]. Required by `leak_lambda`.
        leak_lambda (`float`, *optional*, defaults to 0.0):
            Scale of the share of the confined corrections' output that lands on deployment inputs. See
            `EfficientAdapterP1Model.confined_leak`.
        timing_kl_lambda (`float`, *optional*, defaults to 0.0):
            Scale of the KL that holds the `PAD` / `EPAD` frames to a reference model. See
            [`Phase1Trainer._timing_kl`].
        timing_reference (`str`, *optional*, defaults to `"corrections_off"`):
            What the timing KL holds the model to. `"corrections_off"` is the model with its backbone
            corrections switched off, which pins what the corrections do. `"start"` is the model with every
            trainable parameter as the run started, which pins the timing of the checkpoint the run continues,
            as in the speaking phase.
        english_guard_path (`str`, *optional*):
            `torch.save` of English utterances with their own speech (`text` and `codes`). Required by
            `english_guard_lambda`.
        english_guard_lambda (`float`, *optional*, defaults to 0.0):
            Scale of the term that keeps the model's answer to English input where it was. See
            [`Phase1Trainer._english_guard`].
        english_guard_batch_size (`int`, *optional*, defaults to 4):
            English utterances scored per step under that term.
        persona (`str`, *optional*):
            Instruction the duplex prefix carries, the collator's default when omitted. A language marker
            goes here: the English preservation states are read under the plain persona, so a Korean
            session that carries the marker is one the rows can tell apart from an English one.
    """

    adapter_class = EfficientAdapterP1Model
    dataset_class = Phase1Dataset
    label_names = ["text_labels"]

    def __init__(
        self,
        model=None,
        args: TrainingArguments | None = None,
        train_dataset=None,
        *,
        model_name_or_path: str = DEFAULT_BASE_MODEL,
        base_vocab_size: int | None = None,
        torch_dtype: str = "bfloat16",
        quantization: str | None = None,
        train_input_embeddings: bool = False,
        train_lm_head: bool = True,
        train_depth_text_embeddings: bool = False,
        shared_token_residual: str | None = None,
        residual_all_frozen_rows: bool = False,
        project_new_rows: bool = False,
        group_gated_softmax: bool = False,
        base_row_mixture: int = 0,
        lora_layers: int = 0,
        lora_rank: int = 16,
        confined_basis: str | None = None,
        head_hidden_size: int | None = None,
        tie_new_input_rows: bool = True,
        tied_backbone_grad: bool = False,
        tied_gradient_gain: float = 1.44,
        adapter_path: str | None = None,
        duplex: bool = False,
        dense: bool = False,
        processor=None,
        score_words_only: bool = True,
        drop_audio: bool = False,
        english_lambda: float = 0.0,
        english_states_path: str | None = None,
        english_frames: int = 2048,
        timing_lambda: float = 0.0,
        share_row_mixture: bool = False,
        partition_direction: bool = False,
        train_rows: bool = True,
        train_corrections: bool = True,
        audio_labels: bool = False,
        text_dataset_path: str | None = None,
        text_weight: float = 1.0,
        text_batch_size: int | None = None,
        persona: str | None = None,
        confined_moments: str | None = None,
        confined_attention_basis: str | None = None,
        session_expert_layers: str | None = None,
        session_expert_rank: int = 256,
        leak_lambda: float = 0.0,
        timing_kl_lambda: float = 0.0,
        timing_reference: str = "corrections_off",
        english_guard_path: str | None = None,
        english_guard_lambda: float = 0.0,
        english_guard_batch_size: int = 4,
        **kwargs,
    ):
        if model is None:
            model, _ = Phase1Assemble(
                model_name_or_path,
                base_vocab_size=base_vocab_size,
                torch_dtype=torch_dtype,
                quantization=quantization,
                train_input_embeddings=train_input_embeddings,
                train_lm_head=train_lm_head,
                train_depth_text_embeddings=train_depth_text_embeddings,
                shared_token_residual=shared_token_residual,
                residual_all_frozen_rows=residual_all_frozen_rows,
                project_new_rows=project_new_rows,
                group_gated_softmax=group_gated_softmax,
                base_row_mixture=base_row_mixture,
                lora_layers=lora_layers,
                lora_rank=lora_rank,
                confined_basis=confined_basis,
                confined_moments=confined_moments,
                confined_attention_basis=confined_attention_basis,
                session_expert_layers=session_expert_layers,
                session_expert_rank=session_expert_rank,
                head_hidden_size=head_hidden_size,
                tie_new_input_rows=tie_new_input_rows,
                tied_backbone_grad=tied_backbone_grad,
                tied_gradient_gain=tied_gradient_gain,
                share_row_mixture=share_row_mixture,
                partition_direction=partition_direction,
                train_rows=train_rows,
                train_corrections=train_corrections,
            ).assemble()
            if adapter_path:
                # Continuing a phase starts from the rows it already moved, not from the expansion.
                load_adapter_slices(model, adapter_path)
        self.adapter_path = adapter_path
        self._check_model(model)
        self._prepare(model, args)

        # Consumed whether or not the layout reads it. The caller passes the argument either way and
        # the base trainer rejects one it does not declare.
        max_frames = kwargs.pop("max_frames", None)
        prefix = {} if persona is None else {"persona": persona}

        if dense:
            if duplex or processor is None:
                raise ValueError("`dense` needs the `processor` and excludes `duplex`.")
            if timing_kl_lambda or english_guard_lambda or text_dataset_path is not None:
                raise ValueError(
                    "`dense` batches hold no PAD frames and no persona, so the timing KL, the English guard and "
                    "the duplex text loader do not apply."
                )
            if args is not None:
                args.remove_unused_columns = False
            kwargs["data_collator"] = dense_collate_fn(processor, model.model)
        elif duplex:
            if processor is None:
                raise ValueError("`duplex` needs the `processor` its layout is built from.")
            # A duplex item carries the acoustic codes and the word spans, none of which are arguments
            # of `forward`. Left on, the trainer wraps the collator in one that drops exactly those keys
            # and the layout fails on the first batch instead of at configuration time.
            if args is not None:
                args.remove_unused_columns = False
            # Built here rather than by the caller: the opening frame and the delay pattern are the
            # model's, and the model is what this constructor assembles.
            kwargs["data_collator"] = duplex_collate_fn(
                processor, model.model, score_words_only=score_words_only, drop_audio=drop_audio,
                max_frames=max_frames, audio_labels=audio_labels, **prefix,
            )
        else:
            self._check_dataset(train_dataset)
            self._check_dataset(kwargs.get("eval_dataset"))

        if audio_labels:
            # The split metrics read text logits, which an audio only batch does not produce.
            if args is not None:
                args.prediction_loss_only = True
        elif kwargs.get("eval_dataset") is not None:
            reduce, compute = Trainer.token_group_metrics(model.config.base_vocab_size)
            kwargs.setdefault("preprocess_logits_for_metrics", reduce)
            kwargs.setdefault("compute_metrics", compute)

        super().__init__(model=model, args=args, train_dataset=train_dataset, **kwargs)

        self.text_weight = float(text_weight)
        self._text_loader, self._text_iterator = None, None
        if text_dataset_path is not None:
            if not duplex:
                raise ValueError("`text_dataset_path` is a duplex cache and needs `duplex`.")
            from ..datasets.duplex_data import DuplexUtteranceDataset

            self._text_loader = torch.utils.data.DataLoader(
                DuplexUtteranceDataset(text_dataset_path),
                batch_size=text_batch_size or self.args.per_device_train_batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=self.args.dataloader_num_workers,
                collate_fn=duplex_collate_fn(
                    processor, model.model, score_words_only=score_words_only, drop_audio=True,
                    max_frames=max_frames, **prefix,
                ),
            )

        logger.info(
            "Phase 1 trainer: %d rows across %d tables are trainable.",
            model.num_trainable_rows(),
            len(model.adapter_parameters()),
        )

        self.leak_lambda = float(leak_lambda)
        self.timing_kl_lambda = float(timing_kl_lambda)
        if timing_reference not in ("corrections_off", "start"):
            raise ValueError(f"`timing_reference` must be 'corrections_off' or 'start', got {timing_reference!r}.")
        self.timing_reference = timing_reference
        self._start_state = model.trainable_state() if timing_reference == "start" else None
        self._prefix_frames = None
        if duplex:
            persona_default = inspect.signature(duplex_collate_fn).parameters["persona"].default
            self._prefix_frames = processor(text_prompt=persona or persona_default)["input_ids"].shape[-1]

        self.english_guard_lambda = float(english_guard_lambda)
        self._guard_loader, self._guard_iterator = None, None
        if self.english_guard_lambda:
            if not duplex or english_guard_path is None:
                raise ValueError("`english_guard_lambda` needs `duplex` and `english_guard_path`.")
            from ..datasets.duplex_data import packed_words

            corpus = torch.load(english_guard_path, weights_only=False)
            items = [
                {"codes": torch.as_tensor(e.codes)[:8].long(), "words": e.text.split(), "starts": []}
                for e in corpus
            ]
            # The English session is read under the plain persona, the one deployment uses. Each batch is laid
            # out twice, once with its own speech and once with a silent assistant: the corrections are
            # trained under silence, so silence alone must not be what switches them on.
            layouts = [
                duplex_collate_fn(
                    processor, model.model, score_words_only=False, drop_audio=silent, max_frames=max_frames,
                    layout=packed_words,
                )
                for silent in (False, True)
            ]
            self._guard_loader = torch.utils.data.DataLoader(
                items,
                batch_size=english_guard_batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=lambda examples: [collate(examples) for collate in layouts],
            )
            self._guard_prefix_frames = processor(
                text_prompt=inspect.signature(duplex_collate_fn).parameters["persona"].default
            )["input_ids"].shape[-1]
            logger.info("English guard: %d utterances, lambda %s.", len(items), self.english_guard_lambda)

        self.english_lambda = float(english_lambda)
        self.timing_lambda = float(timing_lambda)
        self.english_frames = english_frames
        self.english_states, self.english_labels = None, None
        if self.english_lambda or self.timing_lambda:
            if english_states_path is None:
                raise ValueError(
                    "`english_lambda` is nonzero, so the term needs `english_states_path`, the cache "
                    "`python -m polyplex_research.utils.evaluate english-states` writes."
                )
            self.english_states, self.english_labels = load_english_states(english_states_path)
            logger.info(
                "English preservation: %.3f over %d of %d cached frames.",
                self.english_lambda,
                min(self.english_frames, self.english_states.shape[0]),
                self.english_states.shape[0],
            )
        if self.timing_lambda:
            timing = (self.english_labels == PAD_TOKEN_ID) | (self.english_labels == EPAD_TOKEN_ID)
            self._english_word_index = (~timing).nonzero().squeeze(-1)
            self._english_timing_index = timing.nonzero().squeeze(-1)
            if not len(self._english_timing_index):
                raise ValueError(
                    "`timing_lambda` is nonzero but the English cache holds no `PAD` or `EPAD` frame. "
                    "Write it with `python -m polyplex_research.utils.evaluate english-states --score-all-frames`."
                )
            logger.info(
                "Timing preservation: %.3f over %d timing and %d word frames.",
                self.timing_lambda,
                len(self._english_timing_index),
                len(self._english_word_index),
            )

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None):
        r"""
        Leaves the weights as assembly loaded them.

        A phase checkpoint holds only the adapter slice, which the base loader does not recognise. The
        slice is read during assembly by passing the checkpoint as `adapter_path`, so resuming restores
        only the optimizer, scheduler, random state and step here.

        Args:
            resume_from_checkpoint (`str`):
                Checkpoint directory the run resumes from.
            model (`torch.nn.Module`, *optional*):
                Ignored.

        Raises:
            ValueError: If the checkpoint is not the one the adapter slice was loaded from.
        """
        if os.path.realpath(resume_from_checkpoint) != os.path.realpath(self.adapter_path or ""):
            raise ValueError(
                f"Resuming from {resume_from_checkpoint} needs `--adapter_path` set to the same directory."
            )

    def english_penalty(self) -> torch.Tensor:
        r"""
        Scores a sample of the cached English frames through the head the phase is training.

        The logits are formed the way the Korean loss forms them, the frozen pretrained rows and the
        current appended rows in one softmax, so the term reads the same distribution the phase is
        charged on and reaches whatever carries the appended rows: the mixture coefficients under
        `base_row_mixture`, and the rows themselves without it. The pretrained rows are an argument of
        neither, so nothing here moves them.

        A sample rather than the whole cache: the cache holds far more frames than a step needs, and the
        term is meant to cost about what one Korean batch costs.

        Returns:
            `torch.Tensor`: The mean cross entropy over the sampled frames.
        """
        head = self.model.model.get_output_embeddings().weight
        count = min(self.english_frames, self.english_states.shape[0])
        index = torch.randint(0, self.english_states.shape[0], (count,))
        states = self.english_states[index].to(head.device, head.dtype)
        labels = self.english_labels[index].to(head.device)
        return F.cross_entropy(self.model.text_logits(states).float(), labels)

    def _english_cross_entropy(self, pool: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            pool (`torch.LongTensor`):
                Indices of the cached frames to sample from.

        Returns:
            `torch.Tensor`: The mean cross entropy over `english_frames` frames drawn from `pool`.
        """
        head = self.model.model.get_output_embeddings().weight
        count = min(self.english_frames, pool.shape[0])
        index = pool[torch.randint(0, pool.shape[0], (count,))]
        states = self.english_states[index].to(head.device, head.dtype)
        labels = self.english_labels[index].to(head.device)
        return F.cross_entropy(self.model.text_logits(states).float(), labels)

    def _next_text_batch(self) -> dict:
        r"""
        Returns:
            `dict`: The next batch of the text corpus on the training device, starting the corpus over
            once it is exhausted.
        """
        if self._text_iterator is None:
            self._text_iterator = iter(self._text_loader)
        try:
            batch = next(self._text_iterator)
        except StopIteration:
            self._text_iterator = iter(self._text_loader)
            batch = next(self._text_iterator)
        return self._prepare_inputs(batch)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        r"""
        Args:
            model (`torch.nn.Module`):
                The model being trained.
            inputs (`dict`):
                A training batch.
            return_outputs (`bool`, *optional*, defaults to `False`):
                Whether the model outputs are returned next to the loss.

        Returns:
            `torch.Tensor` or `tuple`: The task loss, plus the text term and the English preservation
            term when they are asked for, optionally with the outputs. The two extra terms are added in
            training only, so an evaluation reports the task loss on its own.
        """
        if "audio_labels" in inputs:
            # The transcript of a speaking batch is its conditioning, in the speech corpus's register. Under
            # the timing KL the text head still has to run, so the targets are masked rather than dropped.
            if self.timing_kl_lambda and "text_labels" in inputs:
                inputs = {**inputs, "text_labels": torch.full_like(inputs["text_labels"], -100)}
            else:
                inputs = {key: value for key, value in inputs.items() if key != "text_labels"}
        outputs = model(**inputs)
        loss = outputs.loss
        if not model.training:
            return (loss, outputs) if return_outputs else loss

        # A data parallel run hands the loss the wrapper, which forwards but does not carry the adapter's methods.
        core = self.accelerator.unwrap_model(model)
        logs = {}
        if self._text_loader is not None and self.text_weight:
            text_loss = model(**self._next_text_batch()).loss
            loss = loss + self.text_weight * text_loss
            logs["text_loss"] = text_loss.detach().item()
            logs["task_loss"] = outputs.loss.detach().item()

        if self.timing_lambda:
            word = self._english_cross_entropy(self._english_word_index)
            timing = self._english_cross_entropy(self._english_timing_index)
            loss = loss + self.english_lambda * word.to(loss.device) + self.timing_lambda * timing.to(loss.device)
            logs["english_loss"] = word.detach().item()
            logs["timing_loss"] = timing.detach().item()
        elif self.english_lambda:
            penalty = self.english_penalty()
            loss = loss + self.english_lambda * penalty.to(loss.device)
            logs["english_loss"] = penalty.detach().item()

        if self.leak_lambda:
            leak = core.confined_leak()
            loss = loss + self.leak_lambda * leak.to(loss.device)
            logs["leak"] = leak.detach().item()

        if self.timing_kl_lambda:
            timing_kl = self._timing_kl(core, inputs, outputs.logits)
            loss = loss + self.timing_kl_lambda * timing_kl.to(loss.device)
            logs["timing_kl"] = timing_kl.detach().item()

        if self.english_guard_lambda:
            drift, korean = self._english_guard(core)
            loss = loss + self.english_guard_lambda * (drift + korean).to(loss.device)
            logs["english_drift"] = drift.detach().item()
            logs["english_korean"] = korean.detach().item()

        # `compute_loss` runs once per micro batch, `global_step` only moves on an optimizer step, so
        # the accumulation boundary is what makes this one line per logged step.
        if self.accelerator.sync_gradients and (self.state.global_step % max(1, self.args.logging_steps) == 0):
            if "text_labels" in inputs and outputs.logits is not None:
                logs.update(self.korean_metrics(outputs.logits, inputs["text_labels"]))
            if logs:
                self.log(logs)

        return (loss, outputs) if return_outputs else loss

    def korean_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> dict:
        r"""
        Scores the Korean text of a training batch on its own, split by vocabulary half.

        The training loss carries the English preservation term and whatever else the run adds, so it does
        not show how far the Korean text itself has come. The cross entropy is taken one row at a time
        without gradient, which keeps the full vocabulary logits in float32 for one row only.

        Args:
            logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_size)`):
                Text logits of the batch.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Targets, with `-100` on the positions that are not scored.

        Returns:
            `dict`: `korean_loss` over every scored token, `korean_new_loss` over the appended vocabulary
            and `korean_base_loss` over the pretrained one. A half with no token in the batch is left out.
        """
        base = self.model.config.base_vocab_size
        sums = {"korean_loss": 0.0, "korean_new_loss": 0.0, "korean_base_loss": 0.0}
        counts = dict.fromkeys(sums, 0)
        with torch.no_grad():
            for row_logits, row_labels in zip(logits[:, :-1], labels[:, 1:].to(logits.device)):
                scored = row_labels != -100
                targets = row_labels[scored]
                losses = F.cross_entropy(row_logits[scored].float(), targets, reduction="none")
                appended = targets >= base
                for key, mask in (
                    ("korean_loss", torch.ones_like(appended)),
                    ("korean_new_loss", appended),
                    ("korean_base_loss", ~appended),
                ):
                    sums[key] += losses[mask].sum().item()
                    counts[key] += int(mask.sum())
        return {key: sums[key] / counts[key] for key in sums if counts[key]}

    def _english_guard(self, model) -> tuple:
        r"""
        Scores one batch of English for how far the model's answer moved from the frozen one.

        The batch is read twice, with its own speech and with a silent assistant, and both readings count
        equally. Two parts, on every frame after the persona prefix. The KL from the model with its confined
        corrections off to the model as it is holds the whole distribution in place. The second part,
        `-log(1 - p)` with `p` the probability given to the appended rows, presses that mass to zero:
        the rows answer English input even with the corrections off, and in generation one sampled
        Korean piece pulls every later frame toward Korean.

        Args:
            model (`EfficientAdapterP1Model`):
                The model being trained.

        Returns:
            `tuple`: The mean KL and the mean `-log(1 - p)`, both scalars.
        """
        if self._guard_iterator is None:
            self._guard_iterator = iter(self._guard_loader)
        try:
            readings = next(self._guard_iterator)
        except StopIteration:
            self._guard_iterator = iter(self._guard_loader)
            readings = next(self._guard_iterator)

        base = model.config.base_vocab_size
        drifts, koreans = [], []
        for batch in readings:
            batch = self._prepare_inputs({k: v for k, v in batch.items() if k != "text_labels"})
            frames = batch["attention_mask"][:, 1:].bool().clone()
            frames[:, : self._guard_prefix_frames] = False
            logits = model(**batch).logits[:, :-1][frames].float()
            with torch.no_grad(), model.corrections_disabled():
                reference = model(**batch).logits[:, :-1][frames].float().log_softmax(-1)
            current = logits.log_softmax(-1)
            drifts.append((reference.exp() * (reference - current)).sum(-1).mean())
            koreans.append((logits.logsumexp(-1) - logits[:, :base].logsumexp(-1)).mean())
        return sum(drifts) / len(drifts), sum(koreans) / len(koreans)

    def _timing_kl(self, model, inputs, logits) -> torch.Tensor:
        r"""
        KL from a reference model to the model as it is, on the frames whose next text token is `PAD` or
        `EPAD`. The reference is the model with its confined corrections off, or with every trainable
        parameter as the run started when `timing_reference` is `"start"`.

        Those frames decide when the model speaks, and the corrections are only ever asked to predict
        words. Without this term nothing stops them from moving that decision; with it the backbone's own
        answer is the target, so no conversation data is needed.

        Args:
            model (`EfficientAdapterP1Model`):
                The model being trained.
            inputs (`dict`):
                The training batch.
            logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_size)`):
                The logits the batch was scored with.

        Returns:
            `torch.Tensor`: The mean KL over those frames.
        """
        following = inputs["input_ids"][:, 1:]
        frames = (following == PAD_TOKEN_ID) | (following == EPAD_TOKEN_ID)
        if "attention_mask" in inputs:
            frames &= inputs["attention_mask"][:, 1:].bool()
        # The persona prefix and the opening frame are ours, not a decision the model makes.
        frames[:, : self._prefix_frames] = False
        if not bool(frames.any()):
            return logits.new_zeros((), dtype=torch.float32)

        reference_inputs = {key: value for key, value in inputs.items() if key != "audio_labels"}
        if getattr(self, "timing_reference", "corrections_off") == "start":
            context = model.parameters_replaced(self._start_state)
        else:
            context = model.corrections_disabled()
        with torch.no_grad(), context:
            reference = model(**reference_inputs).logits[:, :-1][frames].float().log_softmax(-1)
        current = logits[:, :-1][frames].float().log_softmax(-1)
        return (reference.exp() * (reference - current)).sum(-1).mean()

    def create_optimizer(self):
        r"""
        Builds the optimizer: Muon for weight matrices, AdamW for everything else, the text tables exempt
        from weight decay.

        Phase 1 trains rows inside tables it does not own, and the leading rows are held fixed by a
        gradient hook rather than by `requires_grad` (`protect_leading_rows`). Decay does not read the
        gradient: it shrinks every row of a tensor it is applied to, so a run with decay on would pull
        the pretrained English rows toward zero on every step while reporting that they are frozen. The
        appended rows are the only thing that may move, and they move by their gradient alone.

        The exemption is the tables themselves, not everything the phase trains. A mixture coefficient is
        a parameter of its own whose decay pulls nothing toward zero except itself, so `weight_decay`
        reaches it the way it reaches any other.

        A weight matrix, such as the output side of a confined correction, is stepped with Muon. A
        coefficient table is not, although it is two dimensional: each of its rows is a separate token,
        and an orthogonalized step would tie unrelated tokens together.

        Returns:
            `torch.optim.Optimizer`: The optimizer, built once and cached as `self.optimizer`.
        """
        if self.optimizer is not None:
            return self.optimizer

        tables = {id(parameter) for parameter in self.model.table_parameters()}
        decayed, undecayed = [], []
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                (undecayed if id(parameter) in tables else decayed).append(parameter)

        # Coefficient tables hold one token per row, so they stay on AdamW with the tables.
        tokens = {id(parameter) for parameter in getattr(self.model, "row_mixture", None) or []}
        matrices = [p for p in decayed if p.ndim == 2 and id(p) not in tokens]
        chosen = {id(p) for p in matrices}
        self.optimizer = MuonWithAdamW(
            matrices,
            [
                {"params": [p for p in decayed if id(p) not in chosen], "weight_decay": self.args.weight_decay},
                {"params": undecayed, "weight_decay": 0.0},
            ],
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
            muon_weight_decay=self.args.weight_decay,
        )
        return self.optimizer

class Phase2Trainer(Trainer):
    r"""
    Trainer of the audio injection phase, adding the orthogonal penalty to the speech loss.

    The adapter keeps the penalty out of its forward, because it constrains the parameter rather than the
    prediction. The trainer is what turns it into part of the objective. Both residual kinds report a
    penalty through the same call, so this trainer does not need to know which one it is driving.

    Args:
        model (`CodeRotationAdapterModel`, *optional*):
            An already assembled Phase 2 model. Assembled from `model_name_or_path` when omitted.
        args (`TrainingArguments`, *optional*):
            Standard training configuration.
        train_dataset (`Phase2Dataset`, *optional*):
            The single-turn TTS dataset of this phase, carried as acoustic codes.
        model_name_or_path (`str`, *optional*, defaults to `"nvidia/personaplex-7b-v1"`):
            Checkpoint the phase starts from, normally the Phase 1 output.
        adapter_kind (`str`, *optional*, defaults to `"rotation"`):
            `"rotation"` is the path the method rests on, `"tied"` is the ablation.
        torch_dtype (`str`, *optional*, defaults to `"bfloat16"`):
            Dtype the frozen weights are loaded in.
        quantization (`str`, *optional*):
            Quantization of the frozen backbone, one of `"nvfp4"` (weights and activations),
            `"int8"` and `"int4"` (weight-only). The tables the adapter binds to are kept in full precision.
        gated_adapter (`bool`, *optional*, defaults to `True`):
            Whether the tied residual uses the gate times orthogonal matrix factorization. Tied kind only.
        tie_self_input (`bool`, *optional*, defaults to `True`):
            Whether the residual also reaches the assistant stream tables, so that the model reads its own
            emitted codes under the convention its head writes them in. Read by both kinds.
        ortho_lambda (`float`, *optional*, defaults to 0.1):
            Scale of the orthogonal penalty.
        nonneg_beta (`float`, *optional*, defaults to 10.0):
            Scale of the penalty on the negative entries of the rotation.
        penalty_fn (`Callable`, *optional*):
            Returns the unscaled orthogonal penalty. Defaults to the penalty of the assembled model.
    """

    adapter_class = CodeRotationAdapterModel
    dataset_class = Phase2Dataset
    label_names = ["audio_labels"]

    def __init__(
        self,
        model=None,
        args: TrainingArguments | None = None,
        train_dataset=None,
        *,
        model_name_or_path: str = DEFAULT_BASE_MODEL,
        adapter_kind: str = "rotation",
        torch_dtype: str = "bfloat16",
        quantization: str | None = None,
        gated_adapter: bool = True,
        tie_self_input: bool = True,
        ortho_lambda: float = 0.1,
        nonneg_beta: float = 10.0,
        penalty_fn=None,
        **kwargs,
    ):
        if model is None:
            model, assembled_penalty_fn = Phase2Assemble(
                model_name_or_path,
                adapter_kind=adapter_kind,
                torch_dtype=torch_dtype,
                quantization=quantization,
                gated_adapter=gated_adapter,
                tie_self_input=tie_self_input,
                ortho_lambda=ortho_lambda,
                nonneg_beta=nonneg_beta,
            ).assemble()
            penalty_fn = penalty_fn if penalty_fn is not None else assembled_penalty_fn
        self._check_model(model)
        self._prepare(model, args)

        # Phase 2 reads the acoustic cache through its own collator, which the caller supplies. The
        # duplex layout belongs to Phase 1, and the copy of that branch that stood here named three
        # arguments this constructor does not take, so any Phase 2 run raised before its first batch.
        self._check_dataset(train_dataset)
        self._check_dataset(kwargs.get("eval_dataset"))

        super().__init__(model=model, args=args, train_dataset=train_dataset, **kwargs)

        self.penalty_fn = penalty_fn if penalty_fn is not None else model.penalty
        self.ortho_lambda = ortho_lambda

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        r"""
        Args:
            model (`torch.nn.Module`):
                The model being trained.
            inputs (`dict`):
                A training batch.
            return_outputs (`bool`, *optional*, defaults to `False`):
                Whether the model outputs are returned next to the loss.

        Returns:
            `torch.Tensor` or `tuple`: The total loss, optionally with the outputs.
        """
        outputs = model(**inputs)
        penalty = self.penalty_fn()
        loss = outputs.loss + self.ortho_lambda * penalty.to(outputs.loss.device)

        # `compute_loss` runs once per micro batch, `global_step` only moves on an optimizer step, so
        # the accumulation boundary is what makes this one line per logged step.
        if self.accelerator.sync_gradients and (
            self.state.global_step % max(1, self.args.logging_steps) == 0
        ):
            scaled = self.ortho_lambda * penalty.detach().item()
            self.log(
                {
                    "ortho_penalty": penalty.detach().item(),
                    # What the constraint is worth against the speech loss, which is what `ortho_lambda`
                    # is tuned on: a share near 1 means the run is fitting the penalty and not the codes.
                    "ortho_share": scaled / max(abs(loss.detach().item()), 1e-9),
                }
            )

        return (loss, outputs) if return_outputs else loss


@dataclass
class PhaseArguments:
    r"""
    Command line arguments selecting the phase and pointing it at its inputs.

    Everything about how a phase assembles its model now lives on its trainer, so this only carries what
    a command line has to state: which phase, which checkpoint, and where the dataset is.

    Args:
        phase (`int`, *optional*, defaults to 2):
            `1` runs text injection, `2` runs audio injection.
        model_name_or_path (`str`, *optional*, defaults to `"nvidia/personaplex-7b-v1"`):
            Checkpoint the phase starts from. Phase 1 expects the expanded checkpoint produced by
            `polyplex_research.utils.assemble`, Phase 2 expects the Phase 1 output.
        quantization (`str`, *optional*):
            Quantization of the frozen backbone, one of `"nvfp4"` (weights and activations), `"int8"` and `"int4"` (weight-only).
        torch_dtype (`str`, *optional*, defaults to `"bfloat16"`):
            Dtype the frozen weights are loaded in.
        eval_dataset_path (`str`, *optional*):
            Directory or Hub repository of a held out split of the phase dataset. Without one a run only
            reports the loss on examples it is training on, which cannot say whether another epoch is
            still learning or starting to memorize.
        dataset_path (`str`, *optional*):
            Directory or Hub repository of the phase dataset, as written by its `save_pretrained`.
        tokenizer_name_or_path (`str`, *optional*):
            Tokenizer of the text stream, defaults to the one shipped with the checkpoint.
        adapter_kind (`str`, *optional*, defaults to `"rotation"`):
            Phase 2. `"rotation"` is the path the method rests on, `"tied"` is the ablation.
        gated_adapter (`bool`, *optional*, defaults to `True`):
            Phase 2. Whether the tied residual uses the gate times orthogonal matrix factorization.
        tie_self_input (`bool`, *optional*, defaults to `True`):
            Phase 2. Whether the residual also reaches the assistant stream tables.
        ortho_lambda (`float`, *optional*, defaults to 0.1):
            Phase 2. Scale of the orthogonal penalty.
        nonneg_beta (`float`, *optional*, defaults to 10.0):
            Phase 2. Scale of the penalty on the negative entries of the rotation.
        duplex (`bool`, *optional*, defaults to `False`):
            Phase 1. Whether `dataset_path` holds the acoustic cache and word spans a duplex batch is
            laid out from, rather than the plain text mixture. See [`DuplexUtteranceDataset`].
        dense (`bool`, *optional*, defaults to `False`):
            Phase 1. Whether `dataset_path` holds packed text rows for the dense stage. See
            [`DenseTextDataset`].
        lora_layers (`int`, *optional*, defaults to 0):
            Phase 1. How many of the backbone's last layers carry a low rank correction. Word order and
            morphology live in the backbone, and a frozen English one has no reason to hold the Korean
            ones.
        lora_rank (`int`, *optional*, defaults to 16):
            Phase 1. Rank of that correction.
        confined_basis (`str`, *optional*):
            Phase 1. Fixed input directions per layer for an FFN correction that deployment does not
            read. See [`EfficientAdapterP1Config`].
        session_expert_layers (`str`, *optional*):
            Phase 1. Layer range of the session expert, a full input low rank FFN correction.
        session_expert_rank (`int`, *optional*, defaults to 256):
            Phase 1. Its rank.
        confined_attention_basis (`str`, *optional*):
            Phase 1. The same for a correction of each listed layer's attention value projection.
        confined_moments (`str`, *optional*):
            Phase 1. Second moments the confined corrections are scored on.
        leak_lambda (`float`, *optional*, defaults to 0.0):
            Phase 1. Scale of the confined corrections' share of output on deployment inputs.
        timing_kl_lambda (`float`, *optional*, defaults to 0.0):
            Phase 1. Scale of the KL holding the `PAD` / `EPAD` frames to a reference model.
        timing_reference (`str`, *optional*, defaults to `"corrections_off"`):
            Phase 1. Reference of the timing KL, see [`Phase1Trainer`].
        english_guard_path (`str`, *optional*):
            Phase 1, duplex layout. `torch.save` of English utterances with their own speech.
        english_guard_lambda (`float`, *optional*, defaults to 0.0):
            Phase 1. Scale of the term holding the answer to English speech in place and the appended
            rows' share of it at zero.
        score_words_only (`bool`, *optional*, defaults to `True`):
            Phase 1, duplex layout. Whether `PAD` and `EPAD` frames are left out of the loss. Most of
            the stream is padding, so scoring it moves the gradient off the words.
        english_lambda (`float`, *optional*, defaults to 0.0):
            Phase 1. Scale of the English preservation term. Rows that learn Korean take probability
            mass from English inside the same softmax, and asking them in the objective to stay quiet on
            cached English states is what costs the least Korean to prevent it.
        english_states_path (`str`, *optional*):
            Phase 1. Cache of backbone states on English frames, built by
            [`write_english_states`]. Required whenever `english_lambda` is nonzero.
        english_frames (`int`, *optional*, defaults to 2048):
            Phase 1. How many cached frames that term is scored on each step.
        timing_lambda (`float`, *optional*, defaults to 0.0):
            Phase 1. Scale of the separate term on the cached English `PAD` and `EPAD` frames.
        train_depth_text_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the depth decoder's text table carries appended rows that train. The speaking
            objective reaches the rows only through it.
        share_row_mixture (`bool`, *optional*, defaults to `False`):
            Whether every trained table builds its appended rows from one set of coefficients.
        train_rows (`bool`, *optional*, defaults to `True`):
            Whether the appended rows train, or only the backbone corrections do.
        train_corrections (`bool`, *optional*, defaults to `True`):
            Whether the backbone corrections train, or only the appended rows do.
        partition_direction (`bool`, *optional*, defaults to `False`):
            Whether one free vector shared by every appended row of the head is added to their mixture.
        audio_labels (`bool`, *optional*, defaults to `False`):
            Whether the training batches are scored on the assistant's speech, the speaking objective.
        text_dataset_path (`str`, *optional*):
            Duplex cache of the text corpus scored next to the speech every step.
        text_weight (`float`, *optional*, defaults to 1.0):
            Scale of that text term.
        text_batch_size (`int`, *optional*):
            Batch size of that text term.
        persona (`str`, *optional*):
            Instruction the duplex prefix carries, where a language marker goes.
        head_hidden_size (`int`, *optional*):
            Phase 1. Width of the hidden layer the appended rows read the state through. A single
            matrix over a frozen state can only score a word by how well it matches that state, which
            ranks words by topic rather than by what the sentence needs next.
        text_max_length (`int`, *optional*):
            Phase 1. Hard cap on the length of a sequence, in tokens. A corpus with a long tail makes
            the batch that holds its longest example decide the memory the whole run needs, and the
            logits of a wide vocabulary are what that memory is spent on.
        text_pad_token_id (`int`, *optional*, defaults to 3):
            Phase 2. Text token filling the frames that carry no word.
        text_alignment (`str`, *optional*, defaults to `"spread"`):
            Phase 2. How the words of an utterance are laid over its frames.
        max_frames (`int`, *optional*):
            Phase 1 duplex layout, and Phase 2.
            Phase 2. Hard cap on the temporal length of a batch.
    """

    phase: int = field(default=2)
    model_name_or_path: str = field(default=DEFAULT_BASE_MODEL)
    adapter_path: str | None = field(default=None)
    early_stopping_patience: int = field(default=0)
    early_stopping_threshold: float = field(default=0.0)
    dataset_path: str | None = field(default=None)
    eval_dataset_path: str | None = field(default=None)
    tokenizer_name_or_path: str | None = field(default=None)
    quantization: str | None = field(default=None)
    torch_dtype: str = field(default="bfloat16")
    shared_token_residual: str | None = field(default=None)
    residual_all_frozen_rows: bool = field(default=False)
    project_new_rows: bool = field(default=False)
    adapter_kind: str = field(default="rotation")
    gated_adapter: bool = field(default=True)
    tie_self_input: bool = field(default=True)
    ortho_lambda: float = field(default=0.1)
    nonneg_beta: float = field(default=10.0)
    text_max_length: int | None = field(default=None)
    duplex: bool = field(default=False)
    dense: bool = field(default=False)
    base_row_mixture: int = field(default=0)
    tie_new_input_rows: bool = field(default=True)
    lora_layers: int = field(default=0)
    lora_rank: int = field(default=16)
    confined_basis: str | None = field(default=None)
    confined_moments: str | None = field(default=None)
    confined_attention_basis: str | None = field(default=None)
    session_expert_layers: str | None = field(default=None)
    session_expert_rank: int = field(default=256)
    leak_lambda: float = field(default=0.0)
    timing_kl_lambda: float = field(default=0.0)
    timing_reference: str = field(default="corrections_off")
    english_guard_path: str | None = field(default=None)
    english_guard_lambda: float = field(default=0.0)
    score_words_only: bool = field(default=True)
    drop_audio: bool = field(default=False)
    english_lambda: float = field(default=0.0)
    english_states_path: str | None = field(default=None)
    english_frames: int = field(default=2048)
    timing_lambda: float = field(default=0.0)
    train_depth_text_embeddings: bool = field(default=False)
    share_row_mixture: bool = field(default=False)
    train_rows: bool = field(default=True)
    train_corrections: bool = field(default=True)
    partition_direction: bool = field(default=False)
    audio_labels: bool = field(default=False)
    text_dataset_path: str | None = field(default=None)
    text_weight: float = field(default=1.0)
    text_batch_size: int | None = field(default=None)
    persona: str | None = field(default=None)
    head_hidden_size: int | None = field(default=None)
    text_pad_token_id: int = field(default=3)
    text_alignment: str = field(default="spread")
    max_frames: int | None = field(default=None)


def dense_mixture(
    sizes: dict, temperature: float = 0.3, english_floor: float = 0.2, languages: tuple | None = None
) -> dict:
    r"""
    Share of the training tokens each language takes (SPEC S13.4): proportional to its bytes to the power
    `temperature`, with English raised to `english_floor` when it falls below it and the others scaled to fill the
    rest.

    Args:
        sizes (`dict`):
            `"en"` and FineWeb-2 language codes mapped to bytes, as
            [`~polyplex_research.datasets.webtext.fineweb2_bytes`] lists them.
        temperature (`float`, *optional*, defaults to 0.3):
            Exponent of the bytes.
        english_floor (`float`, *optional*, defaults to 0.2):
            Smallest share of English.
        languages (`tuple[str]`, *optional*):
            Dense language keys the mixture keeps; the shares are computed over these alone. Every language of
            `sizes` when omitted.

    Returns:
        `dict`: Dense language key ([`~polyplex_research.datasets.webtext.language_key`]) mapped to its share,
        summing to 1.
    """
    raw = {("en" if code == "en" else language_key(code)): size**temperature for code, size in sizes.items()}
    if languages is not None:
        missing = [language for language in languages if language not in raw]
        if missing:
            raise ValueError(f"Languages without sizes: {missing}")
        raw = {language: value for language, value in raw.items() if language in languages}
    total = sum(raw.values())
    shares = {language: value / total for language, value in raw.items()}
    if shares.get("en", 0.0) < english_floor:
        rest = sum(share for language, share in shares.items() if language != "en")
        shares = {language: share * (1 - english_floor) / rest for language, share in shares.items() if language != "en"}
        shares["en"] = english_floor
    return shares


def segment_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, positions: torch.Tensor, first_position: int
) -> tuple[float, int]:
    r"""
    Sums the next-token cross entropy of packed documents, leaving out the first token of every document.

    Args:
        logits (`torch.Tensor` of shape `(batch_size, sequence_length, vocab_size)`):
            Logits, the one at `t` predicting the target at `t + 1`.
        targets (`torch.Tensor` of shape `(batch_size, sequence_length)`):
            Targets aligned with the inputs. `-100` is not scored.
        positions (`torch.Tensor` of shape `(batch_size, sequence_length)`):
            Position of each frame inside its document.
        first_position (`int`):
            Position of a document's first token: 1 behind an opening frame, 0 without one.

    Returns:
        `tuple[float, int]`: The summed cross entropy and the number of scored tokens.
    """
    targets = targets[:, 1:]
    scored = (targets != -100) & (positions[:, 1:] > first_position)
    if not scored.any():
        return 0.0, 0
    total = F.cross_entropy(logits[:, :-1][scored].float(), targets[scored], reduction="sum")
    return total.item(), int(scored.sum())


def language_model_view(batch: dict, row: int) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Takes one row of a dense batch as the text LLM reads it: the scored tokens of each document, with no
    opening frame and no filler, and positions counted from zero per document.

    Args:
        batch (`dict`):
            Batch of [`~polyplex_research.datasets.duplex_data.dense_collate_fn`].
        row (`int`):
            Row of the batch.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: Token ids and positions, each of shape `(1, length)`.
    """
    positions = batch["position_ids"][row]
    kept = (positions > 0) & (batch["text_labels"][row] != -100)
    return batch["input_ids"][row][kept][None], (positions[kept] - 1)[None]


def dense_trainable_parameters(model) -> list[str]:
    r"""
    Freezes a DualFFN model except what the dense check trains (SPEC S13.4). With the text LLM's vocabulary that is
    the normalizations of its language side: each layer's `language_norm` and the `text_norm` the text head reads.
    With the Polyplex vocabulary (a model with a shared text composition, SPEC S12.3a) it is each layer's
    `language_norm`, the stored rows of the LM head and the text input table and the composition coefficients, of
    which [`train_dense`] keeps the base rows fixed; `text_norm` is then Helium's and stays frozen, and the depth
    decoder's text rows move through the coefficients alone. Appended layers (SPEC S11.4a) train whole, and so does the
    language FFN output scale of a model that has one (SPEC S11.2c).

    Args:
        model ([`PolyplexForConditionalGeneration`]):
            The model.

    Returns:
        `list[str]`: Names of the parameters left trainable.
    """
    vocabulary = getattr(model, "text_composition", None) is not None
    tables = ("lm_head.weight", "model.embed_tokens.weight", "text_composition.values")
    appended = tuple(f"model.layers.{index}." for index in model.config.appended_layers)
    names = []
    for name, parameter in model.named_parameters():
        if vocabulary:
            trainable = ".language_norm." in name or name in tables
        else:
            trainable = ".language_norm." in name or name.startswith("model.text_norm.")
        trainable = trainable or name.endswith(".mlp.language_scale")
        # The language stream's own text table trains whole (SPEC S11.2f).
        trainable = trainable or name == "language_embed_tokens.weight"
        # The language FFN's low rank corrections train whole (SPEC S11.2g).
        trainable = trainable or "_lora." in name
        # Appended layers train whole (SPEC S11.4a).
        trainable = trainable or name.startswith(appended)
        parameter.requires_grad_(trainable)
        if parameter.requires_grad:
            names.append(name)
    return names


def frozen_text_rows(model, names: list[str]) -> dict:
    r"""
    Rows of the trained text tables that stay fixed (SPEC S12.3a): the base tokens and everything above the head's
    rows, the reserved id that opens every turn.

    Args:
        model ([`PolyplexForConditionalGeneration`]):
            The model.
        names (`list[str]`):
            Names of the trained parameters.

    Returns:
        `dict`: Each trained text table mapped to the `(base_vocab_size, vocab_size)` bounds of its trained rows.
    """
    base = getattr(model.config, "base_vocab_size", None)
    if not base:
        return {}
    parameters = dict(model.named_parameters())
    tables = ("lm_head.weight", "model.embed_tokens.weight", "text_composition.values")
    return {parameters[name]: (base, model.config.vocab_size) for name in tables if name in names}


@torch.no_grad()
def dense_evaluate(model, collate, rows: list, text_llm: bool = False, micro_batch_size: int = 1) -> float:
    r"""
    Measures the per-token cross entropy of packed rows, leaving out each document's first token.

    Args:
        model (`PreTrainedModel`):
            A DualFFN model, or the text LLM when `text_llm` is set.
        collate (`Callable`):
            [`~polyplex_research.datasets.duplex_data.dense_collate_fn`] of the DualFFN model.
        rows (`list`):
            Rows with a `segments` list of token tensors.
        text_llm (`bool`, *optional*, defaults to `False`):
            Whether `model` is the text LLM, which reads [`language_model_view`] of each row.
        micro_batch_size (`int`, *optional*, defaults to 1):
            Rows per forward pass of the DualFFN model.

    Returns:
        `float`: Mean cross entropy per scored token.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(rows), micro_batch_size):
        batch = collate(rows[start : start + micro_batch_size])
        if text_llm:
            for row in range(batch["input_ids"].shape[0]):
                ids, positions = (tensor.to(device) for tensor in language_model_view(batch, row))
                logits = model(input_ids=ids, position_ids=positions, use_cache=False).logits
                part = segment_cross_entropy(logits, ids, positions, first_position=0)
                total, count = total + part[0], count + part[1]
        else:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("text_labels")
            logits = model(**batch, use_cache=False).logits
            part = segment_cross_entropy(logits, labels, batch["position_ids"], first_position=1)
            total, count = total + part[0], count + part[1]
    model.train(was_training)
    return total / max(count, 1)


BYTE_PIECE = re.compile(r"<0x[0-9A-F]{2}>")


def piece_byte_length(piece: str | None, byte_level: bool = False) -> int:
    r"""
    Args:
        piece (`str`):
            A token string.
        byte_level (`bool`, *optional*, defaults to `False`):
            Whether the tokenizer is byte-level BPE (Qwen3), where each character of a token stands for one byte.

    Returns:
        `int`: UTF-8 bytes of the text the token stands for: 1 for a byte-fallback token, and a word-start mark
        `▁` counts as the space it replaces.
    """
    if piece is None:
        return 0
    if byte_level:
        return len(piece)
    if BYTE_PIECE.fullmatch(piece):
        return 1
    return len(piece.replace("\u2581", " ").encode("utf-8"))


def token_byte_lengths(tokenizer, byte_level: bool = False) -> torch.Tensor:
    r"""
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer.
        byte_level (`bool`, *optional*, defaults to `False`):
            See [`piece_byte_length`].

    Returns:
        `torch.Tensor` of shape `(len(tokenizer),)`: The bytes of each id.
    """
    pieces = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
    return torch.tensor([piece_byte_length(piece, byte_level) for piece in pieces])


@torch.no_grad()
def dense_bits_per_byte(
    model, collate, tokenizer, texts: list, byte_lengths: torch.Tensor, text_llm: bool = False, vocabulary: int | None = None
) -> float:
    r"""
    Bits per byte of texts, comparable across tokenizers (SPEC S13.4): each text is cut by `tokenizer`, the cross
    entropy of every token but the first is summed and divided by the UTF-8 bytes those tokens stand for.

    Args:
        model (`PreTrainedModel`):
            A DualFFN model, read through `collate` with its opening frame, or the text LLM when `text_llm` is set.
        collate (`Callable`):
            [`~polyplex_research.datasets.duplex_data.dense_collate_fn`] of the DualFFN model. Unused for the text
            LLM.
        tokenizer (`PreTrainedTokenizerBase`):
            The model's tokenizer.
        texts (`list[str]`):
            The texts.
        byte_lengths (`torch.Tensor`):
            Bytes of each id, from [`token_byte_lengths`].
        text_llm (`bool`, *optional*, defaults to `False`):
            Whether `model` is the text LLM.
        vocabulary (`int`, *optional*):
            Scores the distribution over the first `vocabulary` ids alone, renormalized, at the positions whose
            target is one of them; the bytes of the other positions are not counted.

    Returns:
        `float`: Bits per byte.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    total, scored_bytes = 0.0, 0
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 2:
            continue
        targets = torch.tensor(ids[1:])
        if text_llm:
            logits = model(input_ids=torch.tensor([ids], device=device), use_cache=False).logits[0, :-1]
        else:
            batch = {key: value.to(device) for key, value in collate([{"segments": [torch.tensor(ids)]}]).items()}
            batch.pop("text_labels")
            # Behind the opening frame, the logit at frame 1 predicts the second token.
            logits = model(**batch, use_cache=False).logits[0, 1:-1]
        if vocabulary is not None:
            scored = targets < vocabulary
            logits, targets = logits[scored.to(logits.device), :vocabulary], targets[scored]
        total += F.cross_entropy(logits.float(), targets.to(device), reduction="sum").item()
        scored_bytes += int(byte_lengths[targets].sum())
    model.train(was_training)
    return total / math.log(2) / max(scored_bytes, 1)


TOKEN_SCRIPTS = (
    ("HANGUL", "hangul"),
    ("CJK", "han"),
    ("HIRAGANA", "hiragana"),
    ("KATAKANA", "katakana"),
    ("LATIN", "latin"),
    ("CYRILLIC", "cyrillic"),
    ("ARABIC", "arabic"),
    ("DEVANAGARI", "devanagari"),
    ("DIGIT", "digit"),
)


def token_script(text: str) -> str:
    r"""
    Args:
        text (`str`):
            The text of one token.

    Returns:
        `str`: The script of its first letter or digit ([`TOKEN_SCRIPTS`]), or `"other"` when it has none.
    """
    import unicodedata

    for character in text:
        if character.isalnum():
            name = unicodedata.name(character, "")
            for prefix, script in TOKEN_SCRIPTS:
                if name.startswith(prefix):
                    return script
            return "other"
    return "other"


def dense_token_losses(model, collate, tokenizer, texts: list, byte_lengths: torch.Tensor, base: int) -> list:
    r"""
    The cross entropy of every token [`dense_bits_per_byte`] scores, with what [`grouped_bits_per_byte`] groups by.

    Args:
        model ([`PolyplexForConditionalGeneration`]):
            The model.
        collate (`Callable`):
            [`~polyplex_research.datasets.duplex_data.dense_collate_fn`] of the model.
        tokenizer (`PreTrainedTokenizerBase`):
            The model's tokenizer.
        texts (`list[str]`):
            The texts.
        byte_lengths (`torch.Tensor`):
            Bytes of each id, from [`token_byte_lengths`].
        base (`int`):
            Size of the base vocabulary, Helium's; ids at or above it are new.

    Returns:
        `list[dict]`: Per token, `loss` (nats), `bytes`, `script` ([`token_script`]), `new`, `position` (index of
        the target in its text) and `id`.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < 2:
                continue
            targets = torch.tensor(ids[1:])
            batch = {key: value.to(device) for key, value in collate([{"segments": [torch.tensor(ids)]}]).items()}
            batch.pop("text_labels")
            logits = model(**batch, use_cache=False).logits[0, 1:-1]
            values = F.cross_entropy(logits.float(), targets.to(device), reduction="none").tolist()
            for position, (target, value) in enumerate(zip(ids[1:], values), start=1):
                losses.append({
                    "loss": value,
                    "bytes": int(byte_lengths[target]),
                    "script": token_script(tokenizer.decode([target])),
                    "new": target >= base,
                    "position": position,
                    "id": target,
                })
    model.train(was_training)
    return losses


def grouped_bits_per_byte(losses: list, positions: tuple = (64, 256), counts=None, exposures: tuple = (1, 10, 100)) -> dict:
    r"""
    Bits per byte of the tokens [`dense_token_losses`] scored, per script, per vocabulary (base or new) and per range
    of positions, with the share of the bytes each group holds. Given `counts`, also per range of how often training
    saw each new token.

    Args:
        losses (`list[dict]`):
            What [`dense_token_losses`] returned.
        positions (`tuple[int]`, *optional*, defaults to `(64, 256)`):
            Bounds of the position ranges.
        counts (`Sequence[int]`, *optional*):
            Occurrences of each id in the training tokens.
        exposures (`tuple[int]`, *optional*, defaults to `(1, 10, 100)`):
            Bounds of the exposure ranges.

    Returns:
        `dict`: `script`, `vocabulary`, `position` and, given `counts`, `exposure`, each mapping a group to
        `bits_per_byte` and `byte_share`.
    """

    def exposure_group(entry: dict) -> str:
        if not entry["new"]:
            return "base"
        count = int(counts[entry["id"]])
        lower = 0
        for bound in exposures:
            if count < bound:
                return f"new {lower}" if bound == lower + 1 else f"new {lower}-{bound - 1}"
            lower = bound
        return f"new >={exposures[-1]}"

    def position_group(position: int) -> str:
        for bound in positions:
            if position < bound:
                return f"<{bound}"
        return f">={positions[-1]}"

    keys = {
        "script": lambda entry: entry["script"],
        "vocabulary": lambda entry: "new" if entry["new"] else "base",
        "position": lambda entry: position_group(entry["position"]),
    }
    if counts is not None:
        keys["exposure"] = exposure_group
    total_bytes = max(sum(entry["bytes"] for entry in losses), 1)
    result = {}
    for kind, key in keys.items():
        sums = {}
        for entry in losses:
            loss, size = sums.get(key(entry), (0.0, 0))
            sums[key(entry)] = (loss + entry["loss"], size + entry["bytes"])
        result[kind] = {
            group: {"bits_per_byte": loss / math.log(2) / max(size, 1), "byte_share": size / total_bytes}
            for group, (loss, size) in sums.items()
        }
    return result


def unigram_bits_per_byte(log_probs: torch.Tensor, tokenizer, texts: list, byte_lengths: torch.Tensor) -> float:
    r"""
    Bits per byte of a unigram model on the tokens [`dense_bits_per_byte`] scores: every token of each text but the
    first, each costing its own log probability whatever comes before it.

    Args:
        log_probs (`torch.Tensor` of shape `(vocab_size,)`):
            Log probability of each id, from [`~polyplex_research.utils.evaluate.unigram_log_probs`].
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer the probabilities are over.
        texts (`list[str]`):
            The texts.
        byte_lengths (`torch.Tensor`):
            Bytes of each id, from [`token_byte_lengths`].

    Returns:
        `float`: Bits per byte.
    """
    total, scored_bytes = 0.0, 0
    for text in texts:
        targets = torch.tensor(tokenizer.encode(text, add_special_tokens=False)[1:], dtype=torch.long)
        total -= log_probs[targets].sum().item()
        scored_bytes += int(byte_lengths[targets].sum())
    return total / math.log(2) / max(scored_bytes, 1)


DENSE_TABLES = (
    "lm_head.weight", "model.embed_tokens.weight", "text_composition.values", "language_embed_tokens.weight"
)
"""Text tables and coefficients a dense run trains row by row, so AdamW steps them (SPEC S7.2)."""

DENSE_FACTORED_TABLES = ("lm_head.weight", "model.embed_tokens.weight", "language_embed_tokens.weight")
"""The two vocabulary sized tables, stepped by FactoredAdam (SPEC S7.2a)."""


def wsd_factor(step: int, warmup_steps: int, total_steps: int, decay_steps: int) -> float:
    r"""
    Learning rate factor of a warmup, stable, decay schedule (SPEC S13.4): linear warmup, then constant, then a cosine
    to 0 over the last `decay_steps` steps.

    Args:
        step (`int`):
            The optimizer step about to be taken, counted from 1 over the whole adaptation.
        warmup_steps (`int`):
            Steps of linear warmup.
        total_steps (`int`):
            Last step of this run.
        decay_steps (`int`):
            Steps of decay that end this run, 0 for none.

    Returns:
        `float`: The factor applied to the peak learning rate.
    """
    if warmup_steps and step <= warmup_steps:
        return step / warmup_steps
    start = total_steps - decay_steps
    if decay_steps and step > start:
        return 0.5 * (1 + math.cos(math.pi * (step - start) / decay_steps))
    return 1.0


def cosine_restart_factor(step: int, warmup_steps: int, save_every: int) -> float:
    r"""
    Learning rate factor of cosine warm restarts that end at every checkpoint (SPEC S13.4a): linear warmup, then a
    cosine from 1 to 0 that reaches 0 at each multiple of `save_every` and restarts at 1 on the next step.

    The factor depends only on `step`, so a run resumed in the middle of a cycle continues the same curve. The first
    cycle runs from the end of warmup to the first multiple of `save_every` after it.

    Args:
        step (`int`):
            The optimizer step about to be taken, counted from 1 over the whole adaptation.
        warmup_steps (`int`):
            Steps of linear warmup.
        save_every (`int`):
            Steps between checkpoints, which end the cycles.

    Returns:
        `float`: The factor applied to the peak learning rate.
    """
    if warmup_steps and step <= warmup_steps:
        return step / warmup_steps
    end = -(-step // save_every) * save_every
    start = max(end - save_every, warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * (step - start) / (end - start)))


LR_SCHEDULES = ("wsd", "cosine-restarts")
"""Learning rate schedules of the dense stage: [`wsd_factor`] and [`cosine_restart_factor`]."""


def dense_masters(trained: dict) -> set:
    r"""
    The trained parameters a dense run steps through a float32 master (SPEC S13.4): vectors, such as the
    normalization weights, whose values near 1 lose updates below bfloat16's spacing, and the composition
    coefficients. Every other parameter is stepped in its own dtype.

    Args:
        trained (`dict`):
            Parameter name mapped to the trained parameter.

    Returns:
        `set[str]`: Names of the parameters with a master.
    """
    return {name for name, parameter in trained.items() if parameter.ndim == 1 or name == "text_composition.values"}


def shared_stream_switch(alignment: str, language_model: str, state: dict):
    r"""
    Builds the call that moves a model's language stream to one shared rotation folded into its language FFN weights
    (SPEC S11.2h, [`~polyplex_research.utils.assemble.fold_language_stream`]). Frozen language FFNs are folded from the
    text LLM's original weights. The gains folded in are kept in `state["gains"]`; when `state` already holds gains, as
    for a resumed folded checkpoint, they are set as each layer's `language_norm` first.

    Args:
        alignment (`str`):
            A shared residual basis alignment file.
        language_model (`str`):
            Hub id of the text LLM.
        state (`dict`):
            Holds `gains`, of shape `(layers, hidden_size)`, or `None` before the switch.

    Returns:
        `Callable`: Takes the model and folds it in place.
    """

    def switch(model) -> None:
        from .assemble import fold_language_stream, load_language_ffn_weights, load_residual_alignment

        rotation, scale, _ = load_residual_alignment(alignment)(0)
        layers = model.model.layers
        with torch.no_grad():
            if state.get("gains") is None:
                state["gains"] = torch.stack([layer.mlp.language_norm.weight.detach().float().cpu() for layer in layers])
            else:
                for layer, gain in zip(layers, state["gains"]):
                    layer.mlp.language_norm.weight.copy_(gain)
        device = layers[0].mlp.language_norm.weight.device
        original = load_language_ffn_weights(language_model, model.config.language_ffn_layers)
        fold_language_stream(model, rotation.to(device), scale, original=original)

    return switch


def reset_optimizer_state(optimizer: MuonWithAdamW, tensors: list) -> None:
    r"""
    Drops the optimizer state of the given tensors, so their next step starts from fresh moments.

    Args:
        optimizer ([`~polyplex_research.utils.optimization.MuonWithAdamW`]):
            The optimizer.
        tensors (`list[torch.Tensor]`):
            Tensors it steps.
    """
    for inner in optimizer._inner():
        if inner is not None:
            for tensor in tensors:
                inner.state.pop(tensor, None)


def dense_optimizer(masters: dict, learning_rate: float) -> MuonWithAdamW:
    r"""
    The optimizer of a dense run (SPEC S7.2): Muon for two dimensional weight matrices, FactoredAdam for the input table
    and the LM head (SPEC S7.2a), AdamW without weight decay for the composition coefficients and every vector. The
    state of each tensor takes its dtype.

    Args:
        masters (`dict`):
            Parameter name mapped to the tensor the optimizer steps, a float32 master or the parameter itself.
        learning_rate (`float`):
            Peak learning rate.

    Returns:
        [`~polyplex_research.utils.optimization.MuonWithAdamW`]: The optimizer.
    """
    matrices = [tensor for name, tensor in masters.items() if tensor.ndim == 2 and name not in DENSE_TABLES]
    tables = [tensor for name, tensor in masters.items() if name in DENSE_FACTORED_TABLES]
    chosen = {id(tensor) for tensor in matrices + tables}
    rest = [tensor for tensor in masters.values() if id(tensor) not in chosen]
    return MuonWithAdamW(matrices, [{"params": rest, "weight_decay": 0.0}], lr=learning_rate, table_params=tables)


def upgrade_dense_optimizer_state(state: dict, masters: dict, optimizer: MuonWithAdamW) -> dict:
    r"""
    Brings an optimizer state saved before SPEC S7.2a, with every table on AdamW, to the layout of [`dense_optimizer`]:
    the input table and the LM head move to FactoredAdam, their first moment kept and their second moment reduced to
    row and column means ([`~polyplex_research.utils.optimization.FactoredAdam.from_adam_state`]). A state that already
    has the tables is returned as it is.

    Args:
        state (`dict`):
            What [`~polyplex_research.utils.optimization.MuonWithAdamW.state_dict`] returned.
        masters (`dict`):
            Parameter name mapped to the tensor the optimizer steps, in the order [`dense_optimizer`] was given them.
        optimizer ([`~polyplex_research.utils.optimization.MuonWithAdamW`]):
            The optimizer the state is loaded into, whose own state gives the new layout.

    Returns:
        `dict`: The state in the new layout.
    """
    from .optimization import FactoredAdam

    if "tables" in state or optimizer.tables is None:
        return state
    matrices = {name for name, tensor in masters.items() if tensor.ndim == 2 and name not in DENSE_TABLES}
    old_rest = [name for name in masters if name not in matrices]
    template = optimizer.state_dict()
    adamw, tables = {}, {}
    rest = [name for name in old_rest if name not in DENSE_FACTORED_TABLES]
    factored = [name for name in old_rest if name in DENSE_FACTORED_TABLES]
    for index, name in enumerate(old_rest):
        entry = state["adamw"]["state"].get(index)
        if entry is None:
            continue
        if name in DENSE_FACTORED_TABLES:
            tables[factored.index(name)] = FactoredAdam.from_adam_state(entry)
        else:
            adamw[rest.index(name)] = entry
    upgraded = {"adamw": {"state": adamw, "param_groups": template["adamw"]["param_groups"]},
                "tables": {"state": tables, "param_groups": template["tables"]["param_groups"]}}
    if "muon" in state:
        upgraded["muon"] = state["muon"]
    return upgraded


def keep_trained_rows(gradient: torch.Tensor, first: int, last: int) -> torch.Tensor:
    r"""
    Leaves only the trained rows `[first, last)` of a table's gradient (SPEC S12.3a), dense or sparse.

    Args:
        gradient (`torch.Tensor`):
            The gradient of a table.
        first (`int`):
            First trained row.
        last (`int`):
            One past the last trained row.

    Returns:
        `torch.Tensor`: The gradient with every other row zero, or for a sparse gradient, absent.
    """
    if gradient.is_sparse:
        gradient = gradient.coalesce()
        index = gradient.indices()[0]
        keep = (index >= first) & (index < last)
        return torch.sparse_coo_tensor(
            gradient.indices()[:, keep], gradient.values()[keep], gradient.shape, is_coalesced=True
        )
    gradient[:first] = 0
    gradient[last:] = 0
    return gradient


def write_checkpoint(
    directory, step: int, trained: dict, masters: dict, optimizer, extra: dict, hand_over=None, buffers: dict | None = None
) -> list:
    r"""
    Writes a dense run's checkpoint (SPEC S13.4) as one file, `checkpoint.safetensors`, holding the trained parameters
    in their own dtype, the float32 masters of those that have one and the optimizer state, and then a small
    manifest with the step. Each is passed to `hand_over` as soon as it is written, the manifest last, so a store
    that holds the manifest holds the whole checkpoint.

    Keys are `weight/<name>`, `master/<name>`, `buffer/<name>` and `optimizer/<side>/<index>/<key>`.

    Args:
        directory (`str` or `Path`):
            Directory written to.
        step (`int`):
            Optimizer steps taken.
        trained (`dict`):
            Parameter name mapped to the trained parameter.
        masters (`dict`):
            Parameter name mapped to its float32 master, for the parameters that have one ([`dense_masters`]).
        optimizer ([`~polyplex_research.utils.optimization.MuonWithAdamW`] or `dict`):
            The optimizer, or its state dict.
        extra (`dict`):
            JSON data kept with the step, such as the documents a run has used.
        hand_over (`Callable`, *optional*):
            Called with the `Path` of each file once it is written.
        buffers (`dict`, *optional*):
            Untrained tensors the run needs to rebuild its model, such as the gains a shared stream rotation was folded
            with (SPEC S11.2h).

    Returns:
        `list[str]`: Names of the files, the manifest last.
    """
    import json
    from pathlib import Path

    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for name in trained:
        tensors[f"weight/{name}"] = trained[name].detach().contiguous().cpu()
        if name in masters:
            tensors[f"master/{name}"] = masters[name].detach().contiguous().cpu()
    for name, tensor in (buffers or {}).items():
        tensors[f"buffer/{name}"] = tensor.detach().contiguous().cpu()
    state = optimizer if isinstance(optimizer, dict) else optimizer.state_dict()
    groups = {}
    for side, inner in state.items():
        groups[side] = inner["param_groups"]
        for index, entries in inner["state"].items():
            for key, value in entries.items():
                tensors[f"optimizer/{side}/{index}/{key}"] = value.detach().contiguous().cpu()
    path = directory / "checkpoint.safetensors"
    save_file(tensors, str(path))
    del tensors
    names = [path.name]
    if hand_over is not None:
        hand_over(path)
    manifest = {"step": step, "files": names, "groups": groups, "extra": extra}
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1))
    names.append(path.name)
    if hand_over is not None:
        hand_over(path)
    return names


def read_checkpoint(directory, device="cpu", tensors: bool = True) -> dict:
    r"""
    Reads what [`write_checkpoint`] wrote: one `checkpoint.safetensors` or, from earlier runs, shards
    (`checkpoint-00001.safetensors`, ...) or one file per tensor group.

    Args:
        directory (`str` or `Path`):
            Directory holding the files.
        device (`str` or `torch.device`, *optional*, defaults to `"cpu"`):
            Device the tensors are loaded onto.
        tensors (`bool`, *optional*, defaults to `True`):
            Whether the tensors are read, or only the step and `extra` of the manifest.

    Returns:
        `dict`: `step`, `weights`, `masters` and `buffers` (name mapped to tensor), `optimizer` (a state dict for
        [`~polyplex_research.utils.optimization.MuonWithAdamW.load_state_dict`]) and `extra`; without `tensors`, only
        `step` and `extra`.
    """
    import json
    from pathlib import Path

    from safetensors.torch import load_file

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if not tensors:
        return {"step": manifest["step"], "extra": manifest["extra"]}
    weights, masters, buffers = {}, {}, {}
    optimizer = {side: {"state": {}, "param_groups": groups} for side, groups in manifest["groups"].items()}
    for name in manifest["files"]:
        loaded = load_file(str(directory / name), device=str(device))
        if name.startswith("checkpoint"):
            for key, tensor in loaded.items():
                kind, _, rest = key.partition("/")
                if kind == "weight":
                    weights[rest] = tensor
                elif kind == "master":
                    masters[rest] = tensor
                elif kind == "buffer":
                    buffers[rest] = tensor
                else:
                    side, index, entry = rest.split("/", 2)
                    optimizer[side]["state"].setdefault(int(index), {})[entry] = tensor
        elif name.startswith("weight-"):
            weights.update(loaded)
        elif name.startswith("master-"):
            masters.update(loaded)
        else:
            _, side, index = name[: -len(".safetensors")].split("-")
            optimizer[side]["state"][int(index)] = loaded
    return {"step": manifest["step"], "weights": weights, "masters": masters, "buffers": buffers,
            "optimizer": optimizer, "extra": manifest["extra"]}


def fold_checkpoint_alignment(source, run_dir, alignment: str) -> str:
    r"""
    Writes a dense checkpoint with a residual basis alignment folded into its language side (SPEC S11.2a), for a run
    that continues from it: every `language_norm` becomes 1 (weight and master), the language FFN weights the
    checkpoint holds (the appended layers', SPEC S11.4a) are folded with their layer's gain
    ([`~polyplex_research.utils.assemble.fold_language_weights`]), and the optimizer state of every changed parameter
    is left out, so it starts afresh in the new basis. The gains are written beside the alignment for the frozen
    layers, which assembly folds ([`~polyplex_research.utils.assemble.graft_layers`]).

    Args:
        source (`str` or `Path`):
            Checkpoint directory ([`write_checkpoint`]).
        run_dir (`str` or `Path`):
            Directory of the run that continues. The checkpoint is written to its `checkpoint-<step>` folder
            ([`save_local_checkpoint`]), so [`resume_checkpoint`] reads it at that step, and the
            alignment with the gains to `alignment.safetensors`.
        alignment (`str`):
            The residual basis alignment file ([`~polyplex_research.utils.assemble.residual_alignment`]).

    Returns:
        `str`: Path of the alignment file with the gains, for [`run_dense_adaptation`].

    Raises:
        ValueError: If a layer of the alignment has no `language_norm` in the checkpoint.
    """
    import re
    from pathlib import Path

    from safetensors import safe_open
    from safetensors.torch import save_file

    from .assemble import fold_language_weights

    run_dir = Path(run_dir)
    state = read_checkpoint(source)
    weights, masters = state["weights"], state["masters"]
    names = list(weights)
    norm = re.compile(r"model\.layers\.(\d+)\.mlp\.language_norm\.weight")
    projection = re.compile(r"model\.layers\.(\d+)\.mlp\.language_mlp\.(gate|up|down)_proj\.weight")
    changed = {name for name in names if norm.fullmatch(name) or projection.fullmatch(name)}

    gains = {}
    for name in names:
        match = norm.fullmatch(name)
        if match:
            gains[int(match.group(1))] = masters.get(name, weights[name]).float()
    rotations = safe_open(alignment, framework="pt")
    layers = rotations.get_slice("scales").get_shape()[0]
    missing = [layer for layer in range(layers) if layer not in gains]
    if missing:
        raise ValueError(f"The checkpoint has no language_norm for layers {missing}.")

    folded_layers = {int(projection.fullmatch(name).group(1)) for name in changed if projection.fullmatch(name)}
    for layer in sorted(folded_layers):
        prefix = f"model.layers.{layer}.mlp.language_mlp."
        parts = [weights[f"{prefix}{kind}_proj.weight"] for kind in ("gate", "up", "down")]
        rotation = rotations.get_slice("rotations")[layer]
        scale = float(rotations.get_slice("scales")[layer])
        for kind, tensor in zip(("gate", "up", "down"), fold_language_weights(*parts, rotation, scale, gains[layer])):
            weights[f"{prefix}{kind}_proj.weight"] = tensor
    for name in names:
        if norm.fullmatch(name):
            weights[name] = torch.ones_like(weights[name])
            if name in masters:
                masters[name] = torch.ones_like(masters[name])

    # The optimizer indices follow `dense_optimizer`'s split of the trained parameters, in their order.
    sides = {"muon": [], "adamw": [], "tables": []}
    for name in names:
        if name in DENSE_FACTORED_TABLES:
            sides["tables"].append(name)
        elif weights[name].ndim == 2 and name not in DENSE_TABLES:
            sides["muon"].append(name)
        else:
            sides["adamw"].append(name)
    optimizer = state["optimizer"]
    dropped = 0
    for side, members in sides.items():
        for index, name in enumerate(members):
            if name in changed and side in optimizer and optimizer[side]["state"].pop(index, None) is not None:
                dropped += 1
    extra = dict(state["extra"], folded_from=str(source), alignment=str(alignment))
    save_local_checkpoint(
        run_dir, state["step"], lambda path: write_checkpoint(path, state["step"], weights, masters, optimizer, extra)
    )

    output = run_dir / "alignment.safetensors"
    save_file(
        {
            "rotations": torch.stack([rotations.get_slice("rotations")[layer] for layer in range(layers)]).contiguous(),
            "scales": rotations.get_slice("scales")[:],
            "gains": torch.stack([gains[layer] for layer in range(layers)]),
        },
        str(output),
    )
    logger.info("Folded the alignment into %d language FFNs and %d gains; %d optimizer states start afresh.",
                len(folded_layers), len(gains), dropped)
    return str(output)


def replace_checkpoint(directory, write) -> None:
    r"""
    Writes a local checkpoint beside the previous one and swaps it in only once it is whole (SPEC S13.4), so a run
    stopped while saving keeps its last complete checkpoint.

    Args:
        directory (`str` or `Path`):
            The checkpoint directory.
        write (`Callable[[Path], object]`):
            Writes a checkpoint into the directory it is given ([`write_checkpoint`]).
    """
    import shutil
    from pathlib import Path

    directory = Path(directory)
    partial = directory.with_name(directory.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    write(partial)
    previous = directory.with_name(directory.name + ".previous")
    if directory.exists():
        directory.rename(previous)
    partial.rename(directory)
    if previous.exists():
        shutil.rmtree(previous)


def save_local_checkpoint(work_dir, step: int, write):
    r"""
    Writes the checkpoint of `step` into its own `checkpoint-<step>` directory under the run directory (SPEC S13.4),
    so every save is kept. The directory appears only once it is whole ([`replace_checkpoint`]).

    Args:
        work_dir (`str` or `Path`):
            Directory of the run.
        step (`int`):
            Step the checkpoint holds.
        write (`Callable[[Path], object]`):
            Writes a checkpoint into the directory it is given ([`write_checkpoint`]).

    Returns:
        `Path`: The checkpoint directory.
    """
    from pathlib import Path

    directory = Path(work_dir) / f"checkpoint-{step}"
    replace_checkpoint(directory, write)
    return directory


def local_checkpoints(run_dir) -> list:
    r"""
    Lists the whole checkpoints a dense run left on the local disk: every `checkpoint-<step>` directory and the
    single `checkpoint` directory older runs wrote, skipping `.partial` and `.previous` directories of a save that
    did not finish.

    Args:
        run_dir (`str` or `Path`):
            Directory of the run.

    Returns:
        `list[Path]`: The checkpoint directories that hold a manifest.
    """
    from pathlib import Path

    return [
        path.parent
        for path in Path(run_dir).glob("checkpoint*/manifest.json")
        if path.parent.name == "checkpoint" or re.fullmatch(r"checkpoint-\d+", path.parent.name)
    ]


def dense_rank() -> tuple[int, int]:
    r"""
    Rank of this process and the number of processes of a data parallel dense run (SPEC S13.4b).

    Returns:
        `tuple[int, int]`: `(rank, world_size)`, `(0, 1)` outside a process group.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


# Bytes of dense gradients reduced in one call, so the flat copy stays small beside a nearly full device.
GRADIENT_BUCKET_BYTES = 256 * 2**20


def average_gradients(parameters: list) -> None:
    r"""
    Averages the gradients of `parameters` over the processes of the group, in place (SPEC S13.4b). Dense gradients
    are reduced in buckets of [`GRADIENT_BUCKET_BYTES`]; a sparse one is gathered as its row indices and values and
    summed.

    Args:
        parameters (`list[torch.nn.Parameter]`):
            The trained parameters, in the same order on every process.
    """
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

    world = torch.distributed.get_world_size()
    bucket, size = [], 0

    def reduce(gradients):
        flat = _flatten_dense_tensors(gradients)
        torch.distributed.all_reduce(flat)
        flat.div_(world)
        for gradient, reduced in zip(gradients, _unflatten_dense_tensors(flat, gradients)):
            gradient.copy_(reduced)

    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        if gradient.is_sparse:
            parameter.grad = gather_sparse_gradient(gradient, world)
            continue
        if bucket and (bucket[0].dtype != gradient.dtype or size + gradient.nbytes > GRADIENT_BUCKET_BYTES):
            reduce(bucket)
            bucket, size = [], 0
        bucket.append(gradient)
        size += gradient.nbytes
    if bucket:
        reduce(bucket)


def gather_sparse_gradient(gradient: torch.Tensor, world: int) -> torch.Tensor:
    r"""
    The mean over the processes of a sparse row gradient, such as an embedding table's.

    Args:
        gradient (`torch.Tensor`):
            Sparse COO gradient whose first dimension indexes rows.
        world (`int`):
            Number of processes.

    Returns:
        `torch.Tensor`: The coalesced sparse mean.
    """
    gradient = gradient.coalesce()
    indices, values = gradient.indices(), gradient.values()
    count = torch.tensor([indices.shape[1]], device=values.device)
    counts = [torch.zeros_like(count) for _ in range(world)]
    torch.distributed.all_gather(counts, count)
    longest = int(max(counts).item())
    padded_indices = indices.new_zeros((indices.shape[0], longest))
    padded_indices[:, : indices.shape[1]] = indices
    padded_values = values.new_zeros((longest, *values.shape[1:]))
    padded_values[: values.shape[0]] = values
    all_indices = [torch.empty_like(padded_indices) for _ in range(world)]
    all_values = [torch.empty_like(padded_values) for _ in range(world)]
    torch.distributed.all_gather(all_indices, padded_indices)
    torch.distributed.all_gather(all_values, padded_values)
    lengths = [int(value.item()) for value in counts]
    indices = torch.cat([part[:, :length] for part, length in zip(all_indices, lengths)], dim=1)
    values = torch.cat([part[:length] for part, length in zip(all_values, lengths)]) / world
    return torch.sparse_coo_tensor(indices, values, gradient.shape).coalesce()


def average_scalar(value: float) -> float:
    r"""
    The mean of a number over the processes of the group.

    Args:
        value (`float`):
            This process's number.

    Returns:
        `float`: The mean.
    """
    rank, world = dense_rank()
    nccl = torch.distributed.get_backend() == "nccl"
    device = f"cuda:{torch.cuda.current_device()}" if nccl else "cpu"
    tensor = torch.tensor([float(value)], device=device, dtype=torch.float64)
    torch.distributed.all_reduce(tensor)
    return tensor.item() / world


def share_from_first(trained: dict, masters: dict) -> None:
    r"""
    Copies the trained parameters and their float32 masters of the first process to the others, so the small numeric
    differences of their optimizer steps do not add up (SPEC S13.4b).

    Args:
        trained (`dict`):
            Trained parameters by name.
        masters (`dict`):
            Float32 masters by name.
    """
    with torch.no_grad():
        for name in sorted(trained):
            torch.distributed.broadcast(trained[name].data, src=0)
        for name in sorted(masters):
            torch.distributed.broadcast(masters[name].data, src=0)


def dense_data_parallel(args: list[str], gpus: int) -> dict:
    r"""
    Runs [`dense_main`] as one process per GPU of this machine, joined by NCCL (SPEC S13.4b), and returns the first
    process's result. SIGTERM and SIGINT are passed on to every process, so they all save the same step and stop.

    Args:
        args (`list[str]`):
            Command line of [`dense_main`] without `--gpus`.
        gpus (`int`):
            Processes started, one per visible GPU.

    Returns:
        `dict`: The result of [`run_dense_adaptation`] on the first process.

    Raises:
        RuntimeError: If a process fails.
    """
    import signal
    import socket
    import subprocess
    import sys
    import tempfile
    import time
    from pathlib import Path

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    root = str(Path(__file__).resolve().parents[2])
    result_path = os.path.join(tempfile.mkdtemp(prefix="dense-result-"), "result.json")
    processes = []
    for rank in range(gpus):
        environment = {
            **os.environ,
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(gpus),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "POLYPLEX_DENSE_RESULT": result_path,
            "PYTHONPATH": os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")])),
        }
        command = [sys.executable, "-c", "import sys; from polyplex_research.utils.phase import dense_main; dense_main(sys.argv[1:])"]
        processes.append(subprocess.Popen(command + list(args), env=environment))

    def forward(signum, frame):
        for process in processes:
            if process.poll() is None:
                process.send_signal(signum)

    previous = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            previous[signum] = signal.signal(signum, forward)
        except ValueError:
            logger.warning("Signal %d is not passed on outside the main thread.", signum)
    try:
        failed_at = None
        while any(process.poll() is None for process in processes):
            if failed_at is None and any(process.poll() not in (None, 0) for process in processes):
                # The others wait in a collective that will not finish.
                failed_at = time.monotonic()
                forward(signal.SIGTERM, None)
            if failed_at is not None and time.monotonic() - failed_at > 600:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
            time.sleep(5)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    codes = [process.returncode for process in processes]
    if any(codes):
        raise RuntimeError(f"Data parallel dense run failed, exit codes by rank: {codes}")
    with open(result_path, encoding="utf-8") as file:
        return json.load(file)


def train_dense(
    model,
    collate,
    train_dataset,
    eval_sets: dict,
    reference: dict,
    steps: int,
    batch_size: int,
    micro_batch_size: int,
    learning_rate: float,
    warmup_steps: int,
    eval_every: int,
    log=None,
    seed: int = 0,
    evaluate_fn=None,
    decay_steps: int = 0,
    save_every: int = 500,
    lr_schedule: str = "wsd",
    checkpoint=None,
    stop=None,
    resume_state: dict | None = None,
    evaluate_at_start: bool = False,
    stream_switch=None,
    switch_step: int = 8000,
) -> dict:
    r"""
    Trains what [`dense_trainable_parameters`] leaves trainable on packed text rows (SPEC S13.4).

    Vectors and the composition coefficients keep a float32 copy the optimizer steps ([`dense_masters`]), written back
    into the model's dtype after every step, since a bfloat16 weight near 1 does not register an update below 1/128.
    The tables and weight matrices are stepped in their own dtype. Every row is read at most once.

    Args:
        model ([`PolyplexForConditionalGeneration`]):
            The assembled model, on its device.
        collate (`Callable`):
            [`~polyplex_research.datasets.duplex_data.dense_collate_fn`] of the model.
        train_dataset (`Dataset`):
            Rows with a `segments` list of token tensors.
        eval_sets (`dict`):
            Rows of each evaluated language, by language.
        reference (`dict`):
            Cross entropy of the text LLM on each evaluated language, by language.
        steps (`int`):
            Optimizer steps, fewer when the rows run out first.
        batch_size (`int`):
            Rows per optimizer step.
        micro_batch_size (`int`):
            Rows per forward pass. Must divide `batch_size`.
        learning_rate (`float`):
            Peak learning rate of [`dense_optimizer`].
        warmup_steps (`int`):
            Linear warmup of the schedule ([`wsd_factor`]), counted over the whole adaptation.
        eval_every (`int`):
            Steps between evaluations. A fresh run is also evaluated before its first step, and every run after its
            last.
        log (`Callable`, *optional*):
            Receives a dict of metrics with a `step` field after every step and every evaluation.
        seed (`int`, *optional*, defaults to 0):
            Seed of the row order.
        evaluate_fn (`Callable`, *optional*):
            Takes the model and returns the metrics of one evaluation, in place of the per-token cross entropy of
            `eval_sets` against `reference`.
        decay_steps (`int`, *optional*, defaults to 0):
            Steps of learning rate decay that end this run. A run without decay can be continued at the same rate.
        save_every (`int`, *optional*, defaults to 500):
            Steps between checkpoints. A checkpoint is also taken after the last step, when `stop` asks to stop, and
            for the last finished step when a step fails.
        lr_schedule (`str`, *optional*, defaults to `"wsd"`):
            `"wsd"` for [`wsd_factor`], `"cosine-restarts"` for [`cosine_restart_factor`] with cycles of `save_every`.
        checkpoint (`Callable`, *optional*):
            Called as `checkpoint(step, trained, masters, optimizer)` with the step reached, the trained parameters
            and the float32 masters by name, and the optimizer.
        stop (`Callable`, *optional*):
            Asked after every step whether to stop, as when the process is told to terminate.
        resume_state (`dict`, *optional*):
            `step`, `weights`, `masters` and `optimizer` of a checkpoint ([`read_checkpoint`]) to continue from.
        evaluate_at_start (`bool`, *optional*, defaults to `False`):
            Whether a resumed run is evaluated before its first step.
        stream_switch (`Callable`, *optional*):
            Called with the model to move its language stream to the shared, folded rotation (SPEC S11.2h). It is
            called before the first step past `switch_step` of a model whose stream is not folded yet; the trained
            parameters it changes (each `language_norm` and the language FFN weights) take their new values as masters
            and start their optimizer state afresh.
        switch_step (`int`, *optional*, defaults to 8000):
            The last step trained with the per-layer rotations.

    Returns:
        `dict`: `steps` (the last step reached, counted over the whole adaptation), `evaluations` (the metrics of
        each evaluation) and `train_loss` (the last step's).
    """
    import time

    from torch.utils.data import DataLoader

    rank, world = dense_rank()
    if batch_size % (micro_batch_size * world):
        raise ValueError(
            f"`micro_batch_size` ({micro_batch_size}) times the {world} processes must divide `batch_size` ({batch_size})."
        )
    log = (log or (lambda metrics: None)) if rank == 0 else (lambda metrics: None)
    stop = stop or (lambda: False)
    device = next(model.parameters()).device
    names = dense_trainable_parameters(model)
    trained_rows = frozen_text_rows(model, names)
    # The input table's gradient holds only the rows a batch read (SPEC S7.2a).
    input_table = model.get_input_embeddings() if "model.embed_tokens.weight" in names else None
    sparse_before = input_table.sparse if input_table is not None else None
    if input_table is not None:
        input_table.sparse = True
    language_table = getattr(model, "language_embed_tokens", None) if "language_embed_tokens.weight" in names else None
    if language_table is not None:
        language_table.sparse = True
    parameters = dict(model.named_parameters())
    trained = {name: parameters[name] for name in names}
    mastered = dense_masters(trained)
    masters = {name: trained[name].detach().float().clone().requires_grad_(True) for name in trained if name in mastered}
    # Mean absolute language FFN output scale (SPEC S11.2c), logged each step.
    scales = [parameter for name, parameter in trained.items() if name.endswith(".mlp.language_scale")]
    start_step = 0
    if resume_state is not None:
        start_step = resume_state["step"]
        with torch.no_grad():
            for name, parameter in trained.items():
                parameter.copy_(resume_state["weights"][name].to(parameter.device))
            for name, master in masters.items():
                master.copy_(resume_state["masters"][name].to(master.device))
                trained[name].copy_(master)
    stepped = {name: masters.get(name, parameter) for name, parameter in trained.items()}
    optimizer = dense_optimizer(stepped, learning_rate)
    if resume_state is not None:
        optimizer.load_state_dict(upgrade_dense_optimizer_state(resume_state["optimizer"], stepped, optimizer))
        for key in ("weights", "masters", "optimizer"):
            resume_state.pop(key, None)
    accumulation = batch_size // (micro_batch_size * world)
    end_step = start_step + min(steps, len(train_dataset) // batch_size)

    def set_learning_rate(step: int) -> float:
        if lr_schedule == "cosine-restarts":
            rate = learning_rate * cosine_restart_factor(step, warmup_steps, save_every)
        else:
            rate = learning_rate * wsd_factor(step, warmup_steps, end_step, decay_steps)
        for group in optimizer.param_groups:
            group["lr"] = rate
        return rate
    # Rows are collated only by the process that reads them, so every process walks the same row order.
    loader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda rows: rows,
        generator=torch.Generator().manual_seed(seed),
    )

    evaluations = []

    def switch_stream(step: int):
        stream_switch(model)
        changed = [name for name in trained if ".mlp.language_norm." in name or ".mlp.language_mlp." in name]
        with torch.no_grad():
            for name in changed:
                if name in masters:
                    masters[name].copy_(trained[name].float())
        reset_optimizer_state(optimizer, [stepped[name] for name in changed])
        logger.info("Step %d moved the language stream to the shared rotation; reset %d parameters.", step, len(changed))

    def evaluate(step: int):
        if rank != 0:
            torch.distributed.barrier()
            return
        metrics = {"step": step}
        if evaluate_fn is not None:
            metrics.update(evaluate_fn(model))
        for language, rows in ({} if evaluate_fn is not None else eval_sets).items():
            loss = dense_evaluate(model, collate, rows, micro_batch_size=micro_batch_size)
            metrics[f"eval/{language}_loss"] = loss
            metrics[f"eval/{language}_gap"] = loss - reference[language]
        evaluations.append(metrics)
        log(metrics)
        if world > 1:
            torch.distributed.barrier()

    if start_step == 0 or evaluate_at_start:
        evaluate(start_step)
    model.train()
    batches = iter(loader)
    train_loss = None
    finished = start_step
    # `POLYPLEX_PROFILE_STEPS=n` profiles steps 3 to 2 + n and logs where the device time goes.
    profiled = int(os.environ.get("POLYPLEX_PROFILE_STEPS", "0"))
    profiler = None
    try:
        for step in range(start_step + 1, end_step + 1):
            if profiled and step == start_step + 3:
                profiler = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
                )
                profiler.__enter__()
            if profiler is not None and step == start_step + 3 + profiled:
                torch.cuda.synchronize()
                profiler.__exit__(None, None, None)
                table = profiler.key_averages().table(
                    sort_by="self_cuda_time_total", row_limit=40, max_name_column_width=90
                )
                # Printed, since the run's logger may have no handler on a letify worker.
                print(f"Profile of {profiled} steps:\n{table}", file=sys.stderr, flush=True)
                profiler = None
            if stream_switch is not None and step > switch_step and not model.config.language_stream_folded:
                switch_stream(step)
            started = time.perf_counter()
            rate = set_learning_rate(step)
            total = 0.0
            for _ in range(accumulation):
                # Process r reads the r-th of every `world` micro batches of the single process order.
                rows = [next(batches) for _ in range(world)][rank]
                batch = {key: value.to(device) for key, value in collate(rows).items()}
                loss = model(**batch, use_cache=False).loss / accumulation
                loss.backward()
                total += loss.item()
            if world > 1:
                average_gradients(list(trained.values()))
                total = average_scalar(total)
            for name, parameter in trained.items():
                if name in masters:
                    masters[name].grad = parameter.grad.float()
                    parameter.grad = None
                if parameter in trained_rows and stepped[name].grad is not None:
                    stepped[name].grad = keep_trained_rows(stepped[name].grad, *trained_rows[parameter])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                for name, master in masters.items():
                    trained[name].copy_(master)
            finished = step
            train_loss = total
            metrics = {
                "step": step,
                "train/loss": total,
                "train/learning_rate": rate,
                "train/batch_size": batch_size,
                "train/seconds_per_step": time.perf_counter() - started,
            }
            if scales:
                metrics["train/language_scale"] = torch.stack([scale.abs().mean() for scale in scales]).mean().item()
            log(metrics)
            stopping = stop()
            if world > 1:
                # Every process stops at the step any of them was asked to.
                stopping = average_scalar(float(stopping)) > 0
            if step % eval_every == 0 or step == end_step:
                evaluate(step)
                model.train()
            if checkpoint is not None and (step % save_every == 0 or step == end_step or stopping):
                if world > 1:
                    share_from_first(trained, masters)
                if rank == 0:
                    checkpoint(step, trained, masters, optimizer)
                if world > 1:
                    torch.distributed.barrier()
            if stopping:
                break
    except BaseException:
        # The masters and the optimizer still hold the last finished step, since a failing step stops before
        # `optimizer.step`.
        if checkpoint is not None and finished > start_step and rank == 0:
            for parameter in trained.values():
                parameter.grad = None
            checkpoint(finished, trained, masters, optimizer)
        raise
    finally:
        if input_table is not None:
            input_table.sparse = sparse_before
    return {"steps": finished, "evaluations": evaluations, "train_loss": train_loss}


def dense_documents(
    language: str,
    split: str,
    tokens: int,
    tokenizer,
    max_files: int = 1,
    skip: int = 0,
    pieces: list | None = None,
    filtered: bool = False,
    exclude: set | None = None,
    consumed: list | None = None,
) -> list:
    r"""
    Reads documents of one language ([`~polyplex_research.datasets.webtext.MultilingualEduCorpus`]) until they hold
    `tokens` pieces of `tokenizer`, streaming the parquet files row group by row group so only what is read is fetched.
    With `POLYPLEX_DOCUMENT_CACHE` set and no `exclude`, the documents of each request are kept there as one JSON file
    and read back from it the next time.

    Args:
        language (`str`):
            Key of [`~polyplex_research.datasets.webtext.DENSE_EVALUATION`] or a FineWeb-2 language code.
        split (`str`):
            `"train"` or `"valid"`, see [`~polyplex_research.datasets.webtext.held_out`].
        tokens (`int`):
            Pieces to collect.
        tokenizer (`PreTrainedTokenizerBase`):
            Tokenizer of the text stream.
        max_files (`int`, *optional*, defaults to 1):
            Parquet files of the source read.
        skip (`int`, *optional*, defaults to 0):
            Leading documents of the source passed over, the ones an earlier run already read.
        pieces (`list`, *optional*):
            When given, the pieces of every returned document, its whitespace collapsed, are appended to it.
        filtered (`bool`, *optional*, defaults to `False`):
            Whether the language's quality filtered source is read (SPEC S13.4). The evaluation reads the unfiltered
            one.
        exclude (`set`, *optional*):
            [`~polyplex_research.datasets.webtext.text_key`] digests of documents left out, the ones a run already
            trained on from another source (AGENTS H18).
        consumed (`list`, *optional*):
            When given, the number of source documents read past `skip`, the returned ones and the excluded ones, is
            appended to it.

    Returns:
        `list`: The documents, fewer than asked when the source runs out.
    """
    import hashlib

    corpus = MultilingualEduCorpus(language, split=split, filtered=filtered, max_files=max_files)
    cache = os.environ.get("POLYPLEX_DOCUMENT_CACHE")
    path = None
    if cache and exclude is None:
        request = [*corpus.request, split, tokens, max_files, skip, getattr(tokenizer, "name_or_path", ""), len(tokenizer)]
        digest = hashlib.sha256(json.dumps(request).encode()).hexdigest()[:16]
        path = os.path.join(cache, f"{language}-{split}-{digest}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as file:
                documents = json.load(file)
            if pieces is not None:
                pieces.extend(tokenizer.encode(" ".join(text.split()), add_special_tokens=False) for text in documents)
            if consumed is not None:
                consumed.append(len(documents))
            return documents
    documents, collected, read = [], 0, 0
    for index, text in enumerate(corpus):
        if index < skip:
            continue
        read += 1
        if exclude is not None and text_key(text) in exclude:
            continue
        documents.append(text)
        encoded = tokenizer.encode(" ".join(text.split()), add_special_tokens=False)
        if pieces is not None:
            pieces.append(encoded)
        collected += len(encoded)
        if collected >= tokens:
            break
    if consumed is not None:
        consumed.append(read)
    if path is not None:
        os.makedirs(cache, exist_ok=True)
        with open(path + ".partial", "w", encoding="utf-8") as file:
            json.dump(documents, file, ensure_ascii=False)
        os.replace(path + ".partial", path)
    return documents


def trained_digests(language: str, count: int, max_files: int = 1) -> set:
    r"""
    Digests of the first `count` training documents of a language's unfiltered source, the ones a run that read that
    source has trained on, so a run moving to the filtered source leaves them out (SPEC S13.4, AGENTS H18).

    Args:
        language (`str`):
            Key of [`~polyplex_research.datasets.webtext.DENSE_EVALUATION`] or a FineWeb-2 language code.
        count (`int`):
            Documents already trained on.
        max_files (`int`, *optional*, defaults to 1):
            Parquet files of the source read.

    Returns:
        `set`: Their [`~polyplex_research.datasets.webtext.text_key`] digests.
    """
    digests = set()
    if count <= 0:
        return digests
    for index, text in enumerate(MultilingualEduCorpus(language, filtered=False, max_files=max_files)):
        if index >= count:
            break
        digests.add(text_key(text))
    return digests


def dense_rows(name: str, documents: list, tokenizer, work_dir: str, row_frames: int, pieces: list | None = None):
    r"""
    Writes documents as packed rows under `work_dir/name` and reads them back.

    Args:
        name (`str`):
            Name of the cache, also the stem of its documents file.
        documents (`list`):
            Document strings.
        tokenizer (`PreTrainedTokenizerBase`):
            Tokenizer of the text stream.
        work_dir (`str`):
            Directory the documents and the cache are written to.
        row_frames (`int`):
            Frames per row, opening frames included.
        pieces (`list`, *optional*):
            Pieces of every document from [`dense_documents`], so the documents are not encoded again.

    Returns:
        [`~polyplex_research.datasets.phase_data.DenseTextDataset`]: The rows.
    """
    from ..datasets.phase_data import DenseTextDataset, write_dense_cache
    from ..datasets.webtext import write_documents

    path = os.path.join(work_dir, f"{name}.jsonl")
    write_documents([documents], path)
    write_dense_cache(path, tokenizer, os.path.join(work_dir, name), max_tokens=row_frames, pieces=pieces)
    return DenseTextDataset(os.path.join(work_dir, name))


def dense_eval_rows(language: str, tokenizer, work_dir: str, eval_rows: int, row_frames: int, max_files: int) -> list:
    r"""
    The held out rows the dense quick check scores one language on: the first `eval_rows` rows packed from the
    language's `"valid"` documents, read in file order.

    Args:
        language (`str`):
            Key of [`~polyplex_research.datasets.webtext.DENSE_EVALUATION`].
        tokenizer (`PreTrainedTokenizerBase`):
            Tokenizer of the text stream.
        work_dir (`str`):
            Directory the cache is written to, as `valid_<language>`.
        eval_rows (`int`):
            Rows taken.
        row_frames (`int`):
            Frames per row, opening frames included.
        max_files (`int`):
            Parquet files read.

    Returns:
        `list`: The rows, each with a `segments` list of token tensors.
    """
    documents = dense_documents(language, "valid", eval_rows * row_frames, tokenizer, max_files)
    rows = dense_rows(f"valid_{language}", documents, tokenizer, work_dir, row_frames)
    return [rows[index] for index in range(min(eval_rows, len(rows)))]


@contextlib.contextmanager
def depth_decoder_on(model, device):
    r"""
    Keeps a DualFFN model's depth decoder on `device` while text alone trains (SPEC S13.4), and puts it back beside the
    backbone on exit. The text loss never reads the depth decoder.

    Args:
        model ([`PolyplexForConditionalGeneration`]):
            The model.
        device (`str` or `torch.device`):
            Where the depth decoder waits.
    """
    home = next(model.model.parameters()).device
    model.depth_decoder.to(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        yield model
    finally:
        model.depth_decoder.to(home)


def gate_language_ffn(model, value: float = 0.0):
    r"""
    Gives every DualFFN layer a per-dimension scale on its language FFN output, filled with `value` (SPEC S11.2c), and
    records it in the configuration so a saved model reloads with it. With the default 0 the model computes exactly
    what it computes with the language FFN off ([`language_ffn_off`]). Call it before the layers are compiled.

    Args:
        model ([`PolyplexForConditionalGeneration`]):
            The model.
        value (`float`, *optional*, defaults to 0.0):
            The scale every dimension starts at.

    Returns:
        [`PolyplexForConditionalGeneration`]: The same model.
    """
    model.config.language_scale = True
    for layer in model.model.layers:
        weight = layer.mlp.language_norm.weight
        layer.mlp.language_scale = nn.Parameter(torch.full_like(weight, value))
    return model


def open_language_stream(model, alignment: str):
    r"""
    Gives a DualFFN model a second residual stream in the text LLM's basis for its language FFNs (SPEC S11.2d), joined
    to the backbone's by the rotations and scales of a residual basis alignment, and a language FFN output scale of 0
    ([`gate_language_ffn`]), so the model starts out computing what it computes with the language FFN off. The
    language FFNs must be the text LLM's own, with no alignment folded in. Call it before the layers are compiled.

    Args:
        model ([`PolyplexForConditionalGeneration`]):
            The model.
        alignment (`str`):
            A file [`~polyplex_research.utils.assemble.residual_alignment`] wrote.

    Returns:
        [`PolyplexForConditionalGeneration`]: The same model.
    """
    from .assemble import load_residual_alignment

    gate_language_ffn(model)
    model.config.language_stream = True
    layer_alignment = load_residual_alignment(alignment)
    for index, layer in enumerate(model.model.layers):
        rotation, scale, _ = layer_alignment(index)
        weight = layer.mlp.language_norm.weight
        layer.mlp.register_buffer("stream_rotation", rotation.to(weight.device, weight.dtype))
        layer.mlp.register_buffer("stream_scale", torch.tensor(scale, device=weight.device, dtype=weight.dtype))
        layer.mlp_has_stream = True
    return model


def use_flex_attention(model):
    r"""
    Runs the attention of a model on packed dense rows with flex attention (SPEC S13.4): the mask that keeps each
    packed document to itself is then a block mask whose empty blocks are skipped, where SDPA reads it as a dense mask
    and falls back to its memory efficient kernel. The weights are unchanged, so inference keeps any implementation.

    Args:
        model ([`PreTrainedModel`]):
            The model.

    Returns:
        [`PreTrainedModel`]: The same model.
    """
    model.set_attn_implementation("flex_attention")
    return model


def open_language_lora(model, rank: int):
    r"""
    Gives every layer's language FFN a low rank correction of rank `rank` beside each projection (SPEC S11.2g), its
    $B$ at zero so the model's output is unchanged. Call it before a checkpoint holding the corrections is read.

    Args:
        model ([`PolyplexForConditionalGeneration`]):
            The model.
        rank (`int`):
            Rank of the corrections.

    Returns:
        [`PolyplexForConditionalGeneration`]: The same model.
    """
    for layer in model.model.layers:
        layer.mlp.language_mlp.add_lora(rank)
    model.config.language_lora_rank = rank
    return model


def open_language_embeddings(model, tokenizer, language_model: str):
    r"""
    Gives the language stream of a DualFFN model a text table of its own in the text LLM's basis (SPEC S11.2f), each
    row the mean of the text LLM's input embeddings over its text ([`~assemble.language_embedding_rows`]). Call it
    after [`open_language_stream`] and before a checkpoint holding the table is read.

    Args:
        model ([`PolyplexForConditionalGeneration`]):
            The model.
        tokenizer ([`PolyplexTokenizer`]):
            The model's text vocabulary.
        language_model (`str`):
            Hub id of the text LLM.

    Returns:
        [`PolyplexForConditionalGeneration`]: The same model.
    """
    from transformers import AutoTokenizer

    from .assemble import language_embedding_rows, load_language_table

    table = model.get_input_embeddings()
    rows = language_embedding_rows(
        tokenizer, AutoTokenizer.from_pretrained(language_model), load_language_table(language_model), table.num_embeddings
    )
    model.config.language_embeddings = True
    model.language_embed_tokens = nn.Embedding(
        rows.shape[0], rows.shape[1], device=table.weight.device, dtype=table.weight.dtype
    )
    with torch.no_grad():
        model.language_embed_tokens.weight.copy_(rows)
    return model


@contextlib.contextmanager
def language_ffn_off(model, layers=None):
    r"""
    Silences the language FFNs of a DualFFN model, so it reads the Helium path alone there, and restores them on exit.
    Each language FFN is swapped for a module that returns zeros, so an NVFP4 projection without a full precision
    weight is silenced the same way, and a compiled layer, which does not see hooks added after it was compiled,
    recompiles for the swap. With a language stream (SPEC S11.2d) a silenced FFN adds nothing to the stream either.

    Args:
        model ([`PolyplexForConditionalGeneration`]):
            The model.
        layers (`Iterable[int]`, *optional*):
            Layers whose language FFN is silenced. Every layer when omitted.
    """
    chosen = range(len(model.model.layers)) if layers is None else set(layers)
    saved = {}
    for index, layer in enumerate(model.model.layers):
        if index in chosen:
            saved[index] = layer.mlp.language_mlp
            layer.mlp.language_mlp = SilentFFN()
    try:
        yield model
    finally:
        for index, mlp in saved.items():
            model.model.layers[index].mlp.language_mlp = mlp


@contextlib.contextmanager
def language_scale_multiplied(model, factor: float, layers=None):
    r"""
    Multiplies the language FFN output scale (SPEC S11.2c) of a DualFFN model by `factor` in place, and restores it on
    exit. The scale is a parameter, so a compiled layer reads the new value without recompiling.

    Args:
        model ([`PolyplexForConditionalGeneration`]):
            The model, gated with [`gate_language_ffn`].
        factor (`float`):
            The factor.
        layers (`Iterable[int]`, *optional*):
            Layers whose scale is multiplied. Every layer when omitted.
    """
    chosen = range(len(model.model.layers)) if layers is None else set(layers)
    saved = []
    with torch.no_grad():
        for index, layer in enumerate(model.model.layers):
            if index in chosen:
                scale = layer.mlp.language_scale
                saved.append((scale, scale.detach().clone()))
                scale.mul_(factor)
    try:
        yield model
    finally:
        with torch.no_grad():
            for scale, value in saved:
                scale.copy_(value)


@contextlib.contextmanager
def language_ffn_scaled(model, scale: float):
    r"""
    Multiplies every language FFN's output of a DualFFN model by `scale`, and restores them on exit. Each language FFN
    is swapped for a module that scales it, so a compiled layer recompiles for the swap ([`language_ffn_off`]).

    Args:
        model ([`PolyplexForConditionalGeneration`]):
            The model.
        scale (`float`):
            The factor.
    """
    saved = [layer.mlp.language_mlp for layer in model.model.layers]
    for layer, mlp in zip(model.model.layers, saved):
        layer.mlp.language_mlp = ScaledFFN(mlp, scale)
    try:
        yield model
    finally:
        for layer, mlp in zip(model.model.layers, saved):
            layer.mlp.language_mlp = mlp


class ScaledFFN(nn.Module):
    r"""
    Stands in for a language FFN whose output is multiplied by `scale` ([`language_ffn_scaled`]).

    Args:
        ffn (`nn.Module`):
            The language FFN.
        scale (`float`):
            The factor.
    """

    def __init__(self, ffn: nn.Module, scale: float):
        super().__init__()
        self.ffn = ffn
        self.scale = scale

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.ffn(hidden_states) * self.scale


class SilentFFN(nn.Module):
    r"""
    Stands in for a language FFN while it is silenced ([`language_ffn_off`]): returns zeros of its input's shape.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(hidden_states)


def language_ffn_off_bits_per_byte(model, collate, tokenizer, texts: list, byte_lengths: torch.Tensor) -> float:
    r"""
    [`dense_bits_per_byte`] of a DualFFN model with every language FFN silenced ([`language_ffn_off`]).

    Args:
        model ([`PolyplexForConditionalGeneration`]):
            The model.
        collate (`Callable`):
            [`~polyplex_research.datasets.duplex_data.dense_collate_fn`] of the model.
        tokenizer (`PreTrainedTokenizerBase`):
            The model's tokenizer.
        texts (`list[str]`):
            The texts.
        byte_lengths (`torch.Tensor`):
            Bytes of each id, from [`token_byte_lengths`].

    Returns:
        `float`: Bits per byte.
    """
    with language_ffn_off(model):
        return dense_bits_per_byte(model, collate, tokenizer, texts, byte_lengths)


@torch.no_grad()
def mean_probabilities(model, collate, tokenizer, texts: list) -> torch.Tensor:
    r"""
    Mean softmax a DualFFN model gives each id, over the positions [`dense_bits_per_byte`] scores.

    Args:
        model ([`PolyplexForConditionalGeneration`]):
            The model.
        collate (`Callable`):
            [`~polyplex_research.datasets.duplex_data.dense_collate_fn`] of the model.
        tokenizer (`PreTrainedTokenizerBase`):
            The model's tokenizer.
        texts (`list[str]`):
            The texts.

    Returns:
        `torch.Tensor` of shape `(vocab_size,)`: The mean probability of each id, on the CPU in float32.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    total, positions = None, 0
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 2:
            continue
        batch = {key: value.to(device) for key, value in collate([{"segments": [torch.tensor(ids)]}]).items()}
        batch.pop("text_labels")
        probabilities = model(**batch, use_cache=False).logits[0, 1:-1].float().softmax(-1)
        summed = probabilities.sum(0).cpu()
        total = summed if total is None else total + summed
        positions += len(probabilities)
    model.train(was_training)
    return total / max(positions, 1)


def new_token_share(model, collate, tokenizer, texts: list, base: int) -> float:
    r"""
    Mean probability a DualFFN model gives the ids past the base vocabulary, at the positions
    [`dense_bits_per_byte`] scores.

    Args:
        model ([`PolyplexForConditionalGeneration`]):
            The model.
        collate (`Callable`):
            [`~polyplex_research.datasets.duplex_data.dense_collate_fn`] of the model.
        tokenizer (`PreTrainedTokenizerBase`):
            The model's tokenizer.
        texts (`list[str]`):
            The texts.
        base (`int`):
            Size of the base vocabulary, Helium's.

    Returns:
        `float`: The mean probability of the ids at or above `base`.
    """
    return mean_probabilities(model, collate, tokenizer, texts)[base:].sum().item()


def norm_summary(norms: torch.Tensor) -> dict:
    r"""
    Args:
        norms (`torch.Tensor` of shape `(rows,)`):
            Row norms.

    Returns:
        `dict`: Their median, 99th percentile and maximum, empty when there are none.
    """
    if not len(norms):
        return {}
    return {"median": norms.median().item(), "p99": norms.quantile(0.99).item(), "max": norms.max().item()}


@torch.no_grad()
def head_share_check(
    personaplex: str = DEFAULT_BASE_MODEL,
    language_model: str = "Qwen/Qwen3-8B",
    tokenizer: str = "RetentionLabs/PolyPlex-14B-CJKE",
    language: str = "en",
    texts: int = 4,
    characters: int = 600,
    k: int = 32,
) -> dict:
    r"""
    Scores head initializations of the new tokens on the Helium path alone, without assembling the DualFFN model.
    With its language FFNs silent the model's states are PersonaPlex's, so PersonaPlex's final states on held out
    documents (cut by Helium's tokenizer) are scored against Helium's head rows followed by the new tokens' rows
    (SPEC S12.3a), each variant giving the bits per byte of the Helium targets and the probability the new rows take.

    Args:
        personaplex (`str`, *optional*):
            Path or hub id of the duplex checkpoint.
        language_model (`str`, *optional*, defaults to `"Qwen/Qwen3-8B"`):
            Path or hub id of the text LLM.
        tokenizer (`str`, *optional*, defaults to `"RetentionLabs/PolyPlex-14B-CJKE"`):
            Path or hub id of the Polyplex tokenizer.
        language (`str`, *optional*, defaults to `"en"`):
            Language of [`~polyplex_research.datasets.webtext.DENSE_EVALUATION`] scored.
        texts (`int`, *optional*, defaults to 4):
            Held out documents.
        characters (`int`, *optional*, defaults to 600):
            Characters of each document scored.
        k (`int`, *optional*, defaults to 32):
            Base tokens per new token.

    Returns:
        `dict`: Each variant's bits per byte, new token share and the new tokens taking the most, the sums of the new
        tokens' coefficients, and the priors of Helium's rows and the new rows along the mean state,
        read on other documents than the scored ones. The coefficients are cached under `HF_HOME`.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, PersonaPlexForConditionalGeneration

    from ..models.polyplex import PolyplexProcessor, PolyplexTokenizer
    from .assemble import (
        calibrate_coefficients,
        compose_rows,
        load_base_tokenizer,
        qwen_piece_ids,
        vocabulary_coefficients,
        with_components,
    )

    polyplex = PolyplexTokenizer.from_pretrained(tokenizer)
    helium = PersonaPlexForConditionalGeneration.from_pretrained(personaplex, torch_dtype=torch.bfloat16)
    helium_tokenizer = load_base_tokenizer(personaplex)
    text_llm = AutoModelForCausalLM.from_pretrained(language_model, torch_dtype=torch.bfloat16)
    qwen_head = text_llm.lm_head.weight.float()
    del text_llm
    qwen_ids = qwen_piece_ids(AutoTokenizer.from_pretrained(language_model))
    base, size = helium.config.vocab_size, len(polyplex)
    helium_head = helium.lm_head.weight.float()

    cache = os.path.join(os.environ.get("HF_HOME", ".cache/huggingface"), "runs", "head-share", f"coefficients-{k}.pt")
    if os.path.exists(cache):
        indices, values = torch.load(cache)
    else:
        indices, values = vocabulary_coefficients(polyplex, base, size, k, qwen_head, qwen_ids)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        torch.save((indices, values), cache)
    calibrated = calibrate_coefficients(indices, values, helium_head, polyplex.blocks, base)[base:size]
    indices, values = indices[base:size], values[base:size]
    sums = values.sum(-1)
    calibrated_rows = compose_rows(indices, calibrated, helium_head)

    states = []
    hook = helium.lm_head.register_forward_hook(lambda module, inputs, output: states.append(inputs[0].detach()))
    processor = PolyplexProcessor.from_pretrained(personaplex)
    collate = dense_collate_fn(processor, helium)
    documents = dense_documents(language, "valid", max(4 * texts * characters, 12000), polyplex)
    byte_lengths = token_byte_lengths(helium_tokenizer)

    def read(chosen):
        states.clear()
        targets = []
        for document in chosen:
            ids = helium_tokenizer.encode(document[:characters], add_special_tokens=False)
            if len(ids) < 2:
                continue
            batch = collate([{"segments": [torch.tensor(ids)]}])
            batch.pop("text_labels")
            helium(**batch, use_cache=False)
            states[-1] = states[-1][0, 1:-1].float()
            targets.append(torch.tensor(ids[1:]))
        return torch.cat(states), torch.cat(targets)

    # The prior direction is read on documents other than the scored ones.
    calibration, _ = read(documents[texts : 2 * texts])
    hidden, targets = read(documents[:texts])
    hook.remove()
    scored_bytes = int(byte_lengths[targets].sum())
    direction = nn.functional.normalize(calibration.mean(0), dim=0)
    helium_priors = helium_head @ direction
    variants = {
        "helium only": None,
        "calibrated": calibrated_rows,
        "calibrated, half length": calibrated_rows * 0.5,
    }
    for quantile in (0.5, 0.1, 0.01, 0.0):
        target = helium_priors.quantile(quantile).item()
        variants[f"calibrated, prior at helium q{quantile:g}"] = with_components(
            calibrated_rows, direction[None], torch.tensor([target])
        )
    prior = helium_priors.min()[None]
    lowest = with_components(calibrated_rows, direction[None], prior)
    for length in (0.75, 0.5):
        variants[f"calibrated, prior at helium q0, length {length:g}"] = with_components(
            calibrated_rows * length, direction[None], prior
        )
    # Principal directions of the states around their mean, orthogonal to the mean.
    centered = calibration - calibration.mean(0)
    centered = centered - (centered @ direction)[:, None] * direction
    principal = torch.linalg.svd(centered, full_matrices=False).Vh
    for count in (4, 16, 64):
        directions = principal[:count]
        variants[f"calibrated, prior at helium q0, {count} principal directions at helium mean"] = with_components(
            lowest, directions, helium_head.mean(0) @ directions.T
        )

    result = {
        "coefficient_sum": {"median": sums.median().item(), "p99": sums.quantile(0.99).item(), "max": sums.max().item()},
        "helium_prior": {
            str(quantile): helium_priors.quantile(quantile).item() for quantile in (0.0, 0.01, 0.1, 0.5, 0.9, 1.0)
        },
        "new_prior_median": (calibrated_rows @ direction).median().item(),
    }
    for name, rows in variants.items():
        head = helium_head if rows is None else torch.cat([helium_head, rows])
        logits = hidden @ head.T
        bits = F.cross_entropy(logits, targets, reduction="sum").item() / math.log(2) / scored_bytes
        probabilities = logits.softmax(-1)
        entry = {"bits_per_byte": bits}
        if rows is not None:
            mean = probabilities[:, base:].mean(0)
            top = mean.topk(10)
            entry["new_token_share"] = mean.sum().item()
            entry["top_new_tokens"] = [
                [token, round(value, 4), round(sums[index].item(), 2)]
                for token, value, index in zip(
                    polyplex.convert_ids_to_tokens((top.indices + base).tolist()),
                    top.values.tolist(),
                    top.indices.tolist(),
                )
            ]
        result[name] = entry
        logger.info("Head share, %s: %s", name, entry)
    return result


def use_expandable_segments() -> None:
    r"""
    Lets the CUDA caching allocator grow its segments instead of cutting new ones, so the memory a freed tensor leaves
    can hold a larger one later. A dense run on a 46 GB card otherwise failed on a 3.4 GB request with 7.5 GB reserved
    but cut into smaller pieces. Called before the first allocation of a run.
    """
    if torch.cuda.is_available():
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")


def step_zero_check(
    personaplex: str = DEFAULT_BASE_MODEL,
    language_model: str = "Qwen/Qwen3-8B",
    tokenizer: str = "RetentionLabs/PolyPlex-14B-CJKE",
    languages: tuple = ("en", "ko"),
    texts: int = 4,
    characters: int = 600,
    appended_layers: int = 4,
    device: str | None = None,
    head_norm_caps: tuple = (),
    quantization: str | None = "nvfp4",
    full_precision_layers: tuple = (0, 1, 28, 29, 30, 31),
    removed_neurons: dict | None = None,
) -> dict:
    r"""
    Scores a freshly assembled DualFFN model with the Polyplex vocabulary before any training, as it trains (SPEC S7.1),
    against PersonaPlex: bits per byte of held out documents for PersonaPlex, for the DualFFN model, and for the
    DualFFN model with its language FFNs silenced ([`language_ffn_off_bits_per_byte`]).

    Args:
        personaplex (`str`, *optional*):
            Path or hub id of the duplex checkpoint.
        language_model (`str`, *optional*, defaults to `"Qwen/Qwen3-8B"`):
            Path or hub id of the text LLM.
        tokenizer (`str`, *optional*, defaults to `"RetentionLabs/PolyPlex-14B-CJKE"`):
            Path or hub id of the Polyplex tokenizer.
        languages (`tuple[str]`, *optional*, defaults to `("en", "ko")`):
            Languages of [`~polyplex_research.datasets.webtext.DENSE_EVALUATION`] scored.
        texts (`int`, *optional*, defaults to 4):
            Held out documents per language.
        characters (`int`, *optional*, defaults to 600):
            Characters of each document scored.
        appended_layers (`int`, *optional*, defaults to 4):
            Copies of Helium's last layers appended after its own (SPEC S11.4a).
        device (`str`, *optional*):
            Device the models run on. The CPU when omitted.
        head_norm_caps (`tuple[float]`, *optional*, defaults to `()`):
            Quantiles of Helium's head row norms. For each, the model is scored again with the new tokens' composed
            head rows capped at that norm ([`~polyplex_research.utils.assemble.cap_composed_head_norms`]).
        quantization (`str`, *optional*, defaults to `"nvfp4"`):
            How the backbone runs (SPEC S7.1), see [`~polyplex_research.utils.assemble.quantize_frozen_backbone`].
        full_precision_layers (`tuple[int]`, *optional*, defaults to `(0, 1, 28, 29, 30, 31)`):
            Layers left in full precision under `quantization`.
        removed_neurons (`dict[int, list[int]]`, *optional*):
            Layer mapped to the language FFN neurons removed at assembly (SPEC S11.2b).

    Returns:
        `dict`: Language mapped to the three scores and the probability the new tokens take with and without the
        language FFNs ([`new_token_share`]) with the new tokens that take the most of it, the norms of the head rows
        per block (`head_norm`), and the scores under each head norm cap.
    """
    from transformers import PersonaPlexForConditionalGeneration

    from ..models.polyplex import PolyplexProcessor, PolyplexTokenizer
    from .assemble import cap_composed_head_norms, compile_backbone, load_base_tokenizer, load_or_assemble_dualffn

    import gc

    use_expandable_segments()
    device = device or "cpu"
    polyplex = PolyplexTokenizer.from_pretrained(tokenizer)
    helium_tokenizer = load_base_tokenizer(personaplex)
    processor = PolyplexProcessor.from_pretrained(personaplex)
    processor.audio_tokenizer.to(device)
    helium_bytes, dual_bytes = token_byte_lengths(helium_tokenizer), token_byte_lengths(polyplex)
    chunks = {}
    for language in languages:
        documents = dense_documents(language, "valid", max(texts * characters, 6000), polyplex)
        chunks[language] = [document[:characters] for document in documents[:texts]]

    # PersonaPlex is scored and released before the DualFFN is assembled, so the two never share the device.
    helium = PersonaPlexForConditionalGeneration.from_pretrained(personaplex, torch_dtype=torch.bfloat16, device_map=device)
    base = helium.config.vocab_size
    collate_helium = dense_collate_fn(processor, helium)
    reference = {
        language: dense_bits_per_byte(helium, collate_helium, helium_tokenizer, chunk, helium_bytes)
        for language, chunk in chunks.items()
    }
    del helium, collate_helium
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = load_or_assemble_dualffn(
        device,
        personaplex=personaplex,
        language_model=language_model,
        tokenizer=tokenizer,
        quantization=quantization,
        full_precision_layers=tuple(full_precision_layers),
        appended_layers=appended_layers,
        calibration_texts=dense_calibration_texts(polyplex, skip=texts, characters=characters),
        **({} if removed_neurons is None else {"removed_neurons": removed_neurons}),
    )
    if torch.cuda.is_available():
        logger.info("DualFFN on the device: %.1f GB allocated.", torch.cuda.memory_allocated() / 1e9)
    compile_backbone(model)
    collate_dual = dense_collate_fn(processor, model)
    head = model.lm_head.composed_weight().float().norm(dim=-1)
    helium_norms = head[:base]
    results = {"head_norm": {"helium": norm_summary(helium_norms)}}
    for block, (start, end) in polyplex.blocks.items():
        results["head_norm"][block] = norm_summary(head[max(start, base) : end])
    logger.info("Step zero head norms: %s", results["head_norm"])
    for language, chunk in chunks.items():
        results[language] = {
            "personaplex": reference[language],
            "dualffn": dense_bits_per_byte(model, collate_dual, polyplex, chunk, dual_bytes),
            "dualffn_language_ffn_off": language_ffn_off_bits_per_byte(model, collate_dual, polyplex, chunk, dual_bytes),
            "new_token_share": new_token_share(model, collate_dual, polyplex, chunk, base),
        }
        with language_ffn_off(model):
            probabilities = mean_probabilities(model, collate_dual, polyplex, chunk)
        results[language]["new_token_share_language_ffn_off"] = probabilities[base:].sum().item()
        top = probabilities[base:].topk(20)
        results[language]["top_new_tokens_language_ffn_off"] = [
            [token, round(value, 4), round(head[base + index].item(), 2)]
            for token, value, index in zip(
                polyplex.convert_ids_to_tokens((top.indices + base).tolist()), top.values.tolist(), top.indices.tolist()
            )
        ]
        logger.info("Step zero, %s: %s", language, results[language])

    saved = model.text_composition.values.detach().clone()
    for quantile in head_norm_caps:
        limit = helium_norms.quantile(quantile).item()
        capped = cap_composed_head_norms(model, limit)
        name = f"head_norm_cap_q{quantile:g}"
        results[name] = {"limit": limit, "tokens_capped": capped}
        for language, chunk in chunks.items():
            results[name][language] = {
                "dualffn": dense_bits_per_byte(model, collate_dual, polyplex, chunk, dual_bytes),
                "dualffn_language_ffn_off": language_ffn_off_bits_per_byte(
                    model, collate_dual, polyplex, chunk, dual_bytes
                ),
                "new_token_share": new_token_share(model, collate_dual, polyplex, chunk, base),
            }
        logger.info("Step zero, %s: %s", name, results[name])
        with torch.no_grad():
            model.text_composition.values.copy_(saved)
    return results


def dense_calibration_texts(tokenizer, skip: int, count: int = 4, characters: int = 600) -> list:
    r"""
    English held out documents Helium reads at assembly to set the new head rows' context-independent logits (SPEC
    S12.3a), taken after the ones a run scores.

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer the document budget is counted in.
        skip (`int`):
            Leading documents left out, the ones the run scores.
        count (`int`, *optional*, defaults to 4):
            Documents taken.
        characters (`int`, *optional*, defaults to 600):
            Characters of each document kept.

    Returns:
        `list[str]`: The texts.
    """
    documents = dense_documents("en", "valid", max((skip + count) * characters * 2, 12000), tokenizer)
    return [document[:characters] for document in documents[skip : skip + count]]


def ephemeral_disk(device: str) -> bool:
    r"""
    Whether a device spec names a rented machine whose disk is discarded with its session, Colab or Kaggle.

    Args:
        device (`str`):
            A `POLYPLEX_DEVICE` spec, `"<provider>.<accelerator>"`.

    Returns:
        `bool`: True for Colab and Kaggle providers.
    """
    return device.partition(".")[0].startswith(("colab", "kaggle"))


def _delete_cached_repos(repo_ids) -> None:
    from huggingface_hub import scan_cache_dir

    cache = scan_cache_dir()
    revisions = [revision.commit_hash for repo in cache.repos if repo.repo_id in set(repo_ids) for revision in repo.revisions]
    if revisions:
        cache.delete_revisions(*revisions).execute()


def free_downloads(repo_ids) -> None:
    r"""
    Deletes the Hugging Face downloads of models or datasets already read, on a disk the run owns
    (`POLYPLEX_EPHEMERAL_DISK`, set by [`launch`] for Colab and Kaggle), so the next downloads and the checkpoints
    have room. Elsewhere the cache is left alone.

    Args:
        repo_ids (`list[str]`):
            Hub ids of the models or datasets.
    """
    if not runs_on_ephemeral_disk():
        return
    _delete_cached_repos(repo_ids)
    logger.info("Deleted the downloads of %s, %.1f GB left on the disk.", list(repo_ids),
                shutil.disk_usage(os.environ.get("HF_HOME", ".")).free / 1e9)


CHECKPOINT_REPO = "RetentionLabs/polyplex-checkpoints"


def checkpoint_repo() -> str:
    r"""
    Returns:
        `str`: The public Hugging Face repository dense checkpoints are copied to (SPEC S13.4),
        `POLYPLEX_CHECKPOINT_REPO` when set.
    """
    return os.environ.get("POLYPLEX_CHECKPOINT_REPO", CHECKPOINT_REPO)


def upload_checkpoint_file(run_name: str, path) -> None:
    r"""
    Uploads one checkpoint file to `<run_name>/checkpoint/<file>` of the public [`checkpoint_repo`], creating the
    repository on first use.

    Args:
        run_name (`str`):
            Name of the run.
        path (`Path`):
            The file.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    repo_id = checkpoint_repo()
    api.create_repo(repo_id, private=False, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=f"{run_name}/checkpoint/{path.name}",
        repo_id=repo_id,
        commit_message=f"{run_name}: {path.name}",
    )


def download_checkpoint_file(run_name: str, name: str, directory) -> None:
    r"""
    Downloads one file [`upload_checkpoint_file`] put in the public [`checkpoint_repo`] into `directory`.

    Args:
        run_name (`str`):
            Name of the run that wrote it.
        name (`str`):
            The file's name.
        directory (`Path`):
            Directory it is written to.
    """
    from huggingface_hub import hf_hub_download

    # Downloaded straight into `directory`, so a session disk never holds a second copy in the Hub cache.
    staging = directory / ".hub"
    fetched = hf_hub_download(checkpoint_repo(), f"{run_name}/checkpoint/{name}", local_dir=staging)
    os.replace(fetched, directory / name)
    shutil.rmtree(staging, ignore_errors=True)


# Runs whose current checkpoint skips the Drive store after a failed handoff there.
_DRIVE_SKIPPED = set()


def stash_checkpoint_file(run_name: str, path) -> None:
    r"""
    Copies one checkpoint file to the public Hugging Face repository ([`upload_checkpoint_file`]) and to the Drive
    store (SPEC S13.4), then deletes it here when either holds it, logging the disk's free space. On the Drive store
    the manifest goes under `<run_name>-checkpoint`, every other file under `<run_name>-checkpoint-<file stem>`. After
    a Drive handoff fails, the rest of the same checkpoint goes to Hugging Face alone, since each failed handoff
    blocks the run for the handoff's timeout. A file neither accepts is kept.

    Args:
        run_name (`str`):
            Name of the run.
        path (`Path`):
            The file.
    """
    import letify

    stored = []
    try:
        upload_checkpoint_file(run_name, path)
        stored.append("Hugging Face")
    except Exception as error:
        logger.warning("%s not uploaded to %s: %s", path.name, checkpoint_repo(), error)
    if run_name not in _DRIVE_SKIPPED:
        name = f"{run_name}-checkpoint" if path.name == "manifest.json" else f"{run_name}-checkpoint-{path.stem}"
        try:
            letify.stash(name, path)
            stored.append("Drive")
        except Exception as error:
            _DRIVE_SKIPPED.add(run_name)
            logger.warning("%s not on the Drive store, nor the rest of this checkpoint: %s", path.name, error)
    if path.name == "manifest.json":
        _DRIVE_SKIPPED.discard(run_name)
    if not stored:
        logger.warning("%s kept at %s only.", path.name, path)
        return
    path.unlink()
    logger.info("%s stored on %s, %.1f GB left on the disk.", path.name, " and ".join(stored),
                shutil.disk_usage(path.parent).free / 1e9)


def unstash_checkpoint(run_name: str, directory):
    r"""
    Fetches the checkpoint [`stash_checkpoint_file`] stored, for [`read_checkpoint`]: from the Drive store or from the
    public Hugging Face repository, whichever holds the later step, since an upload to one of them can fail.

    Args:
        run_name (`str`):
            Name of the run that wrote it.
        directory (`str` or `Path`):
            Directory the files are fetched into.

    Returns:
        `Path`: `directory`.

    Raises:
        FileNotFoundError: If neither holds a checkpoint of that run.
    """
    import json
    from pathlib import Path

    import letify

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    steps = {}
    try:
        if letify.unstash(f"{run_name}-checkpoint", directory / "drive-manifest.json") is not None:
            steps["drive"] = json.loads((directory / "drive-manifest.json").read_text())["step"]
    except Exception as error:
        logger.warning("The Drive store could not give the checkpoint manifest of %s: %s", run_name, error)
    try:
        download_checkpoint_file(run_name, "manifest.json", directory)
        (directory / "manifest.json").rename(directory / "hub-manifest.json")
        steps["hub"] = json.loads((directory / "hub-manifest.json").read_text())["step"]
    except Exception as error:
        logger.info("%s holds no checkpoint manifest of %s: %s", checkpoint_repo(), run_name, error)
    if not steps:
        raise FileNotFoundError(f"Neither the Drive store nor {checkpoint_repo()} holds a checkpoint of run {run_name!r}.")
    source = max(steps, key=lambda store: (steps[store], store == "hub"))
    logger.info("Resuming %s from step %d on the %s store (steps found: %s).", run_name, steps[source], source, steps)
    (directory / f"{source}-manifest.json").rename(directory / "manifest.json")
    for other in directory.glob("*-manifest.json"):
        other.unlink()
    for name in json.loads((directory / "manifest.json").read_text())["files"]:
        if source == "hub":
            download_checkpoint_file(run_name, name, directory)
        elif letify.unstash(f"{run_name}-checkpoint-{Path(name).stem}", directory / name) is None:
            raise FileNotFoundError(f"The Drive store lists {name} of run {run_name!r} but does not hold it.")
    return directory


def reference_key(language_model: str, held_out: dict) -> str:
    r"""
    The key the text LLM's bits per byte on an evaluation set is stored under (SPEC S13.4): a hash of the model's name
    and the evaluation texts of every language.

    Args:
        language_model (`str`):
            Hub id of the text LLM.
        held_out (`dict`):
            Language mapped to its evaluation texts.

    Returns:
        `str`: 16 hexadecimal characters.
    """
    import hashlib
    import json

    payload = json.dumps({"model": language_model, "texts": held_out}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def local_reference_path(key: str):
    r"""
    Args:
        key (`str`):
            [`reference_key`] of the model and the evaluation set.

    Returns:
        `Path`: `$HF_HOME/reference/<key>.json`, where a reference is kept on this disk.
    """
    from pathlib import Path

    return Path(os.environ.get("HF_HOME", ".cache/huggingface")) / "reference" / f"{key}.json"


def keep_reference(key: str, reference: dict) -> None:
    r"""
    Writes a reference to [`local_reference_path`].

    Args:
        key (`str`):
            [`reference_key`] of the model and the evaluation set.
        reference (`dict`):
            Language mapped to its score.
    """
    import json

    path = local_reference_path(key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(reference, indent=1))
    except OSError as error:
        logger.warning("The reference %s was not kept at %s: %s", key, path, error)


def fetch_reference(key: str) -> dict | None:
    r"""
    Returns the text LLM's score stored under `key`: from [`local_reference_path`] when this disk holds it, else from
    the public [`checkpoint_repo`], kept on this disk once read. `None` when neither holds it.

    Args:
        key (`str`):
            [`reference_key`] of the model and the evaluation set.

    Returns:
        `dict` or `None`: Language mapped to bits per byte.
    """
    import json

    from huggingface_hub import hf_hub_download

    path = local_reference_path(key)
    if path.exists():
        return json.loads(path.read_text())
    try:
        reference = json.loads(open(hf_hub_download(checkpoint_repo(), f"reference/{key}.json")).read())
    except Exception as error:
        logger.info("No stored reference for %s: %s", key, error)
        return None
    keep_reference(key, reference)
    return reference


def store_reference(key: str, reference: dict) -> None:
    r"""
    Stores the text LLM's score under `key` on this disk ([`local_reference_path`]) and in the public
    [`checkpoint_repo`], for [`fetch_reference`].

    Args:
        key (`str`):
            [`reference_key`] of the model and the evaluation set.
        reference (`dict`):
            Language mapped to bits per byte.
    """
    import json

    from huggingface_hub import HfApi

    keep_reference(key, reference)
    try:
        api = HfApi()
        api.create_repo(checkpoint_repo(), private=False, exist_ok=True)
        api.upload_file(
            path_or_fileobj=json.dumps(reference, indent=1).encode(),
            path_in_repo=f"reference/{key}.json",
            repo_id=checkpoint_repo(),
            commit_message=f"reference {key}",
        )
    except Exception as error:
        logger.warning("The reference %s was not stored: %s", key, error)


def language_model_tensor(language_model: str, name: str) -> torch.Tensor:
    r"""
    Reads one tensor of a text LLM checkpoint on the Hub, fetching only the shard that holds it.

    Args:
        language_model (`str`):
            Hub id of the text LLM.
        name (`str`):
            The tensor's name in the checkpoint.

    Returns:
        `torch.Tensor`: The tensor, on the CPU.
    """
    import json

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    index = json.loads(open(hf_hub_download(language_model, "model.safetensors.index.json")).read())
    with safe_open(hf_hub_download(language_model, index["weight_map"][name]), framework="pt") as handle:
        return handle.get_tensor(name)


def restore_appended_language_output(
    state: dict, language_model: str, appended_layers: list, language_ffn_layers: list
) -> list:
    r"""
    Puts the text LLM's output projection back into the language FFN of each appended layer of a read checkpoint
    ([`read_checkpoint`]) whose projection is still zero, and drops the optimizer state of those projections (SPEC
    S11.4a). With the output scale of those layers still zero, the model computes exactly what it did before.

    Args:
        state (`dict`):
            What [`read_checkpoint`] returned. Changed in place.
        language_model (`str`):
            Hub id of the text LLM.
        appended_layers (`list[int]`):
            The appended layers.
        language_ffn_layers (`list[int]`):
            For each backbone layer, the text LLM layer its language FFN comes from.

    Returns:
        `list[str]`: Names of the projections restored.
    """
    weights = state["weights"]
    names = list(weights)
    restored = []
    for layer in appended_layers:
        name = f"model.layers.{layer}.mlp.language_mlp.down_proj.weight"
        if name not in weights or weights[name].any():
            continue
        source = language_model_tensor(language_model, f"model.layers.{language_ffn_layers[layer]}.mlp.down_proj.weight")
        weights[name] = source.to(weights[name].device, weights[name].dtype)
        restored.append(name)
    # The optimizer indices follow `dense_optimizer`'s split of the trained parameters, in their order.
    sides = {"muon": [], "adamw": [], "tables": []}
    for name in names:
        if name in DENSE_FACTORED_TABLES:
            sides["tables"].append(name)
        elif weights[name].ndim == 2 and name not in DENSE_TABLES:
            sides["muon"].append(name)
        else:
            sides["adamw"].append(name)
    for side, members in sides.items():
        for index, name in enumerate(members):
            if name in restored and side in state["optimizer"]:
                state["optimizer"][side]["state"].pop(index, None)
    logger.info("Restored the text LLM's output projection in %s.", restored)
    return restored


class CheckpointUploader:
    r"""
    Uploads dense checkpoints on a thread of its own (SPEC S13.4), so the run trains on while they upload. Each
    checkpoint directory given to [`~CheckpointUploader.submit`] is uploaded in turn, its tensor file first and its
    manifest last ([`stash_checkpoint_file`]), and removed once every file is stored.

    Args:
        run_name (`str`):
            Name of the run.
    """

    def __init__(self, run_name: str):
        import queue
        import threading

        self.run_name = run_name
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="checkpoint-upload", daemon=True)
        self._thread.start()

    def submit(self, directory) -> None:
        r"""
        Queues a checkpoint directory [`write_checkpoint`] has finished writing.

        Args:
            directory (`Path`):
                The directory.
        """
        self._queue.put(directory)

    def _run(self) -> None:
        import json

        while True:
            directory = self._queue.get()
            if directory is None:
                return
            try:
                files = json.loads((directory / "manifest.json").read_text())["files"] + ["manifest.json"]
                for name in files:
                    stash_checkpoint_file(self.run_name, directory / name)
                if not any(directory.iterdir()):
                    directory.rmdir()
                logger.info("Checkpoint %s uploaded.", directory.name)
            except Exception as error:
                logger.warning("Checkpoint %s not uploaded: %s", directory, error)

    def close(self, timeout: float = 3600.0) -> bool:
        r"""
        Waits until every queued checkpoint is uploaded, then ends the thread. After `timeout` seconds it stops
        waiting and logs a warning; the files not yet uploaded stay on the disk.

        Args:
            timeout (`float`, *optional*, defaults to 3600.0):
                Seconds to wait for the uploads.

        Returns:
            `bool`: Whether every queued checkpoint was handled within `timeout`.
        """
        self._queue.put(None)
        self._thread.join(timeout)
        if self._thread.is_alive():
            logger.warning("Checkpoints still uploading after %.0f s; the unfinished files stay on the disk.", timeout)
            return False
        return True


def checkpoint_hand_over(run_name: str):
    r"""
    Where a dense run's checkpoints go once written (SPEC S13.4): on a disk the run owns (`POLYPLEX_EPHEMERAL_DISK`,
    set by [`launch`] for Colab and Kaggle) a [`CheckpointUploader`] moves them to the public Hugging Face repository
    and the Drive store while the run trains on; elsewhere they stay on the local disk.

    Args:
        run_name (`str`):
            Name of the run.

    Returns:
        [`CheckpointUploader`] or `None`: The uploader, `None` on a local disk.
    """
    if not runs_on_ephemeral_disk():
        return None
    return CheckpointUploader(run_name)


def init_dense_wandb_run(run_name: str, resume_extra: dict | None, **kwargs):
    r"""
    Starts the W&B run of a dense run in project `polyplex`. A run resuming its own checkpoint (the manifest `extra`
    names the same `run_name` and holds a W&B URL) continues that W&B run, whose id is the URL's last path segment.
    A run resuming another run's checkpoint starts a new W&B run.

    Args:
        run_name (`str`):
            Name of the dense run.
        resume_extra (`dict`, *optional*):
            The `extra` of the resumed checkpoint manifest, `None` for a fresh run.
        **kwargs:
            Passed on to `wandb.init`.

    Returns:
        The `wandb` run.
    """
    import wandb

    url = (resume_extra or {}).get("wandb")
    if url and resume_extra.get("run_name") == run_name:
        kwargs.update(id=url.rstrip("/").rsplit("/", 1)[-1], resume="allow")
    return wandb.init(project="polyplex", name=run_name, **kwargs)


def resume_checkpoint(run_name: str, work_dir: str):
    r"""
    Finds the checkpoint a dense run resumes from (SPEC S13.4): the one run `run_name` left on the local disk, beside
    `work_dir` under the same runs folder, or else the stored one, fetched into `work_dir` ([`unstash_checkpoint`]).

    Args:
        run_name (`str`):
            Name of the run that wrote the checkpoint.
        work_dir (`str`):
            Directory of the resuming run.

    Returns:
        `Path`: The directory holding the checkpoint, for [`read_checkpoint`].
    """
    from pathlib import Path

    run = Path(work_dir).parent / run_name
    saved = local_checkpoints(run)
    if saved:
        local = max(saved, key=lambda path: json.loads((path / "manifest.json").read_text()).get("step", 0))
        logger.info("Resuming from the local checkpoint at %s.", local)
        return local
    return unstash_checkpoint(run_name, os.path.join(work_dir, "resume"))


def run_dense_adaptation(
    personaplex: str = DEFAULT_BASE_MODEL,
    language_model: str = "Qwen/Qwen3-8B",
    work_dir: str | None = None,
    steps: int = 2000,
    batch_size: int = 24,
    micro_batch_size: int = 4,
    row_frames: int = 1024,
    learning_rate: float = 1e-3,
    warmup_steps: int = 100,
    eval_every: int = 500,
    eval_rows: int = 64,
    max_files: int = 1,
    language_ffn_layers: list[int] | None = None,
    quantization: str | None = "nvfp4",
    run_name: str = "dual-ffn",
    seed: int = 0,
    tokenizer: str | None = None,
    eval_texts: int = 64,
    eval_characters: int = 2000,
    full_precision_layers: tuple = (),
    appended_layers: int = 4,
    decay_steps: int = 0,
    save_every: int = 500,
    lr_schedule: str = "wsd",
    resume: str | None = None,
    backward_dtype: str = "bf16",
    alignment: str | None = None,
    evaluate_at_start: bool = False,
    removed_neurons: dict | None = None,
    language_scale: float | None = None,
    language_stream: str | None = None,
    language_embeddings: bool = False,
    language_lora_rank: int = 0,
    cache_backward_weights: bool = False,
    restore_appended_language_down: bool = False,
    mixture: dict | None = None,
    silent_audio: bool = False,
    languages: tuple | None = None,
    shared_language_stream: str | None = None,
    stream_rotation_switch_step: int = 8000,
    filtered_documents: bool = True,
) -> dict:
    r"""
    Runs the dense quick check before v2 (SPEC S13.4): assembles a DualFFN model, measures how far its text
    cross entropy is from the text LLM's, and trains [`dense_trainable_parameters`] to close that gap.

    With the text LLM's vocabulary both models read the same tokens and are compared per token. With the Polyplex
    vocabulary (`tokenizer`) they read different tokens and are compared in bits per byte on the first
    `eval_characters` characters of `eval_texts` held out documents per language, next to a unigram model of
    the language's training documents. After every evaluation the trained weights are saved under `work_dir`
    and put on the Drive store as `<run_name>-weights`, so a run cut short leaves them behind.

    Args:
        personaplex (`str`, *optional*, defaults to `"nvidia/personaplex-7b-v1"`):
            Duplex checkpoint.
        language_model (`str`, *optional*, defaults to `"Qwen/Qwen3-8B"`):
            Text LLM.
        work_dir (`str`, *optional*):
            Directory of the documents and row caches, `$HF_HOME/runs/<run_name>` when omitted.
        steps (`int`, *optional*, defaults to 2000):
            Optimizer steps.
        batch_size (`int`, *optional*, defaults to 24):
            Rows per optimizer step.
        micro_batch_size (`int`, *optional*, defaults to 4):
            Rows per forward pass.
        row_frames (`int`, *optional*, defaults to 1024):
            Frames per row, opening frames included.
        learning_rate (`float`, *optional*, defaults to 1e-3):
            Peak learning rate.
        warmup_steps (`int`, *optional*, defaults to 100):
            Warmup steps.
        eval_every (`int`, *optional*, defaults to 500):
            Steps between evaluations.
        eval_rows (`int`, *optional*, defaults to 64):
            Held out rows per language.
        max_files (`int`, *optional*, defaults to 1):
            Parquet files fetched per language.
        language_ffn_layers (`list[int]`, *optional*):
            Pairing of backbone layers with text LLM layers, see
            [`~polyplex_research.utils.assemble.dualffn_config`].
        quantization (`str`, *optional*, defaults to `"nvfp4"`):
            How the frozen backbone's projections run, see
            [`~polyplex_research.utils.assemble.quantize_frozen_backbone`]. `None` keeps them in bfloat16.
        run_name (`str`, *optional*, defaults to `"dual-ffn"`):
            W&B run name.
        seed (`int`, *optional*, defaults to 0):
            Seed of the row order.
        tokenizer (`str`, *optional*):
            Path of a Polyplex tokenizer the model is assembled with ([`~assemble.assemble_dualffn`]).
        eval_texts (`int`, *optional*, defaults to 64):
            Held out documents per language scored in bits per byte, used with `tokenizer`.
        eval_characters (`int`, *optional*, defaults to 2000):
            Characters of each such document scored, used with `tokenizer`.
        full_precision_layers (`tuple[int]`, *optional*, defaults to `(0, 1, 28, 29, 30, 31)`):
            Layers left in full precision under `quantization` (SPEC S7.1), see
            [`~polyplex_research.utils.assemble.quantize_frozen_backbone`].
        appended_layers (`int`, *optional*, defaults to 4):
            Copies of Helium's last layers appended after its own (SPEC S11.4a).
        decay_steps (`int`, *optional*, defaults to 0):
            Steps of learning rate decay that end this run ([`wsd_factor`]). Without them the run can be continued.
        save_every (`int`, *optional*, defaults to 500):
            Steps between checkpoints, kept in `work_dir`, or on Colab and Kaggle moved to the Drive store file by
            file as it is written ([`checkpoint_hand_over`]). A checkpoint is also taken at the end, on a failing step
            and on SIGTERM or SIGINT.
        lr_schedule (`str`, *optional*, defaults to `"wsd"`):
            Learning rate schedule, one of [`LR_SCHEDULES`] ([`train_dense`]).
        resume (`str`, *optional*):
            Name of a run whose checkpoint ([`resume_checkpoint`]) this run continues from: its step, weights and optimizer state, with the
            training documents it used skipped (AGENTS H18).
        backward_dtype (`str`, *optional*, defaults to `"bf16"`):
            Precision of the backward matmuls of the NVFP4 layers (SPEC S7.1a), see
            [`~polyplex_research.utils.quantization.set_nvfp4_backward`].
        alignment (`str`, *optional*):
            A residual basis alignment file folded into every language FFN at assembly (SPEC S11.2a). A run that
            resumes a checkpoint folded with [`fold_checkpoint_alignment`] takes the file that wrote, with the gains.
        evaluate_at_start (`bool`, *optional*, defaults to `False`):
            Whether a resumed run is evaluated before its first step, as a fresh run always is.
        removed_neurons (`dict[int, list[int]]`, *optional*):
            Layer mapped to the language FFN neurons removed at assembly (SPEC S11.2b).
        language_scale (`float`, *optional*):
            Value the per-dimension language FFN output scale starts at (SPEC S11.2c). Without it the model has none.
        language_stream (`str`, *optional*):
            A residual basis alignment file whose rotations join a second residual stream for the language FFNs
            (SPEC S11.2d). The model is then assembled without the alignment folded in.
        language_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the language stream starts from a text table of its own, initialized from the text LLM's input
            embeddings and trained (SPEC S11.2f). Needs `language_stream` and `tokenizer`.
        language_lora_rank (`int`, *optional*, defaults to 0):
            Rank of the trained low rank corrections beside each language FFN projection (SPEC S11.2g). 0 adds none.
        cache_backward_weights (`bool`, *optional*, defaults to `False`):
            Whether each frozen NVFP4 projection keeps its MXFP8 input gradient weight, quantized once (SPEC S7.1a).
            It costs about one byte per frozen weight on the device.
        restore_appended_language_down (`bool`, *optional*, defaults to `False`):
            Whether a resumed checkpoint whose appended layers still have a zero language FFN output projection gets
            the text LLM's projection back ([`restore_appended_language_output`], SPEC S11.4a).
        mixture (`dict`, *optional*):
            Dense language key mapped to its share of the training tokens. Every language of [`~polyplex_research.datasets.webtext.QWEN3_FINEWEB2`] and
            English by [`dense_mixture`] when omitted (SPEC S13.4).
        silent_audio (`bool`, *optional*, defaults to `False`):
            Whether the dense rows carry codec silence in both acoustic streams. Without it they carry no audio codes
            and the model reads the text embeddings alone (SPEC S13.1a).
        languages (`tuple[str]`, *optional*):
            Dense language keys the default mixture is restricted to, such as `("en", "ko", "ja")`. Every language
            when omitted. Ignored when `mixture` is given.
        shared_language_stream (`str`, *optional*):
            A shared residual basis alignment file, one rotation and one scale for all layers (SPEC S11.2h). The
            language stream moves to it, folded into the language FFN weights, after `stream_rotation_switch_step`
            ([`shared_stream_switch`]). Needs `language_stream`.
        stream_rotation_switch_step (`int`, *optional*, defaults to 8000):
            The last step trained with the per-layer rotations of `language_stream`.
        filtered_documents (`bool`, *optional*, defaults to `True`):
            Whether each language's training documents come from its quality filtered source, where one is public
            ([`~polyplex_research.datasets.webtext.MultilingualEduCorpus`], SPEC S13.4). The held out documents are
            read from the unfiltered source either way.

    Returns:
        `dict`: The text LLM's score per language (`reference`), the unigram one with `tokenizer` (`unigram`), the
        evaluations, the steps run, the data sizes, the W&B URL and the Drive store name of the last checkpoint
        (`checkpoint`). The weights stay on the Drive store.
    """
    import gc
    from pathlib import Path

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ..models.polyplex import PolyplexProcessor, PolyplexTokenizer
    from .assemble import compile_backbone, load_or_assemble_dualffn
    from .evaluate import unigram_log_probs

    torch.manual_seed(seed)
    use_expandable_segments()
    # The index is explicit, since safetensors reads a bare "cuda" as the first card whatever the current device is.
    device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    work_dir = work_dir or os.path.join(os.environ.get("HF_HOME", ".cache/huggingface"), "runs", run_name)
    llm_tokenizer = AutoTokenizer.from_pretrained(language_model)
    text_tokenizer = llm_tokenizer if tokenizer is None else PolyplexTokenizer.from_pretrained(tokenizer)

    os.makedirs(work_dir, exist_ok=True)
    logger.info("Disk at %s: %.1f GB free.", work_dir, shutil.disk_usage(work_dir).free / 1e9)
    rank, world = dense_rank()
    # The first process reads the documents and writes its rows first; the others then find the documents cached and
    # write their rows apart from the ones it reads.
    rows_dir = work_dir if rank == 0 else os.path.join(work_dir, f"rank-{rank}")
    if rank:
        torch.distributed.barrier()
    resume_state = None
    used = {}
    if resume is not None:
        resume_directory = resume_checkpoint(resume, work_dir)
        # The tensors are read after assembly, straight onto the device: beside the sources in host memory they
        # would not fit.
        resume_state = read_checkpoint(resume_directory, tensors=False)
        used = resume_state["extra"]["documents_used"]
        logger.info("Resuming %s from step %d, documents already used: %s", resume, resume_state["step"], used)

    train_tokens = int(steps * batch_size * row_frames * 1.03)
    train_documents, train_pieces, pieces_by_language = [], [], {}
    sizes = {}
    documents_used = dict(used)
    if mixture is None:
        # FineWeb-Edu's bytes are left out, so English takes its floor; its share by bytes falls below it.
        mixture = dense_mixture({"en": 0.0, **fineweb2_bytes()}, languages=languages)
    for language, share in mixture.items():
        pieces, consumed = [], []
        # Documents are counted per source, so a language that moves to its filtered source starts that source's count
        # and leaves out what it already trained on from the unfiltered one (AGENTS H18).
        key = MultilingualEduCorpus.documents_key(language, filtered_documents)
        exclude = None
        if key != language and used.get(language):
            exclude = trained_digests(language, used[language], max_files)
        documents = dense_documents(
            language, "train", int(train_tokens * share), text_tokenizer, max_files, skip=used.get(key, 0),
            pieces=pieces, filtered=filtered_documents, exclude=exclude, consumed=consumed,
        )
        sizes[f"record/train_documents_{language}"] = len(documents)
        pieces_by_language[language] = pieces
        documents_used[key] = used.get(key, 0) + consumed[0]
        train_documents.extend(documents)
        train_pieces.extend(pieces)
    os.makedirs(rows_dir, exist_ok=True)
    train_dataset = dense_rows("train", train_documents, text_tokenizer, rows_dir, row_frames, pieces=train_pieces)
    sizes["record/train_rows"] = len(train_dataset)
    eval_sets, held_out = {}, {}
    for language in DENSE_EVALUATION:
        if tokenizer is None:
            eval_sets[language] = dense_eval_rows(language, text_tokenizer, rows_dir, eval_rows, row_frames, max_files)
            sizes[f"record/eval_rows_{language}"] = len(eval_sets[language])
        else:
            documents = dense_documents(language, "valid", eval_texts * eval_characters, text_tokenizer, max_files)
            held_out[language] = [document[:eval_characters] for document in documents[:eval_texts]]
            sizes[f"record/eval_texts_{language}"] = len(held_out[language])
    logger.info("Dense rows: %s", sizes)
    calibration_texts = None if tokenizer is None else dense_calibration_texts(text_tokenizer, skip=eval_texts)
    if world > 1 and rank == 0:
        torch.distributed.barrier()
    # Every document the run reads is in memory now, and the corpora would leave no room for the models.
    free_downloads(dense_repositories())

    unigram, reference = {}, {}
    if tokenizer is not None:
        text_bytes = token_byte_lengths(text_tokenizer)
        key = reference_key(language_model, held_out)
        reference = fetch_reference(key)
        if reference is None:
            # The text LLM is scored and released before the DualFFN is loaded, so the two never share the device.
            text_llm = AutoModelForCausalLM.from_pretrained(language_model, torch_dtype=torch.bfloat16, device_map=device)
            llm_bytes = token_byte_lengths(llm_tokenizer, byte_level=True)
            reference = {
                language: dense_bits_per_byte(text_llm, None, llm_tokenizer, texts, llm_bytes, text_llm=True)
                for language, texts in held_out.items()
            }
            store_reference(key, reference)
            del text_llm
        else:
            logger.info("Using the stored reference %s.", key)
        for language, texts in held_out.items():
            pieces = [piece for encoded in pieces_by_language.get(language, []) for piece in encoded]
            if pieces:
                log_probs = unigram_log_probs(pieces, len(text_tokenizer))
                unigram[language] = unigram_bits_per_byte(log_probs, text_tokenizer, texts, text_bytes)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    processor = PolyplexProcessor.from_pretrained(personaplex)
    processor.audio_tokenizer.to(device)
    # The backbone is quantized as it is assembled, on the device it trains on (SPEC S7.1).
    model = load_or_assemble_dualffn(
        device,
        backward_dtype,
        personaplex=personaplex,
        language_model=language_model,
        language_ffn_layers=language_ffn_layers,
        tokenizer=tokenizer,
        quantization=quantization,
        full_precision_layers=tuple(full_precision_layers),
        appended_layers=appended_layers,
        calibration_texts=calibration_texts,
        alignment=alignment,
        **({} if removed_neurons is None else {"removed_neurons": removed_neurons}),
        # A language FFN output scale starts at zero, which already makes appended layers the identity (SPEC S11.4a).
        # A resumed run overwrites the appended layers from its checkpoint, so it shares the fresh run's assembly.
        **(
            {"keep_appended_output": True}
            if language_scale is not None or language_stream is not None
            else {}
        ),
    )
    if torch.cuda.is_available():
        logger.info("DualFFN on the device: %.1f GB allocated.", torch.cuda.memory_allocated() / 1e9)
    if quantization == "nvfp4":
        logger.info(
            "%d NVFP4 layers run their backward in %s.",
            set_nvfp4_backward(model, backward_dtype, cache_weights=cache_backward_weights),
            backward_dtype,
        )
    if language_scale is not None:
        gate_language_ffn(model, language_scale)
    if language_stream is not None:
        open_language_stream(model, language_stream)
        if language_embeddings:
            open_language_embeddings(model, text_tokenizer, language_model)
    if language_lora_rank:
        open_language_lora(model, language_lora_rank)
    use_flex_attention(model)
    compile_backbone(model)
    collate = dense_collate_fn(processor, model, silent_audio=silent_audio)
    del processor
    gc.collect()

    if tokenizer is None:
        key = reference_key(
            language_model,
            {
                f"cross_entropy/{language}": [[segment.tolist() for segment in row["segments"]] for row in rows]
                for language, rows in eval_sets.items()
            },
        )
        reference = fetch_reference(key)
        if reference is None:
            text_llm = AutoModelForCausalLM.from_pretrained(
                language_model, torch_dtype=torch.bfloat16, device_map=device
            )
            reference = {
                language: dense_evaluate(text_llm, collate, rows, text_llm=True) for language, rows in eval_sets.items()
            }
            store_reference(key, reference)
            del text_llm
        else:
            logger.info("Using the stored reference %s.", key)
        evaluate_fn = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:

        def evaluate_fn(trained_model) -> dict:
            metrics = {}
            for language, texts in held_out.items():
                score = dense_bits_per_byte(trained_model, collate, text_tokenizer, texts, text_bytes)
                metrics[f"eval/{language}_bits_per_byte"] = score
                metrics[f"eval/{language}_gap"] = score - reference[language]
            return metrics

    logger.info("Text LLM reference: %s, unigram: %s", reference, unigram)
    free_downloads([personaplex, language_model])

    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    run = init_dense_wandb_run(
        run_name,
        None if resume_state is None else resume_state["extra"],
        tags=["phase1", "N14", "D10"] + ([] if tokenizer is None else ["S12"]),
        config={
            "personaplex": personaplex,
            "language_model": language_model,
            "steps": steps,
            "batch_size": batch_size,
            "micro_batch_size": micro_batch_size,
            "row_frames": row_frames,
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "eval_every": eval_every,
            "eval_rows": eval_rows,
            "max_files": max_files,
            "mixture": mixture,
            "silent_audio": silent_audio,
            "sources": {
                language: MultilingualEduCorpus.source_of(language, filtered_documents) for language in mixture
            },
            "language_ffn_layers": model.config.language_ffn_layers,
            "quantization": quantization,
            "full_precision_layers": list(full_precision_layers),
            "appended_layers": model.config.appended_layers,
            "seed": seed,
            "tokenizer": tokenizer,
            "eval_texts": eval_texts,
            "eval_characters": eval_characters,
            "decay_steps": decay_steps,
            "save_every": save_every,
            "lr_schedule": lr_schedule,
            "resume": resume,
            "backward_dtype": backward_dtype,
            "alignment": alignment,
            "removed_neurons": removed_neurons,
            "language_scale": language_scale,
            "language_stream": language_stream,
            "restore_appended_language_down": restore_appended_language_down,
            "start_step": 0 if resume_state is None else resume_state["step"],
            "documents_used": documents_used,
            **{f"reference/{language}_loss": loss for language, loss in reference.items()},
            **{f"unigram/{language}_bits_per_byte": score for language, score in unigram.items()},
            **sizes,
            **git_record(os.path.dirname(os.path.abspath(__file__))),
        },
    )

    def log(metrics: dict):
        metrics = dict(metrics)
        step = metrics.pop("step")
        run.log(metrics, step=step)
        logger.info("step %d %s", step, {key: round(value, 4) for key, value in metrics.items()})

    uploader = checkpoint_hand_over(run_name)

    stream_state = {"gains": None}
    stream_switch = None
    if shared_language_stream is not None:
        stream_switch = shared_stream_switch(shared_language_stream, language_model, stream_state)

    def checkpoint(step, trained, masters, optimizer):
        extra = {
            "documents_used": documents_used,
            "run_name": run_name,
            "wandb": run.url,
            "batch_size": batch_size,
            "row_frames": row_frames,
            "lr_schedule": lr_schedule,
            "language_stream_folded": bool(model.config.language_stream_folded),
        }
        buffers = {} if stream_state["gains"] is None else {"stream_gains": stream_state["gains"]}
        if uploader is None:
            directory = save_local_checkpoint(
                work_dir, step, lambda path: write_checkpoint(path, step, trained, masters, optimizer, extra, buffers=buffers)
            )
            logger.info("Checkpoint of step %d is at %s.", step, directory)
        else:
            # Each checkpoint has its own directory, so a save never writes into one still uploading.
            directory = Path(work_dir) / f"checkpoint-{step}"
            write_checkpoint(directory, step, trained, masters, optimizer, extra, buffers=buffers)
            uploader.submit(directory)
            logger.info("Checkpoint of step %d written; it uploads while the run trains on.", step)

    import signal

    stopping = {"asked": False}

    def ask_to_stop(signum, frame):
        logger.warning("Signal %d received, saving the last finished step and stopping.", signum)
        stopping["asked"] = True

    previous = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            previous[signum] = signal.signal(signum, ask_to_stop)
        except ValueError:
            # Handlers can only be set on the main thread; a run started elsewhere stops on its failure checkpoint.
            logger.warning("Signal %d is not handled outside the main thread.", signum)

    try:
        with depth_decoder_on(model, "cpu"):
            if resume is not None:
                resume_state = read_checkpoint(resume_directory, device=device)
                if restore_appended_language_down:
                    restore_appended_language_output(
                        resume_state, language_model, model.config.appended_layers, model.config.language_ffn_layers
                    )
                if uploader is not None and Path(resume_directory).name == "resume":
                    # A checkpoint fetched onto a session disk is in memory now; the stores keep it.
                    shutil.rmtree(resume_directory)
                if resume_state["extra"].get("language_stream_folded"):
                    if stream_switch is None:
                        raise ValueError("The checkpoint's language stream is folded; pass its shared alignment.")
                    # The frozen layers are folded again with the gains of the step that switched.
                    stream_state["gains"] = resume_state["buffers"]["stream_gains"]
                    stream_switch(model)
            result = train_dense(
                model,
                collate,
                train_dataset,
                eval_sets,
                reference,
                steps=steps,
                batch_size=batch_size,
                micro_batch_size=micro_batch_size,
                learning_rate=learning_rate,
                warmup_steps=warmup_steps,
                eval_every=eval_every,
                log=log,
                seed=seed,
                evaluate_fn=evaluate_fn,
                decay_steps=decay_steps,
                save_every=save_every,
                lr_schedule=lr_schedule,
                checkpoint=checkpoint,
                stop=lambda: stopping["asked"],
                resume_state=resume_state,
                evaluate_at_start=evaluate_at_start,
                stream_switch=stream_switch,
                switch_step=stream_rotation_switch_step,
            )
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        if uploader is not None:
            # The session's disk goes with it, so the run ends only once every checkpoint is uploaded.
            uploader.close()
    url = run.url
    run.finish()
    return {
        **result,
        "reference": reference,
        "unigram": unigram,
        "sizes": sizes,
        "wandb": url,
        "checkpoint": f"{run_name}-checkpoint",
    }


DENSE_SCALE_ABLATIONS = (
    ("trained", 1.0, None),
    ("language_scale_0", 0.0, None),
    ("language_scale_0_backbone", 0.0, "backbone"),
    ("language_scale_0_appended", 0.0, "appended"),
    ("language_scale_2", 2.0, None),
)


def compare_release(model, release_dir: str, held_out: dict, tokenizer, device, texts: int = 4, characters: int = 500) -> dict:
    r"""
    Compares a released model ([`~polyplex_research.utils.assemble.save_dense_release`]), loaded through its remote
    code, with the model it was saved from (SPEC S13.4c). Both read the same held out documents in the dense layout,
    the opening frame and then the text, and their text logits are compared position by position.

    Args:
        model ([`PolyplexForConditionalGeneration`]):
            The model the release was saved from, with the checkpoint's parameters.
        release_dir (`str`):
            The release, with its remote code files.
        held_out (`dict`):
            Language mapped to its held out documents.
        tokenizer ([`PolyplexTokenizer`]):
            The text tokenizer.
        device (`str` or `torch.device`):
            Where the release is loaded.
        texts (`int`, *optional*, defaults to 4):
            Documents read per language.
        characters (`int`, *optional*, defaults to 500):
            Leading characters read of each document.

    Returns:
        `dict`: Per language, the largest absolute difference of the logits and the share of positions whose most
        likely token agrees.
    """
    from transformers import AutoModel

    release = AutoModel.from_pretrained(release_dir, trust_remote_code=True, dtype=torch.bfloat16, device_map=device)
    release.eval()
    model.eval()
    result = {}
    for language, documents in held_out.items():
        differences, agreement = [], []
        for text in documents[:texts]:
            pieces = tokenizer.encode(text[:characters], add_special_tokens=False)
            ids = torch.tensor([[model.config.vocab_size, *pieces]], device=device)
            positions = torch.arange(ids.shape[1], device=device)[None]
            with torch.no_grad():
                expected = model(input_ids=ids, position_ids=positions).logits.float()
                logits = release(input_ids=ids, position_ids=positions).logits.float()
            differences.append((logits - expected).abs().max().item())
            agreement.append((logits.argmax(-1) == expected.argmax(-1)).float().mean().item())
        result[language] = {
            "max_abs_logit_difference": max(differences),
            "argmax_agreement": sum(agreement) / len(agreement),
        }
        logger.info("Release against the trained model, %s: %s", language, result[language])
    return result


def evaluate_dense_checkpoint(
    checkpoint: str | None,
    run_name: str,
    personaplex: str = DEFAULT_BASE_MODEL,
    language_model: str = "Qwen/Qwen3-8B",
    tokenizer: str = "RetentionLabs/PolyPlex-14B-CJKE",
    language_ffn_layers: list | None = None,
    quantization: str | None = "nvfp4",
    full_precision_layers: tuple = (),
    appended_layers: int = 4,
    backward_dtype: str = "mxfp8",
    alignment: str | None = None,
    removed_neurons: dict | None = None,
    language_scale: float | None = None,
    language_stream: str | None = None,
    eval_texts: int = 64,
    eval_characters: int = 2000,
    max_files: int = 1,
    ablations: tuple = DENSE_SCALE_ABLATIONS,
    token_groups: bool = False,
    training_tokens: str | None = None,
    samples: int = 0,
    personaplex_baseline: bool = False,
    language_model_samples: bool = False,
    language_embeddings: bool = False,
    language_lora_rank: int = 0,
    speech_utterances: int = 0,
    nvfp4_layers: tuple = (),
    silent_audio: bool = False,
    language_stream_from_text: bool = True,
    speech_path: str = ".cache/huggingface/runs/data/english_heldout_speech.pt",
    release_dir: str | None = None,
    release_check: bool = False,
    shared_language_stream: str | None = None,
    sample_temperature: float = 0.0,
) -> dict:
    r"""
    Scores a dense checkpoint on the held out documents [`run_dense_adaptation`] scores, in bits per byte, once per
    entry of `ablations`, each multiplying the language FFN output scale ([`language_scale_multiplied`]) of every
    layer, of the backbone's own layers (`"backbone"`) or of the appended ones (`"appended"`). The model is assembled
    with the settings of the run that wrote the checkpoint, so the stored assembly is reused.

    Args:
        checkpoint (`str`):
            Directory [`write_checkpoint`] wrote, or `None` to score the assembled model as it starts training.
        run_name (`str`):
            Name of this evaluation's W&B run.
        ablations (`tuple`, *optional*, defaults to [`DENSE_SCALE_ABLATIONS`]):
            `(label, factor, layers)` entries. `layers` is `None` (every layer), `"backbone"`, `"appended"` or a list
            of layer indices. A `factor` of `None` silences those layers' language FFNs ([`language_ffn_off`]).
        token_groups (`bool`, *optional*, defaults to `False`):
            Also splits the bits per byte of the trained model and of the model with every scale at 0 by script, by
            vocabulary and by position ([`grouped_bits_per_byte`]).
        training_tokens (`str`, *optional*):
            The `tokens.bin` of the run's training rows. With `token_groups`, also splits the new tokens by how often
            training saw them.
        samples (`int`, *optional*, defaults to 0):
            Held out documents per language whose first 200 characters are continued by 60 greedy tokens
            ([`~polyplex_research.utils.evaluate.dense_generate`]).
        personaplex_baseline (`bool`, *optional*, defaults to `False`):
            Also scores PersonaPlex with its own tokenizer on the same documents, before the DualFFN is loaded.
        language_model_samples (`bool`, *optional*, defaults to `False`):
            Also continues the `samples` prompts with the text LLM and its own tokenizer, 60 greedy tokens each
            ([`~polyplex_research.utils.evaluate.language_model_continuation`]), before the DualFFN is loaded.
        language_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the checkpoint holds the language stream's own text table (SPEC S11.2f).
        language_lora_rank (`int`, *optional*, defaults to 0):
            Rank of the language FFN's low rank corrections the checkpoint holds (SPEC S11.2g).
        speech_utterances (`int`, *optional*, defaults to 0):
            Also scores this many examples of the held out English speech set spoken on the assistant side
            ([`~polyplex_research.utils.evaluate.heldout_speech_items`],
            [`~polyplex_research.utils.evaluate.speech_losses`]): the trained model, the model with every language
            FFN output scale at 0, and the model with every language FFN silenced, and PersonaPlex with
            `personaplex_baseline`.
        language_stream_from_text (`bool`, *optional*, defaults to `True`):
            Whether the language stream starts from the text embeddings alone (SPEC S11.2e). `False` for checkpoints
            trained under SPEC S11.2d.
        speech_path (`str`, *optional*):
            The held out English speech set (`train.sh` `SCORE_ENGLISH_SPEECH`).
        silent_audio (`bool`, *optional*, defaults to `False`):
            Whether the scored text carries codec silence in both acoustic streams, the dense input of checkpoints
            trained before SPEC S13.1a (r4 of PR #27 and earlier). Without it the model reads the text embeddings
            alone.
        nvfp4_layers (`tuple[int]`, *optional*, defaults to `()`):
            Layers kept in full precision at assembly that are run in NVFP4 for this evaluation, frozen
            ([`~polyplex_research.utils.quantization.quantize_linears_nvfp4`]).
        sample_temperature (`float`, *optional*, defaults to 0.0):
            Temperature the `samples` continuations of the trained model and the text LLM are drawn at, from the 50
            most likely tokens with seed 0. `0` continues greedily.
        release_dir (`str`, *optional*):
            Directory the model with the checkpoint's parameters is saved to as the released model
            ([`~polyplex_research.utils.assemble.save_dense_release`], SPEC S13.4c), in place of scoring it.
        shared_language_stream (`str`, *optional*):
            The shared alignment file of a checkpoint whose language stream is folded (SPEC S11.2h).
        release_check (`bool`, *optional*, defaults to `False`):
            Compares the release already in `release_dir` with the model it was saved from ([`compare_release`])
            instead of saving it.

        The other arguments are those of [`run_dense_adaptation`], set as the run that wrote the checkpoint set them.

    Returns:
        `dict`: `step`, `reference` (the text LLM's bits per byte), per label the bits per byte of each language, and
        `personaplex`, `token_groups`, `samples` and `speech` when asked for.
    """
    import gc

    import wandb

    from ..models.polyplex import PolyplexProcessor, PolyplexTokenizer
    from .assemble import compile_backbone, load_or_assemble_dualffn

    use_expandable_segments()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_tokenizer = PolyplexTokenizer.from_pretrained(tokenizer)
    held_out = {}
    for language in DENSE_EVALUATION:
        documents = dense_documents(language, "valid", eval_texts * eval_characters, text_tokenizer, max_files)
        held_out[language] = [document[:eval_characters] for document in documents[:eval_texts]]
    calibration_texts = dense_calibration_texts(text_tokenizer, skip=eval_texts)
    reference = fetch_reference(reference_key(language_model, held_out)) or {}
    text_bytes = token_byte_lengths(text_tokenizer)

    processor = PolyplexProcessor.from_pretrained(personaplex)
    processor.audio_tokenizer.to(device)
    speech_items = []
    if speech_utterances:
        import copy

        from ..datasets.duplex_data import duplex_collate_fn, packed_words
        from .evaluate import heldout_speech_items, speech_losses

        speech_items = heldout_speech_items(speech_path, count=speech_utterances)

        def speech_collate(tokenizer, scored_model, base=processor):
            # The same speech and persona, written by the scored model's own tokenizer.
            speech_processor = copy.copy(base)
            speech_processor.tokenizer = tokenizer
            return duplex_collate_fn(speech_processor, scored_model, audio_labels=True, layout=packed_words)

    speech = {}
    baseline = {}
    language_model_texts = {}
    if samples and language_model_samples:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .evaluate import language_model_continuation

        llm_tokenizer = AutoTokenizer.from_pretrained(language_model)
        # The text LLM generates and is released before the DualFFN is loaded, so the two never share the device.
        llm = AutoModelForCausalLM.from_pretrained(language_model, dtype=torch.bfloat16, device_map=device).eval()
        for language, texts in held_out.items():
            language_model_texts[language] = [
                language_model_continuation(llm, llm_tokenizer, text[:200], 60, temperature=sample_temperature)
                for text in texts[:samples]
            ]
            for text in language_model_texts[language]:
                logger.info("Language model sample %s: %s", language, json.dumps(text, ensure_ascii=False))
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if personaplex_baseline:
        from transformers import PersonaPlexForConditionalGeneration

        from .assemble import load_base_tokenizer

        helium_tokenizer = load_base_tokenizer(personaplex)
        helium_bytes = token_byte_lengths(helium_tokenizer)
        # PersonaPlex is scored and released before the DualFFN is loaded, so the two never share the device.
        helium = PersonaPlexForConditionalGeneration.from_pretrained(
            personaplex, torch_dtype=torch.bfloat16, device_map=device
        )
        collate_helium = dense_collate_fn(processor, helium)
        baseline = {
            language: dense_bits_per_byte(helium, collate_helium, helium_tokenizer, texts, helium_bytes)
            for language, texts in held_out.items()
        }
        logger.info("PersonaPlex: %s", {language: round(score, 4) for language, score in baseline.items()})
        if speech_items:
            helium.eval()
            speech["personaplex"] = speech_losses(
                helium, speech_collate(helium_tokenizer, helium), speech_items, helium_bytes
            )
            logger.info("Speech, PersonaPlex: %s", speech["personaplex"])
        del helium, collate_helium
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    model = load_or_assemble_dualffn(
        device,
        backward_dtype,
        personaplex=personaplex,
        language_model=language_model,
        language_ffn_layers=language_ffn_layers,
        tokenizer=tokenizer,
        quantization=quantization,
        full_precision_layers=tuple(full_precision_layers),
        appended_layers=appended_layers,
        calibration_texts=calibration_texts,
        alignment=alignment,
        **({} if removed_neurons is None else {"removed_neurons": removed_neurons}),
        **({"keep_appended_output": True} if language_scale is not None or language_stream is not None else {}),
    )
    if language_scale is not None:
        gate_language_ffn(model, language_scale)
    if language_stream is not None:
        open_language_stream(model, language_stream)
        model.config.language_stream_from_text = language_stream_from_text
        if language_embeddings:
            open_language_embeddings(model, text_tokenizer, language_model)
    if language_lora_rank:
        open_language_lora(model, language_lora_rank)
    step = 0
    if checkpoint is not None:
        state = read_checkpoint(checkpoint)
        if state["extra"].get("language_stream_folded"):
            if shared_language_stream is None:
                raise ValueError("The checkpoint's language stream is folded; pass its shared alignment.")
            shared_stream_switch(shared_language_stream, language_model, {"gains": state["buffers"]["stream_gains"]})(model)
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, weight in state["weights"].items():
                parameters[name].copy_(state["masters"].get(name, weight).to(parameters[name].device))
        step = state["step"]
        del state
    gc.collect()
    if nvfp4_layers:
        from .quantization import quantize_linears_nvfp4

        for index in nvfp4_layers:
            quantize_linears_nvfp4(model.model.layers[index], frozen=True)
    if release_dir is not None and release_check:
        return {"step": step, "release_check": compare_release(model, str(release_dir), held_out, text_tokenizer, device)}
    if release_dir is not None:
        from .assemble import RELEASE_AUTO_MAP, save_dense_release

        save_dense_release(model, str(release_dir), auto_map=RELEASE_AUTO_MAP)
        return {"step": step, "release": str(release_dir)}
    use_flex_attention(model)
    compile_backbone(model)
    collate = dense_collate_fn(processor, model, silent_audio=silent_audio)
    del processor
    model.eval()
    model.config.use_cache = False

    total = len(model.model.layers)
    groups = {"backbone": range(total - appended_layers), "appended": range(total - appended_layers, total)}
    run = wandb.init(
        project="polyplex",
        name=run_name,
        tags=["phase1", "N14", "S11.2c", "ablation"],
        config={"checkpoint": str(checkpoint), "step": step, "ablations": list(ablations), "tokenizer": tokenizer,
                "silent_audio": silent_audio, "nvfp4_layers": list(nvfp4_layers),
                "language_stream": language_stream, "eval_texts": eval_texts, "eval_characters": eval_characters,
                **{f"reference/{language}_bits_per_byte": score for language, score in reference.items()},
                **git_record(os.path.dirname(os.path.abspath(__file__)))},
    )
    result = {"step": step, "reference": reference}
    if baseline:
        result["personaplex"] = baseline
        run.log({f"eval/personaplex/{language}_bits_per_byte": score for language, score in baseline.items()})
    for label, factor, layers in ablations:
        chosen = groups[layers] if isinstance(layers, str) else layers
        # A factor of None silences the language FFNs themselves ([`language_ffn_off`]).
        silence = language_ffn_off(model, chosen) if factor is None else language_scale_multiplied(model, factor, chosen)
        with silence:
            scores = {
                language: dense_bits_per_byte(model, collate, text_tokenizer, texts, text_bytes)
                for language, texts in held_out.items()
            }
        result[label] = scores
        run.log({f"eval/{label}/{language}_bits_per_byte": score for language, score in scores.items()})
        logger.info("%s (factor %s, layers %s): %s", label, factor, layers,
                    {language: round(score, 4) for language, score in scores.items()})
    if speech_items:
        collate_speech = speech_collate(text_tokenizer, model)
        conditions = (
            ("trained", contextlib.nullcontext()),
            ("language_scale_0", language_scale_multiplied(model, 0.0)),
            ("language_ffn_off", language_ffn_off(model)),
        )
        for label, condition in conditions:
            with condition:
                speech[label] = speech_losses(model, collate_speech, speech_items, text_bytes)
            logger.info("Speech, %s: %s", label, speech[label])
        result["speech"] = speech
        for label, losses in speech.items():
            run.log({f"speech/{label}/{key}": value for key, value in losses.items() if not isinstance(value, list)})
    if token_groups:
        base = model.config.base_vocab_size
        counts = None
        if training_tokens:
            import numpy as np

            counts = np.bincount(np.fromfile(training_tokens, dtype=np.int32), minlength=len(text_tokenizer))
        result["token_groups"] = {}
        for label, factor in (("trained", 1.0), ("language_scale_0", 0.0)):
            with language_scale_multiplied(model, factor):
                result["token_groups"][label] = {
                    language: grouped_bits_per_byte(
                        dense_token_losses(model, collate, text_tokenizer, texts, text_bytes, base), counts=counts
                    )
                    for language, texts in held_out.items()
                }
            logger.info("Token groups, %s: %s", label, json.dumps(result["token_groups"][label]))
    if samples:
        from .evaluate import dense_generate

        result["samples"] = {}
        for language, texts in held_out.items():
            result["samples"][language] = []
            for text in texts[:samples]:
                prompt = text_tokenizer.encode(text[:200], add_special_tokens=False)
                pieces = dense_generate(model, collate, torch.tensor(prompt), 60, temperature=sample_temperature)
                entry = {"prompt": text[:200], "continuation": text_tokenizer.decode(pieces.tolist())}
                if language in language_model_texts:
                    entry["language_model"] = language_model_texts[language][len(result["samples"][language])]
                result["samples"][language].append(entry)
                logger.info("Sample %s: %s", language, json.dumps(entry, ensure_ascii=False))
    run.finish()
    return result


def dense_main(args: list[str] | None = None) -> dict:
    r"""
    Parses the command line of the dense quick check and runs [`run_dense_adaptation`].

    Args:
        args (`list[str]`, *optional*):
            Command line arguments after `dense`, `sys.argv[2:]` when omitted.

    Returns:
        `dict`: The result of [`run_dense_adaptation`].
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Dense quick check of a DualFFN model (SPEC S13.4).")
    parser.add_argument("--personaplex", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--language-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--row-frames", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-rows", type=int, default=64)
    parser.add_argument("--max-files", type=int, default=1)
    parser.add_argument("--language-ffn-layers", default=None, help="Comma separated text LLM layers, one per backbone layer.")
    parser.add_argument("--quantization", default="nvfp4", help="Kind the frozen backbone runs in; 'none' keeps bfloat16.")
    parser.add_argument("--run-name", default="dual-ffn")
    parser.add_argument("--document-cache", default=None, help="Directory dense_documents keeps its documents in.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokenizer", default=None, help="Polyplex tokenizer to assemble with; the text LLM's when omitted.")
    parser.add_argument("--eval-texts", type=int, default=64)
    parser.add_argument("--eval-characters", type=int, default=2000)
    parser.add_argument(
        "--full-precision-layers", default="", help="Comma separated layers left in full precision."
    )
    parser.add_argument("--appended-layers", type=int, default=4, help="Copies of Helium's last layers appended.")
    parser.add_argument("--decay-steps", type=int, default=0, help="Learning rate decay steps that end this run.")
    parser.add_argument("--save-every", type=int, default=500, help="Steps between checkpoints moved to Drive.")
    parser.add_argument(
        "--lr-schedule", choices=LR_SCHEDULES, default="wsd", help="Learning rate schedule of the dense stage."
    )
    parser.add_argument("--resume", default=None, help="Run whose Drive checkpoint this run continues from.")
    parser.add_argument("--backward-dtype", default="bf16", help="Backward matmul precision of the NVFP4 layers.")
    parser.add_argument("--alignment", default=None, help="Residual basis alignment folded into the language FFNs.")
    parser.add_argument("--evaluate-at-start", action="store_true", help="Evaluate a resumed run before it steps.")
    parser.add_argument("--removed-neurons", default=None, help="JSON file of language FFN neurons removed per layer.")
    parser.add_argument("--language-scale", type=float, default=None, help="Start of the language FFN output scale.")
    parser.add_argument("--language-stream", default=None, help="Alignment file joining a second residual stream.")
    parser.add_argument(
        "--cache-backward-weights",
        action="store_true",
        help="Keep each frozen NVFP4 projection's MXFP8 backward weight, about 1 byte per weight on the device.",
    )
    parser.add_argument(
        "--language-lora-rank",
        type=int,
        default=0,
        help="Rank of the trained low rank corrections beside each language FFN projection (SPEC S11.2g).",
    )
    parser.add_argument(
        "--language-embeddings",
        action="store_true",
        help="Start the language stream from its own text table, initialized from the text LLM's (SPEC S11.2f).",
    )
    parser.add_argument(
        "--restore-appended-language-down",
        action="store_true",
        help="Put the text LLM's output projection back into appended layers of a resumed checkpoint.",
    )
    parser.add_argument(
        "--silent-audio", action="store_true", help="Fill both acoustic streams of the dense rows with codec silence."
    )
    parser.add_argument(
        "--dense-languages", default=None, help="Comma separated dense language keys the training mixture keeps."
    )
    parser.add_argument(
        "--shared-language-stream", default=None, help="Shared alignment file the language stream switches to."
    )
    parser.add_argument(
        "--stream-rotation-switch-step",
        type=int,
        default=8000,
        help="Last step trained with the per-layer stream rotations (SPEC S11.2h).",
    )
    parser.add_argument(
        "--unfiltered-documents",
        action="store_true",
        help="Train on FineWeb-2 for every language but English instead of the quality filtered sources (SPEC S13.4).",
    )
    parser.add_argument("--gpus", type=int, default=1, help="GPUs of this machine trained on in data parallel.")
    args = sys.argv[2:] if args is None else [str(arg) for arg in args]
    arguments = parser.parse_args(args)
    inputs = os.path.join(os.environ.get("HF_HOME", ".cache/huggingface"), "inputs")
    if arguments.gpus > 1 and "RANK" not in os.environ:
        # letify serves the Drive store only to the process it runs the call in, so the input files are fetched here
        # and the data parallel processes get their local paths.
        index = args.index("--gpus")
        args = args[:index] + args[index + 2 :]
        for flag in DENSE_INPUT_FILES:
            if flag in args:
                position = args.index(flag) + 1
                args[position] = fetch_input_file(args[position], inputs)
        return dense_data_parallel(args, arguments.gpus)
    rank = int(os.environ.get("RANK", "0"))
    if "RANK" in os.environ:
        import datetime

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        # The other processes wait in a barrier while the first evaluates and saves.
        torch.distributed.init_process_group("nccl", timeout=datetime.timedelta(hours=1))
        if rank:
            os.environ["WANDB_MODE"] = "disabled"
    if arguments.document_cache:
        os.environ["POLYPLEX_DOCUMENT_CACHE"] = arguments.document_cache
    # An imported kernel library configures the root logger first, so the run's INFO lines need `force`.
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING, force=True)
    arguments.alignment = fetch_input_file(arguments.alignment, inputs)
    arguments.removed_neurons = fetch_input_file(arguments.removed_neurons, inputs)
    arguments.language_stream = fetch_input_file(arguments.language_stream, inputs)
    arguments.shared_language_stream = fetch_input_file(arguments.shared_language_stream, inputs)
    result = run_dense_adaptation(
        personaplex=arguments.personaplex,
        language_model=arguments.language_model,
        work_dir=arguments.work_dir,
        steps=arguments.steps,
        batch_size=arguments.batch_size,
        micro_batch_size=arguments.micro_batch_size,
        row_frames=arguments.row_frames,
        learning_rate=arguments.learning_rate,
        warmup_steps=arguments.warmup_steps,
        eval_every=arguments.eval_every,
        eval_rows=arguments.eval_rows,
        max_files=arguments.max_files,
        language_ffn_layers=(
            [int(index) for index in arguments.language_ffn_layers.split(",")]
            if arguments.language_ffn_layers
            else None
        ),
        quantization=None if arguments.quantization == "none" else arguments.quantization,
        run_name=arguments.run_name,
        seed=arguments.seed,
        tokenizer=arguments.tokenizer,
        eval_texts=arguments.eval_texts,
        eval_characters=arguments.eval_characters,
        full_precision_layers=tuple(int(part) for part in arguments.full_precision_layers.split(",") if part),
        appended_layers=arguments.appended_layers,
        decay_steps=arguments.decay_steps,
        lr_schedule=arguments.lr_schedule,
        save_every=arguments.save_every,
        resume=arguments.resume,
        backward_dtype=arguments.backward_dtype,
        alignment=arguments.alignment,
        evaluate_at_start=arguments.evaluate_at_start,
        removed_neurons=None if arguments.removed_neurons is None else json.load(open(arguments.removed_neurons)),
        language_scale=arguments.language_scale,
        language_stream=arguments.language_stream,
        language_embeddings=arguments.language_embeddings,
        language_lora_rank=arguments.language_lora_rank,
        cache_backward_weights=arguments.cache_backward_weights,
        restore_appended_language_down=arguments.restore_appended_language_down,
        silent_audio=arguments.silent_audio,
        languages=tuple(arguments.dense_languages.split(",")) if arguments.dense_languages else None,
        shared_language_stream=arguments.shared_language_stream,
        stream_rotation_switch_step=arguments.stream_rotation_switch_step,
        filtered_documents=not arguments.unfiltered_documents,
    )
    if "RANK" in os.environ:
        if rank == 0:
            with open(os.environ["POLYPLEX_DENSE_RESULT"], "w", encoding="utf-8") as file:
                json.dump(result, file)
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return result


def dense_eval_main(args: list[str] | None = None) -> dict:
    r"""
    Parses the command line of a dense checkpoint evaluation and runs [`evaluate_dense_checkpoint`] on the checkpoint
    run `--checkpoint` left, found as [`resume_checkpoint`] finds it: the latest local `checkpoint-<step>` under
    `$HF_HOME/runs/<run>`, or else the stored one.

    Args:
        args (`list[str]`, *optional*):
            Command line arguments after `dense-eval`, `sys.argv[2:]` when omitted.

    Returns:
        `dict`: The result of [`evaluate_dense_checkpoint`].
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Evaluate a dense checkpoint (SPEC S13.4).")
    parser.add_argument("--checkpoint", required=True, help="Run whose checkpoint is scored; 'none' for the assembly.")
    parser.add_argument("--run-name", required=True, help="Name of this evaluation's W&B run.")
    parser.add_argument("--personaplex", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--language-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--tokenizer", default="RetentionLabs/PolyPlex-14B-CJKE")
    parser.add_argument("--language-ffn-layers", default=None, help="Comma separated text LLM layers.")
    parser.add_argument("--quantization", default="nvfp4", help="Kind the frozen backbone runs in; 'none' keeps bfloat16.")
    parser.add_argument("--full-precision-layers", default="", help="Comma separated layers left in full precision.")
    parser.add_argument("--appended-layers", type=int, default=4)
    parser.add_argument("--backward-dtype", default="mxfp8")
    parser.add_argument("--alignment", default=None)
    parser.add_argument("--removed-neurons", default=None, help="JSON file of language FFN neurons removed per layer.")
    parser.add_argument("--language-scale", type=float, default=None)
    parser.add_argument("--language-stream", default=None, help="Alignment file joining a second residual stream.")
    parser.add_argument("--eval-texts", "--eval-rows", dest="eval_texts", type=int, default=64)
    parser.add_argument("--eval-characters", type=int, default=2000)
    parser.add_argument("--max-files", type=int, default=1)
    parser.add_argument("--no-ablations", action="store_true", help="Score the trained model alone.")
    parser.add_argument("--token-groups", action="store_true")
    parser.add_argument("--training-tokens", default=None, help="tokens.bin of the run's training rows.")
    parser.add_argument("--samples", type=int, default=0, help="Held out documents per language continued greedily.")
    parser.add_argument("--personaplex-baseline", action="store_true")
    parser.add_argument("--language-model-samples", action="store_true")
    parser.add_argument("--language-embeddings", action="store_true")
    parser.add_argument("--language-lora-rank", type=int, default=0)
    parser.add_argument("--speech-utterances", type=int, default=0)
    parser.add_argument("--nvfp4-layers", default="", help="Comma separated full precision layers run in NVFP4.")
    parser.add_argument("--silent-audio", action="store_true")
    parser.add_argument("--language-stream-from-embeddings", action="store_true", help="Checkpoints of SPEC S11.2d.")
    parser.add_argument("--speech-path", default=".cache/huggingface/runs/data/english_heldout_speech.pt")
    parser.add_argument("--document-cache", default=None, help="Directory dense_documents keeps its documents in.")
    parser.add_argument("--release-dir", default=None, help="Save the model as the release here instead of scoring it.")
    parser.add_argument("--sample-temperature", type=float, default=0.0, help="Temperature of the samples; 0 is greedy.")
    parser.add_argument("--release-check", action="store_true", help="Compare the release in --release-dir instead.")
    parser.add_argument("--shared-language-stream", default=None, help="Shared alignment of a folded checkpoint.")
    arguments = parser.parse_args(sys.argv[2:] if args is None else args)
    if arguments.document_cache:
        os.environ["POLYPLEX_DOCUMENT_CACHE"] = arguments.document_cache
    logging.basicConfig(level=logging.INFO, force=True)
    runs = os.path.join(os.environ.get("HF_HOME", ".cache/huggingface"), "runs")
    inputs = os.path.join(os.environ.get("HF_HOME", ".cache/huggingface"), "inputs")
    files = {
        flag: fetch_input_file(getattr(arguments, flag), inputs)
        for flag in (
            "alignment", "removed_neurons", "language_stream", "training_tokens", "speech_path", "shared_language_stream"
        )
    }
    checkpoint = None
    if arguments.checkpoint != "none":
        checkpoint = str(resume_checkpoint(arguments.checkpoint, os.path.join(runs, arguments.run_name)))

    def layers(value):
        return tuple(int(part) for part in value.split(",") if part)

    return evaluate_dense_checkpoint(
        checkpoint=checkpoint,
        run_name=arguments.run_name,
        personaplex=arguments.personaplex,
        language_model=arguments.language_model,
        tokenizer=arguments.tokenizer,
        language_ffn_layers=list(layers(arguments.language_ffn_layers)) if arguments.language_ffn_layers else None,
        quantization=None if arguments.quantization == "none" else arguments.quantization,
        full_precision_layers=layers(arguments.full_precision_layers),
        appended_layers=arguments.appended_layers,
        backward_dtype=arguments.backward_dtype,
        alignment=files["alignment"],
        removed_neurons=None if files["removed_neurons"] is None else json.load(open(files["removed_neurons"])),
        language_scale=arguments.language_scale,
        language_stream=files["language_stream"],
        eval_texts=arguments.eval_texts,
        eval_characters=arguments.eval_characters,
        max_files=arguments.max_files,
        ablations=() if arguments.no_ablations else DENSE_SCALE_ABLATIONS,
        token_groups=arguments.token_groups,
        training_tokens=files["training_tokens"],
        samples=arguments.samples,
        personaplex_baseline=arguments.personaplex_baseline,
        language_model_samples=arguments.language_model_samples,
        language_embeddings=arguments.language_embeddings,
        language_lora_rank=arguments.language_lora_rank,
        speech_utterances=arguments.speech_utterances,
        nvfp4_layers=layers(arguments.nvfp4_layers),
        silent_audio=arguments.silent_audio,
        language_stream_from_text=not arguments.language_stream_from_embeddings,
        speech_path=files["speech_path"],
        release_dir=arguments.release_dir,
        sample_temperature=arguments.sample_temperature,
        release_check=arguments.release_check,
        shared_language_stream=files["shared_language_stream"],
    )


DENSE_INPUT_FILES = ("--alignment", "--removed-neurons", "--language-stream", "--shared-language-stream")
DENSE_EVAL_INPUT_FILES = DENSE_INPUT_FILES + ("--training-tokens", "--speech-path")
STORE_PREFIX = "store:"


def stash_input_files(args: list[str], run_name: str, flags: tuple = DENSE_INPUT_FILES) -> list[str]:
    r"""
    Puts the local files a dense command line names ([`DENSE_INPUT_FILES`]) on the Drive store, so a run on a rented
    machine can read them, and names them there instead: `--alignment <path>` becomes
    `--alignment store:<run_name>-alignment`.

    Args:
        args (`list[str]`):
            The arguments of [`dense_main`].
        run_name (`str`):
            Name of the run, the prefix of the store names.
        flags (`tuple[str]`, *optional*, defaults to [`DENSE_INPUT_FILES`]):
            The flags whose values are files.

    Returns:
        `list[str]`: The arguments with the files replaced by their store names.
    """
    import letify

    args = list(args)
    for flag in flags:
        if flag in args and not args[args.index(flag) + 1].startswith(STORE_PREFIX):
            index = args.index(flag) + 1
            name = f"{run_name}-{flag.lstrip('-')}"
            letify.stash(name, args[index])
            logger.info("%s put on the Drive store as %s.", args[index], name)
            args[index] = STORE_PREFIX + name
    return args


def fetch_input_file(value: str | None, directory: str) -> str | None:
    r"""
    Returns a local path for a file argument of [`dense_main`], fetching it from the Drive store when it names one
    ([`stash_input_files`]).

    Args:
        value (`str`, *optional*):
            A path, or `store:<name>`.
        directory (`str`):
            Directory a fetched file is written to.

    Returns:
        `str` or `None`: The local path, or `None` when `value` is.
    """
    if value is None or not value.startswith(STORE_PREFIX):
        return value
    import letify

    name = value[len(STORE_PREFIX):]
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    if letify.unstash(name, path) is None:
        raise FileNotFoundError(f"The Drive store holds no {name!r}.")
    return path


def save_dense_result(result: dict, output_dir: str) -> None:
    r"""
    Writes the result of [`run_dense_adaptation`] to `result.json`. The trained weights are not part of it; they are
    on the Drive store under the checkpoint the result names.

    Args:
        result (`dict`):
            The result.
        output_dir (`str`):
            Directory written to.
    """
    import json

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "result.json"), "w", encoding="utf-8") as sink:
        json.dump(result, sink, indent=2)


def remote_secrets() -> dict:
    r"""
    Collects the credentials a run on a rented machine needs from this one: the Hugging Face token and the W&B
    API key (from `WANDB_API_KEY` or `~/.netrc`).

    Returns:
        `dict`: `HF_TOKEN` and `WANDB_API_KEY`, each only when this machine has it.
    """
    import netrc

    import huggingface_hub

    secrets = {}
    token = huggingface_hub.get_token()
    if token:
        secrets["HF_TOKEN"] = token
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        try:
            entry = netrc.netrc(os.path.join(os.path.expanduser("~"), ".netrc")).authenticators("api.wandb.ai")
        except (OSError, netrc.NetrcParseError):
            entry = None
        key = entry[2] if entry else None
    if key:
        secrets["WANDB_API_KEY"] = key
    return secrets


def package_archive() -> bytes:
    r"""
    Zips the Python sources of the `polyplex_research` package, without bytecode caches.

    Returns:
        `bytes`: The archive, with paths starting at `polyplex_research/`.
    """
    import io
    import zipfile

    package = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parent = os.path.dirname(package)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for directory, subdirectories, files in os.walk(package):
            subdirectories[:] = sorted(name for name in subdirectories if name != "__pycache__")
            for name in sorted(files):
                if name.endswith(".py"):
                    path = os.path.join(directory, name)
                    archive.write(path, os.path.relpath(path, parent))
    return buffer.getvalue()


def locked_extras(extras: tuple) -> list[str]:
    r"""
    Pins the packages of the project's optional dependency groups to the versions installed here.

    Args:
        extras (`tuple`):
            Names of groups under `[project.optional-dependencies]` in `pyproject.toml`.

    Returns:
        `list[str]`: Requirements of the form `name==version`, one per package.
    """
    import importlib.metadata
    import re
    import tomllib

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(root, "pyproject.toml"), "rb") as source:
        groups = tomllib.load(source)["project"]["optional-dependencies"]
    names = sorted({re.split(r"[<>=!~;\[ ]", requirement, maxsplit=1)[0] for extra in extras for requirement in groups[extra]})
    return [f"{name}=={importlib.metadata.version(name)}" for name in names]


def archive_runner():
    r"""
    Builds the function a remote letify call runs: it unpacks a package archive into a fresh directory, puts
    that directory first on `sys.path`, drops the entries `PYTHONPATH` added, sets the given environment variables,
    imports the entry point and calls it with the command line.

    The function is defined inside this one, so cloudpickle sends it by value, and it uses the standard library
    only, so it runs on a machine that does not have this package installed.

    Returns:
        `Callable`: `run(archive, module, function, args, secrets)`, returning what the entry point returns.
    """

    def run(archive: bytes, module: str, function: str, args: list, secrets: dict):
        import importlib
        import io
        import os
        import sys
        import tempfile
        import zipfile

        # A host PYTHONPATH can name another interpreter's site-packages, which would shadow the locked environment
        # in this process and in the compile workers it starts.
        shadowing = {os.path.abspath(entry) for entry in os.environ.pop("PYTHONPATH", "").split(os.pathsep) if entry}
        sys.path[:] = [entry for entry in sys.path if not entry or os.path.abspath(entry) not in shadowing]
        root = tempfile.mkdtemp(prefix="letify-package-")
        zipfile.ZipFile(io.BytesIO(archive)).extractall(root)
        sys.path.insert(0, root)
        os.environ.update(secrets)
        # A path letify sent arrives as a path object; the entry points parse strings.
        args = [os.fspath(arg) if isinstance(arg, os.PathLike) else arg for arg in args]
        return getattr(importlib.import_module(module), function)(args)

    return run


def retry_lost_sessions(call, attempts: int = 5):
    r"""
    Calls `call` again when letify loses the session, up to `attempts` times in all.

    A lost connection or a worker that dies while the session boots is retried. An exception raised by the run's
    own code (`letify.errors.RemoteError`) is not.

    Args:
        call (`Callable`):
            Starts the run and returns its result.
        attempts (`int`, *optional*, defaults to 5):
            Calls made at most.

    Returns:
        What `call` returns.

    Raises:
        letify.errors.LetifyError: The last session failure, once every attempt failed.
    """
    from letify.errors import ProtocolError, RuntimeFailure

    for attempt in range(1, attempts + 1):
        try:
            return call()
        except (ProtocolError, RuntimeFailure) as error:
            if attempt == attempts:
                raise
            logger.warning("Session lost on attempt %d of %d, starting a new one: %s", attempt, attempts, error)


def checkpoint_on_hub(run_name: str) -> bool:
    r"""
    Tells whether the public Hugging Face repository holds a checkpoint manifest of a run.

    Args:
        run_name (`str`):
            Name of the run.

    Returns:
        `bool`: Whether `<run_name>/checkpoint/manifest.json` is in [`checkpoint_repo`].
    """
    from huggingface_hub import HfApi

    try:
        return HfApi().file_exists(checkpoint_repo(), f"{run_name}/checkpoint/manifest.json")
    except Exception as error:
        logger.info("%s could not be checked for a checkpoint of %s: %s", checkpoint_repo(), run_name, error)
        return False


def stored_checkpoint_exists(run_name: str) -> bool:
    r"""
    Tells whether a dense run has a checkpoint [`resume_checkpoint`] can read: on the local disk under
    `$HF_HOME/runs/<run_name>/checkpoint-<step>` (or the older single `checkpoint`), on the Drive store, or on the public Hugging Face repository.

    Args:
        run_name (`str`):
            Name of the run.

    Returns:
        `bool`: Whether any of them holds a checkpoint manifest of the run.
    """
    import tempfile
    from pathlib import Path

    import letify

    runs = Path(os.environ.get("HF_HOME", ".cache/huggingface")) / "runs"
    if local_checkpoints(runs / run_name):
        return True
    with tempfile.TemporaryDirectory() as directory:
        try:
            if letify.unstash(f"{run_name}-checkpoint", Path(directory) / "manifest.json") is not None:
                return True
        except Exception as error:
            logger.info("The Drive store could not be checked for a checkpoint of %s: %s", run_name, error)
    return checkpoint_on_hub(run_name)


def resume_on_retry(args: list[str], run_name: str) -> list[str]:
    r"""
    Command line of a dense run started again after its session was lost: `--resume <run_name>` is added when the
    run stored a checkpoint ([`stored_checkpoint_exists`]) and the command line names no `--resume` of its own, so
    the run continues from its last checkpoint.

    Args:
        args (`list[str]`):
            Command line arguments of [`dense_main`].
        run_name (`str`):
            Name of the run.

    Returns:
        `list[str]`: The command line for the retry.
    """
    if "--resume" in args or not stored_checkpoint_exists(run_name):
        return args
    logger.info("Resuming the retried run from the stored checkpoint of %s.", run_name)
    return [*args, "--resume", run_name]


def resolve_device(launcher, spec: str):
    r"""
    Turns a device spec into the letify instance a run is declared on.

    Args:
        launcher (`letify.Launcher`):
            The launcher whose providers are searched.
        spec (`str`):
            `"<provider>.<accelerator>"`, for example `"local.RTX_PRO_6000"` or `"colab_pro_plus.G4"`. A bare
            provider name takes its first accelerator, and the CPU only when it has none. A `*N` suffix, as in
            `"lab_gpu.RTX_PRO_6000*2"`, takes N cards in one session (SPEC S13.4b).

    Returns:
        `letify.Instance`: The declared accelerator.

    Raises:
        letify.errors.UnknownInstance: If the provider does not offer the accelerator.
    """
    spec, _, count = spec.partition("*")
    alias, _, accelerator = spec.partition(".")
    provider = getattr(launcher.providers, alias)
    if not accelerator:
        names = [name for name in provider.discover() if name != "CPU"]
        accelerator = names[0] if names else "CPU"
    instance = getattr(provider, accelerator)
    return instance * int(count) if count else instance


def launch(args: list[str] | None = None):
    r"""
    Runs [`main`] on the accelerator `POLYPLEX_DEVICE` names, through letify.

    The default, `"local"`, is the first GPU of this machine, where the call runs in a subprocess with this
    environment. A remote provider takes the same declaration and ships the call to its machine, so a run
    moves between the local card and a rented one by changing the variable alone.

    A process started by `torchrun` with more than one worker runs [`main`] directly, since a letify call
    holds a single GPU and the workers of a data parallel run have to share one process group.

    A command line starting with `dense` runs [`dense_main`] instead, and its result is saved on this machine
    under `$HF_HOME/runs/<run name>` with [`save_dense_result`]. One starting with `dense-eval` runs
    [`dense_eval_main`], and its result is saved as `$HF_HOME/runs/<run name>-eval.json`. One starting with
    `residual-alignment` runs that command of [`~polyplex_research.utils.assemble.main`], and the alignment comes back to
    the directory of its `--output`. The Hugging Face token and the W&B key of this
    machine are set in the run's process ([`remote_secrets`]), and the package is sent along ([`archive_runner`]).

    Args:
        args (`list[str]`, *optional*):
            Command line arguments of [`main`], `sys.argv[1:]` when omitted.
    """
    import sys
    from pathlib import Path

    args = sys.argv[1:] if args is None else args
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        # One process per GPU, started by torchrun, which has already chosen the device.
        main(args)
        return

    import letify

    launcher = letify.Launcher()
    device = resolve_device(launcher, os.environ.get("POLYPLEX_DEVICE", "local"))
    # A remote runtime installs the locked dependencies but not this project, so the package travels as an
    # argument of the call, over the same channel as the call itself.
    # A Colab session reached over tcp_punch sometimes drops its SSH connection while it boots.
    # The runtime syncs the lock without extras, so the extras a run imports are installed at their locked versions.
    env = letify.Env.from_lock().pip_install(*locked_extras(("preprocess", "audio", "train")))
    run = launcher.function(device=device, host=letify.remote, env=env, retries=3)(archive_runner())
    module = "polyplex_research.utils.phase"
    if args[:1] == ["dense"]:
        # The Hugging Face chunk cache keeps a second copy of every download, which the dense run's disk cannot hold.
        environment = {**remote_secrets(), "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0"}
        # A local run inherits this process's environment, so the value is always set, never left to the shell.
        environment["POLYPLEX_EPHEMERAL_DISK"] = "1" if ephemeral_disk(os.environ.get("POLYPLEX_DEVICE", "local")) else "0"
        run_name = args[args.index("--run-name") + 1] if "--run-name" in args else "dual-ffn"
        remote = os.environ.get("POLYPLEX_DEVICE", "local").partition(".")[0] != "local"
        dense_args = stash_input_files(args[1:], run_name) if remote else args[1:]
        if device.devices > 1:
            dense_args = dense_args + ["--gpus", str(device.devices)]
        # letify sends the files a call names by `Path` and copies back what the call adds there, so documents read
        # once on any machine are not streamed again.
        documents = Path(os.environ.get("HF_HOME", ".cache/huggingface"), "documents").absolute()
        documents.mkdir(parents=True, exist_ok=True)
        dense_args = dense_args + ["--document-cache", documents]
        # The sources are read once, so a retried session runs the code the run started with.
        archive = package_archive()
        started = []

        def dense_call():
            call_args = resume_on_retry(dense_args, run_name) if started else dense_args
            started.append(True)
            return run(archive, module, "dense_main", call_args, environment)

        result = retry_lost_sessions(dense_call)
        output_dir = os.path.join(os.environ.get("HF_HOME", ".cache/huggingface"), "runs", run_name)
        save_dense_result(result, output_dir)
        logger.info("Dense result saved to %s", output_dir)
        return
    if args[:1] == ["residual-alignment"]:
        # Passed by `Path`, so letify copies the alignment the call writes there back to this machine.
        directory = Path(args[args.index("--output") + 1]).parent.absolute()
        directory.mkdir(parents=True, exist_ok=True)
        alignment_args = [*args, "--output-dir", directory]
        retry_lost_sessions(
            lambda: run(package_archive(), "polyplex_research.utils.assemble", "main", alignment_args, remote_secrets())
        )
        return
    if args[:1] == ["dense-eval"]:
        environment = {**remote_secrets(), "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0"}
        run_name = args[args.index("--run-name") + 1]
        remote = os.environ.get("POLYPLEX_DEVICE", "local").partition(".")[0] != "local"
        eval_args = stash_input_files(args[1:], run_name, DENSE_EVAL_INPUT_FILES) if remote else args[1:]
        documents = Path(os.environ.get("HF_HOME", ".cache/huggingface"), "documents").absolute()
        documents.mkdir(parents=True, exist_ok=True)
        eval_args = eval_args + ["--document-cache", documents]
        if "--release-dir" in eval_args:
            # Passed by `Path`, so letify copies the saved release back to this machine.
            index = eval_args.index("--release-dir") + 1
            eval_args[index] = Path(eval_args[index]).absolute()
            eval_args[index].mkdir(parents=True, exist_ok=True)
        archive = package_archive()
        result = retry_lost_sessions(lambda: run(archive, module, "dense_eval_main", eval_args, environment))
        output = Path(os.environ.get("HF_HOME", ".cache/huggingface"), "runs", f"{run_name}-eval.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("Dense evaluation saved to %s", output)
        return
    retry_lost_sessions(lambda: run(package_archive(), module, "main", args, remote_secrets()))


def run_record(phase_args: PhaseArguments, train_size: int | None, eval_size: int | None, text_size: int | None) -> dict:
    r"""
    Collects what a W&B run needs beyond the trainer's own arguments.

    Args:
        phase_args (`PhaseArguments`):
            The phase options, recorded in full under `phase/`.
        train_size (`int`, *optional*):
            Examples in the training set.
        eval_size (`int`, *optional*):
            Examples in the evaluation set.
        text_size (`int`, *optional*):
            Examples in the silent text set scored next to every batch.

    Returns:
        `dict`: The phase options, the data sizes that are known and the commit and branch of the checkout.
    """
    record = {f"phase/{key}": value for key, value in asdict(phase_args).items()}
    for key, size in (("train", train_size), ("eval", eval_size), ("text", text_size)):
        if size is not None:
            record[f"record/{key}_data_size"] = size
    record.update(git_record(os.path.dirname(os.path.abspath(__file__))))
    return record


def main(args: list[str] | None = None):
    r"""
    Parses the command line and runs the selected phase.

    Args:
        args (`list[str]`, *optional*):
            Command line arguments, `sys.argv[1:]` when omitted.

    Raises:
        ValueError: If `phase` is neither 1 nor 2, or if no dataset was given.
    """
    from .assemble import load_base_config, load_base_tokenizer

    parser = HfArgumentParser((PhaseArguments, TrainingArguments))
    phase_args, training_args = parser.parse_args_into_dataclasses(args=args)
    # Sets this worker's GPU as the current one before the processor and the model are placed on "cuda".
    _ = training_args.device

    if phase_args.phase not in (1, 2):
        raise ValueError(f"`phase` must be 1 or 2, got {phase_args.phase}.")
    if phase_args.dataset_path is None:
        raise ValueError(
            "Pass `--dataset_path`, the directory or Hub repository a phase dataset was published to."
        )

    tokenizer = load_base_tokenizer(
        phase_args.tokenizer_name_or_path or phase_args.model_name_or_path
    )

    if phase_args.phase == 1:
        if phase_args.dense:
            from ..datasets.phase_data import DenseTextDataset
            from ..models.polyplex import PolyplexProcessor

            processor = PolyplexProcessor.from_pretrained(phase_args.model_name_or_path)
            processor.audio_tokenizer.to("cuda" if torch.cuda.is_available() else "cpu")
            read = DenseTextDataset
            collator = None
        elif phase_args.duplex:
            from ..datasets.duplex_data import DuplexUtteranceDataset
            from ..models.polyplex import PolyplexProcessor

            processor = PolyplexProcessor.from_pretrained(phase_args.model_name_or_path)
            processor.audio_tokenizer.to("cuda" if torch.cuda.is_available() else "cpu")
            read = DuplexUtteranceDataset
            collator = None
        else:
            processor = None
            read = Phase1Dataset.from_pretrained
            collator = phase1_collate_fn(tokenizer, max_length=phase_args.text_max_length)

        trainer = Phase1Trainer(
            args=training_args,
            train_dataset=read(phase_args.dataset_path),
            eval_dataset=read(phase_args.eval_dataset_path) if phase_args.eval_dataset_path else None,
            data_collator=collator,
            duplex=phase_args.duplex,
            dense=phase_args.dense,
            processor=processor,
            score_words_only=phase_args.score_words_only,
            drop_audio=phase_args.drop_audio,
            max_frames=phase_args.max_frames if phase_args.duplex else None,
            processing_class=tokenizer,
            model_name_or_path=phase_args.model_name_or_path,
            adapter_path=phase_args.adapter_path,
            quantization=phase_args.quantization,
            torch_dtype=phase_args.torch_dtype,
            shared_token_residual=phase_args.shared_token_residual,
            residual_all_frozen_rows=phase_args.residual_all_frozen_rows,
            project_new_rows=phase_args.project_new_rows,
            base_row_mixture=phase_args.base_row_mixture,
            tie_new_input_rows=phase_args.tie_new_input_rows,
            lora_layers=phase_args.lora_layers,
            lora_rank=phase_args.lora_rank,
            confined_basis=phase_args.confined_basis,
            confined_moments=phase_args.confined_moments,
            confined_attention_basis=phase_args.confined_attention_basis,
            session_expert_layers=phase_args.session_expert_layers,
            session_expert_rank=phase_args.session_expert_rank,
            leak_lambda=phase_args.leak_lambda,
            timing_kl_lambda=phase_args.timing_kl_lambda,
            timing_reference=phase_args.timing_reference,
            english_guard_path=phase_args.english_guard_path,
            english_guard_lambda=phase_args.english_guard_lambda,
            head_hidden_size=phase_args.head_hidden_size,
            english_lambda=phase_args.english_lambda,
            english_states_path=phase_args.english_states_path,
            english_frames=phase_args.english_frames,
            timing_lambda=phase_args.timing_lambda,
            train_depth_text_embeddings=phase_args.train_depth_text_embeddings,
            share_row_mixture=phase_args.share_row_mixture,
            train_rows=phase_args.train_rows,
            train_corrections=phase_args.train_corrections,
            partition_direction=phase_args.partition_direction,
            audio_labels=phase_args.audio_labels,
            text_dataset_path=phase_args.text_dataset_path,
            text_weight=phase_args.text_weight,
            text_batch_size=phase_args.text_batch_size,
            persona=phase_args.persona,
        )
    else:
        # The reserved acoustic row the collator pads with is the one the checkpoint reserves, so the
        # configuration is read here rather than assumed.
        base_config = load_base_config(phase_args.model_name_or_path)
        trainer = Phase2Trainer(
            args=training_args,
            train_dataset=Phase2Dataset.from_pretrained(phase_args.dataset_path),
            eval_dataset=(
                Phase2Dataset.from_pretrained(phase_args.eval_dataset_path)
                if phase_args.eval_dataset_path
                else None
            ),
            data_collator=phase2_collate_fn(
                tokenizer,
                audio_vocab_size=base_config.audio_vocab_size,
                text_pad_token_id=phase_args.text_pad_token_id,
                max_frames=phase_args.max_frames,
                text_alignment=phase_args.text_alignment,
            ),
            processing_class=tokenizer,
            model_name_or_path=phase_args.model_name_or_path,
            quantization=phase_args.quantization,
            torch_dtype=phase_args.torch_dtype,
            adapter_kind=phase_args.adapter_kind,
            gated_adapter=phase_args.gated_adapter,
            tie_self_input=phase_args.tie_self_input,
            ortho_lambda=phase_args.ortho_lambda,
            nonneg_beta=phase_args.nonneg_beta,
        )

    if phase_args.early_stopping_patience:
        if phase_args.eval_dataset_path is None:
            raise ValueError("`--early_stopping_patience` needs `--eval_dataset_path` to watch.")
        training_args.metric_for_best_model = "eval_loss"
        training_args.greater_is_better = False
        # The callback counts a failure to improve against `state.best_metric`, which only the best
        # model bookkeeping fills in. Without it the metric stays unset, every evaluation looks like an
        # improvement, and the run never stops. That bookkeeping needs checkpoints to point at, so the
        # save strategy is brought in line with the evaluation one.
        training_args.load_best_model_at_end = True
        if training_args.save_strategy != training_args.eval_strategy:
            training_args.save_strategy = training_args.eval_strategy
            training_args.save_steps = training_args.eval_steps
        training_args.save_total_limit = min(training_args.save_total_limit or 2, 2)
        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=phase_args.early_stopping_patience,
                early_stopping_threshold=phase_args.early_stopping_threshold,
            )
        )

    trainer.add_callback(ResearchTreeCallback())
    text_loader = getattr(trainer, "_text_loader", None)
    trainer.add_callback(
        RunRecordCallback(
            run_record(
                phase_args,
                len(trainer.train_dataset) if trainer.train_dataset is not None else None,
                len(trainer.eval_dataset) if trainer.eval_dataset is not None else None,
                len(text_loader.dataset) if text_loader is not None else None,
            )
        )
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    # Both phases move a small slice of the checkpoint, so only that slice is written out.
    trainer.model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    launch()


__all__ = [
    "dense_mixture",
    "archive_runner",
    "collect_english_states",
    "dense_documents",
    "dense_eval_rows",
    "dense_eval_main",
    "dense_main",
    "dense_rows",
    "dense_evaluate",
    "dense_trainable_parameters",
    "evaluate_dense_checkpoint",
    "init_dense_wandb_run",
    "language_model_view",
    "run_dense_adaptation",
    "segment_cross_entropy",
    "train_dense",
    "load_english_states",
    "write_english_states",
    "Phase1Trainer",
    "Phase2Trainer",
    "PhaseArguments",
    "launch",
    "locked_extras",
    "main",
    "package_archive",
    "remote_secrets",
    "resolve_device",
    "resume_on_retry",
    "retry_lost_sessions",
    "save_dense_result",
    "stored_checkpoint_exists",
]
