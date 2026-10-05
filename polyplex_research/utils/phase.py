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


"""Text tables and coefficients a dense run trains row by row, so AdamW steps them (SPEC S7.2)."""

"""The two vocabulary sized tables, stepped by FactoredAdam (SPEC S7.2a)."""


"""Learning rate schedules of the dense stage: [`wsd_factor`] and [`cosine_restart_factor`]."""


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
    "collect_english_states",
    "load_english_states",
    "write_english_states",
    "Phase1Trainer",
    "Phase2Trainer",
    "PhaseArguments",
    "main",
]
