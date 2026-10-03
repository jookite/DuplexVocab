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
"""Assembly of the model each training phase trains."""

from __future__ import annotations

import json
import logging
import os
import shutil
import unicodedata
from typing import Callable

import torch
from torch import nn
from tokenizers import pre_tokenizers
from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TorchAoConfig

from ..adapters.phase_1_efficient_adapter import EfficientAdapterP1Config, EfficientAdapterP1Model
from ..adapters.phase_2_code_rotation_adapter import CodeRotationAdapterConfig, CodeRotationAdapterModel
from huggingface_hub.errors import HFValidationError

from ..models.polyplex import PolyplexConfig, PolyplexForConditionalGeneration, PolyplexProcessor, PolyplexTokenizer
from ..models.polyplex.tokenization_polyplex import WORD_START
from .quantization import quantize_linears_nvfp4


logger = logging.getLogger(__name__)

BASE_VOCAB_SIZE_KEY = "base_vocab_size"


# ---------------------------------------------------------------------------
# Token list I/O
# ---------------------------------------------------------------------------


def load_base_tokenizer(model_name_or_path: str) -> PreTrainedTokenizerBase:
    r"""
    Loads the text tokenizer of a duplex checkpoint.

    A checkpoint this project wrote carries a [`PolyplexProcessor`], so it loads directly. The released
    PersonaPlex repository instead ships a raw SentencePiece model, which the processor converts. Either way the
    tokenizer that comes back is a [`PolyplexTokenizer`].

    Args:
        model_name_or_path (`str`):
            Checkpoint path or hub id.

    Returns:
        `PolyplexTokenizer`: The tokenizer of the text stream.
    """
    try:
        return PolyplexProcessor.from_pretrained(model_name_or_path).tokenizer
    except (OSError, ValueError, HFValidationError):
        # A directory written before the processor was introduced holds the tokenizer files alone.
        return PolyplexTokenizer.from_pretrained(model_name_or_path)


def load_base_config(model_name_or_path: str) -> PolyplexConfig:
    r"""
    Reads the configuration of a duplex checkpoint.

    The collators pad with the rows a checkpoint reserves rather than with a number written into the
    code, so the sizes they need are read off the checkpoint itself.

    Args:
        model_name_or_path (`str`):
            Checkpoint path or hub id.

    Returns:
        `PolyplexConfig`: The configuration of the checkpoint.
    """
    return PolyplexConfig.from_pretrained(model_name_or_path)


def _tokenize_with_base(
    token: str,
    base_tokenizer: PreTrainedTokenizerBase,
) -> list[int]:
    r"""
    Decomposes one surface-form token through the base tokenizer.

    The base tokenizer treats the token as free text and breaks it into its own
    sub-word vocabulary. The resulting ids are used to look up the existing
    embedding rows that carry the closest semantics.

    Args:
        token (`str`):
            A single custom token to decompose.
        base_tokenizer (`PreTrainedTokenizerBase`):
            The original tokenizer before vocabulary expansion.

    Returns:
        `list[int]`: Sub-word ids from the base vocabulary.
    """
    ids = base_tokenizer.encode(token, add_special_tokens=False)
    return ids if ids else [base_tokenizer.eos_token_id or 0]


def focus_embeddings(
    tokens: list[str],
    base_tokenizer: PreTrainedTokenizerBase,
    embedding_weight: torch.Tensor,
    method: str = "mean",
) -> torch.Tensor:
    r"""
    Initializes embedding vectors for new tokens via FOCUS decomposition.

    Each new token is encoded through the base tokenizer into one or more
    existing sub-word ids. The embedding of the new token is then set to a
    summary of those sub-word embeddings. Two summary modes are supported:

    - ``"mean"``: arithmetic mean of the sub-word embeddings. This is the
      standard FOCUS initialization that places the new token at the centroid
      of its sub-word constituents.
    - ``"geometric"``: element-wise geometric mean of the absolute values,
      with signs restored from the arithmetic mean. Geometric averaging
      compresses the magnitude range across sub-words and can help when the
      norms of sub-word embeddings differ by a wide margin.

    Args:
        tokens (`list[str]`):
            New tokens to initialize, in the order they will be appended.
        base_tokenizer (`PreTrainedTokenizerBase`):
            Tokenizer representing the original vocabulary.
        embedding_weight (`torch.Tensor` of shape `(base_vocab_size, hidden_size)`):
            Embedding matrix of the pretrained model before expansion.
        method (`str`, *optional*, defaults to `"mean"`):
            Aggregation method, either `"mean"` or `"geometric"`.

    Returns:
        `torch.Tensor` of shape `(len(tokens), hidden_size)`: The initialized
        embedding rows for the new tokens.

    Raises:
        ValueError: If `method` is not `"mean"` or `"geometric"`.
    """
    if method not in ("mean", "geometric"):
        raise ValueError(f"`method` must be 'mean' or 'geometric', got '{method}'.")

    hidden_size = embedding_weight.shape[1]
    new_embeddings = torch.zeros(len(tokens), hidden_size, dtype=torch.float32)

    for idx, token in enumerate(tokens):
        sub_ids = _tokenize_with_base(token, base_tokenizer)
        sub_embeds = embedding_weight[sub_ids].float()

        if method == "mean":
            new_embeddings[idx] = sub_embeds.mean(dim=0)
        else:
            signs = sub_embeds.mean(dim=0).sign()
            log_abs = sub_embeds.abs().clamp(min=1e-12).log()
            geo = log_abs.mean(dim=0).exp()
            new_embeddings[idx] = signs * geo

    return new_embeddings


# ---------------------------------------------------------------------------
# Checkpoint output
# ---------------------------------------------------------------------------


def save_merged(model: PreTrainedModel, source: str, output_dir: str):
    r"""
    Writes a folded checkpoint next to the tokenizer of the checkpoint it was folded into.

    The text stream is unchanged by a fold, so the tokenizer travels along unread. A source that ships
    none, which a bare `save_pretrained` of the model class produces, leaves the weights on their own.

    Args:
        model (`PreTrainedModel`):
            The merged model.
        source (`str`):
            Checkpoint the tokenizer is taken from.
        output_dir (`str`):
            Destination directory.
    """
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    try:
        load_base_tokenizer(source).save_pretrained(output_dir)
    except Exception:
        logger.warning("No tokenizer found next to '%s', writing the weights alone.", source)
    model.save_pretrained(output_dir)
    logger.info("Saved merged checkpoint to %s.", output_dir)


# ---------------------------------------------------------------------------
# Phase assembly
# ---------------------------------------------------------------------------


# Tables the Phase 2 residual is bound to. The rotation kind rewrites these weights through
# `parametrize`, and `matmul` against a quantized tensor dequantizes: storage would still shrink, but the
# packed kernel is lost and a full precision copy is materialized every forward. Excluding them keeps the
# fast path where it pays, on the backbone, which is where nearly all of the 7B sits anyway.
ADAPTER_BOUND_MODULES = ["lm_head", "lm_heads", "embed_tokens", "text_embed_tokens"]

QUANTIZATION_KINDS = {
    "int8": Int8WeightOnlyConfig,
    "int4": Int4WeightOnlyConfig,
}


def frozen_backbone_quantization(quantization: str | None):
    r"""
    Builds the quantization configuration the frozen backbone is loaded under.

    Both phases freeze every pretrained weight, so the backbone is only ever read. That is what makes
    weight-only quantization safe here: no optimizer ever has to step a packed tensor, and the residual
    that does move is kept in full precision next to it.

    Args:
        quantization (`str` or `None`):
            One of `"none"`, `"nvfp4"`, `"int8"` and `"int4"`. `None` is read as `"none"`.

    Returns:
        `TorchAoConfig` or `None`: The configuration to pass to `from_pretrained`, `None` when the
        backbone is loaded unquantized.

    Raises:
        ValueError: If `quantization` names no known kind.
    """
    if quantization in (None, "none", "nvfp4"):
        # NVFP4 replaces the backbone's linears after loading, see [`quantize_frozen_backbone`].
        return None
    if quantization not in QUANTIZATION_KINDS:
        raise ValueError(
            f"`quantization` must be one of {('none', 'nvfp4', *QUANTIZATION_KINDS)}, got {quantization!r}."
        )

    quant_type = QUANTIZATION_KINDS[quantization]()
    return TorchAoConfig(quant_type=quant_type, modules_to_not_convert=list(ADAPTER_BOUND_MODULES))


def quantize_frozen_backbone(model, quantization: str | None, full_precision_layers: tuple = ()):
    r"""
    Runs the temporal backbone's linear projections in NVFP4 when `quantization` is `"nvfp4"` (SPEC S7.1).

    Both operands of every projection are quantized, on Hadamard rotated groups, so the matmul runs on the
    FP4 tensor cores. Each weight is quantized once and its full precision copy is dropped
    ([`~quantization.FrozenNVFP4Linear`]); a gradient still reaches the input of every layer, which is what a
    correction inside the backbone trains through. The tables, the head and the depth decoder are left as they are.

    Args:
        model (`PolyplexForConditionalGeneration`):
            The loaded model, modified in place.
        quantization (`str` or `None`):
            The kind the model was assembled with. Anything but `"nvfp4"` leaves the model alone.
        full_precision_layers (`tuple[int]`, *optional*):
            Layers left in full precision, their attention and both FFNs together.

    Appended layers (SPEC S11.4a) train: their projections run in NVFP4 and keep their weights trainable
    ([`~quantization.HadamardNVFP4Linear`]).
    """
    if quantization != "nvfp4":
        return
    kept, appended = set(full_precision_layers), set(model.config.appended_layers)
    for index, layer in enumerate(model.model.layers):
        if index not in kept:
            quantize_linears_nvfp4(layer, frozen=index not in appended)


RUN_OPTIONS = (
    "base_row_mixture",
    "tie_new_input_rows",
    "partition_direction",
    "confined_basis",
    "confined_attention_basis",
    "session_expert_layers",
    "session_expert_rank",
)
"""Fields of a run's `config.json` that change how its Phase 1 model is assembled."""


def run_options(run_dir: str) -> dict:
    r"""
    Args:
        run_dir (`str`):
            Directory of a Phase 1 run or checkpoint, holding the adapter's `config.json`.

    Returns:
        `dict`: The [`RUN_OPTIONS`] the run was trained with, to be passed to [`Phase1Assemble`].
    """

    with open(os.path.join(run_dir, "config.json")) as handle:
        config = json.load(handle)
    return {key: config[key] for key in RUN_OPTIONS if key in config}


class Phase1Assemble:
    r"""
    Assembles the model Phase 1 trains: an expanded checkpoint whose new text rows are the only movers.

    Phase 1 declares no new parameter. The vocabulary surgery has already appended rows to the text
    tables, and this class loads that expanded checkpoint and wraps it in an
    [`EfficientAdapterP1Model`], which freezes everything and protects the pretrained rows with a
    gradient hook. [`~assemble_v1.expand_checkpoint`] produces the checkpoint this reads.

    Args:
        model_name_or_path (`str`):
            Checkpoint of the *expanded* duplex model.
        base_vocab_size (`int`, *optional*):
            Vocabulary size before the expansion. Read from the checkpoint config when omitted, where
            [`~assemble_v1.expand_vocabulary`] recorded it.
        torch_dtype (`str`, *optional*, defaults to `"bfloat16"`):
            Dtype the frozen weights are loaded in.
        quantization (`str`, *optional*):
            Quantization of the frozen backbone, one of `"nvfp4"`, `"int8"` and `"int4"`. `"nvfp4"` runs
            the temporal backbone's projections in NVFP4, weights and activations both, see
            [`quantize_frozen_backbone`]; the other two are weight-only. The tables the adapter binds to
            are excluded.
        train_input_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the new rows of the backbone text input embedding are trained.
        train_lm_head (`bool`, *optional*, defaults to `True`):
            Whether the new rows of the text output head are trained.
        train_depth_text_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the new rows of the depth decoder's text embedding are trained.
        share_row_mixture (`bool`, *optional*, defaults to `False`):
            Whether every trained table builds its appended rows from one set of coefficients, see
            [`EfficientAdapterP1Config`].
        train_rows (`bool`, *optional*, defaults to `True`):
            Whether the appended rows train, or stay as loaded while only the backbone corrections train,
            see [`EfficientAdapterP1Config`].
        train_corrections (`bool`, *optional*, defaults to `True`):
            Whether the backbone corrections train, or stay as loaded while only the rows train, see
            [`EfficientAdapterP1Config`].
        partition_direction (`bool`, *optional*, defaults to `False`):
            Whether one free vector shared by every appended row of the head is added to their mixture,
            see [`EfficientAdapterP1Config`].
    """

    def __init__(
        self,
        model_name_or_path: str,
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
        confined_moments: str | None = None,
        confined_attention_basis: str | None = None,
        session_expert_layers: str | None = None,
        session_expert_rank: int = 256,
        head_hidden_size: int | None = None,
        tie_new_input_rows: bool = True,
        tied_backbone_grad: bool = False,
        tied_gradient_gain: float = 1.44,
        share_row_mixture: bool = False,
        partition_direction: bool = False,
        train_rows: bool = True,
        train_corrections: bool = True,
    ):
        self.model_name_or_path = model_name_or_path
        self.share_row_mixture = share_row_mixture
        self.train_rows = train_rows
        self.train_corrections = train_corrections
        self.partition_direction = partition_direction
        self.base_vocab_size = base_vocab_size
        self.torch_dtype = torch_dtype
        self.quantization = quantization
        self.train_input_embeddings = train_input_embeddings
        self.train_lm_head = train_lm_head
        self.train_depth_text_embeddings = train_depth_text_embeddings
        self.shared_token_residual = shared_token_residual
        self.residual_all_frozen_rows = residual_all_frozen_rows
        self.project_new_rows = project_new_rows
        self.group_gated_softmax = group_gated_softmax
        self.base_row_mixture = base_row_mixture
        self.lora_layers = lora_layers
        self.lora_rank = lora_rank
        self.confined_basis = confined_basis
        self.confined_moments = confined_moments
        self.confined_attention_basis = confined_attention_basis
        self.session_expert_layers = session_expert_layers
        self.session_expert_rank = session_expert_rank
        self.head_hidden_size = head_hidden_size
        self.tie_new_input_rows = tie_new_input_rows
        self.tied_backbone_grad = tied_backbone_grad
        self.tied_gradient_gain = tied_gradient_gain

    @classmethod
    def from_run(cls, model_name_or_path: str, run_dir: str) -> EfficientAdapterP1Model:
        r"""
        Rebuilds a trained Phase 1 model from its run directory, for evaluation.

        Args:
            model_name_or_path (`str`):
                Expanded checkpoint the run was trained on.
            run_dir (`str`):
                Directory of the run or one of its checkpoints.

        Returns:
            `EfficientAdapterP1Model`: The model with the run's trained slices loaded, in evaluation mode.
        """
        from .phase import load_adapter_slices

        model, _ = cls(model_name_or_path, **run_options(run_dir)).assemble()
        load_adapter_slices(model, run_dir)
        return model.eval()

    def assemble(self) -> tuple[EfficientAdapterP1Model, Callable[[], torch.Tensor]]:
        r"""
        Loads the expanded checkpoint and returns it with only its new text rows trainable.

        Returns:
            `tuple`: The assembled [`EfficientAdapterP1Model`] and a callable returning its penalty,
            which is zero. The penalty is returned anyway so that both phases hand the training loop the
            same pair.

        Raises:
            ValueError: If `base_vocab_size` was neither given nor recorded on the checkpoint.
        """
        model = PolyplexForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            dtype=getattr(torch, self.torch_dtype),
            quantization_config=frozen_backbone_quantization(self.quantization),
        )
        quantize_frozen_backbone(model, self.quantization)

        base_vocab_size = self.base_vocab_size
        if base_vocab_size is None:
            base_vocab_size = getattr(model.config, BASE_VOCAB_SIZE_KEY, None)
        if base_vocab_size is None:
            raise ValueError(
                f"'{self.model_name_or_path}' records no `{BASE_VOCAB_SIZE_KEY}`, so the boundary between "
                "the pretrained rows and the appended ones is unknown. Pass `base_vocab_size` explicitly, "
                "or produce the checkpoint with `assemble_v1.expand_checkpoint`."
            )

        config = EfficientAdapterP1Config.from_base_model(
            model,
            base_vocab_size=base_vocab_size,
            train_input_embeddings=self.train_input_embeddings,
            train_lm_head=self.train_lm_head,
            train_depth_text_embeddings=self.train_depth_text_embeddings,
            shared_token_residual=self.shared_token_residual,
            residual_all_frozen_rows=self.residual_all_frozen_rows,
            project_new_rows=self.project_new_rows,
            group_gated_softmax=self.group_gated_softmax,
            base_row_mixture=self.base_row_mixture,
            share_row_mixture=self.share_row_mixture,
            partition_direction=self.partition_direction,
            train_rows=self.train_rows,
            train_corrections=self.train_corrections,
            lora_layers=self.lora_layers,
            lora_rank=self.lora_rank,
            confined_basis=self.confined_basis,
            confined_moments=self.confined_moments,
            confined_attention_basis=self.confined_attention_basis,
            session_expert_layers=self.session_expert_layers,
            session_expert_rank=self.session_expert_rank,
            head_hidden_size=self.head_hidden_size,
            tie_new_input_rows=self.tie_new_input_rows,
            tied_backbone_grad=self.tied_backbone_grad,
            tied_gradient_gain=self.tied_gradient_gain,
        )
        assembled = EfficientAdapterP1Model(model, config)
        assembled.freeze_base()
        logger.info(
            "Phase 1: %d new rows over %d tables are trainable, rows below %d are protected.",
            config.num_new_rows,
            len(assembled.adapter_parameters()),
            base_vocab_size,
        )
        return assembled, assembled.penalty

    def dessemble(
        self, adapter_name_or_path: str, output_dir: str | None = None
    ) -> PolyplexForConditionalGeneration:
        r"""
        Folds a trained Phase 1 slice back into its checkpoint, undoing [`assemble`].

        A phase saves the tensors it moved, not the model it moved them in, so what a run leaves behind
        loads through the adapter and not through `from_pretrained`. This reads both halves back and
        writes the single checkpoint the next stage expects, which for Phase 1 is what Phase 2 starts
        from.

        Args:
            adapter_name_or_path (`str`):
                Directory or Hub repository the Phase 1 run wrote, holding the trained row slices.
            output_dir (`str`, *optional*):
                Directory the merged checkpoint and the expanded tokenizer are written to. The model is
                only returned when omitted.

        Returns:
            `PolyplexForConditionalGeneration`: The checkpoint carrying the trained rows.
        """
        # Loaded unquantized whatever the run trained under: folding writes weights back, and a packed
        # tensor has no room to hold them.
        model = PolyplexForConditionalGeneration.from_pretrained(
            self.model_name_or_path, dtype=getattr(torch, self.torch_dtype)
        )
        adapter = EfficientAdapterP1Model.from_pretrained(adapter_name_or_path, model=model)
        merged = adapter.fold_into_base()

        logger.info(
            "Phase 1 dessembled: '%s' folded into '%s'.", adapter_name_or_path, self.model_name_or_path
        )
        if output_dir is not None:
            save_merged(merged, self.model_name_or_path, output_dir)
        return merged


class Phase2Assemble:
    r"""
    Assembles the model Phase 2 trains: a frozen duplex model carrying an acoustic residual.

    Both adapter kinds are built on [`PolyplexForConditionalGeneration`], so the two arms of the ablation
    differ only in how the residual is attached and never in the model that carries the frozen English
    behaviour:

    - `"rotation"` registers the code space rotation of [`attach_residual_tying`] as a `parametrize`
      transform on the depth heads and on the audio embeddings that read a code back. This is the path
      the method rests on, and the only one [`dessemble`] can fold into a plain checkpoint.
    - `"tied"` wraps the base in an [`CodeRotationAdapterModel`], which attaches the temporal space
      residual through forward hooks. Ablation only.

    Either way every pretrained weight is frozen and only the residual is left trainable. The orthogonal
    penalty is returned as a callable rather than folded into the forward, because it constrains the
    parameter and not the prediction, so it belongs to the objective the training loop assembles.

    Args:
        model_name_or_path (`str`):
            Checkpoint of the frozen duplex model, normally the Phase 1 output.
        adapter_kind (`str`, *optional*, defaults to `"rotation"`):
            Either `"rotation"` or `"tied"`.
        torch_dtype (`str`, *optional*, defaults to `"bfloat16"`):
            Dtype the frozen weights are loaded in.
        quantization (`str`, *optional*):
            Quantization of the frozen backbone, one of `"nvfp4"`, `"int8"` and `"int4"`. `"nvfp4"` runs
            the temporal backbone's projections in NVFP4, weights and activations both, see
            [`quantize_frozen_backbone`]; the other two are weight-only. The tables the residual binds to
            are excluded.
        gated_adapter (`bool`, *optional*, defaults to `True`):
            Whether the tied residual uses the zero initialized gate times orthogonal matrix
            factorization. Ignored by the rotation kind.
        tie_self_input (`bool`, *optional*, defaults to `True`):
            Whether the residual also reaches the assistant stream tables, so that the model reads its
            own emitted codes under the convention its head now writes them in. Read by both kinds.
        ortho_lambda (`float`, *optional*, defaults to 0.1):
            Scale of the orthogonal penalty, recorded on the adapter configuration so that a stored
            adapter carries the setting it was trained under.
        nonneg_beta (`float`, *optional*, defaults to 10.0):
            Scale of the penalty on the negative entries of the rotation, which drives it to a
            permutation. Rotation kind only.

    Raises:
        ValueError: If `adapter_kind` is neither `"rotation"` nor `"tied"`.
    """

    def __init__(
        self,
        model_name_or_path: str,
        adapter_kind: str = "rotation",
        torch_dtype: str = "bfloat16",
        quantization: str | None = None,
        gated_adapter: bool = True,
        tie_self_input: bool = True,
        ortho_lambda: float = 0.1,
        nonneg_beta: float = 10.0,
    ):
        if adapter_kind not in ("rotation", "tied"):
            raise ValueError(f"`adapter_kind` must be 'tied' or 'rotation', got {adapter_kind}.")

        self.model_name_or_path = model_name_or_path
        self.adapter_kind = adapter_kind
        self.torch_dtype = torch_dtype
        self.quantization = quantization
        self.gated_adapter = gated_adapter
        self.tie_self_input = tie_self_input
        self.ortho_lambda = ortho_lambda
        self.nonneg_beta = nonneg_beta

    def assemble(self) -> tuple[CodeRotationAdapterModel, Callable[[], torch.Tensor]]:
        r"""
        Loads the checkpoint, attaches the residual and freezes everything else.

        Returns:
            `tuple`: The assembled [`CodeRotationAdapterModel`] and a callable returning the unscaled
            orthogonal penalty.
        """
        base = PolyplexForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            dtype=getattr(torch, self.torch_dtype),
            quantization_config=frozen_backbone_quantization(self.quantization),
        )
        quantize_frozen_backbone(base, self.quantization)

        config = CodeRotationAdapterConfig.from_base_model(
            base,
            adapter_kind=self.adapter_kind,
            gated=self.gated_adapter,
            tie_self_input=self.tie_self_input,
            ortho_lambda=self.ortho_lambda,
            nonneg_beta=self.nonneg_beta,
        )
        model = CodeRotationAdapterModel(base, config)
        # The residual is the only thing that moves and it carries an orthogonality constraint, so it is
        # kept in fp32 while the frozen backbone stays in whatever dtype, or packing, it was loaded with.
        model.adapter.float()
        model.freeze_base()
        logger.info(
            "Phase 2: '%s' residual with %d trainable tensors on a frozen base.",
            self.adapter_kind,
            len(model.adapter_parameters()),
        )
        return model, model.penalty

    def dessemble(
        self, adapter_name_or_path: str, output_dir: str | None = None
    ) -> PolyplexForConditionalGeneration:
        r"""
        Folds a trained Phase 2 residual back into its checkpoint, undoing [`assemble`].

        The rotation acts on the weights, so storing each rotated weight leaves a checkpoint that
        generates Korean through the stock `generate` of the base class, with no adapter attached and no
        code path of this repository involved. That checkpoint is the deliverable of the pipeline.

        Args:
            adapter_name_or_path (`str`):
                Directory or Hub repository the Phase 2 run wrote, holding the trained residual.
            output_dir (`str`, *optional*):
                Directory the merged checkpoint and the tokenizer are written to. The model is only
                returned when omitted.

        Returns:
            `PolyplexForConditionalGeneration`: The checkpoint carrying the rotation.

        Raises:
            ValueError: If the stored adapter is the tied kind, which has no weight space form.
        """
        # Loaded unquantized whatever the run trained under: folding writes weights back, and a packed
        # tensor has no room to hold them.
        base = PolyplexForConditionalGeneration.from_pretrained(
            self.model_name_or_path, dtype=getattr(torch, self.torch_dtype)
        )
        adapter = CodeRotationAdapterModel.from_pretrained(adapter_name_or_path, model=base)
        merged = adapter.fold_into_base()

        logger.info(
            "Phase 2 dessembled: '%s' folded into '%s'.", adapter_name_or_path, self.model_name_or_path
        )
        if output_dir is not None:
            save_merged(merged, self.model_name_or_path, output_dir)
        return merged


# ---------------------------------------------------------------------------
# DualFFN grafting
# ---------------------------------------------------------------------------


def dualffn_config(
    helium_config,
    language_config,
    language_ffn_layers: list[int] | None = None,
    vocab_size: int | None = None,
    composition_size: int | None = None,
    appended_layers: int = 0,
) -> PolyplexConfig:
    r"""
    Builds the configuration of a DualFFN model from the duplex checkpoint and the text LLM it is grafted with.

    Everything about the audio streams, the attention and the depth decoder is the duplex checkpoint's. The FFN
    width, the text vocabulary and the normalization epsilon of the language side are the text LLM's. Helium's
    gated MLP fuses gate and up into one projection of `ffn_dim`, so its own width is half of that.

    Args:
        helium_config (`PersonaPlexConfig`):
            Configuration of the duplex checkpoint.
        language_config (`Qwen3Config`):
            Configuration of the text LLM.
        language_ffn_layers (`list[int]`, *optional*):
            For each backbone layer, the text LLM layer whose FFN it receives. The first layers in order when
            omitted.
        vocab_size (`int`, *optional*):
            Rows of the LM head. The text LLM's vocabulary when omitted.
        composition_size (`int`, *optional*):
            Base tokens per text token in the shared composition over the duplex checkpoint's vocabulary (SPEC
            S12.3a). No composition when omitted.
        appended_layers (`int`, *optional*, defaults to 0):
            Copies of the duplex checkpoint's last layers appended after its own (SPEC S11.4a).

    Returns:
        [`PolyplexConfig`]: The configuration, with no language holding reserved neurons yet.

    Raises:
        ValueError: If the two models differ in width, if the text LLM's FFN is narrower than Helium's, or if
            the pairing names layers the text LLM does not have.
    """
    if helium_config.hidden_size != language_config.hidden_size:
        raise ValueError(
            f"The duplex backbone is {helium_config.hidden_size} wide and the text LLM "
            f"{language_config.hidden_size}; their FFN outputs cannot be summed."
        )
    pretrained_size = helium_config.ffn_dim // 2
    if language_config.intermediate_size < pretrained_size:
        raise ValueError(
            f"The text LLM's FFN ({language_config.intermediate_size}) is narrower than Helium's ({pretrained_size})."
        )
    if not 0 <= appended_layers <= helium_config.num_hidden_layers:
        raise ValueError(f"`appended_layers` must lie in [0, {helium_config.num_hidden_layers}], got {appended_layers}.")
    count = helium_config.num_hidden_layers + appended_layers
    layers = list(range(count)) if language_ffn_layers is None else list(language_ffn_layers)
    if len(layers) != count or not all(0 <= index < language_config.num_hidden_layers for index in layers):
        raise ValueError(
            f"`language_ffn_layers` must name {count} layers of the text LLM's "
            f"{language_config.num_hidden_layers}, got {layers}."
        )

    values = helium_config.to_dict()
    for key in ("model_type", "architectures", "transformers_version"):
        values.pop(key, None)
    # The depth decoder's text table follows the backbone vocabulary, which the config mirrors itself.
    values["depth_decoder_config"].pop("vocab_size", None)
    values.pop("layer_types", None)
    values.update(
        num_hidden_layers=count,
        appended_layers=list(range(helium_config.num_hidden_layers, count)),
        vocab_size=language_config.vocab_size if vocab_size is None else vocab_size,
        text_composition_size=composition_size,
        base_vocab_size=helium_config.vocab_size if composition_size else None,
        intermediate_size=language_config.intermediate_size,
        duplex_pretrained_size=pretrained_size,
        duplex_ffn_languages={},
        language_ffn_layers=layers,
        language_rms_norm_eps=language_config.rms_norm_eps,
    )
    return PolyplexConfig(**values)


@torch.no_grad()
def graft_helium_layer(ours, theirs, pretrained_size: int, appended: bool) -> None:
    r"""
    Copies one Helium layer into a DualFFN layer: the attention and its normalization, and the gated MLP as the first
    `pretrained_size` neurons of the duplex FFN with its normalization. The reserved neurons are zero.

    Args:
        ours ([`PolyplexDecoderLayer`]):
            The layer to fill.
        theirs:
            The Helium layer it takes.
        pretrained_size (`int`):
            Neurons of Helium's gated MLP, `duplex_pretrained_size`.
        appended (`bool`):
            Whether the layer is appended after Helium's own (SPEC S11.4a). Its output projections are then zero, so
            it starts as the identity.
    """
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        source = getattr(theirs.self_attn, name)
        getattr(ours.self_attn, name).weight.copy_(getattr(source, "linear", source).weight)
    ours.input_layernorm.weight.copy_(theirs.input_layernorm.weight)
    ours.mlp.duplex_norm.weight.copy_(theirs.post_attention_layernorm.weight)
    # Helium's `fc1` holds the SiLU gate in its first half and the linear branch in its second.
    fused = theirs.mlp.fc1.weight
    duplex = ours.mlp.duplex_mlp
    duplex.gate_proj.weight[:pretrained_size].copy_(fused[:pretrained_size])
    duplex.up_proj.weight[:pretrained_size].copy_(fused[pretrained_size:])
    duplex.down_proj.weight[:, :pretrained_size].copy_(theirs.mlp.fc2.weight)
    # The reserved neurons are zero: no output and no gradient until a language is given a slice of them.
    duplex.gate_proj.weight[pretrained_size:] = 0
    duplex.up_proj.weight[pretrained_size:] = 0
    duplex.down_proj.weight[:, pretrained_size:] = 0
    if appended:
        # Zero output projections make the copy the identity until it trains.
        ours.self_attn.o_proj.weight.zero_()
        duplex.down_proj.weight.zero_()


@torch.no_grad()
def graft_helium_rest(model: PolyplexForConditionalGeneration, helium) -> None:
    r"""
    Copies into a DualFFN model what it takes from the duplex checkpoint outside its layers: the final normalization
    the depth decoder reads, the reserved text id the opening frame uses, the audio embeddings and the depth decoder.
    The depth decoder's text table is copied only when the two vocabularies have the same size; otherwise it is built
    with [`depth_text_rows`] or [`graft_vocabulary`].

    Args:
        model ([`PolyplexForConditionalGeneration`]):
            The model to fill, built from [`dualffn_config`].
        helium (`PersonaPlexForConditionalGeneration`):
            The duplex checkpoint. Its layers are not read.

    Raises:
        ValueError: If the two depth decoders hold different weights.
    """
    model.model.norm.weight.copy_(helium.model.norm.weight)
    model.model.embed_tokens.weight[-1].copy_(helium.model.embed_tokens.weight[-1])
    for ours, theirs in zip(model.embed_tokens, helium.embed_tokens, strict=True):
        ours.weight.copy_(theirs.weight)

    text_table = "model.text_embed_tokens.weight"
    state = helium.depth_decoder.state_dict()
    if state[text_table].shape != model.depth_decoder.state_dict()[text_table].shape:
        del state[text_table]
    missing, unexpected = model.depth_decoder.load_state_dict(state, strict=False)
    if unexpected or set(missing) - {text_table}:
        raise ValueError(f"The depth decoders differ: missing {missing}, unexpected {unexpected}.")


@torch.no_grad()
def graft_helium(model: PolyplexForConditionalGeneration, helium) -> None:
    r"""
    Copies into a DualFFN model everything it takes from the duplex checkpoint: every layer
    ([`graft_helium_layer`]), an appended layer copying one of Helium's last layers in order (SPEC S11.4a), and the
    rest ([`graft_helium_rest`]).

    Args:
        model ([`PolyplexForConditionalGeneration`]):
            The model to fill, built from [`dualffn_config`].
        helium (`PersonaPlexForConditionalGeneration`):
            The duplex checkpoint.
    """
    appended = set(model.config.appended_layers)
    for index, ours in enumerate(model.model.layers):
        theirs = helium.model.layers[helium_source_layer(model.config, len(helium.model.layers), index)]
        graft_helium_layer(ours, theirs, model.config.duplex_pretrained_size, index in appended)
    graft_helium_rest(model, helium)


def helium_source_layer(config: PolyplexConfig, helium_layers: int, index: int) -> int:
    r"""
    Args:
        config ([`PolyplexConfig`]):
            The DualFFN configuration.
        helium_layers (`int`):
            Layers of the duplex checkpoint.
        index (`int`):
            A DualFFN layer.

    Returns:
        `int`: The Helium layer it is copied from; an appended layer copies one of Helium's last layers, in order
        (SPEC S11.4a).
    """
    return index if index < helium_layers else index - len(config.appended_layers)


@torch.no_grad()
def graft_language_layer(
    ours,
    theirs,
    appended: bool,
    alignment: tuple | None = None,
    removed_neurons: list | None = None,
    keep_appended_output: bool = False,
) -> None:
    r"""
    Copies the FFN of one text LLM layer, with the normalization before it, into a DualFFN layer's language FFN.

    Args:
        ours ([`PolyplexDecoderLayer`]):
            The layer to fill.
        theirs:
            The paired text LLM layer.
        appended (`bool`):
            Whether the layer is appended (SPEC S11.4a). Its language FFN's output projection is then zero.
        alignment (`tuple`, *optional*):
            `(rotation, scale, gain)` of the layer's residual basis alignment (SPEC S11.2a), folded in with
            [`fold_residual_alignment`]. `gain` may be `None`, which folds the text LLM's own normalization weight.
        removed_neurons (`list[int]`, *optional*):
            Intermediate neurons of the language FFN removed ([`remove_language_neurons`], SPEC S11.2b).
        keep_appended_output (`bool`, *optional*, defaults to `False`):
            Whether an appended layer keeps the text LLM's output projection, for a model whose language FFN output
            scale starts at zero (SPEC S11.4a, S11.2c).
    """
    ours.mlp.language_mlp.load_state_dict(theirs.mlp.state_dict())
    if removed_neurons:
        remove_language_neurons(ours.mlp.language_mlp, removed_neurons)
    ours.mlp.language_norm.weight.copy_(theirs.post_attention_layernorm.weight)
    if appended and not keep_appended_output:
        ours.mlp.language_mlp.down_proj.weight.zero_()
    if alignment is not None:
        rotation, scale, gain = alignment
        fold_residual_alignment(ours.mlp.language_mlp, ours.mlp.language_norm, rotation, scale, gain)


@torch.no_grad()
def remove_language_neurons(mlp, neurons: list) -> None:
    r"""
    Removes intermediate neurons of a language FFN (SPEC S11.2b): their gate and up rows and their down column become
    zero, so each adds exactly nothing to the output and the module keeps its shape.

    Args:
        mlp (`Qwen3MLP`):
            The language FFN, changed in place.
        neurons (`list[int]`):
            The neurons.
    """
    index = torch.tensor(neurons, device=mlp.gate_proj.weight.device)
    mlp.gate_proj.weight[index] = 0
    mlp.up_proj.weight[index] = 0
    mlp.down_proj.weight[:, index] = 0


def procrustes_rotation(cross: torch.Tensor) -> torch.Tensor:
    r"""
    The orthogonal matrix $R$ that minimizes $\sum_i \lVert R h_i - q_i \rVert^2$, from the cross covariance
    $\sum_i q_i h_i^\top = U \Sigma V^\top$: $R = U V^\top$ (orthogonal Procrustes).

    Args:
        cross (`torch.Tensor` of shape `(width, width)`):
            The cross covariance of the targets and the sources.

    Returns:
        `torch.Tensor` of shape `(width, width)`: The rotation, in float32.
    """
    left, _, right = torch.linalg.svd(cross.double())
    return (left @ right).float()


@torch.no_grad()
def fold_residual_alignment(mlp, norm, rotation: torch.Tensor, scale: float, gain: torch.Tensor | None = None) -> None:
    r"""
    Folds a residual basis alignment into a language FFN (SPEC S11.2a), so that it reads the Helium residual rotated
    into the text LLM's basis and writes its output rotated back: $W_\text{gate} \leftarrow W_\text{gate}\,
    \mathrm{diag}(\gamma)\,R$, $W_\text{up} \leftarrow W_\text{up}\,\mathrm{diag}(\gamma)\,R$, $W_\text{down} \leftarrow
    s R^\top W_\text{down}$, and the normalization weight becomes 1. RMSNorm keeps a vector's size under a rotation, so
    the layer computes $s R^\top \mathrm{FFN}(\gamma \odot R \hat h)$.

    Args:
        mlp (`Qwen3MLP`):
            The language FFN, in full precision. Changed in place.
        norm (`nn.Module`):
            The normalization before it. Its weight is set to 1.
        rotation (`torch.Tensor` of shape `(hidden_size, hidden_size)`):
            $R$, from the Helium basis to the text LLM's.
        scale (`float`):
            $s$, the size of the Helium residual over the text LLM's.
        gain (`torch.Tensor` of shape `(hidden_size,)`, *optional*):
            $\gamma$. The normalization's own weight when omitted.
    """
    gain = norm.weight if gain is None else gain
    folded = fold_language_weights(
        mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight, rotation, scale, gain
    )
    for projection, weight in zip((mlp.gate_proj, mlp.up_proj, mlp.down_proj), folded):
        projection.weight.copy_(weight)
    norm.weight.fill_(1.0)


@torch.no_grad()
def fold_language_weights(gate, up, down, rotation: torch.Tensor, scale: float, gain: torch.Tensor) -> tuple:
    r"""
    The language FFN weights with a residual basis alignment folded in (SPEC S11.2a), computed in float32 on the
    weights' device and returned in their dtype: `gate` and `up` times $\mathrm{diag}(\gamma) R$, and `down` as
    $s R^\top$ `down`.

    Args:
        gate (`torch.Tensor` of shape `(intermediate_size, hidden_size)`):
            The gate projection.
        up (`torch.Tensor` of shape `(intermediate_size, hidden_size)`):
            The up projection.
        down (`torch.Tensor` of shape `(hidden_size, intermediate_size)`):
            The down projection.
        rotation (`torch.Tensor` of shape `(hidden_size, hidden_size)`):
            $R$.
        scale (`float`):
            $s$.
        gain (`torch.Tensor` of shape `(hidden_size,)`):
            $\gamma$.

    Returns:
        `tuple[torch.Tensor, torch.Tensor, torch.Tensor]`: The folded gate, up and down projections.
    """
    device = gate.device
    rotation = rotation.float().to(device)
    reading = gain.float().to(device)[:, None] * rotation
    return (
        (gate.float() @ reading).to(gate.dtype),
        (up.float() @ reading).to(up.dtype),
        (scale * (rotation.T @ down.float())).to(down.dtype),
    )


def aligned_positions(text: str, first_tokenizer, second_tokenizer) -> tuple:
    r"""
    Cuts a text with two tokenizers and pairs the tokens that end at the same character, where both models have read
    exactly the same text.

    Args:
        text (`str`):
            The text.
        first_tokenizer (`PreTrainedTokenizerBase`):
            A fast tokenizer.
        second_tokenizer (`PreTrainedTokenizerBase`):
            Another fast tokenizer.

    Returns:
        `tuple`: The ids of each cut, then the index in each cut of every paired token.
    """
    first = first_tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    second = second_tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    second_ends = {end: index for index, (_, end) in enumerate(second["offset_mapping"])}
    pairs = [(index, second_ends[end]) for index, (_, end) in enumerate(first["offset_mapping"]) if end in second_ends]
    return (
        first["input_ids"],
        second["input_ids"],
        [index for index, _ in pairs],
        [index for _, index in pairs],
    )


@torch.no_grad()
def graft_language_rest(model: PolyplexForConditionalGeneration, language_model) -> None:
    r"""
    Copies the text LLM's final normalization, text embedding and LM head into a DualFFN model that keeps the text
    LLM's vocabulary. The reserved text id above the vocabulary is left as it is.

    Args:
        model ([`PolyplexForConditionalGeneration`]):
            The model to fill, built from [`dualffn_config`].
        language_model (`Qwen3ForCausalLM`):
            The text LLM. Its layers are not read.
    """
    model.model.text_norm.weight.copy_(language_model.model.norm.weight)
    rows = language_model.model.embed_tokens.weight.shape[0]
    model.model.embed_tokens.weight[:rows].copy_(language_model.model.embed_tokens.weight)
    model.lm_head.weight[:rows].copy_(language_model.lm_head.weight)


@torch.no_grad()
def graft_language_model(model: PolyplexForConditionalGeneration, language_model, vocabulary: bool = True) -> None:
    r"""
    Copies into a DualFFN model everything it takes from the text LLM: for each backbone layer, the FFN of the paired
    text LLM layer ([`graft_language_layer`]), and with `vocabulary` the rest ([`graft_language_rest`]).

    Args:
        model ([`PolyplexForConditionalGeneration`]):
            The model to fill, built from [`dualffn_config`].
        language_model (`Qwen3ForCausalLM`):
            The text LLM.
        vocabulary (`bool`, *optional*, defaults to `True`):
            Whether the text embedding, the LM head and the final normalization are copied too. Left out when the
            model takes the Polyplex vocabulary ([`graft_vocabulary`]).
    """
    appended = set(model.config.appended_layers)
    for layer, (ours, index) in enumerate(zip(model.model.layers, model.config.language_ffn_layers, strict=True)):
        graft_language_layer(ours, language_model.model.layers[index], layer in appended)
    if vocabulary:
        graft_language_rest(model, language_model)


def _materialize(module: nn.Module, device) -> None:
    module.to_empty(device=device)
    with torch.no_grad():
        for tensor in [*module.parameters(), *module.buffers()]:
            tensor.zero_()


def materialize_outside_layers(model: PolyplexForConditionalGeneration, device) -> None:
    r"""
    Allocates, as zeros on `device`, every part of a DualFFN model built on the meta device except its backbone layers,
    and rebuilds the rotary embedding, whose frequencies are computed rather than loaded.

    Args:
        model ([`PolyplexForConditionalGeneration`]):
            The model, built on the meta device.
        device (`str` or `torch.device`):
            Where the parts are allocated.
    """
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    layers = model.model.layers
    model.model.layers = nn.ModuleList()
    try:
        _materialize(model, device)
    finally:
        model.model.layers = layers
    model.model.rotary_emb = Qwen3RotaryEmbedding(config=model.config, device=device)


def save_assembled_dualffn(
    model: PolyplexForConditionalGeneration, output_dir: str, full_precision_layers=(), max_shard_size: str = "2GB"
) -> None:
    r"""
    Saves a DualFFN model assembled in NVFP4 as it is (SPEC S11.2): the frozen projections as their four bit codes and
    scales, the rest in its own dtype, and in the config which layers stayed in full precision.

    Args:
        model ([`PolyplexForConditionalGeneration`]):
            The assembled model.
        output_dir (`str`):
            Directory written to.
        full_precision_layers (`tuple[int]`, *optional*):
            Layers the assembly left in full precision.
        max_shard_size (`str`, *optional*, defaults to `"2GB"`):
            Largest file written.
    """
    model.config.nvfp4_full_precision_layers = sorted(int(index) for index in full_precision_layers)
    os.makedirs(output_dir, exist_ok=True)
    # Small shards keep each one's host copy well under the 28 GB the machine allows.
    model.save_pretrained(output_dir, max_shard_size=max_shard_size)


@torch.no_grad()
def fold_language_stream(
    model: PolyplexForConditionalGeneration, rotation: torch.Tensor, scale: float, original=None
) -> PolyplexForConditionalGeneration:
    r"""
    Moves a model with a language stream (SPEC S11.2d) to one rotation and scale shared by all layers, folded into the
    language FFN weights (SPEC S11.2h): each layer's language FFN becomes $W_\text{gate}\,\mathrm{diag}(\gamma_l) R$,
    $W_\text{up}\,\mathrm{diag}(\gamma_l) R$ and $s R^\top W_\text{down}$ ([`fold_language_weights`]), its
    `language_norm` goes to 1 and its per-layer rotation buffers are dropped, so the language stream runs in the
    backbone's basis. A frozen NVFP4 projection is folded from its original weight and quantized again; a trained one
    from its own weight.

    Args:
        model ([`PolyplexForConditionalGeneration`]):
            A model with an unfolded language stream. Changed in place.
        rotation (`torch.Tensor` of shape `(hidden_size, hidden_size)`):
            The shared $R$, from the backbone's basis to the text LLM's.
        scale (`float`):
            The shared $s$.
        original (`Callable`, *optional*):
            `original(layer, name)` returns the full precision weight of a frozen NVFP4 projection (`name` is `"gate"`,
            `"up"` or `"down"`), which holds no such weight of its own. That weight quantized again must give the
            projection's codes.

    Returns:
        [`PolyplexForConditionalGeneration`]: The same model.

    Raises:
        ValueError: If the model has no unfolded language stream, carries language FFN low rank corrections, or a frozen
            projection has no original weight or one that does not give its codes.
    """
    from .quantization import FrozenNVFP4Linear, HadamardNVFP4Linear, dequantize_nvfp4

    if not model.config.language_stream or model.config.language_stream_folded:
        raise ValueError("The model has no unfolded language stream.")
    for index, layer in enumerate(model.model.layers):
        ffn = layer.mlp
        mlp = ffn.language_mlp
        if hasattr(mlp, "gate_lora"):
            raise ValueError("Language FFN low rank corrections cannot be folded.")
        projections = [getattr(mlp, f"{name}_proj") for name in ("gate", "up", "down")]
        weights = []
        for name, projection in zip(("gate", "up", "down"), projections):
            if isinstance(projection, FrozenNVFP4Linear):
                if original is None:
                    raise ValueError(f"Layer {index} {name}_proj is frozen in NVFP4 and needs its original weight.")
                weight = original(index, name).to(projection.quantized_weight.device)
                check = FrozenNVFP4Linear.from_linear(_linear(weight)).quantized_weight
                if not torch.equal(
                    dequantize_nvfp4(check, torch.float32), dequantize_nvfp4(projection.quantized_weight, torch.float32)
                ):
                    raise ValueError(f"Layer {index} {name}_proj does not hold the original weight given for it.")
                weights.append(weight)
            else:
                weights.append(projection.weight)
        folded = fold_language_weights(*weights, rotation, scale, ffn.language_norm.weight)
        for name, projection, weight in zip(("gate", "up", "down"), projections, folded):
            if isinstance(projection, FrozenNVFP4Linear):
                frozen = FrozenNVFP4Linear.from_linear(_linear(weight), backward_dtype=projection.backward_dtype)
                setattr(mlp, f"{name}_proj", frozen)
            else:
                projection.weight.copy_(weight)
                if isinstance(projection, HadamardNVFP4Linear):
                    # The cached codes belong to the weight before folding.
                    projection._quantized_key = None
        ffn.language_norm.weight.fill_(1.0)
        del ffn.stream_rotation, ffn.stream_scale
        ffn.stream_folded = True
    model.config.language_stream_folded = True
    return model


def _linear(weight: torch.Tensor) -> nn.Linear:
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False, device="meta")
    linear.weight = nn.Parameter(weight, requires_grad=False)
    return linear


RELEASE_AUTO_MAP = {
    "AutoConfig": "configuration_polyplex.PolyplexConfig",
    "AutoModel": "modeling_polyplex.PolyplexForConditionalGeneration",
}
"""The remote code classes the released model loads with (SPEC S13.4c)."""


def save_dense_release(
    model: PolyplexForConditionalGeneration, output_dir: str, auto_map: dict | None = None, max_shard_size: str = "5GB"
) -> None:
    r"""
    Saves a dense checkpoint as the released model (SPEC S13.4c). Every trained NVFP4 projection is replaced by a frozen
    one holding the codes its forward computes, so the backbone is saved as four bit codes and scales alone, and the
    config marks every NVFP4 projection as frozen.

    Args:
        model ([`PolyplexForConditionalGeneration`]):
            The assembled model with the checkpoint's parameters loaded. Its trained projections are replaced in place.
        output_dir (`str`):
            Directory written to.
        auto_map (`dict`, *optional*):
            The `auto_map` the config carries, naming the remote code classes the release loads with.
        max_shard_size (`str`, *optional*, defaults to `"5GB"`):
            Largest file written.
    """
    from .quantization import FrozenNVFP4Linear, HadamardNVFP4Linear

    full_precision = []
    for index, layer in enumerate(model.model.layers):
        for parent in list(layer.modules()):
            for name, child in list(parent.named_children()):
                if isinstance(child, HadamardNVFP4Linear):
                    setattr(parent, name, FrozenNVFP4Linear.from_linear(child.requires_grad_(False), child.hadamard_group_size))
        if not any(isinstance(module, FrozenNVFP4Linear) for module in layer.modules()):
            full_precision.append(index)
    model.config.nvfp4_release = True
    if auto_map is not None:
        model.config.auto_map = dict(auto_map)
    save_assembled_dualffn(model, output_dir, full_precision_layers=full_precision, max_shard_size=max_shard_size)


def load_assembled_dualffn(path: str, device, backward_dtype: str = "bf16") -> PolyplexForConditionalGeneration:
    r"""
    Loads a DualFFN model [`save_assembled_dualffn`] saved, straight onto `device` (SPEC S11.2). The model is built on
    the meta device with every projection [`quantize_frozen_backbone`] would replace already in its NVFP4 form, so the
    frozen weights load as four bit codes and no full precision copy of them is made.

    Args:
        path (`str`):
            The saved directory.
        device (`str` or `torch.device`):
            Where the model is loaded.
        backward_dtype (`str`, *optional*, defaults to `"bf16"`):
            Backward precision of the NVFP4 projections (SPEC S7.1a).

    Returns:
        [`PolyplexForConditionalGeneration`]: The model.
    """
    from safetensors.torch import load_file
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    from .quantization import (
        NVFP4_BLOCK_SIZE,
        FrozenNVFP4Linear,
        HadamardNVFP4Linear,
        _hadamard_matrix,
        _random_hadamard,
    )

    config = PolyplexConfig.from_pretrained(path)
    kept = set(getattr(config, "nvfp4_full_precision_layers", None) or ())
    # A release holds every NVFP4 projection frozen ([`save_dense_release`]).
    appended = set() if getattr(config, "nvfp4_release", False) else set(config.appended_layers)
    with torch.device("meta"):
        model = PolyplexForConditionalGeneration._from_config(config, torch_dtype=torch.bfloat16)

    def replace(parent: nn.Module, trained: bool) -> None:
        for name, child in parent.named_children():
            if isinstance(child, nn.Linear) and not child.in_features % NVFP4_BLOCK_SIZE:
                if trained:
                    projection = HadamardNVFP4Linear.from_linear(child, backward_dtype=backward_dtype)
                else:
                    projection = FrozenNVFP4Linear.empty(
                        child.in_features, child.out_features, child.bias is not None, backward_dtype=backward_dtype
                    )
                setattr(parent, name, projection)
            else:
                replace(child, trained)

    if getattr(config, "nvfp4_full_precision_layers", None) is not None:
        for index, layer in enumerate(model.model.layers):
            if index not in kept:
                replace(layer, index in appended)

    index_file = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file) as handle:
            files = sorted(set(json.load(handle)["weight_map"].values()))
    else:
        files = ["model.safetensors"]
    # One projection's codes and scales can sit in different shards, and are read together.
    state = {}
    for file in files:
        state.update(load_file(os.path.join(path, file), device=str(device)))
    unexpected = model.load_state_dict(state, strict=False, assign=True).unexpected_keys
    del state
    if unexpected:
        raise ValueError(f"Unexpected keys in {path}: {unexpected[:5]}")

    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, HadamardNVFP4Linear):
                weight = module.weight
                module.forward_transform = _hadamard_matrix(module.hadamard_group_size, weight.dtype, weight.device)
                module.wgrad_transform = _random_hadamard(NVFP4_BLOCK_SIZE, weight.dtype, weight.device)
            elif isinstance(module, FrozenNVFP4Linear):
                module.forward_transform = module.forward_transform.to(device)
    model.model.rotary_emb = Qwen3RotaryEmbedding(config=config, device=device)
    left = [name for name, tensor in [*model.named_parameters(), *model.named_buffers()] if tensor.is_meta]
    if left:
        raise ValueError(f"{path} does not hold {left[:5]}")
    return model.eval()


def compile_backbone(model: PolyplexForConditionalGeneration) -> PolyplexForConditionalGeneration:
    r"""
    Compiles each backbone layer with `torch.compile` in place (AGENTS H31, SPEC S7.1), so the NVFP4 rotation, per
    token scaling and quantization fuse into a few kernels. The layers share their shapes, so one compiled graph serves
    them all; the tables, the head and the depth decoder run as they are.

    Args:
        model ([`PolyplexForConditionalGeneration`]):
            The model.

    Returns:
        [`PolyplexForConditionalGeneration`]: The same model.
    """
    for layer in model.model.layers:
        layer.compile()
    return model


@torch.no_grad()
def graft_layers(
    model: PolyplexForConditionalGeneration,
    helium,
    language_model,
    device,
    quantization: str | None = None,
    full_precision_layers: tuple = (),
    alignment=None,
    removed_neurons: dict | None = None,
    keep_appended_output: bool = False,
) -> None:
    r"""
    Builds a DualFFN model's backbone layers one at a time (SPEC S11.2): each layer is allocated on `device`, filled
    from Helium ([`graft_helium_layer`]) and the text LLM ([`graft_language_layer`]), quantized as SPEC S7.1 says
    ([`quantize_frozen_backbone`]), and the source layers no later layer reads are replaced by `nn.Identity`, so the
    full precision DualFFN is never held at once.

    Args:
        model ([`PolyplexForConditionalGeneration`]):
            The model, built on the meta device. Its layers are filled in place.
        helium (`PersonaPlexForConditionalGeneration`):
            The duplex checkpoint. Its layers are dropped.
        language_model (`Qwen3ForCausalLM`):
            The text LLM. The layers the pairing uses are dropped.
        device (`str` or `torch.device`):
            Where the layers are built.
        quantization (`str`, *optional*):
            `"nvfp4"` quantizes each layer as it is built.
        full_precision_layers (`tuple[int]`, *optional*):
            Layers left in full precision, see [`quantize_frozen_backbone`].
        alignment (`Callable[[int], tuple]`, *optional*):
            The residual basis alignment of each layer ([`load_residual_alignment`]), folded into its language FFN
            before it is quantized (SPEC S11.2a).
        removed_neurons (`dict[int, list[int]]`, *optional*):
            Layer mapped to the language FFN neurons removed before it is quantized (SPEC S11.2b).
        keep_appended_output (`bool`, *optional*, defaults to `False`):
            Whether appended layers keep the text LLM's language FFN output projection ([`graft_language_layer`]).
    """
    config = model.config
    appended, kept = set(config.appended_layers), set(full_precision_layers)
    helium_layers = len(helium.model.layers)
    sources = [helium_source_layer(config, helium_layers, index) for index in range(len(model.model.layers))]
    last_helium = {source: index for index, source in enumerate(sources)}
    last_language = {source: index for index, source in enumerate(config.language_ffn_layers)}
    for index, layer in enumerate(model.model.layers):
        _materialize(layer, device)
        graft_helium_layer(layer, helium.model.layers[sources[index]], config.duplex_pretrained_size, index in appended)
        language_source = config.language_ffn_layers[index]
        graft_language_layer(
            layer,
            language_model.model.layers[language_source],
            index in appended,
            None if alignment is None else alignment(index),
            (removed_neurons or {}).get(index),
            keep_appended_output,
        )
        if quantization == "nvfp4" and index not in kept:
            quantize_linears_nvfp4(layer, frozen=index not in appended)
        if last_helium[sources[index]] == index:
            helium.model.layers[sources[index]] = nn.Identity()
        if last_language[language_source] == index:
            language_model.model.layers[language_source] = nn.Identity()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def depth_text_rows(
    tokenizer: PreTrainedTokenizerBase,
    size: int,
    helium_tokenizer: PreTrainedTokenizerBase,
    helium_table: torch.Tensor,
) -> torch.Tensor:
    r"""
    Builds the depth decoder's text rows for a vocabulary it was not trained on.

    The depth decoder was trained on Helium's text ids. Each id of the new vocabulary is decoded to its surface
    text, cut into Helium's pieces, and given the mean of those pieces' rows ([`focus_embeddings`]).

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            Tokenizer of the new vocabulary.
        size (`int`):
            Number of ids to build, from 0.
        helium_tokenizer (`PreTrainedTokenizerBase`):
            Tokenizer the depth decoder was trained with.
        helium_table (`torch.Tensor` of shape `(helium_vocab_size, depth_hidden_size)`):
            The depth decoder's text table under Helium's vocabulary.

    Returns:
        `torch.Tensor` of shape `(size, depth_hidden_size)`: The rows, in float32.
    """
    surfaces = [tokenizer.decode([index]) for index in range(size)]
    return focus_embeddings(surfaces, helium_tokenizer, helium_table)


def qwen_piece_ids(tokenizer: PreTrainedTokenizerBase) -> dict:
    r"""
    Maps Qwen3's text tokens to their ids under the Polyplex spelling (SPEC S12.1b): the leading space written as
    `▁`, in NFC. Added tokens and byte fragments are left out.

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The Qwen3 tokenizer.

    Returns:
        `dict`: Piece mapped to its Qwen3 id, the lowest id when two tokens spell the same piece.
    """
    added = tokenizer.get_added_vocab()
    ids = {}
    for token, index in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]):
        if token in added:
            continue
        surface = tokenizer.convert_tokens_to_string([token])
        if not surface or "�" in surface:
            continue
        piece = WORD_START + surface[1:] if surface.startswith(" ") else surface
        ids.setdefault(unicodedata.normalize("NFC", piece), index)
    return ids


def fit_linear(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    r"""
    Args:
        inputs (`torch.Tensor` of shape `(rows, in_features)`):
            Rows the map reads.
        targets (`torch.Tensor` of shape `(rows, out_features)`):
            Rows it should give.

    Returns:
        `torch.Tensor` of shape `(in_features, out_features)`: The least squares map, in float32.
    """
    return torch.linalg.lstsq(inputs.double(), targets.double()).solution.float()


def piece_ids(tokenizer: PolyplexTokenizer, ids: list[int], kept: list[int]) -> list[list[int]]:
    r"""
    Cuts each token's string by the vocabulary restricted to `kept`, with the same scores (SPEC S12.2).

    Args:
        tokenizer ([`PolyplexTokenizer`]):
            The full tokenizer.
        ids (`list[int]`):
            Tokens to cut.
        kept (`list[int]`):
            Tokens the cut may use. Special tokens are always available.

    Returns:
        `list[list[int]]`: The pieces of each token, as ids of the full tokenizer.
    """
    vocab = json.loads(tokenizer.backend_tokenizer.to_str())["model"]["vocab"]
    kept = sorted(set(kept) | set(tokenizer.all_special_ids))
    restricted = PolyplexTokenizer(vocab=[vocab[index] for index in kept], unk_token=tokenizer.unk_token)
    # A token is cut as it is spelled; a word-start mark is part of the string when the token has one.
    restricted.backend_tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement=WORD_START, prepend_scheme="never", split=False
    )
    strings = tokenizer.convert_ids_to_tokens(list(ids))
    encodings = restricted.backend_tokenizer.encode_batch(strings, add_special_tokens=False)
    return [[kept[index] for index in encoding.ids] for encoding in encodings]


def piece_mean_rows(
    tokenizer: PolyplexTokenizer, ids: list[int], kept: list[int], table: torch.Tensor
) -> torch.Tensor:
    r"""
    Rows for tokens a table has none for (SPEC S12.2): each token takes the mean of the rows of its pieces
    ([`piece_ids`]).

    Args:
        tokenizer ([`PolyplexTokenizer`]):
            The full tokenizer.
        ids (`list[int]`):
            Tokens to build rows for.
        kept (`list[int]`):
            Tokens the cut may use, the ones `table` has rows for. Special tokens are always available.
        table (`torch.Tensor` of shape `(rows, hidden_size)`):
            Rows indexed by the full tokenizer's ids, read only at `kept`.

    Returns:
        `torch.Tensor` of shape `(len(ids), hidden_size)`: The rows, in float32.
    """
    return torch.stack([table[pieces].float().mean(0) for pieces in piece_ids(tokenizer, ids, kept)])


@torch.no_grad()
def omp_coefficients(
    targets: torch.Tensor, basis: torch.Tensor, k: int, batch_size: int = 256, ridge: float = 1e-10
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Writes each target as a sparse combination of basis rows by orthogonal matching pursuit (SPEC S12.3a): `k` times,
    the row best aligned with what is left is added, and the coefficients of the chosen rows are refitted by least
    squares.

    Args:
        targets (`torch.Tensor` of shape `(rows, hidden_size)`):
            Rows to write.
        basis (`torch.Tensor` of shape `(basis_rows, hidden_size)`):
            Rows to write them with.
        k (`int`):
            Basis rows per target.
        batch_size (`int`, *optional*, defaults to 256):
            Targets solved at once.
        ridge (`float`, *optional*, defaults to 1e-10):
            Ridge of the least squares refit, relative to the mean squared norm of the chosen rows. A large ridge
            keeps the coefficients from growing where chosen rows nearly cancel each other.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: The chosen basis rows, of shape `(rows, k)`, and their coefficients, of
        shape `(rows, k)` in float32, both on the CPU. The search runs on the basis's device.
    """
    device = basis.device
    directions = nn.functional.normalize(basis.float(), dim=-1)
    basis = basis.double()
    all_indices, all_values = [], []
    for start in range(0, len(targets), batch_size):
        target = targets[start : start + batch_size].to(device).double()
        residual = target.clone()
        chosen = torch.zeros(len(target), 0, dtype=torch.long, device=device)
        for _ in range(k):
            scores = (residual.float() @ directions.T).abs()
            scores.scatter_(1, chosen, -1.0)
            chosen = torch.cat([chosen, scores.argmax(-1, keepdim=True)], 1)
            rows = basis[chosen]
            gram = rows @ rows.transpose(1, 2)
            shrink = ridge * gram.diagonal(dim1=1, dim2=2).mean(-1)[:, None, None]
            gram = gram + shrink * torch.eye(chosen.shape[1], dtype=gram.dtype, device=device)
            values = torch.linalg.solve(gram, (rows @ target[..., None]))[..., 0]
            residual = target - (values[..., None] * rows).sum(1)
        all_indices.append(chosen.cpu())
        all_values.append(values.float().cpu())
    return torch.cat(all_indices), torch.cat(all_values)


@torch.no_grad()
def composition_reconstruction(
    donor_input: torch.Tensor,
    donor_head: torch.Tensor,
    input_table: torch.Tensor,
    head_table: torch.Tensor,
    held_out: torch.Tensor,
    k: int = 32,
    piece_rows: dict | None = None,
    ridge: float = 1e-10,
) -> dict:
    r"""
    Checks the premise of shared composition (SPEC S12.3a) on tokens two models share: coefficients found in a donor
    table, over the shared tokens that are not held out, are applied to the other model's input and head tables and
    compared with that model's own rows of the held-out tokens.

    Row `t` of every table is the same token. Two baselines are measured on the same rows: the least squares map from
    the head table to the input table (the linear part of S12.3), and the one from the donor head to the head table
    (S12.1a).

    Args:
        donor_input (`torch.Tensor` of shape `(rows, donor_hidden_size)`):
            The donor's input rows, Qwen3's.
        donor_head (`torch.Tensor` of shape `(rows, donor_hidden_size)`):
            The donor's head rows.
        input_table (`torch.Tensor` of shape `(rows, hidden_size)`):
            The input rows to reproduce, Helium's.
        head_table (`torch.Tensor` of shape `(rows, hidden_size)`):
            The head rows to reproduce.
        held_out (`torch.Tensor` of shape `(held_out_rows,)`):
            Rows measured. All other rows form the basis and fit the baselines.
        k (`int`, *optional*, defaults to 32):
            Basis rows per token.
        piece_rows (`dict`, *optional*):
            `"input"` and `"head"` mapped to the S12.2 piece mean of each held-out row, measured as a third baseline.
        ridge (`float`, *optional*, defaults to 1e-10):
            Ridge of the coefficient fit, see [`omp_coefficients`].

    Returns:
        `dict`: The median cosine with the true rows, keyed by method and table. For the compositions also the
        median and 99th percentile of each rebuilt row's norm over the true row's (`..., norm ratio median`,
        `..., norm ratio p99`).
    """
    basis = torch.ones(len(input_table), dtype=torch.bool)
    basis[held_out] = False
    tables = {"input": input_table.float(), "head": head_table.float()}

    def median_cosine(rows, table):
        return torch.nn.functional.cosine_similarity(rows, tables[table][held_out], dim=-1).median().item()

    result = {}
    for donor_name, donor in (("donor input", donor_input), ("donor head", donor_head)):
        indices, values = omp_coefficients(donor[held_out].float(), donor[basis].float(), k, ridge=ridge)
        for table in tables:
            rows = (values[..., None] * tables[table][basis][indices]).sum(1)
            name = f"composition from {donor_name}, {table}"
            result[name] = median_cosine(rows, table)
            ratio = rows.norm(dim=-1) / tables[table][held_out].norm(dim=-1).clamp_min(1e-12)
            result[f"{name}, norm ratio median"] = ratio.median().item()
            result[f"{name}, norm ratio p99"] = ratio.quantile(0.99).item()
    mapping = fit_linear(tables["head"][basis], tables["input"][basis])
    result["linear head to input, input"] = median_cosine(tables["head"][held_out] @ mapping, "input")
    mapping = fit_linear(donor_head[basis].float(), tables["head"][basis])
    result["linear donor head to head, head"] = median_cosine(donor_head[held_out].float() @ mapping, "head")
    for table, rows in (piece_rows or {}).items():
        result[f"piece mean, {table}"] = median_cosine(rows.float(), table)
    return result


def compose_rows(indices: torch.Tensor, values: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    r"""
    Args:
        indices (`torch.Tensor` of shape `(rows, k)`):
            Base tokens of each row.
        values (`torch.Tensor` of shape `(rows, k)`):
            Their coefficients.
        basis (`torch.Tensor` of shape `(base_vocab_size, width)`):
            The base rows of a table.

    Returns:
        `torch.Tensor` of shape `(rows, width)`: Each row's combination of base rows (SPEC S12.3a), on the basis's
        device.
    """
    values = values.to(basis.device, basis.dtype)
    return nn.functional.embedding_bag(indices.to(basis.device), basis, per_sample_weights=values, mode="sum")


def composed_norms(indices: torch.Tensor, values: torch.Tensor, basis: torch.Tensor, chunk: int = 16384) -> torch.Tensor:
    r"""
    Norms of the rows [`compose_rows`] gives, computed `chunk` rows at a time so the rows themselves are never held at
    once.

    Args:
        indices (`torch.Tensor` of shape `(rows, k)`):
            Base tokens of each row.
        values (`torch.Tensor` of shape `(rows, k)`):
            Their coefficients.
        basis (`torch.Tensor` of shape `(base_vocab_size, width)`):
            The base rows of a table.
        chunk (`int`, *optional*, defaults to 16384):
            Rows composed at once.

    Returns:
        `torch.Tensor` of shape `(rows,)`: The norms, on the basis's device.
    """
    return torch.cat(
        [
            compose_rows(indices[start : start + chunk], values[start : start + chunk], basis).norm(dim=-1)
            for start in range(0, len(indices), chunk)
        ]
    )


@torch.no_grad()
def vocabulary_coefficients(
    tokenizer: PolyplexTokenizer, base: int, rows: int, k: int, qwen_head: torch.Tensor, qwen_ids: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Composition coefficients of the Polyplex vocabulary over Helium's tokens, before their norms are calibrated
    (SPEC S12.3a). A token Qwen3 has is its Qwen3 head row written by [`omp_coefficients`] over the shared tokens'
    Qwen3 head rows. Every other token takes the mean of its pieces' coefficients (SPEC S12.2), a Helium piece counting
    as itself with coefficient 1; coefficients on the same Helium token are summed and the `k` largest in magnitude
    kept. Helium's tokens and the rows past the tokenizer have none.

    Args:
        tokenizer ([`PolyplexTokenizer`]):
            The vocabulary, Helium's tokens first.
        base (`int`):
            Helium's vocabulary size.
        rows (`int`):
            Rows of the coefficient table, the input tables' rows.
        k (`int`):
            Base tokens per token.
        qwen_head (`torch.Tensor` of shape `(qwen_vocab_size, hidden_size)`):
            Qwen3's LM head.
        qwen_ids (`dict`):
            Piece mapped to its Qwen3 id, from [`qwen_piece_ids`].

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: The base tokens and the coefficients, each of shape `(rows, k)`.
    """
    size = len(tokenizer)
    pieces = tokenizer.convert_ids_to_tokens(list(range(size)))
    indices = torch.zeros(rows, k, dtype=torch.long)
    values = torch.zeros(rows, k)

    shared = [(index, qwen_ids[piece]) for index, piece in enumerate(pieces[:base]) if piece in qwen_ids]
    shared_helium = torch.tensor([index for index, _ in shared])
    shared_rows = qwen_head[[qwen for _, qwen in shared]].float()
    from_qwen = [index for index in range(base, size) if pieces[index] in qwen_ids]
    if from_qwen:
        # The targets stay in the head's dtype; each batch is widened as it is solved.
        found, coefficients = omp_coefficients(qwen_head[[qwen_ids[pieces[index]] for index in from_qwen]], shared_rows, k)
        indices[from_qwen] = shared_helium[found]
        values[from_qwen] = coefficients

    composed = set(from_qwen)
    rest = sorted(set(range(base, size)) - composed)
    for index, cut in zip(rest, piece_ids(tokenizer, rest, list(range(base)) + from_qwen) if rest else []):
        cut = [piece for piece in cut if piece < base or piece in composed]
        combined = {}
        for piece in cut:
            if piece < base:
                combined[piece] = combined.get(piece, 0.0) + 1.0 / len(cut)
                continue
            for basis, value in zip(indices[piece].tolist(), values[piece].tolist()):
                combined[basis] = combined.get(basis, 0.0) + value / len(cut)
        kept = sorted(combined.items(), key=lambda item: -abs(item[1]))[:k]
        if kept:
            indices[index, : len(kept)] = torch.tensor([basis for basis, _ in kept])
            values[index, : len(kept)] = torch.tensor([value for _, value in kept])
    return indices, values


@torch.no_grad()
def calibrate_coefficients(
    indices: torch.Tensor, values: torch.Tensor, helium_head: torch.Tensor, blocks: dict, base: int
) -> torch.Tensor:
    r"""
    Scales composition coefficients in Helium's head space (SPEC S12.3a): each block's tokens past `base` take one
    factor, so the median norm of their composed head rows is that of Helium's head rows; then no composed head row
    is left longer than Helium's longest.

    Args:
        indices (`torch.Tensor` of shape `(rows, k)`):
            Base tokens of each row.
        values (`torch.Tensor` of shape `(rows, k)`):
            Their coefficients.
        helium_head (`torch.Tensor` of shape `(base, hidden_size)`):
            Helium's LM head.
        blocks (`dict`):
            Block name mapped to its `[start, end)` ids, from the tokenizer.
        base (`int`):
            Helium's vocabulary size.

    Returns:
        `torch.Tensor` of shape `(rows, k)`: The scaled coefficients, on the device of `values`.
    """
    device = values.device
    values = values.to(helium_head.device, torch.float32, copy=True)
    helium_norms = helium_head.norm(dim=-1)
    norms = composed_norms(indices, values, helium_head)
    for start, end in blocks.values():
        members = torch.arange(max(start, base), min(end, len(norms)), device=norms.device)
        members = members[norms[members] > 0]
        if len(members):
            values[members] *= helium_norms.median() / norms[members].median()
    norms = composed_norms(indices, values, helium_head)
    values *= (helium_norms.max() / norms.clamp_min(1e-12)).clamp(max=1.0)[:, None]
    return values.to(device)


def with_components(rows: torch.Tensor, directions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    r"""
    Sets each row's components along orthonormal `directions` to `targets` and keeps the rest of the row.

    Args:
        rows (`torch.Tensor` of shape `(rows, hidden_size)`):
            Head rows.
        directions (`torch.Tensor` of shape `(count, hidden_size)`):
            Orthonormal directions.
        targets (`torch.Tensor` of shape `(count,)`):
            The component along each direction every row takes.

    Returns:
        `torch.Tensor` of shape `(rows, hidden_size)`: The moved rows.
    """
    return rows + (targets - rows @ directions.T) @ directions


def prior_components(states: torch.Tensor, helium_head: torch.Tensor, count: int = 64) -> tuple:
    r"""
    The directions and values [`head_prior_correction`] sets (SPEC S12.3a): the mean of the states the head reads,
    taking the lowest of Helium's rows along it, and the `count` main directions of the states around their mean,
    orthogonal to it, taking Helium's mean head row along them.

    Args:
        states (`torch.Tensor` of shape `(positions, hidden_size)`):
            States the head reads, from [`helium_text_states`].
        helium_head (`torch.Tensor` of shape `(base, hidden_size)`):
            Helium's LM head.
        count (`int`, *optional*, defaults to 64):
            Main directions of the states set to Helium's mean.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: Orthonormal directions of shape `(count + 1, hidden_size)` and the
        component each takes, in float32 on the head's device.
    """
    states, helium_head = states.to(helium_head.device, torch.float32), helium_head.float()
    direction = nn.functional.normalize(states.mean(0), dim=0)
    centered = states - states.mean(0)
    centered = centered - (centered @ direction)[:, None] * direction
    principal = torch.linalg.svd(centered, full_matrices=False).Vh[:count]
    directions = torch.cat([direction[None], principal])
    targets = torch.cat([(helium_head @ direction).min()[None], helium_head.mean(0) @ principal.T])
    return directions, targets


def head_prior_correction(
    rows: torch.Tensor, states: torch.Tensor, helium_head: torch.Tensor, count: int = 64
) -> torch.Tensor:
    r"""
    Gives new head rows Helium's context-independent logits (SPEC S12.3a). The component along the mean of the states
    the head reads, a token's logit prior, becomes the lowest of Helium's rows; the components along the `count` main
    directions of the states around their mean, orthogonal to it, become those of Helium's mean head row
    ([`prior_components`]). The rest of each row is kept.

    Args:
        rows (`torch.Tensor` of shape `(rows, hidden_size)`):
            New head rows.
        states (`torch.Tensor` of shape `(positions, hidden_size)`):
            States the head reads, from [`helium_text_states`].
        helium_head (`torch.Tensor` of shape `(base, hidden_size)`):
            Helium's LM head.
        count (`int`, *optional*, defaults to 64):
            Main directions of the states set to Helium's mean.

    Returns:
        `torch.Tensor` of shape `(rows, hidden_size)`: The corrected rows, in float32 on the device of `rows`.
    """
    directions, targets = prior_components(states, helium_head.to(rows.device), count)
    return with_components(rows.float(), directions, targets)


@torch.no_grad()
def helium_text_states(helium, processor, tokenizer, texts: list) -> torch.Tensor:
    r"""
    The states Helium's text head reads on texts given as dense rows behind the opening frame, at every position that
    predicts a token of the text.

    Args:
        helium (`PersonaPlexForConditionalGeneration`):
            The duplex checkpoint.
        processor:
            Gives the codec silence the rows carry, see
            [`~polyplex_research.datasets.duplex_data.dense_collate_fn`].
        tokenizer (`PreTrainedTokenizerBase`):
            Helium's tokenizer.
        texts (`list[str]`):
            The texts.

    Returns:
        `torch.Tensor` of shape `(positions, hidden_size)`: The states, in float32 on the CPU.
    """
    from ..datasets.duplex_data import dense_collate_fn

    collate = dense_collate_fn(processor, helium)
    device = next(helium.parameters()).device
    captured = []
    hook = helium.lm_head.register_forward_hook(lambda module, inputs, output: captured.append(inputs[0][0, 1:-1]))
    try:
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < 2:
                continue
            batch = {key: value.to(device) for key, value in collate([{"segments": [torch.tensor(ids)]}]).items()}
            batch.pop("text_labels")
            helium(**batch, use_cache=False)
    finally:
        hook.remove()
    return torch.cat(captured).float().cpu()


@torch.no_grad()
def residual_alignment(
    personaplex: str,
    language_model: str,
    texts: list,
    language_ffn_layers: list[int] | None = None,
    appended_layers: int = 4,
    held_out: float = 0.1,
    skip_tokens: int = 4,
    device: str | None = None,
    shared: bool = False,
) -> dict:
    r"""
    Finds the residual basis alignment of every DualFFN layer (SPEC S11.2a): the texts go through PersonaPlex (as
    dense rows behind the opening frame) and the text LLM, and at every token both cut end at the same character the
    residuals before the paired FFNs are compared. The rotation of each layer comes from the RMS normalized residuals
    of the first texts ([`procrustes_rotation`]), the scale is the median ratio of their RMS, and the last `held_out`
    share of the texts measures the fit.

    Args:
        personaplex (`str`):
            Path or hub id of the duplex checkpoint.
        language_model (`str`):
            Path or hub id of the text LLM.
        texts (`list[str]`):
            English texts.
        language_ffn_layers (`list[int]`, *optional*):
            Pairing of backbone layers with text LLM layers, see [`dualffn_config`].
        appended_layers (`int`, *optional*, defaults to 4):
            Layers appended after Helium's own (SPEC S11.4a). They are compared with Helium's last residual.
        held_out (`float`, *optional*, defaults to 0.1):
            Share of the texts kept out of the fit and measured.
        skip_tokens (`int`, *optional*, defaults to 4):
            Leading tokens of each text left out, whose residuals carry the attention sink.
        device (`str`, *optional*):
            Device both models run on. The CPU when omitted.
        shared (`bool`, *optional*, defaults to `False`):
            Fits one rotation and one scale for all layers together (SPEC S11.2h): the residual pairs of every layer
            enter one Procrustes problem, and the scale is the median ratio over every layer and position.

    Returns:
        `dict`: `rotations` of shape `(layers, hidden_size, hidden_size)` and `scales` of shape `(layers,)` in float32
        on the CPU, one entry of each when `shared`, and `report`, per layer the mean held out cosine before and after
        the rotation and the positions fitted.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, PersonaPlexForConditionalGeneration

    from ..datasets.duplex_data import dense_collate_fn

    device = device or "cpu"
    helium = PersonaPlexForConditionalGeneration.from_pretrained(personaplex, torch_dtype=torch.bfloat16, device_map=device)
    text_llm = AutoModelForCausalLM.from_pretrained(language_model, torch_dtype=torch.bfloat16, device_map=device)
    config = dualffn_config(helium.config, text_llm.config, language_ffn_layers, appended_layers=appended_layers)
    processor = PolyplexProcessor.from_pretrained(personaplex)
    processor.audio_tokenizer.to(device)
    collate = dense_collate_fn(processor, helium)
    helium_tokenizer = load_base_tokenizer(personaplex)
    text_tokenizer = AutoTokenizer.from_pretrained(language_model)
    helium_layers = len(helium.model.layers)

    captured = {"helium": {}, "text": {}}
    hooks = []
    for index, layer in enumerate(helium.model.layers):
        hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(
            lambda module, inputs, index=index: captured["helium"].__setitem__(index, inputs[0][0])
        ))
    hooks.append(helium.model.norm.register_forward_pre_hook(
        lambda module, inputs: captured["helium"].__setitem__("final", inputs[0][0])
    ))
    for index, layer in enumerate(text_llm.model.layers):
        hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(
            lambda module, inputs, index=index: captured["text"].__setitem__(index, inputs[0][0])
        ))
    sources = [index if index < helium_layers else "final" for index in range(len(config.language_ffn_layers))]

    def normalized(states):
        states = states.float()
        return states / states.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)

    def read(text):
        helium_ids, text_ids, helium_index, text_index = aligned_positions(text, helium_tokenizer, text_tokenizer)
        keep = [(a, b) for a, b in zip(helium_index, text_index) if a >= skip_tokens and b >= skip_tokens]
        if not keep:
            return None
        batch = {key: value.to(device) for key, value in collate([{"segments": [torch.tensor(helium_ids)]}]).items()}
        batch.pop("text_labels")
        helium(**batch, use_cache=False)
        text_llm(input_ids=torch.tensor([text_ids], device=device), use_cache=False)
        # Behind the opening frame, Helium reads token `i` at frame `i + 1`.
        frames = torch.tensor([a + 1 for a, _ in keep], device=device)
        positions = torch.tensor([b for _, b in keep], device=device)
        return [
            (captured["helium"][source][frames], captured["text"][paired][positions])
            for source, paired in zip(sources, config.language_ffn_layers)
        ]

    layers = len(config.language_ffn_layers)
    width = helium.config.hidden_size
    split = len(texts) - max(1, int(len(texts) * held_out))
    cross = torch.zeros(layers, width, width, device=device)
    ratios = [[] for _ in range(layers)]
    fitted = 0
    try:
        for number, text in enumerate(texts[:split], start=1):
            pairs = read(text)
            if number % 32 == 0:
                logger.info("Residual alignment: read %d of %d texts, %d positions.", number, split, fitted)
            if pairs is None:
                continue
            fitted += len(pairs[0][0])
            for index, (helium_states, text_states) in enumerate(pairs):
                cross[index] += normalized(text_states).T @ normalized(helium_states)
                rms = lambda states: states.float().square().mean(-1).sqrt()
                ratios[index].append((rms(helium_states) / rms(text_states).clamp(min=1e-6)).cpu())
        if shared:
            cross = cross.sum(0, keepdim=True)
            ratios = [[value for values in ratios for value in values]]
        logger.info("Residual alignment: solving %d rotations.", len(cross))
        # The SVD runs on the CPU, whose float64 is far faster than a consumer GPU's.
        rotations = torch.stack([procrustes_rotation(matrix.cpu()) for matrix in cross]).to(device)
        del cross
        before = torch.zeros(layers)
        after = torch.zeros(layers)
        measured = 0
        for text in texts[split:]:
            pairs = read(text)
            if pairs is None:
                continue
            measured += len(pairs[0][0])
            for index, (helium_states, text_states) in enumerate(pairs):
                source, target = normalized(helium_states), normalized(text_states)
                cosine = lambda a, b: nn.functional.cosine_similarity(a, b, dim=-1).sum().item()
                before[index] += cosine(source, target)
                after[index] += cosine(source @ rotations[0 if shared else index].T, target)
    finally:
        for hook in hooks:
            hook.remove()
    scales = torch.stack([torch.cat(values).median() for values in ratios]).float()
    report = [
        {
            "layer": index,
            "text_layer": config.language_ffn_layers[index],
            "cosine_before": round(float(before[index]) / max(measured, 1), 4),
            "cosine_after": round(float(after[index]) / max(measured, 1), 4),
            "scale": round(float(scales[0 if shared else index]), 4),
        }
        for index in range(layers)
    ]
    logger.info("Residual alignment fitted on %d positions, measured on %d.", fitted, measured)
    return {"rotations": rotations.float().cpu(), "scales": scales, "report": report,
            "fitted": fitted, "measured": measured}


@torch.no_grad()
def language_embedding_rows(
    tokenizer: PreTrainedTokenizerBase, language_tokenizer: PreTrainedTokenizerBase, language_table: torch.Tensor, rows: int
) -> torch.Tensor:
    r"""
    Rows of the language stream's own text table (SPEC S11.2f): each token takes the mean of the text LLM's input
    embeddings over the text LLM's tokens of the token's text. Tokens whose text the text LLM reads as nothing, and the
    rows past the vocabulary, are zero.

    Args:
        tokenizer ([`PreTrainedTokenizerBase`]):
            The model's text vocabulary.
        language_tokenizer ([`PreTrainedTokenizerBase`]):
            The text LLM's tokenizer.
        language_table (`torch.Tensor` of shape `(language_vocab_size, hidden_size)`):
            The text LLM's input embeddings.
        rows (`int`):
            Rows of the table, the model's input table's rows.

    Returns:
        `torch.Tensor` of shape `(rows, hidden_size)`: The rows, in the text LLM table's dtype.
    """
    size = len(tokenizer)
    pieces = tokenizer.convert_ids_to_tokens(list(range(size)))
    flat, offsets, owners = [], [], []
    for index, piece in enumerate(pieces):
        ids = language_tokenizer.encode(tokenizer.convert_tokens_to_string([piece]), add_special_tokens=False)
        if ids:
            owners.append(index)
            offsets.append(len(flat))
            flat.extend(ids)
    table = torch.zeros(rows, language_table.shape[-1], dtype=language_table.dtype)
    if owners:
        means = torch.nn.functional.embedding_bag(
            torch.tensor(flat), language_table.float(), torch.tensor(offsets), mode="mean"
        )
        table[torch.tensor(owners)] = means.to(language_table.dtype)
    return table


def load_language_table(language_model: str) -> torch.Tensor:
    r"""
    Reads the input embeddings of a text LLM on the Hub, without loading the rest of it.

    Args:
        language_model (`str`):
            Hub id of the text LLM.

    Returns:
        `torch.Tensor` of shape `(vocab_size, hidden_size)`: Its `model.embed_tokens.weight`.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    name = "model.embed_tokens.weight"
    index = json.load(open(hf_hub_download(language_model, "model.safetensors.index.json")))
    with safe_open(hf_hub_download(language_model, index["weight_map"][name]), framework="pt") as handle:
        return handle.get_tensor(name)


def load_language_ffn_weights(language_model: str, language_ffn_layers: list):
    r"""
    Returns a reader of the text LLM's original MLP weights on the Hub, one projection at a time, for the backbone layer
    each was grafted into.

    Args:
        language_model (`str`):
            Hub id of the text LLM.
        language_ffn_layers (`list[int]`):
            For each backbone layer, the text LLM layer its language FFN came from.

    Returns:
        `Callable[[int, str], torch.Tensor]`: `weights(layer, name)` gives the `gate`, `up` or `down` projection weight
        of backbone layer `layer`.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    index = json.load(open(hf_hub_download(language_model, "model.safetensors.index.json")))["weight_map"]

    def weights(layer: int, name: str) -> torch.Tensor:
        key = f"model.layers.{language_ffn_layers[layer]}.mlp.{name}_proj.weight"
        with safe_open(hf_hub_download(language_model, index[key]), framework="pt") as handle:
            return handle.get_tensor(key)

    return weights


def load_residual_alignment(path: str):
    r"""
    Returns, for each layer index, the `(rotation, scale, gain)` a saved residual basis alignment folds in (SPEC
    S11.2a), read lazily from its safetensors file. `gain` is `None` unless the file carries the gains of a trained
    model. A file with one rotation and one scale is a shared alignment (SPEC S11.2h), and every layer reads that one.

    Args:
        path (`str`):
            A file with `rotations`, `scales` and optionally `gains`.

    Returns:
        `Callable[[int], tuple]`: The alignment of one layer.
    """
    from safetensors import safe_open

    handle = safe_open(path, framework="pt")
    keys = set(handle.keys())

    shared = handle.get_slice("scales").get_shape()[0] == 1

    def layer(index: int) -> tuple:
        gain = handle.get_slice("gains")[index] if "gains" in keys else None
        entry = 0 if shared else index
        return handle.get_slice("rotations")[entry], float(handle.get_slice("scales")[entry]), gain

    return layer


@torch.no_grad()
def graft_vocabulary(
    model: PolyplexForConditionalGeneration,
    tokenizer: PolyplexTokenizer,
    helium,
    qwen_head: torch.Tensor,
    qwen_ids: dict,
    states: torch.Tensor | None = None,
    principal_directions: int = 64,
) -> None:
    r"""
    Fills a DualFFN model's text tables and their shared composition for the Polyplex vocabulary (SPEC S12.3a).

    - Stored rows: the text input table, the LM head and the depth decoder's text table keep Helium's rows for
      Helium's tokens, and their input tables keep Helium's reserved row for the opening frame. Every other stored
      row, a new token's residual, is zero.
    - Coefficients of a token Qwen3 has: its Qwen3 head row written by [`omp_coefficients`] as a combination of the
      shared tokens' Qwen3 head rows. The coefficients go to those tokens' Helium ids.
    - Coefficients of every other token: the mean of the coefficients of its pieces (SPEC S12.2), a Helium piece
      counting as itself with coefficient 1. Coefficients on the same Helium token are summed and the largest
      `text_composition_size` in magnitude are kept.
    - Norms, in Helium's head space: each script block's new tokens have their coefficients scaled by one factor, so
      the median norm of their composed head rows is that of Helium's head rows; then no composed head row is left
      longer than Helium's longest ([`calibrate_coefficients`]).
    - With `states`, each new token's head residual starts at the correction [`head_prior_correction`] makes to its
      composed head row; otherwise at zero.
    - Helium's tokens, the reserved row and the rows past the tokenizer have no coefficients.
    - The head reads Helium's final normalization (SPEC S11.5): its weight goes into `text_norm`.
    - `base_vocab_size` is set to Helium's vocabulary, the rows training leaves frozen.

    Args:
        model ([`PolyplexForConditionalGeneration`]):
            The model, with `vocab_size` at least the tokenizer's size and a shared text composition over Helium's
            vocabulary.
        tokenizer ([`PolyplexTokenizer`]):
            The vocabulary, Helium's tokens first.
        helium (`PersonaPlexForConditionalGeneration`):
            The duplex checkpoint.
        qwen_head (`torch.Tensor` of shape `(qwen_vocab_size, hidden_size)`):
            Qwen3's LM head.
        qwen_ids (`dict`):
            Piece mapped to its Qwen3 id, from [`qwen_piece_ids`].
        states (`torch.Tensor` of shape `(positions, hidden_size)`, *optional*):
            States Helium's text head reads on calibration texts, from [`helium_text_states`].
        principal_directions (`int`, *optional*, defaults to 64):
            Main directions of `states` the correction sets, see [`head_prior_correction`].

    Raises:
        ValueError: If the model has no shared composition over Helium's vocabulary or fewer head rows than the
            tokenizer has tokens.
    """
    size, base = len(tokenizer), helium.config.vocab_size
    composition = model.text_composition
    if composition is None or composition.base_size != base or model.config.vocab_size < size:
        raise ValueError(
            f"The model needs a shared composition over Helium's {base} tokens and at least {size} head rows, it has "
            f"{model.config.vocab_size}."
        )

    tables = {
        "input": (model.model.embed_tokens.weight, helium.model.embed_tokens.weight),
        "head": (model.lm_head.weight, helium.lm_head.weight),
        "depth": (model.depth_decoder.model.text_embed_tokens.weight, helium.depth_decoder.model.text_embed_tokens.weight),
    }
    for name, (table, helium_table) in tables.items():
        table.zero_()
        table[:base] = helium_table[:base]
        if name != "head":
            table[-1] = helium_table[-1]

    indices, values = vocabulary_coefficients(
        tokenizer, base, len(composition.indices), composition.values.shape[-1], qwen_head, qwen_ids
    )
    values = calibrate_coefficients(indices, values, helium.lm_head.weight.float(), tokenizer.blocks, base)
    composition.indices.copy_(indices)
    composition.values.copy_(values)
    if states is not None:
        head = model.lm_head.weight
        directions, targets = prior_components(states, helium.lm_head.weight, principal_directions)
        basis = head[:base]
        for start in range(base, size, 16384):
            end = min(start + 16384, size)
            composed = compose_rows(composition.indices[start:end], composition.values[start:end], basis).float()
            corrected = with_components(composed, directions, targets)
            head[start:end] = (corrected - composed).to(head.dtype)
    model.model.text_norm.weight.copy_(helium.model.norm.weight)
    setattr(model.config, BASE_VOCAB_SIZE_KEY, base)


@torch.no_grad()
def cap_composed_head_norms(model: PolyplexForConditionalGeneration, limit: float) -> int:
    r"""
    Shrinks the composition coefficients of every token past the base vocabulary whose composed LM head row is longer
    than `limit`, so that row is exactly `limit` long. Meant for an assembled model, whose new tokens' stored head rows
    are still zero; their input and depth rows shrink by the same factor.

    Args:
        model ([`PolyplexForConditionalGeneration`]):
            The model, with a shared text composition.
        limit (`float`):
            The longest a composed head row may be.

    Returns:
        `int`: The number of tokens shrunk.
    """
    composition = model.text_composition
    norms = model.lm_head.composed_weight().float().norm(dim=-1)
    factor = (limit / norms.clamp_min(1e-12)).clamp(max=1.0)
    factor[: composition.base_size] = 1.0
    composition.values[: len(factor)] *= factor[:, None].to(composition.values.dtype)
    return int((factor < 1.0).sum())


def assemble_dualffn(
    personaplex: str,
    language_model: str,
    output_dir: str | None = None,
    language_ffn_layers: list[int] | None = None,
    torch_dtype: str = "bfloat16",
    tokenizer: str | None = None,
    composition_size: int = 32,
    device: str | None = None,
    quantization: str | None = None,
    full_precision_layers: tuple = (),
    appended_layers: int = 0,
    calibration_texts: list | None = None,
    alignment: str | None = None,
    removed_neurons: dict | None = None,
    keep_appended_output: bool = False,
) -> PolyplexForConditionalGeneration:
    r"""
    Builds and saves a DualFFN checkpoint from a duplex checkpoint and a text LLM (SPEC S11, S12).

    Args:
        personaplex (`str`):
            Path or hub id of the duplex checkpoint.
        language_model (`str`):
            Path or hub id of the text LLM, a Qwen3 model.
        output_dir (`str`, *optional*):
            Directory the checkpoint and its tokenizer are saved to. Nothing is saved when omitted.
        language_ffn_layers (`list[int]`, *optional*):
            Pairing of backbone layers with text LLM layers, see [`dualffn_config`].
        torch_dtype (`str`, *optional*, defaults to `"bfloat16"`):
            Precision the weights are loaded and saved in.
        tokenizer (`str`, *optional*):
            Path of a [`PolyplexTokenizer`]. The model then takes that vocabulary ([`graft_vocabulary`]), with the head
            padded to a multiple of 128 rows, and is saved with a [`PolyplexProcessor`]. The text LLM's vocabulary is
            kept as it is when omitted (the D10 check, SPEC S13.4).
        composition_size (`int`, *optional*, defaults to 32):
            Base tokens per text token in the shared composition (SPEC S12.3a), used with `tokenizer`.
        device (`str`, *optional*):
            Device the two checkpoints and the model are assembled on. The CPU when omitted.
        quantization (`str`, *optional*):
            `"nvfp4"` quantizes the frozen backbone as soon as it is grafted, dropping its full precision weights
            ([`quantize_frozen_backbone`], SPEC S7.1). Such a model is saved as it is ([`save_assembled_dualffn`]).
        full_precision_layers (`tuple[int]`, *optional*):
            Layers left in full precision under `quantization`, see [`quantize_frozen_backbone`].
        appended_layers (`int`, *optional*, defaults to 0):
            Copies of Helium's last layers appended after its own (SPEC S11.4a), see [`dualffn_config`].
        calibration_texts (`list[str]`, *optional*):
            English texts Helium reads to set the new head rows' context-independent logits (SPEC S12.3a), used with
            `tokenizer`. The new head residuals start at zero when omitted.
        alignment (`str`, *optional*):
            A residual basis alignment file ([`residual_alignment`]) folded into every language FFN (SPEC S11.2a).
        removed_neurons (`dict[int, list[int]]`, *optional*):
            Layer mapped to the language FFN neurons removed (SPEC S11.2b).
        keep_appended_output (`bool`, *optional*, defaults to `False`):
            Whether appended layers keep the text LLM's language FFN output projection, for a model whose language
            FFN output scale starts at zero (SPEC S11.4a).

    Returns:
        [`PolyplexForConditionalGeneration`]: The assembled model.

    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, PersonaPlexForConditionalGeneration

    dtype = getattr(torch, torch_dtype)
    device = device or "cpu"
    # Both checkpoints load straight onto the device and the DualFFN is built one layer at a time (SPEC S11.2).
    helium = PersonaPlexForConditionalGeneration.from_pretrained(personaplex, torch_dtype=dtype, device_map=device)
    text_llm = AutoModelForCausalLM.from_pretrained(language_model, torch_dtype=dtype, device_map=device)
    polyplex = None if tokenizer is None else PolyplexTokenizer.from_pretrained(tokenizer)
    if polyplex is not None:
        rows = -(-len(polyplex) // 128) * 128
        config = dualffn_config(
            helium.config, text_llm.config, language_ffn_layers, rows, composition_size, appended_layers
        )
    else:
        config = dualffn_config(helium.config, text_llm.config, language_ffn_layers, appended_layers=appended_layers)
    with torch.device("meta"):
        model = PolyplexForConditionalGeneration._from_config(config, torch_dtype=dtype)

    states = None
    if polyplex is not None and calibration_texts:
        # Read while Helium still has all its layers.
        processor = PolyplexProcessor.from_pretrained(personaplex)
        processor.audio_tokenizer.to(device)
        states = helium_text_states(helium, processor, load_base_tokenizer(personaplex), calibration_texts)
        del processor
    graft_layers(
        model,
        helium,
        text_llm,
        device,
        quantization,
        full_precision_layers,
        None if alignment is None else load_residual_alignment(alignment),
        None if removed_neurons is None else {int(layer): list(neurons) for layer, neurons in removed_neurons.items()},
        keep_appended_output,
    )
    materialize_outside_layers(model, device)
    graft_helium_rest(model, helium)

    if polyplex is not None:
        qwen_ids = qwen_piece_ids(AutoTokenizer.from_pretrained(language_model))
        graft_vocabulary(model, polyplex, helium, text_llm.lm_head.weight, qwen_ids, states=states)
        del helium, text_llm
        logger.info("DualFFN assembled with the Polyplex vocabulary: %d tokens, %d head rows.", len(polyplex), rows)
        if output_dir is not None:
            if quantization == "nvfp4":
                save_assembled_dualffn(model, output_dir, full_precision_layers)
            else:
                os.makedirs(output_dir, exist_ok=True)
                model.save_pretrained(output_dir)
            PolyplexProcessor(tokenizer=polyplex).save_pretrained(output_dir)
        return model

    graft_language_rest(model, text_llm)
    helium_tokenizer = load_base_tokenizer(personaplex)
    tokenizer = AutoTokenizer.from_pretrained(language_model)
    helium_table = helium.depth_decoder.model.text_embed_tokens.weight
    table = model.depth_decoder.model.text_embed_tokens.weight
    named = min(len(tokenizer), config.vocab_size)
    with torch.no_grad():
        table[:named] = depth_text_rows(tokenizer, named, helium_tokenizer, helium_table[:-1]).to(table.dtype)
        # Ids the tokenizer never emits keep the mean row, and the reserved id keeps Helium's.
        table[named:-1] = helium_table[:-1].float().mean(0).to(table.dtype)
        table[-1] = helium_table[-1]
    del helium, text_llm
    logger.info(
        "DualFFN assembled: %d layers, FFN width %d (%d from Helium), vocabulary %d, layers paired %s.",
        config.num_hidden_layers,
        config.intermediate_size,
        config.duplex_pretrained_size,
        config.vocab_size,
        config.language_ffn_layers,
    )
    if output_dir is not None:
        if quantization == "nvfp4":
            save_assembled_dualffn(model, output_dir, full_precision_layers)
        else:
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    return model


def runs_on_ephemeral_disk() -> bool:
    r"""
    Tells whether this process runs on a disk the run owns, which is lost when the session ends: Colab or Kaggle,
    marked by `POLYPLEX_EPHEMERAL_DISK=1`, which [`~polyplex_research.utils.phase.launch`] sets.

    Returns:
        `bool`: Whether the disk is the session's own.
    """
    return os.environ.get("POLYPLEX_EPHEMERAL_DISK") == "1"


def assembly_name(settings: dict) -> str:
    r"""
    The name an assembly is saved and stored under (SPEC S11.2): `dualffn-` and 16 hexadecimal characters of the
    SHA-256 of `settings` as sorted JSON. A setting that is the path of an existing file is hashed as the SHA-256 of
    the file's content, so the name follows what the file holds and not where it lies. Every other setting, a
    directory or a Hub id among them, is hashed as it is.

    Args:
        settings (`dict`):
            The arguments of [`assemble_dualffn`] other than `device` and `output_dir`.

    Returns:
        `str`: The name.
    """
    import hashlib

    hashed = {}
    for setting, value in settings.items():
        if isinstance(value, (str, os.PathLike)) and os.path.isfile(value):
            digest = hashlib.sha256()
            with open(value, "rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            value = f"sha256:{digest.hexdigest()}"
        hashed[setting] = value
    return "dualffn-" + hashlib.sha256(json.dumps(hashed, sort_keys=True, default=str).encode()).hexdigest()[:16]


def load_or_assemble_dualffn(device, backward_dtype: str = "bf16", **settings) -> PolyplexForConditionalGeneration:
    r"""
    Returns the DualFFN model `settings` describe, assembled once (SPEC S11.2): a model saved under
    `$HF_HOME/assembled` for the same settings is loaded ([`load_assembled_dualffn`]), and otherwise the model is
    assembled ([`assemble_dualffn`]) and, when quantized, saved there first ([`save_assembled_dualffn`]). On a disk
    the run owns (`POLYPLEX_EPHEMERAL_DISK`) the saved model also goes to the Drive store ([`store_assembly`]), and
    a model missing here is fetched from it first ([`fetch_stored_assembly`]).

    Args:
        device (`str` or `torch.device`):
            Where the model is assembled or loaded.
        backward_dtype (`str`, *optional*, defaults to `"bf16"`):
            Backward precision of the NVFP4 projections of a loaded model (SPEC S7.1a).
        **settings:
            The arguments of [`assemble_dualffn`] other than `device` and `output_dir`.

    Returns:
        [`PolyplexForConditionalGeneration`]: The model.
    """
    from .quantization import set_nvfp4_backward

    name = assembly_name(settings)
    path = os.path.join(os.environ.get("HF_HOME", ".cache/huggingface"), "assembled", name)
    ephemeral = runs_on_ephemeral_disk()
    if not os.path.exists(os.path.join(path, "config.json")) and ephemeral:
        fetch_stored_assembly(name, path)
    if os.path.exists(os.path.join(path, "config.json")):
        logger.info("Loading the DualFFN assembled before at %s.", path)
        model = load_assembled_dualffn(path, device, backward_dtype)
        if ephemeral:
            # The Drive store keeps it; the session disk needs the room for checkpoints.
            shutil.rmtree(path)
        return model
    model = assemble_dualffn(device=device, **settings)
    if settings.get("quantization") == "nvfp4":
        save_assembled_dualffn(model, path, settings.get("full_precision_layers", ()))
        with open(os.path.join(path, "assembly.json"), "w") as handle:
            json.dump(settings, handle, indent=1, default=str)
        logger.info("Saved the assembled DualFFN to %s.", path)
        if ephemeral:
            from .phase import free_downloads

            # The handoff writes a second copy of the assembly to the disk, which the sources would not leave room for.
            free_downloads([repo for repo in (settings.get("personaplex"), settings.get("language_model")) if repo])
            try:
                store_assembly(name, path)
            except Exception as error:
                # The model is already on the device; the store only spares the next run an assembly.
                logger.warning("The assembled DualFFN %s was not stored on the Drive store: %s", name, error)
            shutil.rmtree(path)
    set_nvfp4_backward(model, backward_dtype)
    return model


def store_assembly(name: str, path: str) -> None:
    r"""
    Puts a saved assembly on the Drive store (SPEC S11.2): the whole directory as the one handoff `name`.

    Args:
        name (`str`):
            The assembly's name, `dualffn-<settings hash>`.
        path (`str`):
            The directory [`save_assembled_dualffn`] wrote.
    """
    import letify

    letify.stash(name, path)
    logger.info("Put the assembled DualFFN on the Drive store as %s.", name)


def fetch_stored_assembly(name: str, path: str) -> bool:
    r"""
    Fetches an assembly [`store_assembly`] put on the Drive store into `path`.

    Args:
        name (`str`):
            The assembly's name, `dualffn-<settings hash>`.
        path (`str`):
            The directory it is fetched into.

    Returns:
        `bool`: Whether the store held it.
    """
    import letify

    fetched = path.rstrip(os.sep) + ".fetched"
    os.makedirs(os.path.dirname(fetched) or ".", exist_ok=True)
    try:
        if letify.unstash(name, fetched) is None:
            return False
    except Exception as error:
        logger.warning("The Drive store could not be read for %s: %s", name, error)
        return False
    if os.path.isdir(fetched):
        shutil.rmtree(path, ignore_errors=True)
        os.replace(fetched, path)
    else:
        # The handoff lists the files, each stored as a handoff of its own.
        with open(fetched) as handle:
            files = json.load(handle)
        os.remove(fetched)
        os.makedirs(path, exist_ok=True)
        for file in files:
            if letify.unstash(f"{name}-{file}", os.path.join(path, file)) is None:
                raise FileNotFoundError(f"The Drive store lists {file} of {name} but does not hold it.")
    logger.info("Fetched the assembled DualFFN %s from the Drive store.", name)
    return True


def composition_check(
    personaplex: str,
    language_model: str,
    tokenizer: str,
    held_out: float = 0.1,
    k: int = 32,
    seed: int = 0,
    ridges: tuple = (1e-10,),
) -> dict:
    r"""
    Runs [`composition_reconstruction`] on the tokens Helium and the text LLM spell the same (SPEC S12.3a), with the
    S12.2 piece mean of each held-out token as a baseline.

    Args:
        personaplex (`str`):
            Path or hub id of the duplex checkpoint.
        language_model (`str`):
            Path or hub id of the text LLM, a Qwen3 model with untied input and head tables.
        tokenizer (`str`):
            Path or hub id of the [`PolyplexTokenizer`], Helium's tokens first.
        held_out (`float`, *optional*, defaults to 0.1):
            Share of the shared tokens measured.
        k (`int`, *optional*, defaults to 32):
            Basis rows per token.
        seed (`int`, *optional*, defaults to 0):
            Seed of the held-out draw.
        ridges (`tuple[float]`, *optional*, defaults to `(1e-10,)`):
            Ridges of the coefficient fit, each measured on the same tokens.

    Returns:
        `dict`: The measurements of each ridge, keyed by the ridge, with the number of shared and held-out tokens.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, PersonaPlexForConditionalGeneration

    helium = PersonaPlexForConditionalGeneration.from_pretrained(personaplex, torch_dtype=torch.bfloat16)
    text_llm = AutoModelForCausalLM.from_pretrained(language_model, torch_dtype=torch.bfloat16)
    polyplex = PolyplexTokenizer.from_pretrained(tokenizer)
    qwen_ids = qwen_piece_ids(AutoTokenizer.from_pretrained(language_model))
    base = helium.config.vocab_size
    pieces = polyplex.convert_ids_to_tokens(list(range(base)))
    shared = [(index, qwen_ids[piece]) for index, piece in enumerate(pieces) if piece in qwen_ids]
    helium_ids = torch.tensor([index for index, _ in shared])
    qwen = torch.tensor([index for _, index in shared])
    generator = torch.Generator().manual_seed(seed)
    measured = torch.randperm(len(shared), generator=generator)[: int(len(shared) * held_out)]

    input_table = helium.model.embed_tokens.weight[:base].float()
    head_table = helium.lm_head.weight.float()
    excluded = set(helium_ids[measured].tolist())
    kept = [index for index in range(base) if index not in excluded]
    piece_rows = {
        "input": piece_mean_rows(polyplex, helium_ids[measured].tolist(), kept, input_table),
        "head": piece_mean_rows(polyplex, helium_ids[measured].tolist(), kept, head_table),
    }
    result = {"shared": len(shared), "held_out": len(measured), "k": k}
    for ridge in ridges:
        result[f"ridge {ridge:g}"] = composition_reconstruction(
            text_llm.model.embed_tokens.weight[qwen].float(),
            text_llm.lm_head.weight[qwen].float(),
            input_table[helium_ids],
            head_table[helium_ids],
            measured,
            k,
            piece_rows,
            ridge,
        )
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


DESSEMBLERS = {1: Phase1Assemble, 2: Phase2Assemble}


def main(args: list[str] | None = None):
    r"""
    Runs the offline weight stages: the DualFFN assembly, the fold of a trained phase and the shared composition
    check.

    Args:
        args (`list[str]`, *optional*):
            Command line arguments, `sys.argv[1:]` when omitted.

    Raises:
        ValueError: If `--phase` names no phase.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Offline weight surgery around the two training phases."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    dessemble = commands.add_parser(
        "dessemble", help="Fold what a phase trained back into a single duplex checkpoint."
    )
    dessemble.add_argument(
        "--phase", type=int, required=True, choices=(1, 2), help="Which phase wrote the adapter."
    )
    dessemble.add_argument(
        "--base-model",
        required=True,
        help="Checkpoint the phase started from, the one its adapter was trained against.",
    )
    dessemble.add_argument(
        "--adapter", required=True, help="Directory or hub id the phase run wrote."
    )
    dessemble.add_argument(
        "--output-dir", required=True, help="Directory to save the merged checkpoint and tokenizer."
    )
    dessemble.add_argument(
        "--torch-dtype",
        default=os.environ.get("POLYPLEX_TORCH_DTYPE", "bfloat16"),
        help="Dtype for loading the pretrained weights.",
    )

    dualffn = commands.add_parser(
        "dualffn", help="Graft a text LLM's FFNs, embedding and head onto a duplex checkpoint (DualFFN)."
    )
    dualffn.add_argument(
        "--personaplex",
        default=os.environ.get("POLYPLEX_BASE_MODEL", "nvidia/personaplex-7b-v1"),
        help="Checkpoint path or hub id of the duplex model.",
    )
    dualffn.add_argument(
        "--language-model",
        default=os.environ.get("POLYPLEX_LANGUAGE_MODEL", "Qwen/Qwen3-8B"),
        help="Checkpoint path or hub id of the text LLM.",
    )
    dualffn.add_argument(
        "--language-ffn-layers",
        default=os.environ.get("POLYPLEX_LANGUAGE_FFN_LAYERS"),
        help="Comma separated text LLM layer for each backbone layer. The first layers when omitted.",
    )
    dualffn.add_argument("--output-dir", required=True, help="Directory to save the assembled checkpoint.")
    dualffn.add_argument(
        "--tokenizer",
        default=os.environ.get("POLYPLEX_TOKENIZER"),
        help="Path of the Polyplex tokenizer the model takes. The text LLM's vocabulary when omitted.",
    )
    dualffn.add_argument(
        "--composition-size", type=int, default=32, help="Base tokens per text token in the shared composition."
    )
    dualffn.add_argument(
        "--appended-layers", type=int, default=4, help="Copies of Helium's last layers appended after its own."
    )
    dualffn.add_argument(
        "--calibration-skip",
        type=int,
        default=64,
        help="English held out documents left out before the four that set the new head rows' priors.",
    )
    dualffn.add_argument(
        "--torch-dtype",
        default=os.environ.get("POLYPLEX_TORCH_DTYPE", "bfloat16"),
        help="Dtype for loading and saving the weights.",
    )

    composition = commands.add_parser(
        "composition-check",
        help="Measure how well coefficients found in the text LLM rebuild Helium's input and head rows (SPEC S12.3a).",
    )
    composition.add_argument(
        "--personaplex",
        default=os.environ.get("POLYPLEX_BASE_MODEL", "nvidia/personaplex-7b-v1"),
        help="Checkpoint path or hub id of the duplex model.",
    )
    composition.add_argument(
        "--language-model",
        default=os.environ.get("POLYPLEX_LANGUAGE_MODEL", "Qwen/Qwen3-8B"),
        help="Checkpoint path or hub id of the text LLM.",
    )
    composition.add_argument(
        "--tokenizer",
        default=os.environ.get("POLYPLEX_TOKENIZER", "RetentionLabs/PolyPlex-14B-CJKE"),
        help="Path or hub id of the Polyplex tokenizer.",
    )
    composition.add_argument("--held-out", type=float, default=0.1, help="Share of the shared tokens measured.")
    composition.add_argument("--k", type=int, default=32, help="Basis rows per token.")
    composition.add_argument(
        "--ridges", default="1e-10", help="Comma separated ridges of the coefficient fit, each measured."
    )

    alignment = commands.add_parser(
        "residual-alignment",
        help="Find the rotation from Helium's residual basis to the text LLM's for every DualFFN layer (SPEC S11.2a).",
    )
    alignment.add_argument(
        "--personaplex",
        default=os.environ.get("POLYPLEX_BASE_MODEL", "nvidia/personaplex-7b-v1"),
        help="Checkpoint path or hub id of the duplex model.",
    )
    alignment.add_argument(
        "--language-model",
        default=os.environ.get("POLYPLEX_LANGUAGE_MODEL", "Qwen/Qwen3-8B"),
        help="Checkpoint path or hub id of the text LLM.",
    )
    alignment.add_argument("--output", required=True, help="Safetensors file the rotations and scales are saved to.")
    alignment.add_argument("--documents", type=int, default=256, help="English documents read.")
    alignment.add_argument("--characters", type=int, default=2000, help="Characters of each document read.")
    alignment.add_argument(
        "--skip", type=int, default=100, help="Leading held out documents left out, the ones runs score."
    )
    alignment.add_argument("--appended-layers", type=int, default=4, help="Layers appended after Helium's own.")
    alignment.add_argument("--shared", action="store_true", help="One rotation and one scale for all layers.")
    alignment.add_argument("--output-dir", default=None, help="Directory the --output file name is written into.")

    args = parser.parse_args(None if args is None else [str(arg) for arg in args])
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.command == "residual-alignment":
        from safetensors.torch import save_file

        if args.output_dir:
            args.output = os.path.join(args.output_dir, os.path.basename(args.output))
        from transformers import AutoTokenizer

        from .phase import dense_documents

        tokenizer = AutoTokenizer.from_pretrained(args.language_model)
        budget = (args.skip + args.documents) * args.characters
        documents = dense_documents("en", "valid", budget, tokenizer)[args.skip : args.skip + args.documents]
        texts = [document[: args.characters] for document in documents]
        result = residual_alignment(
            args.personaplex,
            args.language_model,
            texts,
            appended_layers=args.appended_layers,
            device="cuda" if torch.cuda.is_available() else "cpu",
            shared=args.shared,
        )
        save_file({"rotations": result["rotations"].contiguous(), "scales": result["scales"]}, args.output)
        summary = {key: result[key] for key in ("report", "fitted", "measured")}
        with open(os.path.splitext(args.output)[0] + ".json", "w") as handle:
            json.dump(summary, handle, indent=1)
        print(json.dumps(summary, indent=1))
        return

    if args.command == "composition-check":
        ridges = tuple(float(value) for value in args.ridges.split(","))
        result = composition_check(
            args.personaplex, args.language_model, args.tokenizer, args.held_out, args.k, ridges=ridges
        )
        print(json.dumps(result, indent=1))
        return

    if args.command == "dualffn":
        layers = None
        if args.language_ffn_layers:
            layers = [int(index) for index in args.language_ffn_layers.split(",")]
        calibration_texts = None
        if args.tokenizer:
            from .phase import dense_calibration_texts

            calibration_texts = dense_calibration_texts(
                PolyplexTokenizer.from_pretrained(args.tokenizer), skip=args.calibration_skip
            )
        assemble_dualffn(
            args.personaplex,
            args.language_model,
            args.output_dir,
            layers,
            args.torch_dtype,
            tokenizer=args.tokenizer,
            calibration_texts=calibration_texts,
            composition_size=args.composition_size,
            appended_layers=args.appended_layers,
        )
        return

    assembler = DESSEMBLERS[args.phase](args.base_model, torch_dtype=args.torch_dtype)
    assembler.dessemble(args.adapter, output_dir=args.output_dir)
    logger.info("Phase %d merged into %s.", args.phase, args.output_dir)


if __name__ == "__main__":
    main()
