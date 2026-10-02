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


"""The remote code classes the released model loads with (SPEC S13.4c)."""


def runs_on_ephemeral_disk() -> bool:
    r"""
    Tells whether this process runs on a disk the run owns, which is lost when the session ends: Colab or Kaggle,
    marked by `POLYPLEX_EPHEMERAL_DISK=1`, which [`~polyplex_research.utils.phase.launch`] sets.

    Returns:
        `bool`: Whether the disk is the session's own.
    """
    return os.environ.get("POLYPLEX_EPHEMERAL_DISK") == "1"


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
