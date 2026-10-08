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
"""Phase 1 text injection adapter configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class EfficientAdapterP1Config:
    r"""
    Settings of the Phase 1 text injection adapter.

    Phase 1 adds no new parameter. The vocabulary expansion has already appended rows to the text tables,
    and this configuration only records which of those tables are trained and where the pretrained
    vocabulary ends, so that the rows the base model learned English on can be held fixed.

    The two shape fields are read off the base model and the expanded tokenizer by
    [`EfficientAdapterP1Config.from_base_model`], so a run only has to state the behavioural ones.

    Args:
        base_vocab_size (`int`):
            Number of leading rows that belong to the pretrained vocabulary. Their gradient is zeroed on
            every step, which is what keeps the English text space fixed while the new rows align to it.
        vocab_size (`int`):
            Vocabulary size after the expansion. Rows from `base_vocab_size` to here are the trainable ones.
        train_input_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the new rows of the backbone text input embedding are trained. Only possible on a model
            that ties this table to the output head, since [`EfficientAdapterP1Model.forward`] runs the
            backbone under `torch.no_grad` and the lookup happens inside that block. PersonaPlex keeps the
            two apart, the input table carries the reserved row and the head does not, so this is off by
            default and turning it on is rejected at assembly rather than silently training nothing.
        train_lm_head (`bool`, *optional*, defaults to `True`):
            Whether the new rows of the text output head are trained. This is the table Phase 1 leans on,
            since the objective is decoding into the new language.
        train_depth_text_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the new rows of the depth decoder's own text embedding are trained. The Phase 1 forward
            never enters the depth decoder, so this table belongs to Phase 2 and is likewise rejected here.
        train_rows (`bool`, *optional*, defaults to `True`):
            Whether the selected tables move at all. With `False` the appended rows are read as they were
            loaded and only the backbone corrections train, which needs at least one of `session_expert_layers`,
            `confined_basis`, `confined_attention_basis` or `lora_layers`.
        train_corrections (`bool`, *optional*, defaults to `True`):
            Whether the backbone corrections (`session_expert_layers`, `confined_basis`,
            `confined_attention_basis`, `lora_layers`) train. With `False` they stay as loaded and only the
            appended rows move, which is how the speaking phase keeps what the earlier phases stored there.
        base_row_mixture (`int`, *optional*, defaults to 0):
            How many pretrained rows each appended row is built out of. Text only training leaves the
            appended rows fitted to the states the backbone produces when nothing is being spoken, and
            those rows lose over two nats once speech fills the acoustic stream. The pretrained rows do
            not, since they were fitted together with the backbone and hold in both regimes, and a row
            built out of them inherits that. The coefficients are the parameter and the rows are a
            function of them, so the constraint holds at every point of the trajectory rather than being
            restored after each step; at the end the coefficients are multiplied out and the phase hands
            on an ordinary table. `0` leaves the rows free.
        share_row_mixture (`bool`, *optional*, defaults to `False`):
            Whether every trained table builds its appended rows from one set of coefficients, each over
            its own pretrained rows. A token is then the same blend of pretrained tokens in the head and
            in the depth decoder's text table, so a loss that reaches one table moves the other by value.
            That is how the audio objective, which only reaches the depth table because the backbone runs
            without a graph, still moves the head and, through the tie, the backbone's input. Needs
            `base_row_mixture`.
        partition_direction (`bool`, *optional*, defaults to `False`):
            Whether one free vector is added to every appended row of the head on top of its mixture. A
            row built from pretrained rows can only point where those rows point, and a Korean session
            and an English one are told apart along a direction none of them needs to span. The shared
            vector is that direction: its score is added to every appended logit alike, so it raises the
            whole partition on Korean states and lowers it on English ones without taking anything from
            what the mixtures say about individual pieces. It is part of the row values, so a folded
            checkpoint carries it as an ordinary head. Needs `base_row_mixture`.
        listen_mixture (`int`, *optional*, defaults to 0):
            How many codes of the same codebook each user code's correction is built out of. The user
            side has no partition to train in, every code is shared with English, so the correction is
            one set of coefficients read by value on both sides of the frozen backbone: the depth
            decoder's user code heads and input embeddings carry the gradient, and the backbone's user
            code embedding follows by value, the same way the tie moves the text input. `0` leaves the
            user side as it is.
        listen_basis (`str`, *optional*):
            Safetensors file with the `mean` and `basis` of the state subspace English occupies while the
            user speaks. The backbone side correction is scaled by how much of the previous state lies
            outside it, so it fades where the user is heard speaking English. Without it the correction
            applies everywhere.
        lora_layers (`int`, *optional*, defaults to 0):
            How many of the backbone's last layers carry a low rank correction. Word order and
            morphology live in the backbone, and a frozen English one has no reason to hold the Korean
            ones, which is what a correction there can supply. Counted from the top because the
            gradient only has to travel back as far as the lowest adapted layer, so the depth of the
            correction is also what the step costs. `0` leaves the backbone untouched and keeps the
            forward under `torch.no_grad`.
        confined_basis (`str`, *optional*):
            Safetensors file of fixed input directions per layer, as [`~polyplex_research.utils.evaluate.build_confined_basis`]
            writes it. Each listed layer's FFN gains a linear correction that reads only those
            directions, the ones deployment barely uses, so what it learns from text under a silent
            assistant stream leaves the duplex behaviour where it was. The
            forward then keeps a graph through the backbone from the lowest listed layer up.
        session_expert_layers (`str`, *optional*):
            Inclusive layer range such as `"8-31"`. Each FFN in it gains a low rank correction whose read
            and write sides are both trained over the whole input. It is meant to be switched on by the
            session language, which is given from outside: an English session runs with it off and is the
            backbone exactly, so only the Korean session's own timing needs a term to hold it.
        session_expert_rank (`int`, *optional*, defaults to 256):
            Rank of that correction.
        confined_attention_basis (`str`, *optional*):
            Same as `confined_basis`, for the value projection of each listed layer's attention, read
            from the attention input. Attention mixes the values across frames, so this correction can
            carry what earlier words say to later ones.
        confined_moments (`str`, *optional*):
            Directory [`~polyplex_research.utils.evaluate.CapacityMeter`] wrote, whose second moments let the confined corrections
            be scored on deployment inputs without a forward, see `EfficientAdapterP1Model.confined_leak`.
        lora_rank (`int`, *optional*, defaults to 16):
            Rank of that correction.
        lora_alpha (`float`, *optional*, defaults to 32.0):
            Scale of that correction, applied as `alpha / rank`.
        head_hidden_size (`int`, *optional*):
            Width of the hidden layer the appended rows read the state through. A frozen backbone
            hands the head one state per position, and a single matrix over that state can only score
            words by how well each one matches it, which is a topical judgement rather than a
            grammatical one. One hidden layer is what lets the appended rows condition on combinations
            of directions. The pretrained rows keep reading the state itself, so the extra capacity
            belongs to the appended vocabulary alone. `None` leaves the head linear.
        tie_new_input_rows (`bool`, *optional*, defaults to `False`):
            Whether the new rows of the input table follow the new rows of the head. The base model keeps
            the two tables apart, so nothing updates the new input rows and the model never learns to read
            the new language. Turning this on writes the trained head rows into the input table, which
            gives the one gradient Phase 1 has an effect on both sides of the vocabulary.
        tied_backbone_grad (`bool`, *optional*, defaults to `False`):
            Whether the backbone keeps its graph so the objective also reaches the tied rows through the
            reading direction. The tie already trains them without this, since the head and the input
            table are one tensor and one update serves both. This only adds the second signal, and it
            costs the activation memory the frozen forward otherwise avoids.
        tied_gradient_gain (`float`, *optional*, defaults to 1.44):
            Factor the reading contribution to the tied rows is multiplied by. The reading gradient is
            proportional to `tied_input_scale`, which cannot be raised because it is what keeps the input
            table inside the range the backbone was trained on. Splitting the row into a detached part and
            a differentiable part leaves the forward value untouched and scales only the gradient, so this
            is an exact reweighting of the reading direction rather than a change of representation.
            The default restores the step length the row would take if the backbone kept its graph: the
            writing and reading contributions were measured at 0.80 and 0.73 with a cosine of 0.13, so
            their sum is 1.15 and the writing half alone is 0.69 of it. Only the length is restored, since
            the reading direction is nearly orthogonal to the writing one and cannot be reconstructed
            from it.
        tied_input_scale (`float`, *optional*):
            Factor the head rows are multiplied by on their way into the input table. The two tables sit at
            different scales, so copying raw rows puts the input table outside the range the backbone was
            trained on. Measured from the pretrained rows of both tables when omitted.
    """

    base_vocab_size: int
    vocab_size: int
    train_input_embeddings: bool = False
    train_lm_head: bool = True
    train_depth_text_embeddings: bool = False
    train_rows: bool = True
    train_corrections: bool = True
    shared_token_residual: str | None = None
    residual_all_frozen_rows: bool = False
    project_new_rows: bool = False
    group_gated_softmax: bool = False
    group_prior: float = 0.559
    base_row_mixture: int = 0
    share_row_mixture: bool = False
    partition_direction: bool = False
    listen_mixture: int = 0
    listen_basis: str | None = None
    float32_rows: bool = True
    lora_layers: int = 0
    lora_rank: int = 16
    confined_basis: str | None = None
    confined_attention_basis: str | None = None
    session_expert_layers: str | None = None
    session_expert_rank: int = 256
    confined_moments: str | None = None
    lora_alpha: float = 32.0
    head_hidden_size: int | None = None
    tie_new_input_rows: bool = True
    tied_backbone_grad: bool = False
    tied_gradient_gain: float = 1.44
    tied_input_scale: float | None = None

    def to_dict(self) -> dict:
        r"""
        Returns:
            `dict`: Every field and its value, which is what the `Trainer` logging integrations read.
        """
        return asdict(self)

    @property
    def num_new_rows(self) -> int:
        r"""
        Returns:
            `int`: Number of appended rows, which is what Phase 1 actually trains.
        """
        return self.vocab_size - self.base_vocab_size

    @classmethod
    def from_base_model(cls, model, base_vocab_size: int, **kwargs) -> "EfficientAdapterP1Config":
        r"""
        Args:
            model (`PreTrainedModel`):
                The expanded duplex model the adapter will be assembled onto.
            base_vocab_size (`int`):
                Vocabulary size before the expansion.
            kwargs:
                Behavioural fields overriding the defaults.

        Returns:
            `EfficientAdapterP1Config`: A configuration whose shapes match `model`.

        Raises:
            ValueError: If the model was never expanded, so that there is nothing to train.
        """
        vocab_size = model.config.vocab_size
        if vocab_size <= base_vocab_size:
            raise ValueError(
                f"The model carries {vocab_size} text rows and `base_vocab_size` is {base_vocab_size}, so "
                "the vocabulary was never expanded and Phase 1 has no trainable row. Run the vocabulary "
                "expansion first."
            )
        return cls(base_vocab_size=base_vocab_size, vocab_size=vocab_size, **kwargs)


__all__ = ["EfficientAdapterP1Config"]
