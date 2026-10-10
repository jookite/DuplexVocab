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
"""PyTorch Phase 2 acoustic adapters, assembled onto a frozen duplex model.

Both arms of the ablation live here: the code space rotation the method rests on, applied through
`parametrize`, and the temporal space tied residual, applied through forward hooks. Their shared
objective terms are in `polyplex_research.utils.loss`."""

import os
from dataclasses import asdict, fields
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from huggingface_hub import ModelHubMixin, hf_hub_download
from safetensors.torch import load_file, save_file
from transformers.modeling_outputs import CausalLMOutputWithPast

from ...utils.loss import orthogonal_penalty, partial_isometry_penalty
from .configuration_phase_2_code_rotation_adapter import CodeRotationAdapterConfig


ADAPTER_WEIGHTS_NAME = "adapter.safetensors"

_CONFIG_FIELDS = {field.name for field in fields(CodeRotationAdapterConfig)}


class TiedAcousticAdapter(nn.Module):
    r"""
    Constructs the residual acoustic weight that is shared between listening and speaking.

    One residual `delta_k` of shape `(audio_vocab_size, hidden_size)` is kept per stream codebook. On the
    listening side it is added row by row to the frozen audio input embeddings, so hearing code `c` writes
    `delta_k[c]` into the residual stream. On the speaking side the same matrix maps the temporal hidden state
    to a logit skip term `hidden @ delta_k.T` that is added to the depth decoder head of codebook `k`. Both
    directions therefore use exactly one matrix and its transpose.

    With `gated=True` the residual is factored as `gate[k] * q[k]` where `q[k]` starts row orthonormal and
    `gate[k]` starts at zero, so the model is identical to the frozen base at initialization while the
    orthogonal penalty keeps `q[k]` a partial isometry. With `gated=False` the residual starts at zero and the
    penalty acts on it directly.

    Args:
        num_codebooks (`int`):
            Number of codebooks of one audio stream.
        audio_vocab_size (`int`):
            Number of acoustic codes per codebook. The trailing special row of the embeddings is never touched.
        hidden_size (`int`):
            Dimension of the temporal transformer, shared by the input embeddings and the speaking skip term.
        gated (`bool`, *optional*, defaults to `True`):
            Whether to use the gated factorization.
    """

    def __init__(self, num_codebooks: int, audio_vocab_size: int, hidden_size: int, gated: bool = True):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.audio_vocab_size = audio_vocab_size
        self.hidden_size = hidden_size
        self.gated = gated

        q = torch.empty(num_codebooks, audio_vocab_size, hidden_size)
        if gated:
            for codebook in range(num_codebooks):
                nn.init.orthogonal_(q[codebook])
            self.q = nn.Parameter(q)
            self.gate = nn.Parameter(torch.zeros(num_codebooks))
        else:
            self.q = nn.Parameter(torch.zeros_like(q))
            self.register_buffer("gate", torch.ones(num_codebooks), persistent=False)

    def delta(self, codebook: int) -> torch.Tensor:
        r"""
        Args:
            codebook (`int`):
                Stream codebook index.

        Returns:
            `torch.Tensor` of shape `(audio_vocab_size, hidden_size)`: The residual of that codebook.
        """
        return self.gate[codebook] * self.q[codebook]

    def listen(self, audio_codes: torch.LongTensor, codebook: int) -> torch.Tensor:
        r"""
        Args:
            audio_codes (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Codes of one codebook. Values equal to `audio_vocab_size` address the special row and get a zero term.
            codebook (`int`):
                Stream codebook index.

        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`: The additive listening term.
        """
        valid = (audio_codes < self.audio_vocab_size).unsqueeze(-1)
        term = F.embedding(audio_codes.clamp(max=self.audio_vocab_size - 1), self.delta(codebook))
        return term * valid.to(term.dtype)

    def speak(self, hidden: torch.Tensor, codebook_idx: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            hidden (`torch.Tensor` of shape `(batch_size, hidden_size)`):
                Temporal hidden state of one frame.
            codebook_idx (`torch.Tensor` of shape `(num_positions,)`):
                Depth decoder positions. Position `i` addresses stream codebook `i % num_codebooks`.

        Returns:
            `torch.Tensor` of shape `(batch_size, num_positions, audio_vocab_size)`: The logit skip term.
        """
        delta = (self.gate[:, None, None] * self.q).index_select(0, codebook_idx % self.num_codebooks)
        return torch.einsum("bd,pcd->bpc", hidden.to(delta.dtype), delta).to(hidden.dtype)

    def penalty(self) -> torch.Tensor:
        r"""
        Returns:
            `torch.Tensor`: `sum_k ||M_k M_k^T - I||_F^2` where `M_k` is `q[k]` when gated and `delta_k` otherwise.
        """
        return partial_isometry_penalty(self.q if self.gated else self.gate[:, None, None] * self.q)


class SpeakingRotation(nn.Module):
    r"""
    Rotates the codebook axis of the speaking heads with $I + \Delta W$.

    The depth decoder stacks one head per predicted codebook into a single weight of shape
    `(num_codebooks, audio_vocab_size, hidden_size)`, and every one of them writes a code index that
    something later reads back. They are therefore all rotated, the user stream heads included: the
    frames a caller leaves open are filled from those heads and fed back through the audio embeddings,
    so a head left on the frozen convention would emit an index the rotated embedding then misreads.

    The two channels keep their roles regardless. `W_self` and `W_user` stay separate frozen tensors and
    only the acoustic relabelling on top of them is shared, which is the split the method asks for.

    Args:
        delta_w (`torch.nn.Parameter` of shape `(audio_vocab_size, audio_vocab_size)`):
            Residual weight shared with the listening embeddings.
    """

    def __init__(self, delta_w: nn.Parameter):
        super().__init__()
        # Held in a list so that the shared parameter is registered once, on the model that owns it,
        # instead of once per parametrization, which would make it a shared tensor at save time.
        self._delta_w = [delta_w]

    @property
    def delta_w(self) -> nn.Parameter:
        return self._delta_w[0]

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            weight (`torch.Tensor` of shape `(num_codebooks, audio_vocab_size, hidden_size)`):
                The frozen per-codebook head weight.

        Returns:
            `torch.Tensor`: The head weight with every codebook rotated.
        """
        # `(I + dW) W` is evaluated as `W + dW W`, so no value ever passes through the neighbourhood
        # of 1 in the storage dtype. A bfloat16 identity quantizes every residual below 2^-9 away.
        return weight + torch.matmul(self.delta_w.to(weight.dtype), weight)


class ListeningRotation(nn.Module):
    r"""
    Rotates the codebook axis of an audio embedding with $I + \Delta W$.

    The audio embedding holds one row per acoustic code plus a trailing padding row,
    which is left untouched. Reading code $c$ through the rotated table returns
    $\sum_j (I + \Delta W)_{cj} E_j$, so at a permutation $P$ it returns $E_{\pi(c)}$,
    exactly the row the speaking rotation moved onto index $c$. The same transform therefore
    inverts the head wherever a code is read back, which is every audio embedding the model
    holds: the user channel it listens to, and the assistant channel it feeds its own output into.

    The rotation is applied with a detached $\Delta W$. Reading is what the method claims to get for
    free, not what it optimizes: the same tensor is trained through the speaking head, and the update it
    receives there is what the next forward reads with.

    Args:
        delta_w (`torch.nn.Parameter` of shape `(audio_vocab_size, audio_vocab_size)`):
            Residual weight shared with the speaking head.
        audio_vocab_size (`int`):
            Number of leading rows that hold acoustic codes.
    """

    def __init__(self, delta_w: nn.Parameter, audio_vocab_size: int):
        super().__init__()
        self._delta_w = [delta_w]
        self.audio_vocab_size = audio_vocab_size

    @property
    def delta_w(self) -> nn.Parameter:
        return self._delta_w[0]

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            weight (`torch.Tensor` of shape `(audio_vocab_size + 1, hidden_size)`):
                The frozen audio embedding weight.

        Returns:
            `torch.Tensor`: The embedding weight with the acoustic rows rotated.
        """
        # `delta_w` is detached here and only here. The rotation still applies in full on the listening
        # side, it simply carries no graph, so the residual is trained through the speaking head alone.
        delta_w = self.delta_w.detach().to(weight.dtype)
        listening = weight[: self.audio_vocab_size]
        listening = listening + torch.matmul(delta_w, listening)
        if self.audio_vocab_size == weight.shape[0]:
            return listening
        return torch.cat([listening, weight[self.audio_vocab_size :]], dim=0)


def audio_embedding_streams(model) -> list[tuple[nn.Module, bool]]:
    r"""
    Collects every table that turns an acoustic code back into a vector, tagged by the stream it reads.

    A duplex model holds two such families. The backbone keeps `2 * num_codebooks` embeddings, the
    assistant stream first and the user stream second, one per codebook of each channel. The depth
    decoder keeps its own, one per codebook that can appear as within-frame context, which is every
    predicted codebook but the last; position `j` of that list carries codebook `j`, so the same index
    rule names its stream.

    Args:
        model (`PolyplexForConditionalGeneration`):
            The duplex model whose audio embeddings are collected.

    Returns:
        `list`: Entries of `(module, is_assistant)`.
    """
    num_codebooks = model.num_codebooks
    tables = [
        (embedding, index < num_codebooks) for index, embedding in enumerate(model.embed_tokens)
    ]
    tables += [
        (embedding, index < num_codebooks)
        for index, embedding in enumerate(model.depth_decoder.model.embed_tokens)
    ]
    return tables


def attach_residual_tying(
    model, delta_w: nn.Parameter | None = None, tie_self_input: bool = True
) -> nn.Parameter:
    r"""
    Binds one shared residual weight to the speaking head and to every table that reads a code back.

    The backbone and the depth decoder do not share a hidden size, so the axis the two
    paths have in common is the acoustic code vocabulary. $\Delta W$ therefore lives
    in that space, which is what makes the rotation a true inverse instead of a shape
    coincidence.

    The head rewrites what an emitted index means: at a permutation $P$, index $c$ now carries what the
    frozen model called $\pi(c)$. Every table that turns an index back into a vector therefore has to be
    rotated by the same $I + \Delta W$, or it reads the new index under the old meaning. That is true of
    the user channel, which hears the codes the method transfers, and equally of the assistant channel,
    which reads back the codes this very head emitted. `tie_self_input` is what decides whether the
    assistant side is kept consistent with the head or deliberately left on the frozen convention for the
    ablation that measures what the consistency is worth.

    Args:
        model (`PolyplexForConditionalGeneration`):
            The loaded duplex model, modified in place.
        delta_w (`torch.nn.Parameter` of shape `(audio_vocab_size, audio_vocab_size)`, *optional*):
            An existing residual to bind. When omitted a zero initialized one is created and registered
            on `model`, which is the form a bare training loop wants. When given, ownership stays with
            the caller and nothing is registered here, which is what [`CodebookRotationAdapter`] needs so
            that the parameter is not saved twice.
        tie_self_input (`bool`, *optional*, defaults to `True`):
            Whether the assistant stream tables are rotated as well. The user stream tables always are.

    Returns:
        `torch.nn.Parameter`: The bound residual weight.
    """
    audio_vocab_size = model.config.audio_vocab_size

    if delta_w is None:
        reference = model.depth_decoder.lm_heads.weight
        delta_w = nn.Parameter(
            torch.zeros(audio_vocab_size, audio_vocab_size, dtype=torch.float32, device=reference.device)
        )
        model.register_parameter("delta_w", delta_w)

    parametrize.register_parametrization(
        model.depth_decoder.lm_heads, "weight", SpeakingRotation(delta_w), unsafe=True
    )

    for embedding, is_assistant in audio_embedding_streams(model):
        if is_assistant and not tie_self_input:
            continue
        parametrize.register_parametrization(
            embedding, "weight", ListeningRotation(delta_w, audio_vocab_size), unsafe=True
        )

    return delta_w


class CodebookRotationAdapter(nn.Module):
    r"""
    Owns the code space residual $\Delta W$ and binds it to a duplex model.

    This is the module form of [`attach_residual_tying`]. It exists so that the rotation kind presents
    the same surface as [`TiedAcousticAdapter`]: one module that holds the trainable tensors, reports the
    penalty and can be saved on its own, which is what lets a single Phase 2 model class carry either
    residual.

    Args:
        audio_vocab_size (`int`):
            Number of acoustic codes per codebook, the dimension the rotation acts on.
        nonneg_beta (`float`, *optional*, defaults to 10.0):
            Scale of the penalty on the negative entries of the rotation, which drives it to a permutation.
    """

    def __init__(self, audio_vocab_size: int, nonneg_beta: float = 10.0):
        super().__init__()
        self.audio_vocab_size = audio_vocab_size
        self.nonneg_beta = nonneg_beta
        self.delta_w = nn.Parameter(torch.zeros(audio_vocab_size, audio_vocab_size, dtype=torch.float32))

    def attach(self, model, tie_self_input: bool = True):
        r"""
        Registers the speaking and listening parametrizations on the base model.

        Args:
            model (`PolyplexForConditionalGeneration`):
                The duplex model, modified in place.
            tie_self_input (`bool`, *optional*, defaults to `True`):
                Whether the assistant stream tables are rotated alongside the user stream ones.
        """
        self.delta_w.data = self.delta_w.data.to(model.depth_decoder.lm_heads.weight.device)
        attach_residual_tying(model, delta_w=self.delta_w, tie_self_input=tie_self_input)

    def penalty(self) -> torch.Tensor:
        r"""
        Returns:
            `torch.Tensor`: The unscaled orthogonal penalty of the rotation.
        """
        return orthogonal_penalty(self.delta_w, self.nonneg_beta)


def freeze_for_audio_injection(model, delta_w: nn.Parameter):
    r"""
    Freezes every pretrained weight and leaves only the residual weight trainable.

    Freezing the temporal transformer is what preserves the English turn taking
    behaviour, and freezing the original per channel weights keeps the user and
    assistant identities intact while the shared residual carries the new phonetics.

    Args:
        model (`PolyplexForConditionalGeneration`):
            The model whose parameters are frozen.
        delta_w (`torch.nn.Parameter`):
            The residual weight that stays trainable.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    delta_w.requires_grad_(True)

class CodeRotationAdapterModel(
    nn.Module,
    ModelHubMixin,
    library_name="polyplex-research",
    tags=["polyplex", "phase-2", "audio-injection"],
    coders={
        CodeRotationAdapterConfig: (
            asdict,
            lambda stored: CodeRotationAdapterConfig(
                **{name: value for name, value in stored.items() if name in _CONFIG_FIELDS}
            ),
        )
    },
):
    r"""
    Assembles the Phase 2 acoustic adapter onto a loaded duplex model.

    The class carries either residual, selected by `config.adapter_kind`, so that everything downstream of
    assembly, the trainer and the checkpoint layout included, is the same for both arms of the ablation.
    Nothing about the base model class changes either way: the training structure stays out of the model
    definition, so any `PersonaPlexForConditionalGeneration` and its subclasses can be adapted.

    With `adapter_kind="rotation"`, the method's own path, a [`CodebookRotationAdapter`] binds one code
    space residual to the depth heads and to the audio embeddings through `parametrize`, so the rotation
    is applied to the weights themselves. The head decides what an emitted index means and every table
    that reads an index back is rotated with it, the user channel always and the assistant channel under
    `tie_self_input`.

    With `adapter_kind="tied"`, kept for the ablation, a [`TiedAcousticAdapter`] is attached as forward
    hooks instead:

    - Listening. A hook on each audio input embedding adds `adapter.listen` to the looked up rows. The user
      stream embeddings, which occupy the second half of the list, always receive it; the assistant ones only
      when `tie_self_input` is set.
    - Speaking. A hook on `depth_decoder.lm_heads` adds `adapter.speak` to the logits. It fires inside the
      depth decoder before the loss is taken, so the training objective already sees the adapted logits. The
      temporal hidden state the skip term starts from is captured by a pre-hook on the depth decoder itself,
      which is the only place it is available.

    Only the speaking side carries a gradient. [`forward`] runs the temporal transformer under
    `torch.no_grad` and detaches its hidden states, and the listening contribution is detached in both
    kinds on top of that, so the backbone stores no activations either way. The depth call is the one
    part left outside the block, because it is where the residual actually enters the graph.

    Training one direction and reading the other back is the claim of the method rather than a shortcut:
    the two directions are one matrix, so the step taken on the speaking head is what the next forward
    hears with, and the orthogonal penalty is what makes that transpose an inverse.

    The orthogonal penalty is deliberately not folded into `forward`. It belongs to the objective, so the
    training loop reads it from [`penalty`] and scales it with `config.ortho_lambda`.

    Args:
        model (`PreTrainedModel`):
            The duplex model to adapt, modified in place.
        config (`CodeRotationAdapterConfig`, *optional*):
            Adapter settings. Derived from `model` when omitted.
    """

    def __init__(self, model, config: CodeRotationAdapterConfig | None = None):
        super().__init__()
        self.config = config if config is not None else CodeRotationAdapterConfig.from_base_model(model)
        self.model = model
        if self.config.adapter_kind == "tied":
            self.adapter = TiedAcousticAdapter(
                num_codebooks=self.config.num_codebooks,
                audio_vocab_size=self.config.audio_vocab_size,
                hidden_size=self.config.hidden_size,
                gated=self.config.gated,
            )
        else:
            self.adapter = CodebookRotationAdapter(
                audio_vocab_size=self.config.audio_vocab_size,
                nonneg_beta=self.config.nonneg_beta,
            )
        # Refreshed by the depth decoder pre-hook on every frame, read by the lm_heads hook of that same call.
        self._frame_hidden = None
        self._handles = []
        self._attach()

    # -----------------------------------------------------------------
    # Assembly
    # -----------------------------------------------------------------

    def _attach(self):
        r"""
        Binds the residual to the base model, the way its kind requires.

        The rotation kind rewrites the weights through `parametrize` and registers no hook, so it leaves
        `_handles` empty and [`detach_adapter`] has nothing to undo for it.
        """
        if self.config.adapter_kind == "rotation":
            self.adapter.attach(self.model, tie_self_input=self.config.tie_self_input)
            return

        num_codebooks = self.config.num_codebooks

        for index, embedding in enumerate(self.model.embed_tokens):
            is_user = index >= num_codebooks
            if not is_user and not self.config.tie_self_input:
                continue
            self._handles.append(embedding.register_forward_hook(self._make_listen_hook(index % num_codebooks)))

        depth_decoder = self.model.depth_decoder
        self._handles.append(depth_decoder.register_forward_pre_hook(self._capture_frame_hidden, with_kwargs=True))
        self._handles.append(depth_decoder.lm_heads.register_forward_hook(self._speak_hook))

    def _make_listen_hook(self, codebook: int):
        r"""
        Args:
            codebook (`int`):
                Stream codebook the embedding belongs to.

        Returns:
            `Callable`: A forward hook adding the listening term to the looked up embeddings.
        """

        def hook(module, args, output):
            codes = args[0]
            # Detached on purpose. See the class docstring: the residual is trained through the speaking
            # side alone, and a graph here would hold the whole temporal transformer's activations.
            term = self.adapter.listen(codes.to(self.adapter.q.device), codebook).detach()
            return output + term.to(output.device, output.dtype)

        return hook

    def _capture_frame_hidden(self, module, args, kwargs):
        r"""
        Stores the temporal hidden state the depth decoder is conditioned on.

        Args:
            module (`torch.nn.Module`):
                The depth decoder.
            args (`tuple`):
                Positional arguments of its forward.
            kwargs (`dict`):
                Keyword arguments of its forward.
        """
        hidden = kwargs.get("last_hidden_state")
        if hidden is None and len(args) > 1:
            hidden = args[1]
        self._frame_hidden = hidden
        return None

    def _speak_hook(self, module, args, output):
        r"""
        Args:
            module (`torch.nn.Module`):
                The per-codebook head stack of the depth decoder.
            args (`tuple`):
                Its hidden states and codebook indices.
            output (`torch.Tensor`):
                The logits of the frozen heads.

        Returns:
            `torch.Tensor`: The logits with the speaking skip term added.
        """
        if self._frame_hidden is None:
            raise RuntimeError(
                "The depth decoder ran without a `last_hidden_state`, so the speaking side of the adapter has "
                "nothing to project. Call the base model rather than the depth decoder on its own."
            )
        codebook_idx = args[1]
        term = self.adapter.speak(self._frame_hidden[:, 0].to(self.adapter.q.device), codebook_idx)
        return output + term.to(output.device, output.dtype)

    def detach_adapter(self):
        r"""
        Removes every hook, returning the base model to its frozen behaviour.

        The rotation kind registers no hook, so this is a no-op for it and the parametrization stays.
        """
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def fold_into_base(self):
        r"""
        Writes the residual into the weights it is bound to and returns the plain duplex model.

        The rotation acts on the weights themselves, so evaluating each parametrization once and storing
        its result leaves a checkpoint that any `PersonaPlexForConditionalGeneration` loads and generates
        with, carrying no trace of the adapter.

        The tied kind cannot be folded. Its speaking term is a skip connection from the temporal hidden
        state to the depth logits, which bypasses the depth transformer entirely, so no depth head weight
        reproduces it. That arm of the ablation is evaluated with the adapter attached.

        Returns:
            `PolyplexForConditionalGeneration`: The base model, modified in place.

        Raises:
            ValueError: If the adapter kind carries a term no weight can hold.
        """
        if self.config.adapter_kind != "rotation":
            raise ValueError(
                f"A '{self.config.adapter_kind}' residual has no weight space form: its speaking term "
                "reads the temporal hidden state, which the depth heads never see. Only "
                "`adapter_kind='rotation'` folds into a plain checkpoint."
            )

        self.detach_adapter()
        # Collected before the loop, since removing a parametrization edits the module tree being walked.
        parametrized = [
            module for module in self.model.modules() if parametrize.is_parametrized(module, "weight")
        ]
        for module in parametrized:
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
        return self.model

    # -----------------------------------------------------------------
    # Training surface
    # -----------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.BoolTensor | None = None,
        user_audio_codes: torch.Tensor | None = None,
        assistant_audio_codes: torch.Tensor | None = None,
        audio_labels: torch.LongTensor | None = None,
        text_labels: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        r"""
        Runs the frozen temporal transformer without a graph and the depth path with one.

        The residual is trained through the speaking side, which lives in the depth decoder, so the
        temporal transformer is pure inference here: it is called under `torch.no_grad` and its hidden
        states are detached before the depth path sees them. That is where nearly all of the activation
        memory of a 7B duplex model would otherwise sit, and none of it is needed to differentiate a
        residual that only enters after the frame is decoded.

        The depth call is deliberately left outside that block. It is the one place the adapter carries a
        gradient, through the logit skip term of the tied kind or through the parametrized head weight of
        the rotation kind. The depth transformer's own weights are frozen and its input is detached, so
        autograd still records nothing but the residual itself.

        The label convention is fixed here rather than taken from the base model. The fork ties the depth
        loss to `text_labels`, does not shift the depth target by a frame, so the model is scored on codes
        it was just conditioned on, and its text cross entropy is `NaN` when every text label is masked,
        which is exactly the audio-only case this phase runs in. This method conditions the depth decoder
        on frame `t` and scores it on frame `t + 1`, the convention of the original `LMModel.forward_train`.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Text tokens of every frame.
            attention_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask of the temporal sequence.
            user_audio_codes (`torch.Tensor` of shape `(batch_size, num_codebooks, sequence_length)`):
                User stream codes of every frame, in the delay layout the model generates with.
            assistant_audio_codes (`torch.Tensor` of shape `(batch_size, num_codebooks, sequence_length)`):
                Assistant stream codes of every frame, in the same layout.
            audio_labels (`torch.LongTensor` of shape `(batch_size, num_codebooks, sequence_length)`, *optional*):
                Assistant codes aligned with `assistant_audio_codes`. The shift by one frame happens here.
                Values of `-100` and of `audio_vocab_size` are ignored.
            text_labels (`torch.LongTensor`, *optional*):
                Accepted and ignored. Phase 2 moves no text weight, so scoring the text stream would only
                add frozen compute.
            past_key_values (`Cache`, *optional*):
                Cached keys and values of the temporal transformer.
            use_cache (`bool`, *optional*):
                Whether the temporal transformer returns its cache.

        Returns:
            `CausalLMOutputWithPast`: The depth logits and, when `audio_labels` is given, their cross
            entropy. The orthogonal penalty is not included, see [`penalty`].

        Raises:
            ValueError: If the audio streams are not both given, or `audio_labels` arrives without the
                conditioning it needs.
        """
        if assistant_audio_codes is None or user_audio_codes is None:
            raise ValueError("Both `assistant_audio_codes` and `user_audio_codes` must be given together.")

        base = self.model
        with torch.no_grad():
            audio_codes = torch.cat([assistant_audio_codes, user_audio_codes], dim=1)
            inputs_embeds = base._embed_audio_codes(audio_codes)
            if input_ids is not None:
                text_embeds = base.model.embed_tokens(input_ids)
                inputs_embeds = inputs_embeds + text_embeds.to(inputs_embeds.device)

            decoder_outputs = base.model(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        # Detached explicitly rather than relying on the `no_grad` block alone, so that the boundary
        # survives a caller that re-enables grad around this method.
        hidden = decoder_outputs.last_hidden_state.detach()

        if audio_labels is None:
            return CausalLMOutputWithPast(
                loss=None,
                logits=None,
                past_key_values=decoder_outputs.past_key_values,
            )

        if input_ids is None:
            raise ValueError("`audio_labels` needs `input_ids`, the depth decoder conditions on the frame text.")

        audio_vocab_size = base.config.audio_vocab_size
        num_codebooks = audio_labels.shape[1]

        targets = audio_labels[:, :, 1:].transpose(1, 2).reshape(-1, num_codebooks)
        text_condition = input_ids[:, 1:].reshape(-1, 1)
        code_condition = assistant_audio_codes[:, :, 1:].transpose(1, 2).reshape(-1, num_codebooks)
        code_condition = code_condition.masked_fill(code_condition == -100, audio_vocab_size)
        depth_input_ids = torch.cat([text_condition, code_condition], dim=1)[:, :num_codebooks]
        depth_hidden = hidden[:, :-1].reshape(-1, 1, hidden.shape[-1])

        depth_outputs = self.model.depth_decoder(
            last_hidden_state=depth_hidden,
            input_ids=depth_input_ids,
        )
        logits = depth_outputs.logits if hasattr(depth_outputs, "logits") else depth_outputs

        # Normalized by its own count of scored positions, so a batch whose targets are all masked
        # contributes zero rather than `NaN`.
        flat_targets = targets.masked_fill(targets == audio_vocab_size, -100).to(logits.device)
        loss = F.cross_entropy(
            logits.float().reshape(-1, audio_vocab_size),
            flat_targets.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        ) / (flat_targets != -100).sum().clamp(min=1)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
        )

    def penalty(self) -> torch.Tensor:
        r"""
        Returns:
            `torch.Tensor`: The orthogonal penalty of the residual, unscaled.
        """
        return self.adapter.penalty()

    def adapter_parameters(self) -> list:
        r"""
        Returns:
            `list` of `torch.nn.Parameter`: The trainable parameters of the adapter.
        """
        return list(self.adapter.parameters())

    def freeze_base(self):
        r"""
        Freezes every pretrained weight and leaves only the adapter trainable.
        """
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for parameter in self.adapter_parameters():
            parameter.requires_grad_(True)

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
        Writes only the adapter tensors. The frozen base model is not duplicated.

        This is the hook `ModelHubMixin.save_pretrained` calls once it has created the directory and
        written `config.json`, so the configuration round trip and the upload are not repeated here.

        Args:
            save_directory (`Path`):
                Directory the weights are written to.
        """
        tensors = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in self.adapter.state_dict().items()
        }
        save_file(tensors, os.path.join(save_directory, ADAPTER_WEIGHTS_NAME))

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
        config: CodeRotationAdapterConfig | None = None,
        model=None,
        **kwargs,
    ) -> "CodeRotationAdapterModel":
        r"""
        Assembles a stored adapter onto a base model.

        This is the hook `ModelHubMixin.from_pretrained` calls once it has read `config.json` and decoded
        it back into an [`CodeRotationAdapterConfig`], so a local directory and a Hub repository reach the
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
            config (`CodeRotationAdapterConfig`, *optional*):
                Decoded configuration, supplied by the mixin.
            model (`PreTrainedModel`):
                The duplex model the adapter is attached to, modified in place.

        Returns:
            `CodeRotationAdapterModel`: The assembled model.

        Raises:
            ValueError: If `model` is missing.
        """
        if model is None:
            raise ValueError(
                "Phase 2 stores the residual alone, not a whole model, so loading needs the duplex model "
                "to assemble it onto. Pass `model=...`."
            )

        if os.path.isdir(model_id):
            weights_path = os.path.join(model_id, ADAPTER_WEIGHTS_NAME)
        else:
            weights_path = hf_hub_download(
                repo_id=model_id,
                filename=ADAPTER_WEIGHTS_NAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
            )

        assembled = cls(model, config)
        tensors = load_file(weights_path)
        # Read off a parameter rather than a named one, since the two adapter kinds hold different tensors.
        device = next(assembled.adapter.parameters()).device
        assembled.adapter.load_state_dict({name: tensor.to(device) for name, tensor in tensors.items()})
        return assembled


__all__ = [
    "ADAPTER_WEIGHTS_NAME",
    "CodebookRotationAdapter",
    "CodeRotationAdapterModel",
    "ListeningRotation",
    "SpeakingRotation",
    "TiedAcousticAdapter",
    "attach_residual_tying",
    "audio_embedding_streams",
    "freeze_for_audio_injection",
]
