#!/usr/bin/env bash
#
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
#
# Driver of the two phase pipeline: vocabulary expansion, dataset builds and the two trainings.
#
# Usage:
#   ./train.sh <stage> [extra arguments forwarded to the stage command]
#
# Stages:
#   assemble Assemble the DualFFN checkpoint from the base checkpoint and the text LLM (Qwen3-8B).
#   dense    Quick check of DualFFN before v2 (SPEC S13.4): dense adaptation of the language side norms,
#            run on DENSE_DEVICE, with the result saved under $RUN_DIR/dual-ffn.
#   dense-eval  Score a dense checkpoint (--checkpoint <run> --run-name <name>) on DENSE_DEVICE, with the result
#            saved as $RUN_DIR/<name>-eval.json.
#   score    Score a run on the Korean validation set and the held out English speech.
#   expand   Expand the text vocabulary of the base checkpoint, writing the Phase 1 input.
#   data1    Build and freeze the translation and instruction mixture Phase 1 reads.
#   english  Cache the backbone states Phase 1 keeps English on, read by P1_ENGLISH_LAMBDA.
#   phase1   Text injection. Trains the new text rows on the expanded checkpoint.
#   merge1   Fold the Phase 1 rows back into a checkpoint, which is what Phase 2 loads.
#   data2    Write the Phase 2 speech caches out of an existing KsponSpeech code cache.
#   phase2   Speaking. Continues the Phase 1 rows with the assistant's speech scored through the
#            depth decoder, the text corpus kept, and English preserved.
#   merge2   Fold the Phase 2 rows back into the expanded checkpoint.
#   all      Every stage above, in order.
#
# The code index rotation Phase 2 was first planned around is kept as a comparison arm under
# `cache`, `data2_rotation`, `phase2_rotation` and `merge2_rotation`.
#
# A phase saves only the tensors it moved, so the merge stage after it is what turns that slice into
# a checkpoint `from_pretrained` reads. Skipping one leaves the next stage with nothing to load.
#
# Every path and hyperparameter below is an environment variable, so a run is reproduced by
# exporting the same values. Arguments after the stage name reach the underlying command and
# win over the defaults set here.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Machine-local settings, such as LETIFY_HOME, live in an uncommitted .env next to this script.
if [[ -f "$ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    . "$ROOT/.env"
    set +a
fi

if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
else
    PYTHON="${PYTHON:-python}"
fi

# Weights, corpora, codec downloads and the stored Hub token all land here.
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
# The host's PYTHONPATH points at another interpreter's site-packages, which would be read before the
# environment's own packages.
unset PYTHONPATH

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_MODEL="${BASE_MODEL:-nvidia/personaplex-7b-v1}"
RUN_DIR="${RUN_DIR:-$HF_HOME/runs}"

VOCAB_SIZE="${VOCAB_SIZE:-24000}"
INIT_METHOD="${INIT_METHOD:-mean}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
QUANTIZATION="${QUANTIZATION:-}"

EXPANDED_DIR="${EXPANDED_DIR:-$RUN_DIR/expanded}"
# The text LLM whose FFNs, text embedding and head the DualFFN checkpoint takes, and, per backbone layer, the
# text LLM layer its FFN comes from (comma separated, the first layers when empty).
LANGUAGE_MODEL="${LANGUAGE_MODEL:-Qwen/Qwen3-8B}"
LANGUAGE_FFN_LAYERS="${LANGUAGE_FFN_LAYERS:-}"
DUALFFN_DIR="${DUALFFN_DIR:-$RUN_DIR/dualffn}"
SCORE_MODEL="${SCORE_MODEL:-$EXPANDED_DIR}"
SCORE_ENGLISH_SPEECH="${SCORE_ENGLISH_SPEECH:-$RUN_DIR/data/english_heldout_speech.pt}"
SCORE_KOREAN="${SCORE_KOREAN:-$RUN_DIR/data/phase1_valid_duplex_v2}"
SCORE_EXCLUDE_TRAIN="${SCORE_EXCLUDE_TRAIN:-$RUN_DIR/data/phase1_duplex_v2}"
PHASE1_DATA_DIR="${PHASE1_DATA_DIR:-$RUN_DIR/data/phase1}"
# The pieces the vocabulary grows by are learned from this corpus, which defaults to the phase 1
# mixture and therefore has to be named after that directory is known.
CORPUS_DATASET="${CORPUS_DATASET:-$PHASE1_DATA_DIR}"
# Held out splits. A run without one reports only the loss on what it is training on.
PHASE1_ADAPTER_DIR="${PHASE1_ADAPTER_DIR:-}"
PHASE1_EVAL_DIR="${PHASE1_EVAL_DIR:-}"
PHASE2_EVAL_DIR="${PHASE2_EVAL_DIR:-}"
EVAL_STEPS="${EVAL_STEPS:-500}"
# Stops a run once the held out loss stops moving, so a long one is not held open for nothing.
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
EARLY_STOPPING_THRESHOLD="${EARLY_STOPPING_THRESHOLD:-0.0}"
PHASE2_DATA_DIR="${PHASE2_DATA_DIR:-$RUN_DIR/data/phase2}"
PHASE1_OUT_DIR="${PHASE1_OUT_DIR:-$RUN_DIR/phase1}"
PHASE2_OUT_DIR="${PHASE2_OUT_DIR:-$RUN_DIR/phase2}"
MERGED1_DIR="${MERGED1_DIR:-$RUN_DIR/merged1}"
MERGED2_DIR="${MERGED2_DIR:-$RUN_DIR/merged2}"
MIMI_CACHE_DIR="${MIMI_CACHE_DIR:-$HF_HOME/datasets/ksponspeech_mimi}"

# Phase 1 mixture. The ratio and the seed are what make the mixture reproducible.
TRANSLATION_RATIO="${TRANSLATION_RATIO:-0.30}"
DIALOGUE_RATIO="${DIALOGUE_RATIO:-0.70}"
TOTAL_EXAMPLES="${TOTAL_EXAMPLES:-200000}"
HELD_OUT_EXAMPLES="${HELD_OUT_EXAMPLES:-10000}"
DATA_SEED="${DATA_SEED:-0}"
DOWNLOAD_CORPORA="${DOWNLOAD_CORPORA:-1}"

# KsponSpeech codec pass. The archive and its transcript are placed by hand, the corpus is
# distributed under terms that forbid an automated fetch.
KSPON_ARCHIVE="${KSPON_ARCHIVE:-$HF_HOME/datasets/ksponspeech/KsponSpeech.zip}"
KSPON_TRANSCRIPT="${KSPON_TRANSCRIPT:-$HF_HOME/datasets/ksponspeech/train.trn}"
KSPON_SHARDS="${KSPON_SHARDS:-}"
KSPON_MAX_HOURS="${KSPON_MAX_HOURS:-}"
NUM_CODEBOOKS="${NUM_CODEBOOKS:-8}"
MIN_DURATION="${MIN_DURATION:-0.5}"
MAX_DURATION="${MAX_DURATION:-30.0}"
TRANSCRIPTION_FORM="${TRANSCRIPTION_FORM:-phonetic}"

# Phase 1 training. Only the new text rows move, so the learning rate is the one of a head.
# The backbone runs under `no_grad`, so there is no activation to trade for recompute.
P1_GRADIENT_CHECKPOINTING="${P1_GRADIENT_CHECKPOINTING:-0}"
P1_MAX_LENGTH="${P1_MAX_LENGTH:-512}"
# The duplex layout reads the acoustic cache and word spans rather than the plain text mixture, so it
# is the frame count that caps an item and the text one no longer applies.
# Rows are coefficients over 64 pretrained rows, and
# English is kept by a term in the objective at lambda 5.
P1_BASE_ROW_MIXTURE="${P1_BASE_ROW_MIXTURE:-64}"
# Keeping English in the objective. Rows that learn Korean take probability mass from English inside
# the same softmax, and asking them to stay quiet on cached English states costs the least Korean of
# anything measured. Zero leaves the objective the Korean cross entropy alone.
P1_ENGLISH_LAMBDA="${P1_ENGLISH_LAMBDA:-5}"
P1_ENGLISH_STATES="${P1_ENGLISH_STATES:-$RUN_DIR/data/english_states.safetensors}"
P1_ENGLISH_FRAMES="${P1_ENGLISH_FRAMES:-2048}"
# Separate weight on the English PAD and EPAD frames, so the word term can be eased without easing
# the frames that decide when to speak. Needs a cache built with --score-all-frames.
P1_TIMING_LAMBDA="${P1_TIMING_LAMBDA:-0}"
# Persona the duplex prefix carries in a Korean session, which is where the language marker goes. The
# English preservation states are read under the plain persona. Empty keeps the collator's default.
SESSION_PERSONA="${SESSION_PERSONA:-}"
# One free vector added to every appended head row, the direction Korean and English sessions are told
# apart along, which the pretrained rows a mixture is built from do not span.
P1_PARTITION_DIRECTION="${P1_PARTITION_DIRECTION:-0}"
# Fixed FFN input directions per layer that deployment barely uses (python -m polyplex_research.utils.v1.evaluate basis). Each
# listed layer gains a correction reading only them. Empty leaves the backbone untouched.
P1_CONFINED_BASIS="${P1_CONFINED_BASIS:-}"
# The same for the attention value projection of each listed layer.
P1_CONFINED_ATTENTION_BASIS="${P1_CONFINED_ATTENTION_BASIS:-}"
# Layer range such as 8-31 whose FFNs gain a full input low rank correction, and its rank. The correction
# is meant for the language's own sessions; an English session runs with it off.
P1_SESSION_EXPERT_LAYERS="${P1_SESSION_EXPERT_LAYERS:-}"
P1_SESSION_EXPERT_RANK="${P1_SESSION_EXPERT_RANK:-256}"
# Moments of python -m polyplex_research.utils.v1.evaluate capacity, and the scales of the two terms that keep the corrections off
# deployment: their share of output on deployment inputs, and the KL on the PAD / EPAD frames.
P1_CONFINED_MOMENTS="${P1_CONFINED_MOMENTS:-}"
P1_LEAK_LAMBDA="${P1_LEAK_LAMBDA:-0}"
P1_TIMING_KL_LAMBDA="${P1_TIMING_KL_LAMBDA:-0}"
# Scale of the term that holds the answer to English speech (ENGLISH_CORPUS) in place and keeps the
# appended rows out of it.
P1_ENGLISH_GUARD_LAMBDA="${P1_ENGLISH_GUARD_LAMBDA:-0}"
ENGLISH_CORPUS="${ENGLISH_CORPUS:-$RUN_DIR/data/english_duplex.pt}"
P1_DROP_AUDIO="${P1_DROP_AUDIO:-0}"
# Phase 1 trains in the layout deployment reads. A plain text sequence does not transfer to it, so the
# duplex layout is the default rather than an option.
P1_DUPLEX="${P1_DUPLEX:-1}"
# The dense stage of Phase 1 (1.0): packed text rows with no PAD frame, read from a cache written by
# `duplex_data dense`. It takes precedence over P1_DUPLEX and needs P1_TIMING_KL_LAMBDA=0.
P1_DENSE="${P1_DENSE:-0}"
# GPUs of this machine one Phase 1 run trains on. Above 1 the run starts as torchrun data parallel workers
# rather than through letify, and P1_BATCH_SIZE is per GPU, so the batch of one update is that times the count.
P1_NPROC="${P1_NPROC:-1}"
# Phase 1.75 (SPEC S9.4a): P1_TRAIN_ROWS=0 keeps the appended rows as PHASE1_ADAPTER_DIR holds them and
# trains only the backbone corrections. P1_TEXT_DIR is a silent duplex cache scored next to every batch
# with weight P1_TEXT_WEIGHT and P1_TEXT_BATCH_SIZE examples per micro batch.
P1_TRAIN_ROWS="${P1_TRAIN_ROWS:-1}"
P1_TEXT_DIR="${P1_TEXT_DIR:-}"
P1_TEXT_WEIGHT="${P1_TEXT_WEIGHT:-1.0}"
P1_TEXT_BATCH_SIZE="${P1_TEXT_BATCH_SIZE:-}"
P1_MAX_FRAMES="${P1_MAX_FRAMES:-}"
P1_EPOCHS="${P1_EPOCHS:-1}"
P1_BATCH_SIZE="${P1_BATCH_SIZE:-8}"
P1_GRAD_ACCUM="${P1_GRAD_ACCUM:-4}"
P1_LR="${P1_LR:-1e-4}"
P1_WARMUP_RATIO="${P1_WARMUP_RATIO:-0.03}"

# Phase 2 training. The residual starts at zero and carries the penalties, hence the larger rate.
P2_EPOCHS="${P2_EPOCHS:-1}"
P2_BATCH_SIZE="${P2_BATCH_SIZE:-2}"
P2_GRAD_ACCUM="${P2_GRAD_ACCUM:-8}"
P2_LR="${P2_LR:-3e-4}"
P2_WARMUP_RATIO="${P2_WARMUP_RATIO:-0.03}"
P2_MAX_FRAMES="${P2_MAX_FRAMES:-750}"
P2_TEXT_PAD_TOKEN_ID="${P2_TEXT_PAD_TOKEN_ID:-3}"
P2_TEXT_ALIGNMENT="${P2_TEXT_ALIGNMENT:-spread}"
P2_ADAPTER_KIND="${P2_ADAPTER_KIND:-rotation}"
P2_TIE_SELF_INPUT="${P2_TIE_SELF_INPUT:-True}"
P2_GATED_ADAPTER="${P2_GATED_ADAPTER:-True}"
P2_ORTHO_LAMBDA="${P2_ORTHO_LAMBDA:-0.1}"
P2_NONNEG_BETA="${P2_NONNEG_BETA:-10.0}"

# Phase 2 speaking. The speech comes from a KsponSpeech code cache already encoded with every clip at
# one loudness, the text term from the corpus Phase 1 was fitted on.
KSPON_CODE_CACHE="${KSPON_CODE_CACHE:-/home/work/duplex_research/haan/data/cache_v2/kspon}"
PHASE2_KSPON_PREFIX="${PHASE2_KSPON_PREFIX:-$RUN_DIR/data/phase2_kspon}"
PHASE2_TRAIN_DIR="${PHASE2_TRAIN_DIR:-${PHASE2_KSPON_PREFIX}_train}"
PHASE2_VALID_DIR="${PHASE2_VALID_DIR:-${PHASE2_KSPON_PREFIX}_valid}"
P2_MIN_DURATION="${P2_MIN_DURATION:-2.0}"
P2_MAX_DURATION="${P2_MAX_DURATION:-12.0}"
# A silent duplex cache the speaking run also scores, to keep the text. It must hold dialogue the
# starting model never trained on (AGENTS.md H18), so there is no default.
P2_TEXT_DIR="${P2_TEXT_DIR:-}"
P2_TEXT_WEIGHT="${P2_TEXT_WEIGHT:-1.0}"
P2_TEXT_BATCH_SIZE="${P2_TEXT_BATCH_SIZE:-}"
P2_ENGLISH_LAMBDA="${P2_ENGLISH_LAMBDA:-$P1_ENGLISH_LAMBDA}"
P2_TIMING_LAMBDA="${P2_TIMING_LAMBDA:-$P1_TIMING_LAMBDA}"
# SPEC S9.1: the session expert of the starting checkpoint (P1_SESSION_EXPERT_LAYERS) is loaded and kept
# fixed unless P2_TRAIN_CORRECTIONS=1, and the timing KL holds PAD/EPAD to the rows the run started from.
P2_TRAIN_CORRECTIONS="${P2_TRAIN_CORRECTIONS:-0}"
P2_TIMING_KL_LAMBDA="${P2_TIMING_KL_LAMBDA:-1}"
P2_SPEAK_BATCH_SIZE="${P2_SPEAK_BATCH_SIZE:-32}"
P2_SPEAK_LR="${P2_SPEAK_LR:-1e-3}"
P2_SPEAK_EPOCHS="${P2_SPEAK_EPOCHS:-1}"
P2_SPEAK_MAX_FRAMES="${P2_SPEAK_MAX_FRAMES:-320}"

# Batching by length, which cuts the padding a batch of mixed lengths spends most of itself on.
GROUP_BY_LENGTH="${GROUP_BY_LENGTH:-1}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SEED="${SEED:-42}"
REPORT_TO="${REPORT_TO:-wandb}"
# Curves go to this W&B project, under the name of the experiment branch the run starts from.
export WANDB_PROJECT="${WANDB_PROJECT:-polyplex}"
# Tags of the run, comma separated, read by wandb itself: the phase and the claims it tests (AGENTS.md 4).
# An empty WANDB_TAGS is rejected by wandb, so it is exported only when set.
[ -n "${WANDB_TAGS:-}" ] && export WANDB_TAGS
# Accelerator the trainer is declared on through letify. "local" is the first GPU of this machine; a
# remote one is named "<provider>.<accelerator>", for example "colab_pro_plus.G4".
export POLYPLEX_DEVICE="${POLYPLEX_DEVICE:-local}"
# Device of the dense quick check, a letify alias and accelerator.
DENSE_DEVICE="${DENSE_DEVICE:-colab_pro_plus.RTX_PRO_6000}"
BRANCH="$(git -C "$ROOT" branch --show-current 2>/dev/null || true)"
RUN_NAME="${RUN_NAME:-${BRANCH#experiment/}}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() {
    printf '\n[train.sh] %s\n' "$*"
}

require_file() {
    if [[ ! -e "$1" ]]; then
        echo "[train.sh] missing $2: $1" >&2
        echo "[train.sh] set the matching environment variable, or place the file there." >&2
        exit 1
    fi
}

# Turns a held out split into the flags the trainer reads, and stays silent when there is none.
eval_args() {
    if [[ -n "$1" ]]; then
        printf '%s\n' --eval_dataset_path "$1" --eval_strategy steps --eval_steps "$EVAL_STEPS"
        if [[ "$EARLY_STOPPING_PATIENCE" != "0" ]]; then
            printf '%s\n' --early_stopping_patience "$EARLY_STOPPING_PATIENCE" \
                --early_stopping_threshold "$EARLY_STOPPING_THRESHOLD"
        fi
    fi
}

# Datasets and checkpoints are loaded by `from_pretrained`, which reads a local directory or
# downloads an `owner/name` Hub repository. Only the local form can be checked here.
require_input() {
    if [[ -e "$1" ]]; then
        return
    fi
    if [[ "$1" == */* && "$1" != /* && "$1" != .* && "$1" != */*/* ]]; then
        return
    fi
    echo "[train.sh] missing $2: $1" >&2
    echo "[train.sh] pass a directory that exists, or the 'owner/name' of a Hub repository." >&2
    exit 1
}

# The two trainings share every `TrainingArguments` flag that is not a rate or a batch size.
common_training_args() {
    local checkpointing="${1:-$GRADIENT_CHECKPOINTING}"
    local flags=(
        --logging_steps "$LOGGING_STEPS"
        --save_steps "$SAVE_STEPS"
        --save_total_limit 2
        --lr_scheduler_type cosine
        --seed "$SEED"
        --report_to "$REPORT_TO"
        --run_name "${RUN_NAME:-run}"
        --remove_unused_columns False
        --dataloader_num_workers 4
    )
    if [[ "$TORCH_DTYPE" == "bfloat16" ]]; then
        flags+=(--bf16 True)
    elif [[ "$TORCH_DTYPE" == "float16" ]]; then
        flags+=(--fp16 True)
    fi
    if [[ "$checkpointing" == "1" ]]; then
        flags+=(--gradient_checkpointing True)
    fi
    if [[ "$GROUP_BY_LENGTH" == "1" ]]; then
        flags+=(--train_sampling_strategy group_by_length)
    fi
    printf '%s\n' "${flags[@]}"
}

quantization_args() {
    if [[ -n "$QUANTIZATION" ]]; then
        printf '%s\n' --quantization "$QUANTIZATION"
    fi
}

# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

stage_assemble() {
    log "assemble: $BASE_MODEL + $LANGUAGE_MODEL -> $DUALFFN_DIR"
    local layers=()
    if [[ -n "$LANGUAGE_FFN_LAYERS" ]]; then
        layers=(--language-ffn-layers "$LANGUAGE_FFN_LAYERS")
    fi
    "$PYTHON" -m polyplex_research.utils.assemble dualffn \
        --personaplex "$BASE_MODEL" \
        --language-model "$LANGUAGE_MODEL" \
        --output-dir "$DUALFFN_DIR" \
        --torch-dtype "$TORCH_DTYPE" \
        "${layers[@]}" \
        "$@"
}

stage_dense() {
    log "dense: $BASE_MODEL + $LANGUAGE_MODEL on $DENSE_DEVICE"
    POLYPLEX_DEVICE="$DENSE_DEVICE" "$PYTHON" -m polyplex_research.utils.phase dense \
        --personaplex "$BASE_MODEL" \
        --language-model "$LANGUAGE_MODEL" \
        "$@"
}

stage_dense_eval() {
    log "dense-eval: $* on $DENSE_DEVICE"
    POLYPLEX_DEVICE="$DENSE_DEVICE" "$PYTHON" -m polyplex_research.utils.phase dense-eval \
        --personaplex "$BASE_MODEL" \
        --language-model "$LANGUAGE_MODEL" \
        "$@"
}

stage_score() {
    log "score: $* against $SCORE_MODEL"
    "$PYTHON" -m polyplex_research.utils.evaluate score \
        --model "$SCORE_MODEL" \
        --english-speech "$SCORE_ENGLISH_SPEECH" \
        --korean "$SCORE_KOREAN" --drop-audio \
        --exclude-train "$SCORE_EXCLUDE_TRAIN" \
        "$@"
}

stage_expand() {
    require_input "$CORPUS_DATASET" "corpus dataset"
    log "expand: $BASE_MODEL -> $EXPANDED_DIR"
    "$PYTHON" -m polyplex_research.utils.v1.assemble expand \
        --base-model "$BASE_MODEL" \
        --corpus-dataset "$CORPUS_DATASET" \
        --vocab-size "$VOCAB_SIZE" \
        --output-dir "$EXPANDED_DIR" \
        --method "$INIT_METHOD" \
        --torch-dtype "$TORCH_DTYPE" \
        "$@"
}

stage_data1() {
    log "data1: dialogue and translation mixture -> $PHASE1_DATA_DIR"
    TRANSLATION_RATIO="$TRANSLATION_RATIO" \
    DIALOGUE_RATIO="$DIALOGUE_RATIO" \
    TOTAL_EXAMPLES="$TOTAL_EXAMPLES" \
    HELD_OUT_EXAMPLES="$HELD_OUT_EXAMPLES" \
    DATA_SEED="$DATA_SEED" \
    DOWNLOAD_CORPORA="$DOWNLOAD_CORPORA" \
    PHASE1_DATA_DIR="$PHASE1_DATA_DIR" \
    "$PYTHON" - <<'PY'
import os

from polyplex_research.datasets import Phase1Dataset

total = int(os.environ["TOTAL_EXAMPLES"])
held_out = int(os.environ["HELD_OUT_EXAMPLES"])
base = os.environ["PHASE1_DATA_DIR"]

# The dialogue corpus ships one held out directory and the phase needs two, so `valid` and `test` are
# its halves, cut by conversation so that no dialogue is scored after being trained on.
for split, size, suffix in (
    ("train", total, ""),
    ("valid", held_out, "_valid"),
    ("test", held_out, "_test"),
):
    dataset = Phase1Dataset(
        split=split,
        direction="en2ko",
        translation_ratio=float(os.environ["TRANSLATION_RATIO"]),
        dialogue_ratio=float(os.environ["DIALOGUE_RATIO"]),
        total_examples=size if size > 0 else None,
        download=os.environ["DOWNLOAD_CORPORA"] == "1",
        seed=int(os.environ["DATA_SEED"]),
    )
    directory = dataset.save_pretrained(base + suffix)
    print(f"phase 1 {split}: {len(dataset)} examples in {directory}", flush=True)
PY
}

stage_phase1() {
    require_input "$EXPANDED_DIR" "expanded checkpoint"
    require_input "$PHASE1_DATA_DIR" "phase 1 dataset"
    log "phase1: text injection -> $PHASE1_OUT_DIR"
    mapfile -t training_flags < <(common_training_args "$P1_GRADIENT_CHECKPOINTING")
    mapfile -t quantization_flags < <(quantization_args)
    mapfile -t eval_flags < <(eval_args "${PHASE1_EVAL_DIR}")
    local resume=()
    if [[ -n "$PHASE1_ADAPTER_DIR" ]]; then
        resume=(--adapter_path "$PHASE1_ADAPTER_DIR")
    fi
    local preserve=()
    if [[ "$P1_ENGLISH_LAMBDA" != "0" && "$P1_ENGLISH_LAMBDA" != "0.0" ]] \
        || [[ "$P1_TIMING_LAMBDA" != "0" && "$P1_TIMING_LAMBDA" != "0.0" ]]; then
        require_file "$P1_ENGLISH_STATES" "english state cache"
        preserve=(
            --english_lambda "$P1_ENGLISH_LAMBDA"
            --timing_lambda "$P1_TIMING_LAMBDA"
            --english_states_path "$P1_ENGLISH_STATES"
            --english_frames "$P1_ENGLISH_FRAMES"
        )
    fi
    local layout=(--text_max_length "$P1_MAX_LENGTH" --base_row_mixture "$P1_BASE_ROW_MIXTURE")
    if [[ "$P1_DENSE" == "1" ]]; then
        layout=(--dense True --base_row_mixture "$P1_BASE_ROW_MIXTURE")
    elif [[ "$P1_DUPLEX" == "1" ]]; then
        # The frame layout replaces the text one, but how the rows are parameterised is orthogonal to it.
        layout=(--duplex True --base_row_mixture "$P1_BASE_ROW_MIXTURE")
        if [[ "$P1_DROP_AUDIO" == "1" ]]; then
            layout+=(--drop_audio True)
        fi
        if [[ -n "$P1_MAX_FRAMES" ]]; then
            layout+=(--max_frames "$P1_MAX_FRAMES")
        fi
        if [[ -n "$SESSION_PERSONA" ]]; then
            layout+=(--persona "$SESSION_PERSONA")
        fi
    fi
    if [[ "$P1_PARTITION_DIRECTION" == "1" ]]; then
        layout+=(--partition_direction True)
    fi
    if [[ -n "$P1_CONFINED_BASIS" ]]; then
        layout+=(--confined_basis "$P1_CONFINED_BASIS")
    fi
    if [[ -n "$P1_SESSION_EXPERT_LAYERS" ]]; then
        layout+=(--session_expert_layers "$P1_SESSION_EXPERT_LAYERS" --session_expert_rank "$P1_SESSION_EXPERT_RANK")
    fi
    if [[ -n "$P1_CONFINED_ATTENTION_BASIS" ]]; then
        layout+=(--confined_attention_basis "$P1_CONFINED_ATTENTION_BASIS")
    fi
    if [[ -n "$P1_CONFINED_MOMENTS" ]]; then
        layout+=(--confined_moments "$P1_CONFINED_MOMENTS")
    fi
    if [[ "$P1_TRAIN_ROWS" == "0" ]]; then
        layout+=(--train_rows False)
    fi
    if [[ -n "$P1_TEXT_DIR" ]]; then
        require_input "$P1_TEXT_DIR" "phase 1 text cache"
        layout+=(--text_dataset_path "$P1_TEXT_DIR" --text_weight "$P1_TEXT_WEIGHT")
        if [[ -n "$P1_TEXT_BATCH_SIZE" ]]; then
            layout+=(--text_batch_size "$P1_TEXT_BATCH_SIZE")
        fi
    fi
    layout+=(--leak_lambda "$P1_LEAK_LAMBDA" --timing_kl_lambda "$P1_TIMING_KL_LAMBDA")
    layout+=(--english_guard_lambda "$P1_ENGLISH_GUARD_LAMBDA" --english_guard_path "$ENGLISH_CORPUS")
    local launcher=("$PYTHON")
    if [[ "$P1_NPROC" -gt 1 ]]; then
        launcher=("$PYTHON" -m torch.distributed.run --standalone --nproc_per_node "$P1_NPROC")
    fi
    "${launcher[@]}" -m polyplex_research.utils.v1.phase \
        --phase 1 \
        --model_name_or_path "$EXPANDED_DIR" \
        --dataset_path "$PHASE1_DATA_DIR" \
        "${layout[@]}" \
        --output_dir "$PHASE1_OUT_DIR" \
        --num_train_epochs "$P1_EPOCHS" \
        --per_device_train_batch_size "$P1_BATCH_SIZE" \
        --gradient_accumulation_steps "$P1_GRAD_ACCUM" \
        --learning_rate "$P1_LR" \
        --warmup_ratio "$P1_WARMUP_RATIO" \
        "${training_flags[@]}" \
        ${quantization_flags[@]+"${quantization_flags[@]}"} \
        ${eval_flags[@]+"${eval_flags[@]}"} \
        ${resume[@]+"${resume[@]}"} \
        ${preserve[@]+"${preserve[@]}"} \
        "$@"
}

stage_english() {
    require_input "$EXPANDED_DIR" "expanded checkpoint"
    require_file "$ENGLISH_CORPUS" "english duplex corpus"
    log "english: backbone states -> $P1_ENGLISH_STATES"
    "$PYTHON" -m polyplex_research.utils.v1.evaluate english-states \
        --model "$EXPANDED_DIR" \
        --corpus "$ENGLISH_CORPUS" \
        --output "$P1_ENGLISH_STATES" \
        "$@"
}

stage_merge1() {
    require_input "$EXPANDED_DIR" "expanded checkpoint"
    require_input "$PHASE1_OUT_DIR" "phase 1 adapter"
    log "merge1: phase 1 rows -> $MERGED1_DIR"
    "$PYTHON" -m polyplex_research.utils.v1.assemble dessemble \
        --phase 1 \
        --base-model "$EXPANDED_DIR" \
        --adapter "$PHASE1_OUT_DIR" \
        --output-dir "$MERGED1_DIR" \
        --torch-dtype "$TORCH_DTYPE" \
        "$@"
}

stage_merge2_rotation() {
    require_input "$MERGED1_DIR" "merged phase 1 checkpoint"
    require_input "$PHASE2_OUT_DIR" "phase 2 adapter"
    log "merge2: phase 2 residual -> $MERGED2_DIR"
    "$PYTHON" -m polyplex_research.utils.v1.assemble dessemble \
        --phase 2 \
        --base-model "$MERGED1_DIR" \
        --adapter "$PHASE2_OUT_DIR" \
        --output-dir "$MERGED2_DIR" \
        --torch-dtype "$TORCH_DTYPE" \
        "$@"
}

stage_cache() {
    require_file "$KSPON_ARCHIVE" "KsponSpeech archive"
    require_file "$KSPON_TRANSCRIPT" "KsponSpeech transcript"
    log "cache: Mimi codes -> $MIMI_CACHE_DIR"
    local optional=()
    if [[ -n "$KSPON_SHARDS" ]]; then
        # Word split is what turns "KsponSpeech_01 KsponSpeech_02" into two arguments.
        optional+=(--shards $KSPON_SHARDS)
    fi
    if [[ -n "$KSPON_MAX_HOURS" ]]; then
        optional+=(--max-hours "$KSPON_MAX_HOURS")
    fi
    "$PYTHON" -m polyplex_research.datasets.ksponspeech mimi \
        --archive "$KSPON_ARCHIVE" \
        --transcript "$KSPON_TRANSCRIPT" \
        --output "$MIMI_CACHE_DIR" \
        --num-codebooks "$NUM_CODEBOOKS" \
        --min-duration "$MIN_DURATION" \
        --max-duration "$MAX_DURATION" \
        --transcription-form "$TRANSCRIPTION_FORM" \
        ${optional[@]+"${optional[@]}"} \
        "$@"
}

stage_data2_rotation() {
    require_file "$MIMI_CACHE_DIR" "Mimi code cache"
    log "data2: single-turn TTS codes -> $PHASE2_DATA_DIR"
    MIMI_CACHE_DIR="$MIMI_CACHE_DIR" \
    PHASE2_DATA_DIR="$PHASE2_DATA_DIR" \
    NUM_CODEBOOKS="$NUM_CODEBOOKS" \
    MIN_DURATION="$MIN_DURATION" \
    MAX_DURATION="$MAX_DURATION" \
    "$PYTHON" - <<'PY'
import os

from polyplex_research.datasets import Phase2Dataset

dataset = Phase2Dataset(
    cache_dir=os.environ["MIMI_CACHE_DIR"],
    min_duration=float(os.environ["MIN_DURATION"]),
    max_duration=float(os.environ["MAX_DURATION"]),
    num_codebooks=int(os.environ["NUM_CODEBOOKS"]),
)
directory = dataset.save_pretrained(os.environ["PHASE2_DATA_DIR"])
print(f"phase 2 dataset: {len(dataset)} utterances in {directory}")
PY
}

stage_phase2_rotation() {
    require_input "$MERGED1_DIR" "merged phase 1 checkpoint"
    require_input "$PHASE2_DATA_DIR" "phase 2 dataset"
    log "phase2: audio injection -> $PHASE2_OUT_DIR"
    mapfile -t training_flags < <(common_training_args)
    mapfile -t quantization_flags < <(quantization_args)
    mapfile -t eval_flags < <(eval_args "${PHASE2_EVAL_DIR}")
    "$PYTHON" -m polyplex_research.utils.v1.phase \
        --phase 2 \
        --model_name_or_path "$MERGED1_DIR" \
        --tokenizer_name_or_path "$MERGED1_DIR" \
        --dataset_path "$PHASE2_DATA_DIR" \
        --output_dir "$PHASE2_OUT_DIR" \
        --torch_dtype "$TORCH_DTYPE" \
        --max_frames "$P2_MAX_FRAMES" \
        --text_pad_token_id "$P2_TEXT_PAD_TOKEN_ID" \
        --text_alignment "$P2_TEXT_ALIGNMENT" \
        --adapter_kind "$P2_ADAPTER_KIND" \
        --tie_self_input "$P2_TIE_SELF_INPUT" \
        --gated_adapter "$P2_GATED_ADAPTER" \
        --ortho_lambda "$P2_ORTHO_LAMBDA" \
        --nonneg_beta "$P2_NONNEG_BETA" \
        --num_train_epochs "$P2_EPOCHS" \
        --per_device_train_batch_size "$P2_BATCH_SIZE" \
        --gradient_accumulation_steps "$P2_GRAD_ACCUM" \
        --learning_rate "$P2_LR" \
        --warmup_ratio "$P2_WARMUP_RATIO" \
        "${training_flags[@]}" \
        ${quantization_flags[@]+"${quantization_flags[@]}"} \
        ${eval_flags[@]+"${eval_flags[@]}"} \
        "$@"
}

stage_data2() {
    require_input "$KSPON_CODE_CACHE" "KsponSpeech code cache"
    log "data2: KsponSpeech codes -> ${PHASE2_KSPON_PREFIX}_{train,valid,test}"
    "$PYTHON" -m polyplex_research.datasets.ksponspeech import \
        --source "$KSPON_CODE_CACHE" \
        --output "$PHASE2_KSPON_PREFIX" \
        --min-duration "$P2_MIN_DURATION" \
        --max-duration "$P2_MAX_DURATION" \
        "$@"
}

stage_phase2() {
    require_input "$EXPANDED_DIR" "expanded checkpoint"
    require_input "$PHASE1_OUT_DIR" "phase 1 adapter"
    require_input "$PHASE2_TRAIN_DIR" "phase 2 speech cache"
    log "phase2: speaking through the text partition -> $PHASE2_OUT_DIR"
    mapfile -t training_flags < <(common_training_args "$P1_GRADIENT_CHECKPOINTING")
    mapfile -t quantization_flags < <(quantization_args)
    mapfile -t eval_flags < <(eval_args "${PHASE2_VALID_DIR}")
    local preserve=()
    if [[ "$P2_ENGLISH_LAMBDA" != "0" && "$P2_ENGLISH_LAMBDA" != "0.0" ]] \
        || [[ "$P2_TIMING_LAMBDA" != "0" && "$P2_TIMING_LAMBDA" != "0.0" ]]; then
        require_file "$P1_ENGLISH_STATES" "english state cache"
        preserve=(
            --english_lambda "$P2_ENGLISH_LAMBDA"
            --timing_lambda "$P2_TIMING_LAMBDA"
            --english_states_path "$P1_ENGLISH_STATES"
            --english_frames "$P1_ENGLISH_FRAMES"
        )
    fi
    local text=()
    if [[ -n "$P2_TEXT_DIR" && "$P2_TEXT_WEIGHT" != "0" && "$P2_TEXT_WEIGHT" != "0.0" ]]; then
        require_input "$P2_TEXT_DIR" "phase 1 text cache"
        text=(--text_dataset_path "$P2_TEXT_DIR" --text_weight "$P2_TEXT_WEIGHT")
        if [[ -n "$P2_TEXT_BATCH_SIZE" ]]; then
            text+=(--text_batch_size "$P2_TEXT_BATCH_SIZE")
        fi
    fi
    if [[ -n "$SESSION_PERSONA" ]]; then
        text+=(--persona "$SESSION_PERSONA")
    fi
    if [[ "$P1_PARTITION_DIRECTION" == "1" ]]; then
        text+=(--partition_direction True)
    fi
    if [[ -n "$P1_SESSION_EXPERT_LAYERS" ]]; then
        text+=(--session_expert_layers "$P1_SESSION_EXPERT_LAYERS" --session_expert_rank "$P1_SESSION_EXPERT_RANK")
        if [[ "$P2_TRAIN_CORRECTIONS" != "1" ]]; then
            text+=(--train_corrections False)
        fi
    fi
    if [[ "$P2_TIMING_KL_LAMBDA" != "0" && "$P2_TIMING_KL_LAMBDA" != "0.0" ]]; then
        text+=(--timing_kl_lambda "$P2_TIMING_KL_LAMBDA" --timing_reference start)
    fi
    # Phase 2 runs on the Phase 1 trainer: the rows it moves are the same ones, now shared with the
    # depth decoder's table so that the audio objective reaches them.
    "$PYTHON" -m polyplex_research.utils.v1.phase \
        --phase 1 \
        --model_name_or_path "$EXPANDED_DIR" \
        --adapter_path "$PHASE1_OUT_DIR" \
        --dataset_path "$PHASE2_TRAIN_DIR" \
        --duplex True \
        --audio_labels True \
        --base_row_mixture "$P1_BASE_ROW_MIXTURE" \
        --share_row_mixture True \
        --train_depth_text_embeddings True \
        --max_frames "$P2_SPEAK_MAX_FRAMES" \
        --output_dir "$PHASE2_OUT_DIR" \
        --num_train_epochs "$P2_SPEAK_EPOCHS" \
        --per_device_train_batch_size "$P2_SPEAK_BATCH_SIZE" \
        --learning_rate "$P2_SPEAK_LR" \
        --warmup_ratio "$P2_WARMUP_RATIO" \
        "${training_flags[@]}" \
        ${quantization_flags[@]+"${quantization_flags[@]}"} \
        ${eval_flags[@]+"${eval_flags[@]}"} \
        ${preserve[@]+"${preserve[@]}"} \
        ${text[@]+"${text[@]}"} \
        "$@"
}

stage_merge2() {
    require_input "$EXPANDED_DIR" "expanded checkpoint"
    require_input "$PHASE2_OUT_DIR" "phase 2 adapter"
    log "merge2: phase 2 rows -> $MERGED2_DIR"
    # The Phase 2 adapter holds every row Phase 1 held and more, so it folds onto the expansion alone.
    "$PYTHON" -m polyplex_research.utils.v1.assemble dessemble \
        --phase 1 \
        --base-model "$EXPANDED_DIR" \
        --adapter "$PHASE2_OUT_DIR" \
        --output-dir "$MERGED2_DIR" \
        --torch-dtype "$TORCH_DTYPE" \
        "$@"
}

stage_all() {
    stage_expand
    stage_data1
    stage_phase1
    stage_merge1
    stage_data2
    stage_phase2
    stage_merge2
}

usage() {
    sed -n '19,44p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

main() {
    local stage="${1:-}"
    shift || true
    case "$stage" in
        assemble) stage_assemble "$@" ;;
        dense) stage_dense "$@" ;;
        dense-eval) stage_dense_eval "$@" ;;
        score) stage_score "$@" ;;
        expand) stage_expand "$@" ;;
        data1) stage_data1 "$@" ;;
        english) stage_english "$@" ;;
        phase1) stage_phase1 "$@" ;;
        merge1) stage_merge1 "$@" ;;
        cache) stage_cache "$@" ;;
        data2) stage_data2 "$@" ;;
        phase2) stage_phase2 "$@" ;;
        merge2) stage_merge2 "$@" ;;
        data2_rotation) stage_data2_rotation "$@" ;;
        phase2_rotation) stage_phase2_rotation "$@" ;;
        merge2_rotation) stage_merge2_rotation "$@" ;;
        all) stage_all ;;
        ""|-h|--help|help) usage ;;
        *)
            echo "[train.sh] unknown stage: $stage" >&2
            usage >&2
            exit 2
            ;;
    esac
}

main "$@"
