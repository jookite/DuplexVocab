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
"""Metrics of the transfer evaluation, gathered behind one `Evaluator`."""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter

import torch


GENERATION_OPENINGS = [
    "안녕하세요, 오늘 무엇을",
    "네, 말씀하신 영화는",
    "제가 추천드리고 싶은 여행지는",
    "그 부분은 조금 더",
    "죄송하지만 지금은",
    "김치찌개를 맛있게 끓이려면",
    "내일 서울 날씨는",
    "그 책의 주인공은",
    "운동을 꾸준히 하려면",
    "와, 정말 축하드려요!",
    "음, 제 생각에는",
    "회의는 오후 세 시에",
    "그거 진짜 재밌겠다,",
    "처음 오셨다면 먼저",
    "요즘 많이 힘드시죠?",
    "이 문제를 풀려면",
]

_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")
_SENTENCE_END = re.compile(r"[.?!]")
_LONG_NUMBER = re.compile(r"\d{6,}")


def generation_metrics(pairs: list) -> dict:
    r"""
    Counts the failure modes of Korean generation over what each output added after its opening.

    Args:
        pairs (`list[tuple[str, str]]`):
            `(opening, output)` pairs, where `output` starts with the forced opening.

    Returns:
        `dict`: The share of outputs with an English word (`english`), whose first sentence is closed by
        `.`, `?` or `!` (`sentence_end`), that repeat a word 3-gram (`repeat`), and that hold a run of six or
        more digits (`long_number`).
    """
    counts = Counter()
    for opening, output in pairs:
        added = output[len(opening):] if output.startswith(opening) else output
        words = added.split()
        grams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
        counts["english"] += bool(_LATIN_WORD.search(added))
        counts["sentence_end"] += bool(_SENTENCE_END.search(added))
        counts["repeat"] += len(grams) != len(set(grams))
        counts["long_number"] += bool(_LONG_NUMBER.search(added))
    total = max(1, len(pairs))
    return {key: counts[key] / total for key in ("english", "sentence_end", "repeat", "long_number")}


def read_speak_log(path: str) -> list:
    r"""
    Reads the samples an `evaluate speak` run printed.

    Args:
        path (`str`):
            Log of the run.

    Returns:
        `list` of `dict`: `opening`, `seed`, `cer`, `text` and `heard` per sample, in the order they were
        generated, which is also the order of the `<opening index>_<seed>.wav` files the run saved.
    """
    header = re.compile(r"^\[speak '(.*)' #(\d+)\] cer ([0-9.]+)$")
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            match = header.match(line)
            if match:
                records.append({"opening": match.group(1), "seed": int(match.group(2)), "cer": float(match.group(3))})
            elif records and line.startswith("    text : "):
                records[-1]["text"] = line[len("    text : ") :]
            elif records and line.startswith("    heard: "):
                records[-1]["heard"] = line[len("    heard: ") :]
    return records


class Evaluator:
    r"""
    Computes the three metric groups the evaluation reports.

    - Transcription. Character and word error rate, over an ASR pass of generated speech. Character rate
      is the one that carries for Korean, where a word is a whitespace token of variable morpheme count.
    - Generation quality. Corpus BLEU and ROUGE-L over decoded text.
    - Duplex behaviour. Turn taking latency and overlap ratio, which is what tells whether the frozen
      temporal transformer kept the timing it was pretrained with.

    Every metric is implemented here rather than pulled from a package. The evaluation is meant to add no
    dependency of its own, and the two text metrics are small enough that an inline version is easier to
    hold to the tokenization the rest of the pipeline uses.

    Args:
        frame_rate (`float`, *optional*, defaults to 12.5):
            Frames per second of the code streams, used to turn frame counts into seconds.
        vad_relative_threshold (`float`, *optional*, defaults to 30.0):
            How many dB below an utterance's loudest frame still counts as speech.
        vad_frame_length (`float`, *optional*, defaults to 0.02):
            Length in seconds of one energy frame.
    """

    def __init__(
        self,
        frame_rate: float = 12.5,
        vad_relative_threshold: float = 30.0,
        vad_frame_length: float = 0.02,
    ):
        self.frame_rate = frame_rate
        self.vad_relative_threshold = vad_relative_threshold
        self.vad_frame_length = vad_frame_length

    # -----------------------------------------------------------------
    # Transcription
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # Generation quality
    # -----------------------------------------------------------------

    @staticmethod
    def _ngrams(tokens: list, order: int) -> Counter:
        r"""
        Args:
            tokens (`list`):
                Token sequence.
            order (`int`):
                Length of the counted n-grams.

        Returns:
            `Counter`: Counts of each n-gram of that order.
        """
        return Counter(tuple(tokens[i : i + order]) for i in range(len(tokens) - order + 1))

    @classmethod
    def bleu(cls, references: list, hypotheses: list, max_order: int = 4) -> float:
        r"""
        Corpus BLEU with add-one smoothing on the higher orders.

        Smoothing matters here because the evaluation sets are small and a single short segment with no
        4-gram match would otherwise take the whole corpus score to zero.

        Args:
            references (`list` of `str`):
                Reference translations.
            hypotheses (`list` of `str`):
                Decoded translations, aligned with `references`.
            max_order (`int`, *optional*, defaults to 4):
                Highest n-gram order entering the score.

        Returns:
            `float`: BLEU in `[0, 1]`.

        Raises:
            ValueError: If the two lists differ in length.
        """
        if len(references) != len(hypotheses):
            raise ValueError(
                f"Got {len(references)} references and {len(hypotheses)} hypotheses, they must be aligned."
            )

        matches = [0] * max_order
        totals = [0] * max_order
        reference_length = 0
        hypothesis_length = 0

        for reference, hypothesis in zip(references, hypotheses):
            reference_tokens = reference.split()
            hypothesis_tokens = hypothesis.split()
            reference_length += len(reference_tokens)
            hypothesis_length += len(hypothesis_tokens)

            for order in range(1, max_order + 1):
                hypothesis_ngrams = cls._ngrams(hypothesis_tokens, order)
                reference_ngrams = cls._ngrams(reference_tokens, order)
                overlap = hypothesis_ngrams & reference_ngrams
                matches[order - 1] += sum(overlap.values())
                totals[order - 1] += max(sum(hypothesis_ngrams.values()), 0)

        if not hypothesis_length:
            return 0.0

        log_precision = 0.0
        for order in range(max_order):
            # Orders above the first are smoothed, the unigram precision is left exact so that an empty
            # overlap still reads as zero rather than as a small positive score.
            numerator = matches[order] + (1 if order else 0)
            denominator = totals[order] + (1 if order else 0)
            if denominator == 0 or numerator == 0:
                return 0.0
            log_precision += math.log(numerator / denominator)
        log_precision /= max_order

        brevity = 1.0 if hypothesis_length > reference_length else math.exp(
            1.0 - reference_length / hypothesis_length
        )
        return brevity * math.exp(log_precision)

    @staticmethod
    def _lcs_length(first: list, second: list) -> int:
        r"""
        Args:
            first (`list`):
                First token sequence.
            second (`list`):
                Second token sequence.

        Returns:
            `int`: Length of the longest common subsequence.
        """
        previous = [0] * (len(second) + 1)
        for first_token in first:
            current = [0]
            for index, second_token in enumerate(second):
                if first_token == second_token:
                    current.append(previous[index] + 1)
                else:
                    current.append(max(previous[index + 1], current[index]))
            previous = current
        return previous[-1]

    @classmethod
    def rouge_l(cls, references: list, hypotheses: list, beta: float = 1.2) -> float:
        r"""
        Sentence level ROUGE-L, averaged over the corpus.

        Args:
            references (`list` of `str`):
                Reference texts.
            hypotheses (`list` of `str`):
                Generated texts, aligned with `references`.
            beta (`float`, *optional*, defaults to 1.2):
                Weight of recall against precision in the F measure.

        Returns:
            `float`: Mean ROUGE-L F score in `[0, 1]`.

        Raises:
            ValueError: If the two lists differ in length.
        """
        if len(references) != len(hypotheses):
            raise ValueError(
                f"Got {len(references)} references and {len(hypotheses)} hypotheses, they must be aligned."
            )
        if not references:
            return 0.0

        scores = []
        for reference, hypothesis in zip(references, hypotheses):
            reference_tokens = reference.split()
            hypothesis_tokens = hypothesis.split()
            if not reference_tokens or not hypothesis_tokens:
                scores.append(0.0)
                continue
            common = cls._lcs_length(reference_tokens, hypothesis_tokens)
            if not common:
                scores.append(0.0)
                continue
            precision = common / len(hypothesis_tokens)
            recall = common / len(reference_tokens)
            scores.append(
                (1 + beta**2) * precision * recall / (recall + beta**2 * precision)
            )
        return sum(scores) / len(scores)

    # -----------------------------------------------------------------
    # Duplex behaviour
    # -----------------------------------------------------------------

    def energy_vad(self, waveform: torch.Tensor, sampling_rate: int) -> torch.BoolTensor:
        r"""
        Marks the speech frames of a waveform by short-term energy.

        The threshold is relative to the loudest frame of the same utterance rather than absolute, so an
        utterance that was recorded quietly is not read as silence throughout.

        Args:
            waveform (`torch.Tensor` of shape `(1, num_samples)` or `(num_samples,)`):
                Mono waveform.
            sampling_rate (`int`):
                Sampling rate of `waveform`.

        Returns:
            `torch.BoolTensor` of shape `(num_frames,)`: Whether each frame carries speech.
        """
        samples = waveform.reshape(-1).float()
        frame_length = max(1, int(self.vad_frame_length * sampling_rate))
        num_frames = samples.shape[0] // frame_length
        if num_frames == 0:
            return torch.zeros(0, dtype=torch.bool)

        frames = samples[: num_frames * frame_length].reshape(num_frames, frame_length)
        energy = frames.pow(2).mean(dim=1).clamp(min=1e-12)
        decibels = 10.0 * energy.log10()
        return decibels > (decibels.max() - self.vad_relative_threshold)

    @staticmethod
    def _segments(activity: torch.Tensor) -> list:
        r"""
        Args:
            activity (`torch.Tensor` of shape `(num_frames,)`):
                Per-frame speech flags.

        Returns:
            `list`: Half open `(start, end)` frame ranges of the active runs.
        """
        flags = activity.reshape(-1).bool().tolist()
        segments = []
        start = None
        for index, active in enumerate(flags):
            if active and start is None:
                start = index
            elif not active and start is not None:
                segments.append((start, index))
                start = None
        if start is not None:
            segments.append((start, len(flags)))
        return segments

    def turn_taking(
        self,
        user_activity: torch.Tensor,
        assistant_activity: torch.Tensor,
    ) -> dict:
        r"""
        Measures how the assistant times its speech against the user's.

        Latency is taken per user turn: the gap between the end of a user segment and the first assistant
        frame after it. A user segment the assistant never answers contributes nothing rather than an
        arbitrary large number, and is counted separately as a miss.

        Overlap is the share of speaking time the two streams spend talking at once. Some overlap is
        correct duplex behaviour, a collapse to zero means the model has stopped interrupting and
        backchanneling at all, and a large value means it talks over the user.

        Args:
            user_activity (`torch.Tensor` of shape `(num_frames,)`):
                Per-frame speech flags of the user stream.
            assistant_activity (`torch.Tensor` of shape `(num_frames,)`):
                Per-frame speech flags of the assistant stream.

        Returns:
            `dict`: `mean_latency`, `median_latency` in seconds, `overlap_ratio`, `response_rate`, and
            `num_turns`.

        Raises:
            ValueError: If the two streams differ in length.
        """
        user = user_activity.reshape(-1).bool()
        assistant = assistant_activity.reshape(-1).bool()
        if user.shape[0] != assistant.shape[0]:
            raise ValueError(
                f"The streams differ in length, {user.shape[0]} user frames against "
                f"{assistant.shape[0]} assistant frames."
            )

        latencies = []
        answered = 0
        turns = self._segments(user)
        for _, end in turns:
            following = assistant[end:].nonzero()
            if following.numel() == 0:
                continue
            answered += 1
            latencies.append(int(following[0].item()) / self.frame_rate)

        both = (user & assistant).sum().item()
        either = (user | assistant).sum().item()
        ordered = sorted(latencies)

        return {
            "mean_latency": sum(latencies) / len(latencies) if latencies else float("nan"),
            "median_latency": ordered[len(ordered) // 2] if ordered else float("nan"),
            "overlap_ratio": both / either if either else 0.0,
            "response_rate": answered / len(turns) if turns else 0.0,
            "num_turns": len(turns),
        }

    # -----------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------

    def evaluate(
        self,
        references: list | None = None,
        hypotheses: list | None = None,
        user_activity: torch.Tensor | None = None,
        assistant_activity: torch.Tensor | None = None,
    ) -> dict:
        r"""
        Runs every metric the given inputs support.

        Args:
            references (`list` of `str`, *optional*):
                Ground truth texts.
            hypotheses (`list` of `str`, *optional*):
                Predicted texts, aligned with `references`.
            user_activity (`torch.Tensor` of shape `(num_frames,)`, *optional*):
                Per-frame speech flags of the user stream.
            assistant_activity (`torch.Tensor` of shape `(num_frames,)`, *optional*):
                Per-frame speech flags of the assistant stream.

        Returns:
            `dict`: The metrics that could be computed, keyed by name.
        """
        metrics = {}
        if references is not None and hypotheses is not None:
            metrics["cer"] = self.character_error_rate(references, hypotheses)
            metrics["wer"] = self.word_error_rate(references, hypotheses)
            metrics["bleu"] = self.bleu(references, hypotheses)
            metrics["rouge_l"] = self.rouge_l(references, hypotheses)
        if user_activity is not None and assistant_activity is not None:
            metrics.update(self.turn_taking(user_activity, assistant_activity))
        return metrics


class ForceTextLogitsProcessor:
    r"""
    Constructs a logits processor that pins the text token of the first generated frames to a fixed stream,
    then leaves the rest free. Used to start generation from the opening words of a sentence.

    Args:
        stream (`list[int]`):
            Text token of every forced frame, in order.
    """

    def __init__(self, stream: list):
        self.stream = stream
        self.start = None

    def __call__(self, input_ids, scores):
        if self.start is None:
            self.start = input_ids.shape[-1]
        step = input_ids.shape[-1] - self.start
        if step < len(self.stream):
            forced = torch.full_like(scores, float("-inf"))
            forced[:, self.stream[step]] = 0.0
            return forced
        return scores


class SilentAssistantLogitsProcessor:
    r"""
    Constructs a depth decoder logits processor that pins every assistant codebook to the codec silence a
    silent training item carries, so the text continues under the condition Phase 1 trains in.

    The depth decoder writes one frame as the text token followed by the assistant codebooks in order, so the
    position of the next token says which codebook it is. The codec's silence is a short settling sequence,
    not one repeated frame, so the frame index is counted and the sequence holds its last frame once it runs out.

    Args:
        silence (`list[list[int]]`):
            The silence sequence, one list of frame codes per assistant codebook, as
            `processor.get_silence_audio_codes` returns it.
    """

    def __init__(self, silence: list):
        self.silence = silence
        self.frame = -1

    def __call__(self, input_ids, scores):
        codebook = input_ids.shape[-1] - 1
        if codebook == 0:
            self.frame += 1
        if 0 <= codebook < len(self.silence):
            codes = self.silence[codebook]
            forced = torch.full_like(scores, float("-inf"))
            forced[:, codes[min(self.frame, len(codes) - 1)]] = 0.0
            return forced
        return scores


"""Frame groups of the capacity measurement. The two `deploy_` groups are what deployment reads (English with its
own speech, and `PAD` frames of an assistant that is listening), `empty_ko` and `empty_en` are word and `EPAD`
frames under a silent assistant stream, and `speech_ko` are Korean word frames with their own speech."""

"""The held out English speech set (`train.sh` `SCORE_ENGLISH_SPEECH`)."""


"""NVFP4 variants `nvfp4-check` compares with full precision: what S7.1 runs, and the usual exceptions."""


if __name__ == "__main__":
    main()


__all__ = [
    "Evaluator",
    "ForceTextLogitsProcessor",
    "SilentAssistantLogitsProcessor",
]
