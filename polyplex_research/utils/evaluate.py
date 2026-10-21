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


def character_error_rate(reference: str, hypothesis: str) -> float:
    r"""
    Edit distance between two texts over the reference length, counted on characters with spaces and
    punctuation removed, since Korean spacing and the marks a recognizer adds are not what is being judged.

    Args:
        reference (`str`):
            The text that should have been said.
        hypothesis (`str`):
            The text recognized.

    Returns:
        `float`: Substitutions, insertions and deletions divided by the reference length, `1.0` for an empty
        reference with a non empty hypothesis.
    """
    reference = [character for character in reference if character.isalnum()]
    hypothesis = [character for character in hypothesis if character.isalnum()]
    if not reference:
        return 0.0 if not hypothesis else 1.0
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, found in enumerate(hypothesis, start=1):
            current.append(min(previous[column] + 1, current[column - 1] + 1, previous[column - 1] + (expected != found)))
        previous = current
    return previous[-1] / len(reference)


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

    @staticmethod
    def _edit_distance(reference: list, hypothesis: list) -> int:
        r"""
        Args:
            reference (`list`):
                Reference tokens.
            hypothesis (`list`):
                Hypothesis tokens.

        Returns:
            `int`: The Levenshtein distance between the two sequences.
        """
        previous = list(range(len(hypothesis) + 1))
        for i, reference_token in enumerate(reference, start=1):
            current = [i]
            for j, hypothesis_token in enumerate(hypothesis, start=1):
                current.append(
                    min(
                        previous[j] + 1,
                        current[j - 1] + 1,
                        previous[j - 1] + (reference_token != hypothesis_token),
                    )
                )
            previous = current
        return previous[-1]

    @classmethod
    def _error_rate(cls, references: list, hypotheses: list, tokenize) -> float:
        r"""
        Args:
            references (`list` of `str`):
                Ground truth strings.
            hypotheses (`list` of `str`):
                Predicted strings, aligned with `references`.
            tokenize (`Callable`):
                Splits one string into the units the rate counts.

        Returns:
            `float`: Total edits over total reference units, `0.0` when the references are empty.

        Raises:
            ValueError: If the two lists differ in length.
        """
        if len(references) != len(hypotheses):
            raise ValueError(
                f"Got {len(references)} references and {len(hypotheses)} hypotheses, they must be aligned."
            )
        edits = 0
        units = 0
        for reference, hypothesis in zip(references, hypotheses):
            reference_tokens = tokenize(reference)
            edits += cls._edit_distance(reference_tokens, tokenize(hypothesis))
            units += len(reference_tokens)
        return edits / units if units else 0.0

    @classmethod
    def character_error_rate(cls, references: list, hypotheses: list) -> float:
        r"""
        Args:
            references (`list` of `str`):
                Ground truth transcriptions.
            hypotheses (`list` of `str`):
                ASR output of the generated speech.

        Returns:
            `float`: Corpus level CER. Whitespace is collapsed but not dropped.
        """
        return cls._error_rate(references, hypotheses, lambda text: list(" ".join(text.split())))

    @classmethod
    def word_error_rate(cls, references: list, hypotheses: list) -> float:
        r"""
        Args:
            references (`list` of `str`):
                Ground truth transcriptions.
            hypotheses (`list` of `str`):
                ASR output of the generated speech.

        Returns:
            `float`: Corpus level WER over whitespace tokens.
        """
        return cls._error_rate(references, hypotheses, lambda text: text.split())

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


class Phase1Scorer:
    r"""
    Constructs the scorer of Phase 1 runs: Korean, English and timing losses, the same-meaning comparison with
    English, and generation checks.

    Every number is read off the model's own forward, so whatever a run adds inside the backbone is part of
    what is scored.

    Args:
        model_name_or_path (`str`):
            Expanded checkpoint the runs were trained on.
        batch_size (`int`, *optional*, defaults to 16):
            Utterances scored at a time.
        max_frames (`int`, *optional*, defaults to 320):
            Hard cap on the length of an item, in frames.
        persona (`str`, *optional*, defaults to `"You are a helpful assistant."`):
            Instruction of the persona prefix used in generation.
    """

    def __init__(
        self,
        model_name_or_path: str,
        batch_size: int = 16,
        max_frames: int = 320,
        persona: str = "You are a helpful assistant.",
    ):
        from ..models.polyplex import PolyplexProcessor

        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.max_frames = max_frames
        self.persona = persona
        self.processor = PolyplexProcessor.from_pretrained(model_name_or_path)
        self.processor.audio_tokenizer.to("cuda")

    def load_run(self, run_dir: str):
        r"""
        Args:
            run_dir (`str`):
                Directory of a Phase 1 run or checkpoint.

        Returns:
            `EfficientAdapterP1Model`: The run's model on the GPU, in evaluation mode.
        """
        from .assemble import Phase1Assemble

        return Phase1Assemble.from_run(self.model_name_or_path, run_dir).cuda()

    @staticmethod
    def korean_items(path: str, exclude: str | None = None) -> tuple:
        r"""
        Args:
            path (`str`):
                Duplex cache of the Korean set.
            exclude (`str`, *optional*):
                Duplex cache whose texts are left out, so a validation set does not score sentences the run
                trained on.

        Returns:
            `tuple`: The items and their total character count without spaces.
        """
        from ..datasets.duplex_data import DuplexUtteranceDataset

        dataset = DuplexUtteranceDataset(path)
        banned = set()
        if exclude:
            with open(os.path.join(exclude, "index.jsonl")) as handle:
                banned = {json.loads(line)["text"] for line in handle}
        keep = [index for index, entry in enumerate(dataset.entries) if entry["text"] not in banned]
        characters = sum(len(dataset.entries[index]["text"].replace(" ", "")) for index in keep)
        return [dataset[index] for index in keep], characters

    @staticmethod
    def english_items(path: str) -> list:
        r"""
        Args:
            path (`str`):
                `torch.save` of English utterances carrying `text` and their own `codes`.

        Returns:
            `list`: Items in the form the collator reads.
        """
        return [
            {"codes": torch.as_tensor(e.codes)[:8].long(), "words": e.text.split(), "starts": [0.0] * len(e.text.split())}
            for e in torch.load(path, weights_only=False)
        ]

    @staticmethod
    def paced_item(text: str, tokenizer, frames_per_piece: float = 4.049) -> dict:
        r"""
        Lays a sentence out at a fixed rhythm, for comparing languages on the same meaning.

        Args:
            text (`str`):
                The sentence.
            tokenizer (`PreTrainedTokenizerBase`):
                Tokenizer of the text stream.
            frames_per_piece (`float`, *optional*, defaults to 4.049):
                Frames each text piece is given.

        Returns:
            `dict`: An item with placeholder `codes` (silence replaces them), `words` and `starts`.
        """
        words, starts, pieces = text.split(), [], 0
        for word in words:
            # The first word is announced on the frame before it, so it starts a few frames in.
            starts.append(pieces * frames_per_piece / 12.5 + 0.4)
            pieces += max(1, len(tokenizer.encode(" " + word, add_special_tokens=False)))
        frames = int((pieces + 2) * frames_per_piece) + 8
        return {"codes": torch.zeros(8, frames, dtype=torch.long), "words": words, "starts": starts}

    def _collate(self, model, drop_audio: bool, layout=None):
        from ..datasets.duplex_data import duplex_collate_fn

        return duplex_collate_fn(
            self.processor, model.model, score_words_only=False, drop_audio=drop_audio,
            max_frames=self.max_frames, layout=layout,
        )

    @torch.no_grad()
    def score_korean(self, model, items: list, drop_audio: bool) -> dict:
        r"""
        Args:
            model (`EfficientAdapterP1Model`):
                The run's model.
            items (`list`):
                Korean items, see [`~Phase1Scorer.korean_items`].
            drop_audio (`bool`):
                Whether the assistant stream is codec silence (the condition Phase 1 trains in) or the
                utterance's own speech.

        Returns:
            `dict`: `korean` (word frame loss), `korean_new` (word frames whose target is a new token),
            `korean_sum`, `timing` (`PAD` and `EPAD` frames), `timing_cost` (the same minus its value with the
            new embeddings left out) and `word_frames`.
        """
        import torch.nn.functional as F

        from ..datasets.duplex_data import EPAD_TOKEN_ID, PAD_TOKEN_ID

        collate = self._collate(model, drop_audio)
        base = model.config.base_vocab_size
        sums = dict.fromkeys(("word", "new", "timing", "timing_alone"), 0.0)
        counts = dict.fromkeys(("word", "new", "timing"), 0)
        for start in range(0, len(items), self.batch_size):
            batch = {key: value.cuda() for key, value in collate(items[start : start + self.batch_size]).items()}
            labels = batch.pop("text_labels")[:, 1:]
            logits = model(**batch).logits[:, :-1].float()
            scored = labels != -100
            timing = scored & ((labels == PAD_TOKEN_ID) | (labels == EPAD_TOKEN_ID))
            word = scored & ~timing
            new = word & (labels >= base)
            loss = F.cross_entropy(logits.transpose(1, 2), labels.clamp(min=0), reduction="none")
            alone = F.cross_entropy(
                logits[..., :base].transpose(1, 2), labels.clamp(min=0, max=base - 1), reduction="none"
            )
            sums["word"] += loss[word].sum().item()
            counts["word"] += int(word.sum())
            sums["new"] += loss[new].sum().item()
            counts["new"] += int(new.sum())
            sums["timing"] += loss[timing].sum().item()
            sums["timing_alone"] += alone[timing].sum().item()
            counts["timing"] += int(timing.sum())
        return {
            "korean": sums["word"] / counts["word"],
            "korean_new": sums["new"] / max(counts["new"], 1),
            "korean_sum": sums["word"],
            "timing": sums["timing"] / counts["timing"],
            "timing_cost": (sums["timing"] - sums["timing_alone"]) / counts["timing"],
            "word_frames": counts["word"],
        }

    @torch.no_grad()
    def score_english(self, model, items: list) -> dict:
        r"""
        Args:
            model (`EfficientAdapterP1Model`):
                The run's model.
            items (`list`):
                English items with their own speech, see [`~Phase1Scorer.english_items`].

        Returns:
            `dict`: `english` (word frame loss), `english_cost` (minus its value with the new embeddings left
            out), and `drift_word` and `drift_timing` (KL from the model with its backbone corrections off to
            the model as it is, on the pretrained vocabulary).
        """
        import torch.nn.functional as F

        from ..datasets.duplex_data import EPAD_TOKEN_ID, PAD_TOKEN_ID, packed_words

        collate = self._collate(model, drop_audio=False, layout=packed_words)
        base = model.config.base_vocab_size
        sums = dict.fromkeys(("full", "alone", "drift_word", "drift_timing"), 0.0)
        counts = dict.fromkeys(("word", "timing"), 0)
        for start in range(0, len(items), self.batch_size):
            batch = {key: value.cuda() for key, value in collate(items[start : start + self.batch_size]).items()}
            labels = batch.pop("text_labels")[:, 1:]
            logits = model(**batch).logits[:, :-1].float()
            with model.corrections_disabled():
                original = model(**batch).logits[:, :-1].float()
            scored = labels != -100
            timing = scored & ((labels == PAD_TOKEN_ID) | (labels == EPAD_TOKEN_ID))
            word = scored & ~timing
            sums["full"] += F.cross_entropy(logits[word], labels[word], reduction="sum").item()
            sums["alone"] += F.cross_entropy(logits[word][:, :base], labels[word], reduction="sum").item()
            reference = original[..., :base].log_softmax(-1)
            current = logits[..., :base].log_softmax(-1)
            kl = (reference.exp() * (reference - current)).sum(-1)
            sums["drift_word"] += kl[word].sum().item()
            sums["drift_timing"] += kl[timing].sum().item()
            counts["word"] += int(word.sum())
            counts["timing"] += int(timing.sum())
        return {
            "english": sums["full"] / counts["word"],
            "english_cost": (sums["full"] - sums["alone"]) / counts["word"],
            "drift_word": sums["drift_word"] / counts["word"],
            "drift_timing": sums["drift_timing"] / max(counts["timing"], 1),
        }

    @torch.no_grad()
    def score_parallel(self, model, pairs: list, frames_per_piece: float = 4.049) -> dict:
        r"""
        Scores Korean against English on the same meaning, under the silent condition.

        Args:
            model (`EfficientAdapterP1Model`):
                The run's model.
            pairs (`list`):
                `{"ko": ..., "en": ...}` sentence pairs.
            frames_per_piece (`float`, *optional*, defaults to 4.049):
                Rhythm both sides are laid out at, see [`~Phase1Scorer.paced_item`].

        Returns:
            `dict`: For `korean`, `english_off` (backbone corrections off) and `english_on`, the word frame
            loss per token and per sentence.
        """
        import torch.nn.functional as F

        from ..datasets.duplex_data import EPAD_TOKEN_ID, PAD_TOKEN_ID

        tokenizer = self.processor.tokenizer
        collate = self._collate(model, drop_audio=True)

        def score(items):
            total = count = 0.0
            for item in items:
                batch = {key: value.cuda() for key, value in collate([item]).items()}
                labels = batch.pop("text_labels")[:, 1:]
                logits = model(**batch).logits[:, :-1].float()
                word = (labels != -100) & (labels != PAD_TOKEN_ID) & (labels != EPAD_TOKEN_ID)
                loss = F.cross_entropy(logits.transpose(1, 2), labels.clamp(min=0), reduction="none")[word]
                total += loss.sum().item()
                count += word.sum().item()
            return {"per_token": total / count, "per_sentence": total / len(items)}

        korean = [self.paced_item(pair["ko"], tokenizer, frames_per_piece) for pair in pairs]
        english = [self.paced_item(pair["en"], tokenizer, frames_per_piece) for pair in pairs]
        result = {"korean": score(korean)}
        with model.corrections_disabled():
            result["english_off"] = score(english)
        result["english_on"] = score(english)
        return result

    def opening_stream(self, words: list, frames_per_piece: float = 4.049) -> list:
        r"""
        Args:
            words (`list`):
                Opening words to force.
            frames_per_piece (`float`, *optional*, defaults to 4.049):
                Rhythm the words are laid out at.

        Returns:
            `list`: Text token of every forced frame.
        """
        from ..datasets.duplex_data import text_stream

        tokenizer = self.processor.tokenizer
        starts, pieces = [], 0
        for word in words:
            starts.append(pieces * frames_per_piece / 12.5)
            pieces += max(1, len(tokenizer.encode(" " + word, add_special_tokens=False)))
        frames = int((pieces + 1) * frames_per_piece) + 1
        # Offset 1 leaves room for the announcing EPAD of the first word.
        return text_stream(words, starts, frames, tokenizer, offset=1).tolist()

    @torch.no_grad()
    def generate(
        self, model, steps: int, seed: int, stream: list | None = None, silent: bool = False, return_codes: bool = False
    ):
        r"""
        Generates with the official `generate`, optionally forcing the opening words and the silent condition.

        Args:
            model (`PolyplexForConditionalGeneration`):
                The folded model, see [`EfficientAdapterP1Model.fold_into_base`].
            steps (`int`):
                Frames generated.
            seed (`int`):
                Sampling seed.
            stream (`list`, *optional*):
                Text tokens forced on the first frames, see [`~Phase1Scorer.opening_stream`].
            silent (`bool`, *optional*, defaults to `False`):
                Whether the assistant codebooks are pinned to codec silence.
            return_codes (`bool`, *optional*, defaults to `False`):
                Whether the assistant's generated codes are returned with the text.

        Returns:
            `str`, or `tuple` of the text and the codes of shape `(num_codebooks, steps)` with `return_codes`.
        """
        from transformers import LogitsProcessorList

        torch.manual_seed(seed)
        inputs = self.processor(text_prompt=self.persona).to("cuda")
        user = self.processor.get_silence_audio_codes(steps + 1, batch_size=1).to("cuda")
        inputs["user_audio_codes"] = torch.cat([inputs["user_audio_codes"], user], dim=2)
        extra = {}
        if stream is not None:
            extra["logits_processor"] = LogitsProcessorList([ForceTextLogitsProcessor(stream)])
        if silent:
            silence = self.processor.get_silence_audio_codes(steps + 1)[0].tolist()
            extra["depth_decoder_logits_processor"] = LogitsProcessorList([SilentAssistantLogitsProcessor(silence)])
        output = model.generate(
            **inputs, max_new_tokens=steps, do_sample=True, temperature=0.8, top_k=250, return_audio_codes=True, **extra
        )
        prompt = inputs["input_ids"].shape[-1]
        text = self.processor.tokenizer.decode(output.sequences[0, prompt:], skip_special_tokens=True).strip()
        if not return_codes:
            return text
        return text, output.audio_codes[0, :, -steps:]


CAPACITY_GROUPS = ("deploy_en_speech", "deploy_silent_pad", "empty_ko", "empty_en", "speech_ko")
"""Frame groups of the capacity measurement. The two `deploy_` groups are what deployment reads (English with its
own speech, and `PAD` frames of an assistant that is listening), `empty_ko` and `empty_en` are word and `EPAD`
frames under a silent assistant stream, and `speech_ko` are Korean word frames with their own speech."""

CAPACITY_MODULES = ("attention", "ffn")


class CapacityMeter:
    r"""
    Constructs a meter of how much room each backbone layer leaves for capacity that only the empty space reads.

    Nothing is trained. The frozen backbone is run over the frame groups of [`CAPACITY_GROUPS`], and the input of
    every layer's attention (after `input_layernorm`, shared by q, k and v) and of its FFN (after
    `post_attention_layernorm`) is accumulated as an uncentered second moment per group. A linear correction
    reading an input `x` answers `W x`, so its output on a group is governed by that group's second moment.

    Args:
        model (`EfficientAdapterP1Model`):
            The assembled Phase 1 model, in evaluation mode on the GPU.
        processor (`PolyplexProcessor`):
            Processor of the expanded checkpoint.
        max_frames (`int`, *optional*, defaults to 320):
            Hard cap on the length of an item, in frames.
        batch_size (`int`, *optional*, defaults to 16):
            Utterances run at a time.
    """

    def __init__(self, model, processor, max_frames: int = 320, batch_size: int = 16):
        self.model = model
        self.processor = processor
        self.max_frames = max_frames
        self.batch_size = batch_size
        self.layers = model.model.model.layers
        hidden_size = self.layers[0].hidden_size
        device = next(self.layers.parameters()).device
        self.moments = {
            (group, index, module): torch.zeros(hidden_size, hidden_size, dtype=torch.float32, device=device)
            for group in CAPACITY_GROUPS
            for index in range(len(self.layers))
            for module in CAPACITY_MODULES
        }
        self.counts = dict.fromkeys(CAPACITY_GROUPS, 0)
        self.masks = {}
        self.handles = []
        for index, layer in enumerate(self.layers):
            self.handles.append(layer.self_attn.q_proj.register_forward_pre_hook(self._hook(index, "attention")))
            self.handles.append(layer.mlp.fc1.register_forward_pre_hook(self._hook(index, "ffn")))

    def _hook(self, index: int, module: str):
        def hook(_, args):
            hidden = args[0]
            for group, mask in self.masks.items():
                rows = hidden[mask].float()
                if rows.shape[0]:
                    self.moments[(group, index, module)].addmm_(rows.T, rows)

        return hook

    @staticmethod
    def frame_masks(batch: dict, kind: str) -> dict:
        r"""
        Args:
            batch (`dict`):
                A collated duplex batch.
            kind (`str`):
                Which corpus the batch came from: `"en_speech"`, `"en_silent"`, `"ko_silent"` or `"ko_speech"`.

        Returns:
            `dict`: Boolean `(batch_size, sequence_length)` masks of the groups this batch feeds, selected by the
            text token each frame reads.
        """
        from ..datasets.duplex_data import PAD_TOKEN_ID

        ids = batch["input_ids"]
        body = batch["text_labels"] != -100
        pad = body & (ids == PAD_TOKEN_ID)
        words = body & (ids != PAD_TOKEN_ID)
        if kind == "en_speech":
            return {"deploy_en_speech": body}
        if kind == "en_silent":
            return {"deploy_silent_pad": pad, "empty_en": words}
        if kind == "ko_silent":
            return {"deploy_silent_pad": pad, "empty_ko": words}
        return {"speech_ko": words}

    @torch.no_grad()
    def collect(self, items: list, kind: str, drop_audio: bool, layout=None):
        r"""
        Runs the backbone over one corpus and adds its frames to the groups `kind` feeds.

        Args:
            items (`list`):
                Utterances with `codes`, `words` and `starts`.
            kind (`str`):
                See [`~CapacityMeter.frame_masks`].
            drop_audio (`bool`):
                Whether the assistant stream is replaced by codec silence.
            layout (`Callable`, *optional*):
                Text layout passed to [`duplex_collate_fn`], the aligned one when omitted.
        """
        from ..datasets.duplex_data import duplex_collate_fn

        collate = duplex_collate_fn(
            self.processor, self.model.model, score_words_only=False, drop_audio=drop_audio,
            max_frames=self.max_frames, layout=layout,
        )
        for start in range(0, len(items), self.batch_size):
            batch = {key: value.cuda() for key, value in collate(items[start : start + self.batch_size]).items()}
            self.masks = self.frame_masks(batch, kind)
            for group, mask in self.masks.items():
                self.counts[group] += int(mask.sum())
            batch.pop("text_labels")
            self.model(**batch)
        self.masks = {}

    def close(self):
        r"""
        Removes the hooks from the backbone.
        """
        for handle in self.handles:
            handle.remove()

    @staticmethod
    def analyze(moment: dict, counts: dict) -> dict:
        r"""
        Args:
            moment (`dict`):
                Second moment of every group for one layer and module.
            counts (`dict`):
                Frames behind every group.

        Returns:
            `dict`: `protected_rank_99` and `_999` (directions holding that share of the deploy energy),
            `empty_outside_*` (share of the Korean silent energy outside them), `allowed_rank` (complement
            directions where the Korean silent energy is at least 100 times the deploy energy and at least 0.1%
            of the total), `allowed_empty_share`, and `wiring_reach` (share of the Korean speech energy on the
            allowed directions).
        """
        normalized = {group: moment[group] / max(counts[group], 1) for group in CAPACITY_GROUPS}
        deploy = 0.5 * (normalized["deploy_en_speech"] + normalized["deploy_silent_pad"])
        values, vectors = torch.linalg.eigh(deploy.double())
        values, vectors = values.flip(0).clamp(min=0), vectors.flip(1)
        share = values.cumsum(0) / values.sum()

        empty = normalized["empty_ko"].double()
        speech = normalized["speech_ko"].double()
        english = normalized["empty_en"].double()
        report = {}
        for threshold, name in ((0.99, "99"), (0.999, "999")):
            rank = int((share < threshold).sum()) + 1
            basis = vectors[:, :rank]
            outside = torch.eye(basis.shape[0], dtype=basis.dtype, device=basis.device) - basis @ basis.T
            report[f"protected_rank_{name}"] = rank
            report[f"empty_outside_{name}"] = float(torch.trace(outside @ empty @ outside) / torch.trace(empty))
            report[f"empty_en_outside_{name}"] = float(
                torch.trace(outside @ english @ outside) / torch.trace(english).clamp(min=1e-12)
            )
            report[f"speech_ko_outside_{name}"] = float(torch.trace(outside @ speech @ outside) / torch.trace(speech))

        basis = vectors[:, : report["protected_rank_99"]]
        outside = torch.eye(basis.shape[0], dtype=basis.dtype, device=basis.device) - basis @ basis.T
        residual_values, residual_vectors = torch.linalg.eigh(outside @ empty @ outside)
        residual_values, residual_vectors = residual_values.flip(0), residual_vectors.flip(1)
        deploy_energy = torch.einsum("hd,he,ed->d", residual_vectors, deploy.double(), residual_vectors)
        empty_energy = residual_values.clamp(min=0)
        allowed = (empty_energy >= 1e-3 * torch.trace(empty)) & (empty_energy >= 100 * deploy_energy.clamp(min=1e-12))
        allowed_vectors = residual_vectors[:, allowed]
        report["allowed_rank"] = int(allowed.sum())
        report["allowed_empty_share"] = float(empty_energy[allowed].sum() / torch.trace(empty))
        report["wiring_reach"] = (
            float(torch.trace(allowed_vectors.T @ speech @ allowed_vectors) / torch.trace(speech))
            if allowed_vectors.shape[1]
            else 0.0
        )
        return report

    def write(self, output_dir: str) -> dict:
        r"""
        Writes the per layer second moments (`moments_<layer>_<module>.pt`) and the analysis (`report.json`).

        Args:
            output_dir (`str`):
                Directory written to.

        Returns:
            `dict`: The report.
        """
        os.makedirs(output_dir, exist_ok=True)
        report = {"counts": self.counts, "layers": []}
        for index in range(len(self.layers)):
            row = {"layer": index}
            for module in CAPACITY_MODULES:
                moment = {group: self.moments[(group, index, module)] for group in CAPACITY_GROUPS}
                row[module] = self.analyze(moment, self.counts)
                torch.save(
                    {group: (value / max(self.counts[group], 1)).half().cpu() for group, value in moment.items()},
                    os.path.join(output_dir, f"moments_{index:02d}_{module}.pt"),
                )
            report["layers"].append(row)
        with open(os.path.join(output_dir, "report.json"), "w") as handle:
            json.dump(report, handle, indent=1)
        return report


def measure_capacity(
    model_dir: str,
    adapter_dir: str,
    output_dir: str,
    english_speech: str,
    korean_text: str,
    korean_speech: str,
    limit: int = 800,
    batch_size: int = 16,
    max_frames: int = 320,
) -> dict:
    r"""
    Runs [`CapacityMeter`] over the four corpora and writes its moments and report.

    Args:
        model_dir (`str`):
            Expanded checkpoint.
        adapter_dir (`str`):
            Phase 1 run whose new embeddings the Korean tokens are read with.
        output_dir (`str`):
            Directory the report and the moments are written to.
        english_speech (`str`):
            `torch.save` of English utterances with `codes` and `text`.
        korean_text (`str`):
            Duplex cache of the Korean text corpus.
        korean_speech (`str`):
            Duplex cache of Korean utterances with speech.
        limit (`int`, *optional*, defaults to 800):
            Utterances read from each corpus.
        batch_size (`int`, *optional*, defaults to 16):
            Utterances run at a time.
        max_frames (`int`, *optional*, defaults to 320):
            Hard cap on the length of an item, in frames.

    Returns:
        `dict`: The report.
    """
    from ..datasets.duplex_data import DuplexUtteranceDataset, packed_words
    from ..models.polyplex import PolyplexProcessor
    from .assemble import Phase1Assemble
    from .phase import load_adapter_slices

    model, _ = Phase1Assemble(model_dir, base_row_mixture=64, tie_new_input_rows=True).assemble()
    load_adapter_slices(model, adapter_dir)
    model = model.cuda().eval()
    processor = PolyplexProcessor.from_pretrained(model_dir)
    processor.audio_tokenizer.to("cuda")

    english = [
        {"codes": torch.as_tensor(e.codes)[:8].long(), "words": e.text.split(), "starts": [0.0] * len(e.text.split())}
        for e in torch.load(english_speech, weights_only=False)[:limit]
    ]

    def sample(path):
        dataset = DuplexUtteranceDataset(path)
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(0))[: min(limit, len(dataset))]
        return [dataset[int(index)] for index in order]

    meter = CapacityMeter(model, processor, max_frames, batch_size)
    meter.collect(english, "en_speech", drop_audio=False, layout=packed_words)
    meter.collect(english, "en_silent", drop_audio=True, layout=packed_words)
    meter.collect(sample(korean_text), "ko_silent", drop_audio=True)
    meter.collect(sample(korean_speech), "ko_speech", drop_audio=False)
    meter.close()
    return meter.write(output_dir)


def build_confined_basis(moments_dir: str, layers: str, rank: int, module: str, output: str) -> int:
    r"""
    Writes the input directions a confined correction reads at each chosen layer, for `confined_basis`.

    Args:
        moments_dir (`str`):
            Directory [`CapacityMeter.write`] wrote.
        layers (`str`):
            Inclusive layer range such as `"8-31"`.
        rank (`int`):
            Directions per layer.
        module (`str`):
            `"ffn"` or `"attention"`.
        output (`str`):
            Safetensors file written, keyed `layer_<index>`.

    Returns:
        `int`: Number of layers written.
    """
    from safetensors.torch import save_file

    from ..adapters.phase_1_efficient_adapter.modeling_phase_1_efficient_adapter import ConfinedCapacity

    first, last = (int(value) for value in layers.split("-"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensors = {}
    for index in range(first, last + 1):
        moments = torch.load(os.path.join(moments_dir, f"moments_{index:02d}_{module}.pt"))
        moments = {group: value.to(device) for group, value in moments.items()}
        tensors[f"layer_{index}"] = ConfinedCapacity.directions_from_moments(moments, rank).cpu().contiguous()
    save_file(tensors, output)
    return len(tensors)


"""The held out English speech set (`train.sh` `SCORE_ENGLISH_SPEECH`)."""


"""NVFP4 variants `nvfp4-check` compares with full precision: what S7.1 runs, and the usual exceptions."""


if __name__ == "__main__":
    main()


__all__ = [
    "CAPACITY_GROUPS",
    "CapacityMeter",
    "Evaluator",
    "ForceTextLogitsProcessor",
    "Phase1Scorer",
    "SilentAssistantLogitsProcessor",
    "build_confined_basis",
    "measure_capacity",
]
