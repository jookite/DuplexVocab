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


def language_model_continuation(
    model, tokenizer, prompt: str, max_new_tokens: int = 60, temperature: float = 0.0, top_k: int = 50, seed: int = 0
) -> str:
    r"""
    Continues a text prompt with a text LLM and its own tokenizer, the reference the dense samples are read against:
    greedily, or sampled from the `top_k` most likely tokens at `temperature`.

    Args:
        model (`PreTrainedModel`):
            The text LLM, in evaluation mode.
        tokenizer (`PreTrainedTokenizer`):
            Its tokenizer.
        prompt (`str`):
            The opening text.
        max_new_tokens (`int`, *optional*, defaults to 60):
            Tokens generated after the prompt.
        temperature (`float`, *optional*, defaults to 0.0):
            Sampling temperature. `0` takes the most likely token.
        top_k (`int`, *optional*, defaults to 50):
            Tokens sampled from at each step.
        seed (`int`, *optional*, defaults to 0):
            Seed of the sampling.

    Returns:
        `str`: The generated text, without the prompt.
    """
    ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=model.device)
    sampling = {"do_sample": True, "temperature": temperature, "top_k": top_k} if temperature > 0 else {"do_sample": False}
    if temperature > 0:
        torch.manual_seed(seed)
    with torch.no_grad():
        output = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,
            **sampling,
        )
    return tokenizer.decode(output[0, ids.shape[1] :].tolist())


def dense_generate(
    model, collate, prompt: torch.Tensor, max_new_tokens: int = 80, temperature: float = 0.8, top_k: int = 50, seed: int = 0
) -> torch.Tensor:
    r"""
    Continues a text prompt in the dense layout the first stage of Phase 1 trains in, with no PAD frames.

    Every step reads the whole sequence again through [`~duplex_data.dense_collate_fn`], so the opening frame
    and the codec silence are those of training. The appended vocabulary and the pretrained one are sampled
    alike.

    Args:
        model (`EfficientAdapterP1Model`):
            The model, in evaluation mode.
        collate (`Callable`):
            The collator of the dense stage.
        prompt (`torch.Tensor` of shape `(length,)`):
            Pieces of the opening text.
        max_new_tokens (`int`, *optional*, defaults to 80):
            Pieces generated after the prompt.
        temperature (`float`, *optional*, defaults to 0.8):
            Sampling temperature. `0` takes the most likely piece.
        top_k (`int`, *optional*, defaults to 50):
            Pieces sampled from at each step.
        seed (`int`, *optional*, defaults to 0):
            Seed of the sampling.

    Returns:
        `torch.Tensor` of shape `(max_new_tokens,)`: The generated pieces.
    """
    from ..datasets.duplex_data import PAD_TOKEN_ID

    device = next(model.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(seed)
    vocab = model.config.vocab_size
    pieces = torch.as_tensor(prompt, dtype=torch.long).cpu()
    produced = []
    for _ in range(max_new_tokens):
        # The delay pattern rewrites the acoustic codes of a row's last frame, a frame training never reads a
        # prediction from, so one padding frame follows the prompt and the prediction is read before it.
        padded = torch.cat([pieces, torch.tensor([PAD_TOKEN_ID])])
        batch = collate([{"segments": [padded]}])
        inputs = {k: v.to(device) for k, v in batch.items() if k != "text_labels"}
        with torch.no_grad():
            logits = model(**inputs).logits[0, len(pieces), :vocab].float().cpu()
        if temperature <= 0:
            choice = int(logits.argmax())
        else:
            values, indices = logits.topk(min(top_k, vocab))
            probs = torch.softmax(values / temperature, dim=-1)
            choice = int(indices[torch.multinomial(probs, 1, generator=generator)])
        produced.append(choice)
        pieces = torch.cat([pieces, torch.tensor([choice])])
    return torch.tensor(produced, dtype=torch.long)


_HANGUL = re.compile("[\uac00-\ud7a3]")
_DIGITS = re.compile(r"▁?[0-9]+")
_LATIN = re.compile("[A-Za-z]")


"""The held out English speech set (`train.sh` `SCORE_ENGLISH_SPEECH`)."""


def piece_kind(piece: str, previous: str | None) -> str:
    r"""
    Names what a text piece is, for scoring the dense stage by kind of piece.

    A Hangul or Latin piece starts a word when it carries the space marker `▁` or follows a piece that is the
    marker alone. Every other such piece continues the word before it.

    Args:
        piece (`str`):
            The piece as the tokenizer spells it.
        previous (`str`, *optional*):
            The piece before it, `None` at the start of a document.

    Returns:
        `str`: One of `hangul_word_start`, `hangul_inside_word`, `latin_word_start`, `latin_inside_word`, `space`,
        `digits`, `byte` and `punctuation`.
    """
    if piece.startswith("<0x"):
        return "byte"
    if piece == "▁":
        return "space"
    if _HANGUL.search(piece):
        if piece.startswith("▁") or previous is None or previous == "▁":
            return "hangul_word_start"
        return "hangul_inside_word"
    if _DIGITS.fullmatch(piece):
        return "digits"
    if _LATIN.search(piece):
        if piece.startswith("▁") or previous is None or previous == "▁":
            return "latin_word_start"
        return "latin_inside_word"
    return "punctuation"


def piece_kind_scores(model, collate, rows: list, tokenizer, batch_size: int = 8, base_only: bool = False) -> dict:
    r"""
    Scores packed dense rows by kind of target piece (see [`piece_kind`]).

    Args:
        model (`EfficientAdapterP1Model`):
            The model, in evaluation mode.
        collate (`Callable`):
            The collator of the dense stage.
        rows (`list[dict]`):
            Rows of [`~phase_data.DenseTextDataset`].
        tokenizer (`PreTrainedTokenizerBase`):
            Tokenizer of the text stream.
        batch_size (`int`, *optional*, defaults to 8):
            Rows per forward pass.
        base_only (`bool`, *optional*, defaults to `False`):
            Score the pretrained vocabulary alone: the softmax runs over the pretrained rows and appended
            targets are skipped. With the corrections disabled this is the original model.

    Returns:
        `dict`: For each kind, its `share` of the scored pieces, mean `loss`, `ppl` and `top1` accuracy, plus
        `all`.
    """
    import torch.nn.functional as F

    device = next(model.parameters()).device
    vocab = model.config.vocab_size
    names = tokenizer.convert_ids_to_tokens(list(range(vocab)))
    sums = {}
    for start in range(0, len(rows), batch_size):
        batch = collate(rows[start : start + batch_size])
        inputs = {k: v.to(device) for k, v in batch.items() if k != "text_labels"}
        with torch.no_grad():
            logits = model(**inputs).logits[:, :-1].float()
        labels = batch["text_labels"][:, 1:].clone()
        if base_only:
            base = model.config.base_vocab_size
            logits = logits[..., :base]
            labels[labels >= base] = -100
        previous = batch["input_ids"][:, :-1]
        losses = F.cross_entropy(logits.transpose(1, 2), labels.to(device).clamp(min=0), reduction="none").cpu()
        hits = (logits.argmax(-1).cpu() == labels)
        for row, column in (labels != -100).nonzero().tolist():
            before = int(previous[row, column])
            kind = piece_kind(names[int(labels[row, column])], names[before] if before < vocab else None)
            for key in (kind, "all"):
                total = sums.setdefault(key, [0.0, 0, 0])
                total[0] += float(losses[row, column])
                total[1] += 1
                total[2] += int(hits[row, column])
    count = sums["all"][1]
    return {
        key: {"share": n / count, "loss": s / n, "ppl": math.exp(s / n), "top1": h / n}
        for key, (s, n, h) in sums.items()
    }


def dense_checks(model, collate, rows: list, base_vocab_size: int, unigram_log_probs: torch.Tensor, seed: int = 0) -> dict:
    r"""
    Checks that a loss on packed dense rows measures prediction from context and nothing else.

    Four readings on the same rows. The loss itself, split into appended and pretrained vocabulary targets.
    The loss with the pieces of every document shuffled, which rises well above it only if the model reads
    its context. The loss a unigram model gives the same targets. And two leaks, as the largest change of a
    logit: on the frames before a point when every piece after it is replaced (`future_leak`), and on the
    second document of the first row when the first document is replaced (`document_leak`).

    Args:
        model (`EfficientAdapterP1Model`):
            The model, in evaluation mode.
        collate (`Callable`):
            The collator of the dense stage, see [`~duplex_data.dense_collate_fn`].
        rows (`list[dict]`):
            Rows of [`~phase_data.DenseTextDataset`]. The first one must hold two documents at least.
        base_vocab_size (`int`):
            First row of the appended vocabulary.
        unigram_log_probs (`torch.Tensor` of shape `(vocab_size,)`):
            Log probability of each piece under a unigram model of the training text.
        seed (`int`, *optional*, defaults to 0):
            Seed of the shuffle and of the replacement pieces.

    Returns:
        `dict`: `loss`, `loss_new`, `loss_base`, `shuffled`, `unigram`, `scored`, `future_leak` and
        `document_leak`.
    """
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed)

    def logits_of(batch):
        inputs = {k: v.to(device) for k, v in batch.items() if k != "text_labels"}
        with torch.no_grad():
            return model(**inputs).logits.float()

    def losses(batch):
        logits = logits_of(batch)[:, :-1]
        labels = batch["text_labels"][:, 1:].to(device)
        scored = labels != -100
        per_token = torch.nn.functional.cross_entropy(logits[scored], labels[scored], reduction="none")
        return per_token, labels[scored]

    batch = collate(rows)
    per_token, targets = losses(batch)
    new = targets >= base_vocab_size
    shuffled_rows = [
        {"segments": [s[torch.randperm(len(s), generator=generator)] for s in row["segments"]]} for row in rows
    ]
    shuffled, _ = losses(collate(shuffled_rows))
    unigram = -unigram_log_probs.to(device)[targets]

    first = rows[0]["segments"]
    reference = logits_of(collate([{"segments": first}]))[0]
    cut = 1 + len(first[0]) // 2
    replaced = [torch.cat([first[0][: cut - 1], torch.randint(0, base_vocab_size, (len(first[0]) - cut + 1,), generator=generator)])]
    replaced += [torch.randint(0, base_vocab_size, (len(s),), generator=generator) for s in first[1:]]
    future = logits_of(collate([{"segments": replaced}]))[0]
    other = [torch.randint(0, base_vocab_size, (len(first[0]),), generator=generator)] + list(first[1:])
    isolated = logits_of(collate([{"segments": other}]))[0]
    second = 1 + len(first[0])

    return {
        "loss": per_token.mean().item(),
        "loss_new": per_token[new].mean().item() if new.any() else float("nan"),
        "loss_base": per_token[~new].mean().item() if (~new).any() else float("nan"),
        "shuffled": shuffled.mean().item(),
        "unigram": unigram.mean().item(),
        "scored": int(targets.numel()),
        "future_leak": (future[:cut] - reference[:cut]).abs().max().item(),
        "document_leak": (isolated[second:] - reference[second:]).abs().max().item(),
    }


def unigram_log_probs(tokens, vocab_size: int, limit: int = 20_000_000) -> torch.Tensor:
    r"""
    Args:
        tokens (`numpy.ndarray`):
            Pieces of the training text end to end.
        vocab_size (`int`):
            Size of the vocabulary.
        limit (`int`, *optional*, defaults to 20,000,000):
            How many pieces from the start are counted.

    Returns:
        `torch.Tensor` of shape `(vocab_size,)`: Add-one smoothed log probability of every piece.
    """
    import numpy as np

    counts = np.bincount(np.asarray(tokens[:limit], dtype=np.int64), minlength=vocab_size)[:vocab_size] + 1
    return torch.from_numpy(np.log(counts / counts.sum())).float()


def dense_unigram_baseline(rows: list, log_probs: torch.Tensor, tokenizer) -> dict:
    r"""
    Scores packed rows with a unigram model, the cross entropy a model reaches without reading any context.

    The first piece of every document is left out, as the dense quick check leaves it out.

    Args:
        rows (`list`):
            Rows with a `segments` list of token tensors, one per document.
        log_probs (`torch.Tensor` of shape `(vocab_size,)`):
            Log probability of every piece, from [`unigram_log_probs`].
        tokenizer (`PreTrainedTokenizerBase`):
            Tokenizer the pieces come from, to measure the text they spell.

    Returns:
        `dict`: `loss` (mean cross entropy per scored piece, in nats), `tokens` (scored pieces), and
        `bytes_per_token` and `characters_per_token` of the scored text in UTF-8.
    """
    total, count, characters, utf8 = 0.0, 0, 0, 0
    for row in rows:
        for segment in row["segments"]:
            scored = segment[1:].long()
            if scored.numel() == 0:
                continue
            total -= log_probs[scored].double().sum().item()
            count += scored.numel()
            text = tokenizer.decode(scored.tolist())
            characters += len(text)
            utf8 += len(text.encode("utf-8"))
    count = max(count, 1)
    return {
        "loss": total / count,
        "tokens": count,
        "bytes_per_token": utf8 / count,
        "characters_per_token": characters / count,
    }


def frame_embedding_norms(text_embeds: torch.Tensor, audio_embeds: torch.Tensor, mask: torch.Tensor) -> dict:
    r"""
    Compares the two parts of a frame's input: the text embedding and the summed audio embeddings added to it.

    Args:
        text_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Text embedding of every frame.
        audio_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sum of the audio codebook embeddings of every frame.
        mask (`torch.Tensor` of shape `(batch_size, sequence_length)`):
            Frames to measure.

    Returns:
        `dict`: Mean L2 norm of the text part (`text`) and of the audio part (`audio`) over the frames, and
        their ratio (`audio_over_text`).
    """
    text = text_embeds.float().norm(dim=-1)[mask].mean().item()
    audio = audio_embeds.float().norm(dim=-1)[mask].mean().item()
    return {"text": text, "audio": audio, "audio_over_text": audio / text}


def dense_baselines(
    personaplex: str,
    language_model: str,
    work_dir: str,
    eval_rows: int = 64,
    row_frames: int = 1024,
    train_tokens: int = 2_000_000,
    max_files: int = 1,
) -> dict:
    r"""
    Reference points for the dense quick check, on the rows it scores: a unigram model per language, the text
    each piece spells, and how large the silent audio input is next to the text input of the assembled model.

    Args:
        personaplex (`str`):
            Duplex checkpoint, for its audio embeddings and the codec silence.
        language_model (`str`):
            Text LLM, for its tokenizer and its text embedding.
        work_dir (`str`):
            Directory the row caches are written to.
        eval_rows (`int`, *optional*, defaults to 64):
            Held out rows per language.
        row_frames (`int`, *optional*, defaults to 1024):
            Frames per row.
        train_tokens (`int`, *optional*, defaults to 2,000,000):
            Training pieces each language's unigram model is counted on.
        max_files (`int`, *optional*, defaults to 1):
            Parquet files read per language.

    Returns:
        `dict`: Per language, the result of [`dense_unigram_baseline`]; under `input_norms`, the result of
        [`frame_embedding_norms`] on the Korean rows and the mean row norm of Helium's own text table.
    """
    import numpy as np
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoTokenizer, PersonaPlexForConditionalGeneration

    from ..datasets.duplex_data import dense_collate_fn
    from ..models.polyplex import PolyplexProcessor
    from ..datasets.webtext import DENSE_EVALUATION
    from .phase import dense_documents, dense_eval_rows

    tokenizer = AutoTokenizer.from_pretrained(language_model)
    vocab_size = len(tokenizer)
    results, rows_by_language = {}, {}
    for language in DENSE_EVALUATION:
        rows = dense_eval_rows(language, tokenizer, work_dir, eval_rows, row_frames, max_files)
        rows_by_language[language] = rows
        pieces = [
            piece
            for text in dense_documents(language, "train", train_tokens, tokenizer, max_files)
            for piece in tokenizer.encode(" ".join(text.split()), add_special_tokens=False)
        ]
        log_probs = unigram_log_probs(np.asarray(pieces), vocab_size)
        results[language] = dense_unigram_baseline(rows, log_probs, tokenizer)
        print(f"[dense-baselines] {language}: {results[language]}", flush=True)

    index = json.load(open(hf_hub_download(language_model, "model.safetensors.index.json")))
    shard = hf_hub_download(language_model, index["weight_map"]["model.embed_tokens.weight"])
    with safe_open(shard, "pt") as weights:
        text_table = weights.get_tensor("model.embed_tokens.weight").float()

    helium = PersonaPlexForConditionalGeneration.from_pretrained(personaplex, torch_dtype=torch.bfloat16)
    processor = PolyplexProcessor.from_pretrained(personaplex)
    batch = dense_collate_fn(processor, helium)(rows_by_language["ko"][:8])
    with torch.no_grad():
        codes = torch.cat([batch["assistant_audio_codes"], batch["user_audio_codes"]], dim=1)
        audio = helium._embed_audio_codes(codes)
        opening = batch["position_ids"] == 0
        text = text_table[batch["input_ids"].clamp(max=text_table.shape[0] - 1)]
    results["input_norms"] = {
        **frame_embedding_norms(text, audio, ~opening),
        "helium_text_rows": helium.model.embed_tokens.weight.float().norm(dim=-1).mean().item(),
        "qwen_text_rows": text_table.norm(dim=-1).mean().item(),
    }
    return results


NVFP4_VARIANTS = {
    "W4A4": {},
    "W4 only": {"quantize_activations": False},
    "W4A4, down_proj full": {"skip": "down_proj"},
    "W4A4, MLP full": {"skip": "mlp"},
    "W4A4, first and last 2 layers full": {"skip": "edges"},
    "W4A4, Hadamard 128": {"hadamard_group_size": 128},
}
"""NVFP4 variants `nvfp4-check` compares with full precision: what S7.1 runs, and the usual exceptions."""


@torch.no_grad()
def nvfp4_variant_losses(model, rows: list, variants: dict | None = None) -> dict:
    r"""
    Per-token cross entropy of a causal LM on `rows`, in full precision and with its decoder layers' projections
    run through emulated NVFP4 ([`~quantization.emulate_linears_nvfp4`]) under each variant.

    Args:
        model (`PreTrainedModel`):
            A causal LM whose decoder layers are `model.model.layers`.
        rows (`list[list[int]]`):
            Token rows. The first token of each row is context only.
        variants (`dict`, *optional*):
            Name mapped to options: `quantize_activations`, `hadamard_group_size`, and `skip`, one of `"down_proj"`,
            `"mlp"` (every FFN projection) and `"edges"` (the first and last two layers). [`NVFP4_VARIANTS`] when
            omitted.

    Returns:
        `dict`: `"full precision"` and each variant mapped to the mean cross entropy per predicted token.
    """
    from .quantization import emulate_linears_nvfp4

    layers = model.model.layers
    count = len(layers)
    skips = {
        "down_proj": lambda name: name.endswith("down_proj"),
        "mlp": lambda name: ".mlp." in name,
        "edges": lambda name: int(name.split(".")[0]) in {0, 1, count - 2, count - 1},
    }

    def loss() -> float:
        total = tokens = 0.0
        for row in rows:
            ids = torch.tensor([row], device=model.device)
            logits = model(input_ids=ids).logits[0, :-1].float()
            total += torch.nn.functional.cross_entropy(logits, ids[0, 1:], reduction="sum").item()
            tokens += len(row) - 1
        return total / tokens

    results = {"full precision": loss()}
    for name, options in (NVFP4_VARIANTS if variants is None else variants).items():
        options = dict(options)
        skip = skips[options.pop("skip")] if "skip" in options else None
        restore = emulate_linears_nvfp4(layers, skip=skip, **options)
        try:
            results[name] = loss()
        finally:
            restore()
    return results


class FFNRecorder(torch.nn.Module):
    r"""
    Wraps one FFN of a DualFFN layer and adds up, over every position after the first `skip` of a row, the squared
    size of its input residual and of its output, and the sum and squared sum of its output per dimension
    ([`language_ffn_report`]).

    Args:
        ffn (`nn.Module`):
            The FFN.
        skip (`int`):
            Leading positions of each row left out.
        keep (`int`, *optional*, defaults to 0):
            Inputs kept, on the CPU, for a later look at the FFN's neurons ([`ffn_neuron_energy`]).
    """

    def __init__(self, ffn: torch.nn.Module, skip: int, keep: int = 0):
        super().__init__()
        self.ffn = ffn
        self.skip = skip
        self.keep = keep
        self.reset()

    def reset(self):
        self.count = 0
        self.output_sum = None
        self.output_squares = None
        self.inputs = []

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if sum(len(part) for part in self.inputs) < self.keep:
            self.inputs.append(hidden_states[:, self.skip :].reshape(-1, hidden_states.shape[-1]).float().cpu())
        output = self.ffn(hidden_states)
        kept = output[:, self.skip :].float().reshape(-1, output.shape[-1])
        self.count += kept.shape[0]
        sums, squares = kept.sum(0), kept.square().sum(0)
        self.output_sum = sums if self.output_sum is None else self.output_sum + sums
        self.output_squares = squares if self.output_squares is None else self.output_squares + squares
        return output


@torch.no_grad()
def language_ffn_report(model, collate, tokenizer, texts: list, skip: int = 4, top: int = 8, keep: dict | None = None) -> list:
    r"""
    Measures, layer by layer, how large the language FFN's output is beside the duplex FFN's and how much of it is the
    same vector at every position (SPEC S11.2): texts are read as dense rows and both FFN outputs are recorded.

    Args:
        model ([`PolyplexForConditionalGeneration`]):
            The DualFFN model.
        collate (`Callable`):
            [`~polyplex_research.datasets.duplex_data.dense_collate_fn`] of the model.
        tokenizer (`PreTrainedTokenizerBase`):
            The model's tokenizer.
        texts (`list[str]`):
            The texts.
        skip (`int`, *optional*, defaults to 4):
            Leading positions of each row left out.
        top (`int`, *optional*, defaults to 8):
            Dimensions counted in the share of the output's energy its largest dimensions carry.
        keep (`dict[int, int]`, *optional*):
            Layer mapped to how many language FFN inputs to return for it, under `inputs` in its entry.

    Returns:
        `list[dict]`: Per layer, the RMS of each FFN's output, the language FFN's over the duplex FFN's, the share of
        the language FFN's energy in its mean vector (the part every position shares) and in its `top` largest
        dimensions, and those dimensions.
    """
    device = next(model.parameters()).device
    layers = model.model.layers
    recorders = []
    for layer in layers:
        kept = (keep or {}).get(len(recorders), 0)
        recorders.append((FFNRecorder(layer.mlp.language_mlp, skip, kept), FFNRecorder(layer.mlp.duplex_mlp, skip)))
        layer.mlp.language_mlp, layer.mlp.duplex_mlp = recorders[-1]
    try:
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            batch = {key: value.to(device) for key, value in collate([{"segments": [torch.tensor(ids)]}]).items()}
            batch.pop("text_labels")
            model(**batch, use_cache=False)
    finally:
        for layer, (language, duplex) in zip(layers, recorders):
            layer.mlp.language_mlp, layer.mlp.duplex_mlp = language.ffn, duplex.ffn
    report = []
    for index, (language, duplex) in enumerate(recorders):
        energy = language.output_squares / language.count
        mean = language.output_sum / language.count
        rms = energy.sum().sqrt() / energy.numel() ** 0.5
        duplex_rms = (duplex.output_squares.sum() / duplex.count).sqrt() / energy.numel() ** 0.5
        largest = energy.topk(top)
        report.append(
            {
                "layer": index,
                "language_rms": round(float(rms), 4),
                "duplex_rms": round(float(duplex_rms), 4),
                "language_over_duplex": round(float(rms / duplex_rms), 2),
                "mean_share": round(float(mean.square().sum() / energy.sum()), 3),
                "top_share": round(float(largest.values.sum() / energy.sum()), 3),
                "top_dims": largest.indices.tolist(),
            }
        )
        if language.inputs:
            report[-1]["inputs"] = torch.cat(language.inputs)[: keep[index]]
    return report


@torch.no_grad()
def ffn_neuron_energy(inputs: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor, top: int = 16) -> dict:
    r"""
    Ranks the intermediate neurons of a SwiGLU FFN by the energy they put into its output on given inputs: neuron
    $j$ adds $a_j\,W_\text{down}[:, j]$, so its energy is the mean of $a_j^2$ times $\lVert W_\text{down}[:, j]\rVert^2$.

    Args:
        inputs (`torch.Tensor` of shape `(positions, hidden_size)`):
            The normalized inputs the FFN reads.
        gate (`torch.Tensor` of shape `(intermediate_size, hidden_size)`):
            The gate projection.
        up (`torch.Tensor` of shape `(intermediate_size, hidden_size)`):
            The up projection.
        down (`torch.Tensor` of shape `(hidden_size, intermediate_size)`):
            The down projection.
        top (`int`, *optional*, defaults to 16):
            Neurons reported.

    Returns:
        `dict`: The `top` neurons, their share of the summed neuron energy, the mean activation of each, the share of
        the output's energy that is left once they are removed, and the output dimensions each writes most to.
    """
    inputs = inputs.float()
    activation = torch.nn.functional.silu(inputs @ gate.float().T) * (inputs @ up.float().T)
    down = down.float()
    energy = activation.square().mean(0) * down.square().sum(0)
    largest = energy.topk(top)
    output = activation @ down.T
    kept = activation.clone()
    kept[:, largest.indices] = 0
    left = (kept @ down.T).square().sum() / output.square().sum()
    return {
        "neurons": largest.indices.tolist(),
        "energy_share": [round(float(value / energy.sum()), 4) for value in largest.values],
        "mean_activation": [round(float(activation[:, index].mean()), 3) for index in largest.indices],
        "output_energy_left": round(float(left), 5),
        "output_dims": [down[:, index].abs().topk(3).indices.tolist() for index in largest.indices],
    }


def main():
    r"""
    Command line: `score`, `parallel` and `generate` evaluate Phase 1 runs, `english-states` caches English
    backbone states, `capacity` measures per layer second moments,
    `basis` writes confined correction directions.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1 evaluation and measurement.")
    commands = parser.add_subparsers(dest="command", required=True)

    capacity = commands.add_parser("capacity", help="Per layer second moments of the deploy and empty frames.")
    capacity.add_argument("--model", required=True)
    capacity.add_argument("--adapter", required=True)
    capacity.add_argument("--output", required=True)
    capacity.add_argument("--english-speech", required=True)
    capacity.add_argument("--korean-text", required=True)
    capacity.add_argument("--korean-speech", required=True)
    capacity.add_argument("--limit", type=int, default=800)
    capacity.add_argument("--batch-size", type=int, default=16)
    capacity.add_argument("--max-frames", type=int, default=320)

    basis = commands.add_parser("basis", help="Input directions of a confined correction, per layer.")
    basis.add_argument("--moments", required=True)
    basis.add_argument("--layers", required=True)
    basis.add_argument("--rank", type=int, default=64)
    basis.add_argument("--module", default="ffn", choices=CAPACITY_MODULES)
    basis.add_argument("--output", required=True)

    english = commands.add_parser("english-states", help="Cache of English backbone states for english_lambda.")
    english.add_argument("--model", required=True)
    english.add_argument("--corpus", required=True)
    english.add_argument("--output", required=True)
    english.add_argument("--batch-size", type=int, default=8)
    english.add_argument("--limit", type=int, default=0)
    english.add_argument("--device", default="cuda")
    english.add_argument("--persona", default="You are a helpful assistant.")
    english.add_argument("--score-all-frames", action="store_true")
    english.add_argument("--drop-audio", action="store_true")

    score = commands.add_parser("score", help="Korean, English and timing losses of Phase 1 runs.")
    score.add_argument("--model", required=True)
    score.add_argument("--korean", required=True)
    score.add_argument("--drop-audio", action="store_true")
    score.add_argument("--exclude-train")
    score.add_argument("--english-speech", required=True)
    score.add_argument("--batch-size", type=int, default=16)
    score.add_argument("--max-frames", type=int, default=320)
    score.add_argument("runs", nargs="+")

    parallel = commands.add_parser("parallel", help="Korean against English on the same meaning.")
    parallel.add_argument("--model", required=True)
    parallel.add_argument("--pairs", required=True)
    parallel.add_argument("--frames-per-piece", type=float, default=4.049)
    parallel.add_argument("runs", nargs="+")

    kinds = commands.add_parser("dense-kinds", help="Dense stage loss by kind of target piece.")
    kinds.add_argument("--model", required=True)
    kinds.add_argument("--dataset", required=True)
    kinds.add_argument("--rows", type=int, default=256)
    kinds.add_argument("--original", action="store_true")
    kinds.add_argument("runs", nargs="+")
    dense_gen = commands.add_parser("dense-generate", help="Text continuation in the dense layout of Phase 1.0.")
    dense_gen.add_argument("--model", required=True)
    dense_gen.add_argument("--seeds", type=int, default=4)
    dense_gen.add_argument("--max-new-tokens", type=int, default=80)
    dense_gen.add_argument("--temperature", type=float, default=0.8)
    dense_gen.add_argument("run")
    dense = commands.add_parser("dense-check", help="Leak and context checks of a dense stage checkpoint.")
    dense.add_argument("--model", required=True)
    dense.add_argument("--dataset", required=True)
    dense.add_argument("--train-dataset", required=True)
    dense.add_argument("--rows", type=int, default=8)
    dense.add_argument("--seed", type=int, default=0)
    dense.add_argument("run")
    generate = commands.add_parser("generate", help="Generation in English, while speaking, and in silence.")
    generate.add_argument("--model", required=True)
    generate.add_argument("--steps", type=int, default=120)
    generate.add_argument("--seeds", type=int, default=2)
    generate.add_argument("--frames-per-piece", type=float, default=4.049)
    generate.add_argument("run")

    speak = commands.add_parser("speak", help="Whether the generated speech says the generated text.")
    speak.add_argument("--model", required=True)
    speak.add_argument("--steps", type=int, default=200)
    speak.add_argument("--seeds", type=int, default=2)
    speak.add_argument("--frames-per-piece", type=float, default=4.049)
    speak.add_argument("--whisper", default="openai/whisper-large-v3")
    speak.add_argument("--save-dir")
    speak.add_argument("run")

    audio = commands.add_parser("audio-loss", help="Cross entropy of the assistant's codes, Korean and English.")
    audio.add_argument("--model", required=True)
    audio.add_argument("--korean", required=True)
    audio.add_argument("--english-speech", required=True)
    audio.add_argument("--whisper", default="openai/whisper-large-v3")
    audio.add_argument("--limit", type=int, default=256)
    audio.add_argument("--batch-size", type=int, default=16)
    audio.add_argument("--max-frames", type=int, default=512)
    audio.add_argument("runs", nargs="+")

    baselines = commands.add_parser("dense-baselines", help="Unigram and input scale reference points of the dense check.")
    baselines.add_argument("--personaplex", default="nvidia/personaplex-7b-v1")
    baselines.add_argument("--language-model", default="Qwen/Qwen3-8B")
    baselines.add_argument("--work-dir", required=True)
    baselines.add_argument("--eval-rows", type=int, default=64)
    baselines.add_argument("--train-tokens", type=int, default=2_000_000)
    baselines.add_argument("--output", required=True)
    nvfp4 = commands.add_parser("nvfp4-check", help="Text loss of a causal LM in full precision and emulated NVFP4.")
    nvfp4.add_argument("--model", default="Qwen/Qwen3-8B")
    nvfp4.add_argument("--languages", default="en,ko", help="Comma separated keys of the dense sources.")
    nvfp4.add_argument("--rows", type=int, default=4, help="Rows per language.")
    nvfp4.add_argument("--row-tokens", type=int, default=512)
    nvfp4.add_argument("--variants", default=None, help="Semicolon separated names of NVFP4_VARIANTS to run; all when omitted.")
    nvfp4.add_argument("--output", required=True)
    upload = commands.add_parser("speak-upload", help="Put the audio of an `evaluate speak` run on W&B.")
    upload.add_argument("--log", required=True)
    upload.add_argument("--audio-dir", required=True)
    upload.add_argument("--run-id", required=True, help="W&B id of the training run the checkpoint came from.")
    upload.add_argument("--name", help="Name of the evaluation run, `<training run>-speak` by default.")
    upload.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "polyplex"))
    upload.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    upload.add_argument("--key", default="speak/samples")
    upload.add_argument("--sampling-rate", type=int, default=24000)

    arguments = parser.parse_args()
    if arguments.command == "nvfp4-check":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .phase import dense_documents

        tokenizer = AutoTokenizer.from_pretrained(arguments.model)
        model = AutoModelForCausalLM.from_pretrained(arguments.model, dtype=torch.float32).eval()
        results = {}
        for language in arguments.languages.split(","):
            pieces = []
            for text in dense_documents(language, "valid", arguments.rows * arguments.row_tokens * 2, tokenizer):
                pieces += tokenizer.encode(" ".join(text.split()), add_special_tokens=False)
            rows = [pieces[start : start + arguments.row_tokens] for start in range(0, len(pieces), arguments.row_tokens)]
            rows = [row for row in rows if len(row) == arguments.row_tokens][: arguments.rows]
            chosen = None
            if arguments.variants:
                chosen = {name.strip(): NVFP4_VARIANTS[name.strip()] for name in arguments.variants.split(";")}
            results[language] = nvfp4_variant_losses(model, rows, chosen)
            print(f"[nvfp4-check] {language} {len(rows)} rows of {arguments.row_tokens}: "
                  + ", ".join(f"{name} {value:.4f}" for name, value in results[language].items()), flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(arguments.output)), exist_ok=True)
        with open(arguments.output, "w", encoding="utf-8") as sink:
            json.dump(results, sink, indent=2)
        return
    if arguments.command == "dense-baselines":
        results = dense_baselines(
            arguments.personaplex,
            arguments.language_model,
            arguments.work_dir,
            eval_rows=arguments.eval_rows,
            train_tokens=arguments.train_tokens,
        )
        os.makedirs(os.path.dirname(os.path.abspath(arguments.output)), exist_ok=True)
        with open(arguments.output, "w", encoding="utf-8") as sink:
            json.dump(results, sink, indent=2)
        print(json.dumps(results["input_norms"]), flush=True)
        return
    if arguments.command == "speak-upload":
        import wandb

        records = read_speak_log(arguments.log)
        openings = {opening: index for index, opening in enumerate(GENERATION_OPENINGS)}
        table = wandb.Table(columns=["opening", "seed", "cer", "text", "heard", "audio"])
        for record in records:
            path = os.path.join(arguments.audio_dir, f"{openings[record['opening']]:02d}_{record['seed']}.wav")
            table.add_data(
                record["opening"], record["seed"], record["cer"], record.get("text", ""), record.get("heard", ""),
                wandb.Audio(path, sample_rate=arguments.sampling_rate, caption=record.get("text", "")),
            )
        from .train import git_record

        # A run of its own rather than a resumed training run, whose recorded commit a resume would replace.
        source = wandb.Api().run(f"{arguments.entity}/{arguments.project}/{arguments.run_id}")
        run = wandb.init(
            project=arguments.project,
            entity=arguments.entity,
            name=arguments.name or f"{source.name}-speak",
            job_type="eval",
            tags=sorted(set(source.tags) | {"speak"}),
            config={"record/source_run": source.url, "record/log": arguments.log, **git_record()},
        )
        run.log({arguments.key: table})
        print(f"[speak-upload] {len(records)} samples -> {run.url}", flush=True)
        run.finish()
        return
    if arguments.command == "score":
        scorer = Phase1Scorer(arguments.model, arguments.batch_size, arguments.max_frames)
        korean, characters = scorer.korean_items(arguments.korean, arguments.exclude_train)
        english = scorer.english_items(arguments.english_speech)
        condition = "silent" if arguments.drop_audio else "own speech"
        print(f"[score] korean {len(korean)} utterances ({condition})", flush=True)
        for run in arguments.runs:
            model = scorer.load_run(run)
            k = scorer.score_korean(model, korean, arguments.drop_audio)
            e = scorer.score_english(model, english)
            print(
                f"[score] {run.rstrip('/').split('/runs/')[-1]}\n"
                f"    korean {k['korean']:.3f}  new {k['korean_new']:.3f}  per char {k['korean_sum'] / characters:.3f}"
                f"  timing {k['timing']:.3f} ({k['timing_cost']:+.3f})  word frames {k['word_frames']}\n"
                f"    english {e['english']:.3f} ({e['english_cost']:+.3f})"
                f"  drift word {e['drift_word']:.4f} timing {e['drift_timing']:.4f}",
                flush=True,
            )
            del model
            torch.cuda.empty_cache()
    elif arguments.command == "parallel":
        scorer = Phase1Scorer(arguments.model)
        with open(arguments.pairs) as handle:
            pairs = json.load(handle)
        print(f"[parallel] {len(pairs)} pairs", flush=True)
        for run in arguments.runs:
            model = scorer.load_run(run)
            r = scorer.score_parallel(model, pairs, arguments.frames_per_piece)
            k, off, on = r["korean"], r["english_off"], r["english_on"]
            print(
                f"[parallel] {run.rstrip('/').split('/runs/')[-1]}\n"
                f"    korean {k['per_token']:.3f} ({k['per_sentence']:.1f} per sentence)  english off "
                f"{off['per_token']:.3f} ({off['per_sentence']:.1f})  on {on['per_token']:.3f} "
                f"({on['per_sentence']:.1f}, {on['per_token'] - off['per_token']:+.3f})",
                flush=True,
            )
            del model
            torch.cuda.empty_cache()
    elif arguments.command == "generate":
        scorer = Phase1Scorer(arguments.model)
        adapter = scorer.load_run(arguments.run)
        model = adapter.fold_into_base().to("cuda", torch.bfloat16).eval()
        print("######## english, corrections on / off", flush=True)
        for seed in range(arguments.seeds + 1):
            on = scorer.generate(model, arguments.steps, seed)
            with adapter.corrections_disabled():
                off = scorer.generate(model, arguments.steps, seed)
            print(f"[english #{seed}] same={on == off}\n    on : {on}\n    off: {off}", flush=True)
        for label, silent in (("speaking", False), ("silent", True)):
            print(f"######## {label}", flush=True)
            pairs = []
            for opening in GENERATION_OPENINGS:
                stream = scorer.opening_stream(opening.split(), arguments.frames_per_piece)
                for seed in range(arguments.seeds):
                    text = scorer.generate(model, arguments.steps, seed, stream, silent)
                    pairs.append((opening, text))
                    print(f"[{label} '{opening}' #{seed}] {text}", flush=True)
            metrics = generation_metrics(pairs)
            print(
                f"[{label} metrics] {len(pairs)} outputs  english {metrics['english']:.2f}  sentence end "
                f"{metrics['sentence_end']:.2f}  repeat {metrics['repeat']:.2f}  long number "
                f"{metrics['long_number']:.2f}",
                flush=True,
            )
    elif arguments.command == "audio-loss":
        import torchaudio
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        from ..datasets.duplex_data import DuplexUtteranceDataset, duplex_collate_fn
        from ..datasets.duplex_data import whisper_alignment

        scorer = Phase1Scorer(arguments.model)
        korean = DuplexUtteranceDataset(arguments.korean)
        korean = [korean[i] for i in range(min(arguments.limit, len(korean)))]
        english = scorer.english_items(arguments.english_speech)[: arguments.limit]
        # The English set carries no word times, so they are read off the decoded speech the way the Korean
        # speaking data was timed.
        recognizer_processor = WhisperProcessor.from_pretrained(arguments.whisper)
        recognizer = WhisperForConditionalGeneration.from_pretrained(
            arguments.whisper, torch_dtype=torch.float16, attn_implementation="eager"
        ).to("cuda").eval()
        resample = torchaudio.transforms.Resample(scorer.processor.audio_tokenizer.config.sampling_rate, 16000)
        for start in range(0, len(english), arguments.batch_size):
            chunk = english[start : start + arguments.batch_size]
            with torch.no_grad():
                waveforms = [
                    resample(scorer.processor.decode_audio(item["codes"][None].cuda()).float().cpu()[0, 0]).numpy()
                    for item in chunk
                ]
            aligned = whisper_alignment(
                recognizer, recognizer_processor, waveforms, [" ".join(item["words"]) for item in chunk], language="en"
            )
            for item, (starts, _, _) in zip(chunk, aligned):
                item["starts"] = starts
        del recognizer
        torch.cuda.empty_cache()

        for run in arguments.runs:
            adapter = scorer.load_run(run).eval()
            collate = duplex_collate_fn(
                scorer.processor, adapter.model, drop_audio=False, max_frames=arguments.max_frames, audio_labels=True
            )

            def mean_loss(items):
                total = count = 0.0
                for start in range(0, len(items), arguments.batch_size):
                    batch = collate(items[start : start + arguments.batch_size])
                    batch.pop("text_labels", None)
                    batch = {key: value.cuda() if torch.is_tensor(value) else value for key, value in batch.items()}
                    with torch.no_grad():
                        loss = adapter(**batch).loss.item()
                    total += loss * len(items[start : start + arguments.batch_size])
                    count += len(items[start : start + arguments.batch_size])
                return total / max(1.0, count)

            korean_on = mean_loss(korean)
            english_on = mean_loss(english)
            with adapter.corrections_disabled():
                korean_off = mean_loss(korean)
                english_off = mean_loss(english)
            print(
                f"[audio-loss] {run.rstrip('/').split('/runs/')[-1]}  (corrections on / off, uniform {math.log(2048):.3f})\n"
                f"    korean {korean_on:.3f} / {korean_off:.3f}  english {english_on:.3f} / {english_off:.3f}",
                flush=True,
            )
            del adapter
            torch.cuda.empty_cache()
    elif arguments.command == "speak":
        import torchaudio
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        scorer = Phase1Scorer(arguments.model)
        adapter = scorer.load_run(arguments.run)
        model = adapter.fold_into_base().to("cuda", torch.bfloat16).eval()
        recognizer_processor = WhisperProcessor.from_pretrained(arguments.whisper)
        recognizer = WhisperForConditionalGeneration.from_pretrained(arguments.whisper, torch_dtype=torch.float16).to("cuda").eval()
        rate = scorer.processor.audio_tokenizer.config.sampling_rate
        resample = torchaudio.transforms.Resample(rate, 16000)
        if arguments.save_dir:
            import soundfile

            os.makedirs(arguments.save_dir, exist_ok=True)
        errors = []
        for index, opening in enumerate(GENERATION_OPENINGS):
            stream = scorer.opening_stream(opening.split(), arguments.frames_per_piece)
            for seed in range(arguments.seeds):
                text, codes = scorer.generate(model, arguments.steps, seed, stream, return_codes=True)
                with torch.no_grad():
                    waveform = scorer.processor.decode_audio(codes[None]).float().cpu()[0, 0]
                    features = recognizer_processor(
                        resample(waveform).numpy(), sampling_rate=16000, return_tensors="pt"
                    ).input_features.to("cuda", torch.float16)
                    ids = recognizer.generate(features, language="korean", task="transcribe")
                heard = recognizer_processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
                error = character_error_rate(text, heard)
                errors.append(error)
                if arguments.save_dir:
                    soundfile.write(os.path.join(arguments.save_dir, f"{index:02d}_{seed}.wav"), waveform.numpy(), rate)
                print(f"[speak '{opening}' #{seed}] cer {error:.2f}\n    text : {text}\n    heard: {heard}", flush=True)
        print(f"[speak metrics] {len(errors)} outputs  mean cer {sum(errors) / max(1, len(errors)):.3f}", flush=True)
    elif arguments.command == "dense-kinds":
        from ..datasets.duplex_data import dense_collate_fn
        from ..datasets.phase_data import DenseTextDataset

        scorer = Phase1Scorer(arguments.model)
        dataset = DenseTextDataset(arguments.dataset)
        rows = [dataset[i] for i in range(min(arguments.rows, len(dataset)))]
        for run in arguments.runs:
            model = scorer.load_run(run)
            collate = dense_collate_fn(scorer.processor, model.model)
            if arguments.original:
                with model.corrections_disabled():
                    report = piece_kind_scores(model, collate, rows, scorer.processor.tokenizer, base_only=True)
            else:
                report = piece_kind_scores(model, collate, rows, scorer.processor.tokenizer)
            print(f"[dense-kinds] {run}", flush=True)
            for key, value in sorted(report.items(), key=lambda item: -item[1]["share"]):
                print(f"    {key:20s} share {value['share']:.3f}  loss {value['loss']:.3f}  ppl {value['ppl']:.1f}  top1 {value['top1']:.3f}", flush=True)
            del model
            torch.cuda.empty_cache()
    elif arguments.command == "dense-generate":
        from ..datasets.duplex_data import dense_collate_fn

        scorer = Phase1Scorer(arguments.model)
        model = scorer.load_run(arguments.run)
        collate = dense_collate_fn(scorer.processor, model.model)
        tokenizer = scorer.processor.tokenizer
        for opening in GENERATION_OPENINGS:
            prompt = torch.tensor(tokenizer.encode(opening, add_special_tokens=False))
            for seed in range(arguments.seeds):
                pieces = dense_generate(model, collate, prompt, arguments.max_new_tokens, arguments.temperature, seed=seed)
                text = tokenizer.decode(pieces.tolist())
                print(f"[dense '{opening}' #{seed}] {opening}{text}", flush=True)
    elif arguments.command == "dense-check":
        from ..datasets.duplex_data import dense_collate_fn
        from ..datasets.phase_data import DenseTextDataset

        scorer = Phase1Scorer(arguments.model)
        model = scorer.load_run(arguments.run)
        dataset = DenseTextDataset(arguments.dataset)
        rows = [dataset[i] for i in range(arguments.rows)]
        if len(rows[0]["segments"]) < 2:
            rows.insert(0, next(dataset[i] for i in range(len(dataset)) if len(dataset[i]["segments"]) > 1))
        unigram = unigram_log_probs(DenseTextDataset(arguments.train_dataset).tokens, model.config.vocab_size)
        report = dense_checks(
            model, dense_collate_fn(scorer.processor, model.model), rows, model.config.base_vocab_size, unigram,
            arguments.seed,
        )
        print(f"[dense-check] {arguments.run}\n    " + "  ".join(f"{k} {v:.4g}" for k, v in report.items()), flush=True)
    elif arguments.command == "english-states":
        from .phase import write_english_states

        frames = write_english_states(
            arguments.model, arguments.corpus, arguments.output, arguments.batch_size, arguments.limit,
            arguments.device, arguments.persona, arguments.score_all_frames, arguments.drop_audio,
        )
        print(f"[english] {frames} frames -> {arguments.output}", flush=True)
    elif arguments.command == "capacity":
        report = measure_capacity(
            arguments.model, arguments.adapter, arguments.output, arguments.english_speech, arguments.korean_text,
            arguments.korean_speech, arguments.limit, arguments.batch_size, arguments.max_frames,
        )
        print(f"[capacity] frames per group {report['counts']} -> {arguments.output}", flush=True)
    else:
        count = build_confined_basis(arguments.moments, arguments.layers, arguments.rank, arguments.module, arguments.output)
        print(f"[basis] {count} layers, rank {arguments.rank} -> {arguments.output}", flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "dense_baselines",
    "dense_unigram_baseline",
    "frame_embedding_norms",
    "CAPACITY_GROUPS",
    "CapacityMeter",
    "Evaluator",
    "ForceTextLogitsProcessor",
    "Phase1Scorer",
    "SilentAssistantLogitsProcessor",
    "build_confined_basis",
    "dense_checks",
    "dense_generate",
    "measure_capacity",
    "piece_kind",
    "piece_kind_scores",
    "unigram_log_probs",
]
