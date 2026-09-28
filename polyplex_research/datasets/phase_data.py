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
"""The dataset each transfer phase trains on, built once and replicated as a folder or a Hub repo."""

from __future__ import annotations

import json
import os
import random
import zlib
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .base import MimiCacheMixin, ReplicableMixin, SingleTurnTTSExample
from .instruction import iter_instruction_texts
from .multisession import iter_wizard_turns
from .translation import ColloquialKoEnDataset


SOURCE_SEPARATOR = "\n"

HELD_OUT_BUCKETS = 2


@dataclass
class Phase1Example:
    r"""
    Container for one Phase 1 text example.

    Phase 1 teaches the text stream to decode into Korean, and the corpora contribute to that in
    different ways, which is why one container carries all of them. A dialogue turn is what the stream
    actually holds at deployment, a chatbot speaking, and it carries the register the phase is fitted
    on. A translation pair conditions on English and is scored on the Korean side, and it is there for
    the vocabulary and domain breadth a single dialogue corpus does not have.

    Args:
        target (`str`):
            Korean text the model is trained to produce.
        source (`str`, *optional*, defaults to `""`):
            English text conditioning the target, ending in the newline that marks where the scored
            half begins. Empty for monolingual examples.
        origin (`str`, *optional*, defaults to `"translation"`):
            Corpus the example came from, `"dialogue"`, `"translation"` or `"instruction"`.
        domain (`str`, *optional*, defaults to `""`):
            Domain the translation pair carries, or the topic of a dialogue turn.
        subdomain (`str`, *optional*, defaults to `""`):
            Finer grained label of the same pair.
        conversation (`str`, *optional*, defaults to `""`):
            Dialogue the turn belongs to. Turns of one conversation repeat persona and topic, so a
            split that cuts between them leaks, and this is the key the split respects.
    """

    target: str
    source: str = ""
    origin: str = "translation"
    domain: str = ""
    subdomain: str = ""
    conversation: str = ""


class Phase1Dataset(ReplicableMixin, Dataset):
    r"""
    Mixes the translation and instruction corpora into the single dataset Phase 1 trains on.

    The two corpora are not interchangeable and neither is sufficient alone. Translation pairs carry the
    decoding supervision the phase is about, English in and Korean out, but the AI Hub corpus is
    colloquial throughout, so a model trained on it alone learns a register rather than a language. The
    instruction sentences are standard register Korean with no source side, and they are mixed in at a
    small share to widen that.

    Because the mixture is a sampling decision, not a property of either corpus, it belongs to the
    dataset rather than to a training script: `translation_ratio` and `seed` fix it, and
    [`save_pretrained`] freezes the result so every run reads the same examples.

    Args:
        translation_root (`str`, *optional*):
            Directory of the extracted translation corpus, defaults to the corpus cache.
        instruction_cache_dir (`str`, *optional*):
            Directory holding the downloaded instruction files, defaults to the corpus cache.
        split (`str`, *optional*, defaults to `"train"`):
            Split of the translation corpus. `"train"` and `"valid"` are the ones the corpus ships,
            `"train_ext"` is both of them together, for a run that has read everything the `"train"`
            split holds and needs more.
        direction (`str`, *optional*, defaults to `"en2ko"`):
            Translation direction, `"en2ko"` teaches decoding into Korean.
        instruction_subsets (`tuple`, *optional*, defaults to `("koalpaca",)`):
            Instruction corpora contributing the standard register share.
        translation_ratio (`float`, *optional*, defaults to 0.85):
            Share of the examples that are translation pairs, the remainder are instruction sentences.
        total_examples (`int`, *optional*):
            Size of the mixture, defaults to as many as the ratio allows from what the corpora hold.
        download (`bool`, *optional*, defaults to `False`):
            Whether missing corpora may be fetched.
        seed (`int`, *optional*, defaults to 0):
            Seed of the shuffle that interleaves the two corpora.
    """

    def __init__(
        self,
        translation_root: str | None = None,
        instruction_cache_dir: str | None = None,
        split: str = "train",
        direction: str = "en2ko",
        instruction_subsets: tuple = ("koalpaca",),
        translation_ratio: float = 0.85,
        dialogue_ratio: float = 0.0,
        dialogue_root: str | None = None,
        total_examples: int | None = None,
        download: bool = False,
        seed: int = 0,
    ):
        if not 0.0 <= translation_ratio <= 1.0:
            raise ValueError(f"`translation_ratio` must be in [0, 1], got {translation_ratio}.")
        if not 0.0 <= dialogue_ratio <= 1.0:
            raise ValueError(f"`dialogue_ratio` must be in [0, 1], got {dialogue_ratio}.")
        if translation_ratio + dialogue_ratio > 1.0:
            raise ValueError(
                "`translation_ratio` and `dialogue_ratio` sum to "
                f"{translation_ratio + dialogue_ratio}, which leaves the instruction share negative."
            )

        self.split = split
        self.direction = direction
        self.instruction_subsets = tuple(instruction_subsets)
        self.translation_ratio = translation_ratio
        self.dialogue_ratio = dialogue_ratio
        self.total_examples = total_examples
        self.seed = seed

        self.examples = self._build(
            translation_root=translation_root,
            instruction_cache_dir=instruction_cache_dir,
            dialogue_root=dialogue_root,
            download=download,
        )

    def _build(
        self,
        translation_root: str | None,
        instruction_cache_dir: str | None,
        dialogue_root: str | None,
        download: bool,
    ) -> list:
        r"""
        Reads the corpora and interleaves them at the configured ratios.

        The requested size is only a target. When one corpus runs short the others are not padded to
        make up the difference, because that would silently move the ratio the run was configured with.

        Args:
            translation_root (`str` or `None`):
                Directory of the extracted translation corpus.
            instruction_cache_dir (`str` or `None`):
                Directory holding the downloaded instruction files.
            dialogue_root (`str` or `None`):
                Directory of the extracted multi session dialogue corpus.
            download (`bool`):
                Whether missing corpora may be fetched.

        Returns:
            `list` of `Phase1Example`: The shuffled mixture.
        """
        translation_target = None
        dialogue_target = None
        instruction_target = None
        if self.total_examples is not None:
            translation_target = int(round(self.total_examples * self.translation_ratio))
            dialogue_target = int(round(self.total_examples * self.dialogue_ratio))
            instruction_target = self.total_examples - translation_target - dialogue_target

        if self.split == "train_ext":
            # The corpus ships two splits and a single run only ever read one of them. The extended
            # split is both, so a held out set has to be carved out of this one rather than taken
            # from the split it no longer leaves untouched.
            pairs = []
            for corpus_split in ("train", "valid"):
                remaining = None if translation_target is None else translation_target - len(pairs)
                if remaining is not None and remaining <= 0:
                    break
                pairs.extend(
                    ColloquialKoEnDataset(
                        root=translation_root,
                        split=corpus_split,
                        direction=self.direction,
                        download=download,
                        max_examples=remaining,
                    )
                )
        elif self.split in ("valid", "test"):
            # The corpus ships no test split, so the two held out sets are the halves of its validation
            # split. The bucket is a checksum of the pair rather than a shuffle, so a pair lands in the
            # same half however many pairs the run happens to read.
            wanted = 0 if self.split == "valid" else 1
            pairs = [
                pair
                for pair in ColloquialKoEnDataset(
                    root=translation_root,
                    split="valid",
                    direction=self.direction,
                    download=download,
                )
                if zlib.crc32((pair.source + pair.target).encode("utf-8")) % HELD_OUT_BUCKETS == wanted
            ]
            if translation_target is not None:
                pairs = pairs[:translation_target]
        else:
            pairs = ColloquialKoEnDataset(
                root=translation_root,
                split=self.split,
                direction=self.direction,
                download=download,
                max_examples=translation_target,
            )
        # The trailing newline is the boundary the scored half starts at, carried in the record so
        # that an export holds the format it was built with.
        examples = [
            Phase1Example(
                target=pair.target,
                source=pair.source + SOURCE_SEPARATOR,
                origin="translation",
                domain=pair.domain,
                subdomain=pair.subdomain,
            )
            for pair in pairs
        ]

        if self.dialogue_ratio > 0.0 and dialogue_target != 0:
            if dialogue_target is None and self.translation_ratio > 0.0:
                dialogue_target = int(round(len(examples) * self.dialogue_ratio / self.translation_ratio))
            examples.extend(
                Phase1Example(
                    target=turn.text,
                    source="",
                    origin="dialogue",
                    domain=turn.topic,
                    conversation=turn.conversation,
                )
                for turn in iter_wizard_turns(
                    root=dialogue_root, split=self.split, max_turns=dialogue_target
                )
            )

        if instruction_target is None:
            # Without a stated size the instruction share is derived from what translation actually
            # yielded, so the ratio holds whatever the corpus turned out to contain.
            ratio = self.translation_ratio
            remainder = 1.0 - ratio - self.dialogue_ratio
            instruction_target = int(round(len(examples) * remainder / ratio)) if ratio > 0.0 else None

        if instruction_target != 0:
            examples.extend(
                Phase1Example(target=sentence, source="", origin="instruction")
                for sentence in iter_instruction_texts(
                    subsets=self.instruction_subsets,
                    cache_dir=instruction_cache_dir,
                    download=download,
                    max_lines=instruction_target,
                )
            )

        random.Random(self.seed).shuffle(examples)
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def example_lengths(self, tokenizer) -> list:
        r"""
        Returns the tokenized length of every example, for a sampler that batches by length.

        The collator pads a batch to its longest sequence, so batching unrelated lengths together
        spends most of the batch on padding. The lengths are the ones the collator will produce, and
        they are computed once and kept, since a length grouped sampler asks for them every epoch.

        Args:
            tokenizer (`PreTrainedTokenizerBase`):
                Tokenizer of the text stream, the one the collator was built with.

        Returns:
            `list` of `int`: One length per example, in dataset order.
        """
        if getattr(self, "_lengths", None) is None:
            self._lengths = [
                len(tokenizer.encode(example.source + example.target, add_special_tokens=False))
                for example in self.examples
            ]
        return self._lengths

    def __getitem__(self, index: int) -> Phase1Example:
        return self.examples[index]

    # -----------------------------------------------------------------
    # Replication
    # -----------------------------------------------------------------

    def export_config(self) -> dict:
        r"""
        Returns:
            `dict`: The mixture settings, enough to describe how the export was produced.
        """
        return {
            "split": self.split,
            "direction": self.direction,
            "instruction_subsets": list(self.instruction_subsets),
            "translation_ratio": self.translation_ratio,
            "dialogue_ratio": self.dialogue_ratio,
            "total_examples": self.total_examples,
            "seed": self.seed,
        }

    def export_records(self):
        r"""
        Yields:
            `dict`: One record per example, already in the shuffled order.
        """
        for example in self.examples:
            yield {
                "target": example.target,
                "source": example.source,
                "origin": example.origin,
                "domain": example.domain,
                "subdomain": example.subdomain,
                "conversation": example.conversation,
            }

    @classmethod
    def from_export(cls, config: dict, records: list, arrays) -> Phase1Dataset:
        r"""
        Rebuilds the mixture from an export, without reading either corpus.

        Args:
            config (`dict`):
                Settings written by [`export_config`].
            records (`list` of `dict`):
                The exported examples, in order.
            arrays (`None`):
                Unused, Phase 1 is text only.

        Returns:
            `Phase1Dataset`: The rebuilt dataset.
        """
        dataset = cls.__new__(cls)
        dataset.split = config.get("split", "train")
        dataset.direction = config.get("direction", "en2ko")
        dataset.instruction_subsets = tuple(config.get("instruction_subsets", ()))
        dataset.translation_ratio = config.get("translation_ratio", 0.85)
        dataset.dialogue_ratio = config.get("dialogue_ratio", 0.0)
        dataset.total_examples = config.get("total_examples")
        dataset.seed = config.get("seed", 0)
        dataset.examples = [
            Phase1Example(
                target=record["target"],
                source=record.get("source", ""),
                origin=record.get("origin", "translation"),
                domain=record.get("domain", ""),
                subdomain=record.get("subdomain", ""),
                conversation=record.get("conversation", ""),
            )
            for record in records
        ]
        return dataset


class Phase2Dataset(ReplicableMixin, MimiCacheMixin, Dataset):
    r"""
    The single-turn TTS dataset Phase 2 trains on, carried as acoustic codes rather than audio.

    Phase 2 never sees a waveform: the residual is trained against Mimi codes, so the codec pass belongs
    to dataset construction and not to the training loop. The dataset is therefore built over a code
    cache, and what it replicates is the codes themselves, which is also the only form of this corpus
    small enough to move.

    The export rewrites the cache into one contiguous array in dataset order, so a reload does not depend
    on how the original run happened to shard its codec pass.

    Args:
        cache_dir (`str`):
            Directory holding the `<shard>.npy` and `<shard>.jsonl` pairs written by the caching script.
        shards (`list` of `str`, *optional*):
            Shard names to read, defaults to every pair found.
        min_duration (`float`, *optional*, defaults to 0.5):
            Utterances shorter than this many seconds are dropped.
        max_duration (`float`, *optional*, defaults to 30.0):
            Utterances longer than this many seconds are dropped.
        num_codebooks (`int`, *optional*, defaults to 8):
            Number of leading codebooks kept per frame.
    """

    def __init__(
        self,
        cache_dir: str,
        shards: list | None = None,
        min_duration: float = 0.5,
        max_duration: float = 30.0,
        num_codebooks: int = 8,
    ):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.entries = self.index_code_cache(
            cache_dir,
            shards=shards,
            min_duration=min_duration,
            max_duration=max_duration,
            num_codebooks=num_codebooks,
        )
        self._exported_codes = None

    def __len__(self) -> int:
        return len(self.entries)

    def example_lengths(self, tokenizer=None) -> list:
        r"""
        Returns the frame count of every utterance, for a sampler that batches by length.

        A Phase 2 batch is as long as its longest utterance, and the frame count is already in the
        manifest, so no tokenizer is involved.

        Args:
            tokenizer (`PreTrainedTokenizerBase`, *optional*):
                Unused, present so both phase datasets answer the same call.

        Returns:
            `list` of `int`: One frame count per utterance, in dataset order.
        """
        return [row["frames"] for _, row in self.entries]

    def __getitem__(self, index: int) -> SingleTurnTTSExample:
        shard, row = self.entries[index]
        if shard is None:
            # Rebuilt from an export: one array, rows already offset into it.
            window = self._exported_codes[
                : self.num_codebooks, row["offset"] : row["offset"] + row["frames"]
            ]
            return SingleTurnTTSExample(
                text=row["text"],
                codes=torch.from_numpy(np.asarray(window, dtype=np.int64)),
                audio_path=row["id"],
            )
        return self.code_example(shard, row)

    # -----------------------------------------------------------------
    # Replication
    # -----------------------------------------------------------------

    def export_config(self) -> dict:
        r"""
        Returns:
            `dict`: The filters and the codebook count the export was written under.
        """
        return {
            "min_duration": self.min_duration,
            "max_duration": self.max_duration,
            "num_codebooks": self.num_codebooks,
        }

    def export_records(self):
        r"""
        Yields:
            `dict`: One manifest row per utterance, offset into the exported array rather than the shard
            it came from.
        """
        offset = 0
        for _, row in self.entries:
            yield {
                "id": row["id"],
                "text": row["text"],
                "duration": row["duration"],
                "offset": offset,
                "frames": row["frames"],
            }
            offset += row["frames"]

    def export_arrays(self) -> np.ndarray:
        r"""
        Returns:
            `np.ndarray` of shape `(num_codebooks, total_frames)`: Every utterance's codes end to end, in
            dataset order and matching the offsets of [`export_records`]. Held as `int16`, which spans
            the acoustic vocabulary, and widened to `int64` when a row is read.
        """
        total_frames = sum(row["frames"] for _, row in self.entries)
        exported = np.empty((self.num_codebooks, total_frames), dtype=np.int16)
        offset = 0
        for shard, row in self.entries:
            source = (
                self._exported_codes if shard is None else self.shard_codes(shard)
            )
            window = source[: self.num_codebooks, row["offset"] : row["offset"] + row["frames"]]
            exported[:, offset : offset + row["frames"]] = window
            offset += row["frames"]
        return exported

    @classmethod
    def from_export(cls, config: dict, records: list, arrays) -> Phase2Dataset:
        r"""
        Rebuilds the dataset from an export, reading no code cache.

        Args:
            config (`dict`):
                Settings written by [`export_config`].
            records (`list` of `dict`):
                The exported manifest rows, in order.
            arrays (`np.ndarray`):
                The exported codes, memory mapped.

        Returns:
            `Phase2Dataset`: The rebuilt dataset.

        Raises:
            ValueError: If the export carries no codes.
        """
        if arrays is None:
            raise ValueError("A Phase 2 export must carry its codes, none were found next to the records.")

        dataset = cls.__new__(cls)
        dataset.min_duration = config.get("min_duration", 0.5)
        dataset.max_duration = config.get("max_duration", 30.0)
        dataset.num_codebooks = config.get("num_codebooks", 8)
        dataset.cache_dir = ""
        dataset._arrays = {}
        dataset._exported_codes = arrays
        dataset.entries = [(None, record) for record in records]
        return dataset


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def build_unseen_dataset(sources: dict, seen: set, seed: int = 0) -> Phase1Dataset:
    r"""
    Builds a Phase 1 training set out of sentences a run has not trained on.

    A sentence is kept once, the first time it appears in any source, and only when it is not in `seen`.

    Args:
        sources (`dict` of `str` to `list` of `Phase1Example`):
            Examples per corpus, in reading order.
        seen (`set` of `str`):
            Target sentences an earlier stage of the same model already trained on.
        seed (`int`, *optional*, defaults to 0):
            Seed of the shuffle over the kept examples.

    Returns:
        `Phase1Dataset`: The kept examples, shuffled.
    """
    kept, taken = [], set(seen)
    for examples in sources.values():
        for example in examples:
            if example.target in taken:
                continue
            taken.add(example.target)
            kept.append(example)
    random.Random(seed).shuffle(kept)

    dataset = Phase1Dataset.__new__(Phase1Dataset)
    dataset.split = "train"
    dataset.direction = "ko"
    dataset.instruction_subsets = ()
    dataset.translation_ratio = None
    dataset.dialogue_ratio = None
    dataset.total_examples = len(kept)
    dataset.seed = seed
    dataset.examples = kept
    return dataset


def phase1_collate_fn(tokenizer, max_length: int | None = None):
    r"""
    Builds the `collate_fn` that turns [`Phase1Example`] into a text batch.

    A translation pair is one sequence, the English side followed by the Korean one, and only the Korean
    half is scored: the phase teaches decoding into Korean, not the reconstruction of the prompt. An
    instruction sentence has no source, so all of it is scored.

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            Tokenizer of the text stream, expanded with the Korean rows.
        max_length (`int`, *optional*):
            Hard cap on the length of a sequence, in tokens.

    Returns:
        `Callable`: A `collate_fn` producing `input_ids`, `attention_mask` and `text_labels`.
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def collate(examples: list) -> dict:
        sequences, targets = [], []
        for example in examples:
            prompt = (
                tokenizer.encode(example.source, add_special_tokens=False) if example.source else []
            )
            target = tokenizer.encode(example.target, add_special_tokens=False)
            sequences.append(prompt + target)
            targets.append([-100] * len(prompt) + target)

        if max_length is not None:
            sequences = [sequence[:max_length] for sequence in sequences]
            targets = [target[:max_length] for target in targets]
        width = max(len(sequence) for sequence in sequences)

        input_ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), width), dtype=torch.bool)
        text_labels = torch.full((len(sequences), width), -100, dtype=torch.long)

        for row, (sequence, target) in enumerate(zip(sequences, targets)):
            length = len(sequence)
            input_ids[row, :length] = torch.tensor(sequence, dtype=torch.long)
            attention_mask[row, :length] = True
            text_labels[row, :length] = torch.tensor(target, dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "text_labels": text_labels}

    return collate


def text_frame_positions(num_tokens: int, frames: int, alignment: str) -> torch.Tensor:
    r"""
    Chooses the frame each text token of an utterance sits on.

    The text stream of a duplex model is one token per frame, and the frames a word does not fall on
    carry the pad token. Where the words go therefore decides how the inner monologue lines up with the
    speech it narrates.

    - `"spread"` places token `i` of `n` on frame `floor(i * frames / n)`. The words are evenly paced
      over the utterance, which is the closest a run without a forced alignment gets to the timing the
      pretrained model reads. The mapping is strictly increasing whenever `n <= frames`, so no two
      tokens land on the same frame and their order survives.
    - `"leading"` packs them onto the opening frames and pads the rest, which asks the model to hold the
      whole sentence in the first moment of the utterance.

    Args:
        num_tokens (`int`):
            Number of text tokens of the utterance.
        frames (`int`):
            Number of frames the utterance occupies.
        alignment (`str`):
            One of `"spread"` and `"leading"`.

    Returns:
        `torch.LongTensor` of shape `(num_tokens,)`: The frame index of every token.

    Raises:
        ValueError: If `alignment` names no layout.
    """
    index = torch.arange(num_tokens, dtype=torch.long)
    if alignment == "leading":
        return index
    if alignment == "spread":
        return (index * frames) // max(num_tokens, 1)
    raise ValueError(f"`text_alignment` must be 'spread' or 'leading', got {alignment!r}.")


def delay_audio_codes(codes: torch.Tensor, bos_token_id: int, pad_token_id: int) -> torch.Tensor:
    r"""
    Rewrites one utterance into the delayed layout the duplex model reads a stream in.

    Codebook 0 stays where it is and every codebook above it is pushed one frame later, so that frame
    $t$ carries the coarse code of acoustic frame $t$ next to the fine codes of frame $t - 1$. The model
    predicts a whole frame at once, and this is what leaves the fine codebooks conditioned on a coarse
    code that was already decided. The opening frame of the shifted codebooks has no earlier acoustic
    frame to carry, so it holds the reserved row, and the closing frame is where codebook 0 runs out.

    An utterance of `frames` codes therefore occupies `frames + 1` frames. This is the array
    `MoshiGenerationMixin.build_delay_pattern_mask` returns as its pattern, for `max_length` one past
    the utterance, which is the form that keeps every acoustic frame.

    Args:
        codes (`torch.LongTensor` of shape `(num_codebooks, frames)`):
            Acoustic codes of one utterance, undelayed, as the codec emits them.
        bos_token_id (`int`):
            Value the shifted codebooks carry on the opening frame.
        pad_token_id (`int`):
            Value codebook 0 carries on the closing frame.

    Returns:
        `torch.LongTensor` of shape `(num_codebooks, frames + 1)`: The delayed codes.
    """
    num_codebooks, frames = codes.shape
    delayed = codes.new_full((num_codebooks, frames + 1), pad_token_id)
    delayed[0, :frames] = codes[0]
    delayed[1:, 1:] = codes[1:]
    delayed[1:, 0] = bos_token_id
    return delayed


def phase2_collate_fn(
    tokenizer,
    audio_vocab_size: int = 2048,
    text_pad_token_id: int = 3,
    max_frames: int | None = None,
    text_alignment: str = "spread",
):
    r"""
    Builds the `collate_fn` that turns [`SingleTurnTTSExample`] into a duplex batch.

    The assistant stream carries the codes of the utterance and the text is laid along the same frames,
    which is what makes the batch a text to speech target. The user stream carries no one talking, so it
    is filled with the reserved row the model reads as an ungenerated frame.

    Both audio streams are written in the delayed layout of [`delay_audio_codes`], which is the only
    layout the frozen model has ever read: it generates in it, and its own training objective builds it
    before scoring. An utterance of `frames` codes therefore occupies `frames + 1` frames, and the batch
    is one wider than its longest utterance. The text stream is not delayed. It runs with codebook 0, so
    a word still sits on the acoustic frame it is spoken on.

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            Tokenizer of the text stream, expanded with the Korean rows.
        audio_vocab_size (`int`, *optional*, defaults to 2048):
            Number of acoustic codes. The row at this index is the reserved one, and is what silence and
            padding are written as. Read off the model configuration rather than left at the default
            whenever the caller has it.
        text_pad_token_id (`int`, *optional*, defaults to 3):
            Text token filling the frames that carry no word.
        max_frames (`int`, *optional*):
            Hard cap on the temporal length of a batch.
        text_alignment (`str`, *optional*, defaults to `"spread"`):
            How the words of an utterance are laid over its frames, see [`text_frame_positions`].

    Returns:
        `Callable`: A `collate_fn` producing `input_ids`, `attention_mask`, `assistant_audio_codes`,
        `user_audio_codes` and `audio_labels`.
    """

    def collate(examples: list) -> dict:
        codes = [example.codes for example in examples]
        if max_frames is not None:
            codes = [entry[:, :max_frames] for entry in codes]

        num_codebooks = codes[0].shape[0]
        # The delay costs one frame, so the batch is one wider than its longest utterance.
        width = max(entry.shape[1] for entry in codes) + 1
        batch_size = len(examples)

        input_ids = torch.full((batch_size, width), text_pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, width), dtype=torch.bool)
        assistant = torch.full((batch_size, num_codebooks, width), audio_vocab_size, dtype=torch.long)
        audio_labels = torch.full((batch_size, num_codebooks, width), -100, dtype=torch.long)

        for row, (example, entry) in enumerate(zip(examples, codes)):
            frames = entry.shape[1]
            span = frames + 1
            assistant[row, :, :span] = delay_audio_codes(entry, audio_vocab_size, audio_vocab_size)
            # The reserved row is what the two ends of the delay hold, and it is not a code anyone has
            # to predict, so the targets mark those frames unscored instead.
            audio_labels[row, :, :span] = delay_audio_codes(entry, -100, -100)
            attention_mask[row, :span] = True

            tokens = tokenizer.encode(example.text, add_special_tokens=False)[:frames]
            if tokens:
                positions = text_frame_positions(len(tokens), frames, text_alignment)
                input_ids[row, positions] = torch.tensor(tokens, dtype=torch.long)

        # Nobody is speaking on the user side of a single-turn utterance.
        user = torch.full_like(assistant, audio_vocab_size)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "assistant_audio_codes": assistant,
            "user_audio_codes": user,
            "audio_labels": audio_labels,
        }

    return collate


DENSE_TOKENS_NAME = "tokens.bin"
DENSE_SEGMENTS_NAME = "segments.npy"
DENSE_ROWS_NAME = "rows.npy"


def write_dense_cache(
    documents_path: str, tokenizer, output_dir: str, max_tokens: int = 1024, pieces: list | None = None
) -> int:
    r"""
    Tokenizes a document corpus and packs it into rows for the dense stage of Phase 1.

    Whitespace is collapsed to single spaces, since the text stream never carries a line break. Every
    document becomes one or more segments, each costing its pieces plus the opening frame it is read after,
    and the segments are packed in order into rows of at most `max_tokens` frames. A document longer than a
    row is cut into consecutive segments. Nothing is repeated or dropped.

    Args:
        documents_path (`str`):
            JSON lines file with one document per line under `text`.
        tokenizer (`PreTrainedTokenizerBase`):
            Tokenizer of the text stream.
        output_dir (`str`):
            Directory the cache is written to: the pieces end to end in `tokens.bin` (`int32`), each segment's
            start and length in `segments.npy`, and each row's first segment in `rows.npy`.
        max_tokens (`int`, *optional*, defaults to 1024):
            Frames a row holds, opening frames included.
        pieces (`list`, *optional*):
            Pieces of every line of `documents_path`, in order, as `tokenizer` encodes the line's text with its
            whitespace collapsed. When given, `tokenizer` is not called.

    Returns:
        `int`: Number of rows written.
    """
    os.makedirs(output_dir, exist_ok=True)
    limit = max_tokens - 1
    segments, rows = [], [0]
    written, used = 0, 0
    with open(os.path.join(output_dir, DENSE_TOKENS_NAME), "wb") as sink, open(documents_path, encoding="utf-8") as lines:
        for index, line in enumerate(lines):
            text = " ".join((json.loads(line).get("text") or "").split())
            if not text:
                continue
            encoded = tokenizer.encode(text, add_special_tokens=False) if pieces is None else pieces[index]
            np.asarray(encoded, dtype=np.int32).tofile(sink)
            for start in range(0, len(encoded), limit):
                length = min(limit, len(encoded) - start)
                if used and used + length + 1 > max_tokens:
                    rows.append(len(segments))
                    used = 0
                segments.append((written + start, length))
                used += length + 1
            written += len(encoded)
    if used:
        rows.append(len(segments))
    np.save(os.path.join(output_dir, DENSE_SEGMENTS_NAME), np.asarray(segments, dtype=np.int64).reshape(-1, 2))
    np.save(os.path.join(output_dir, DENSE_ROWS_NAME), np.asarray(rows, dtype=np.int64))
    return len(rows) - 1


class DenseTextDataset(Dataset):
    r"""
    Reads the rows [`write_dense_cache`] packs, one row per item.

    Args:
        cache_dir (`str`):
            Directory holding `tokens.bin`, `segments.npy` and `rows.npy`.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.tokens = np.memmap(os.path.join(cache_dir, DENSE_TOKENS_NAME), dtype=np.int32, mode="r")
        self.segments = np.load(os.path.join(cache_dir, DENSE_SEGMENTS_NAME))
        self.rows = np.load(os.path.join(cache_dir, DENSE_ROWS_NAME))

    def __len__(self) -> int:
        return len(self.rows) - 1

    def __getitem__(self, index: int) -> dict:
        first, last = self.rows[index], self.rows[index + 1]
        return {
            "segments": [
                torch.from_numpy(np.asarray(self.tokens[start : start + length], dtype=np.int64))
                for start, length in self.segments[first:last]
            ]
        }

    def example_lengths(self) -> list:
        r"""
        Returns:
            `list`: Frames of each row, opening frames included, for a sampler that groups by length.
        """
        per_segment = self.segments[:, 1] + 1
        totals = np.concatenate([[0], np.cumsum(per_segment)])
        return (totals[self.rows[1:]] - totals[self.rows[:-1]]).tolist()


__all__ = [
    "DenseTextDataset",
    "Phase1Dataset",
    "Phase1Example",
    "Phase2Dataset",
    "delay_audio_codes",
    "phase1_collate_fn",
    "phase2_collate_fn",
    "text_frame_positions",
    "write_dense_cache",
]
