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
r"""
Lays an utterance out the way the checkpoint reads one, rather than the way a text model would.

Three streams run at the codec's frame rate and the text one carries only what the assistant says, on
the frames it says it: a word's pieces start on the frame its first sample falls in, `EPAD` sits one
frame ahead of them to close the padding, and `PAD` fills what is left. On top of that the batch has
to open on the frame that says nobody has spoken yet, and both acoustic streams have to arrive in the
delayed layout. Getting either wrong does not raise: it moves every absolute position, and the model
answers from the wrong ones.

The persona prefix is built by the processor and prepended to every item, so the frames a run scores
start after it. What the prefix instructs is a property of the run, not of the corpus, which is why it
is a constructor argument rather than something the cache carries.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


PAD_TOKEN_ID = 3
EPAD_TOKEN_ID = 0

FRAMES_PER_SECOND = 12.5

# Frames a text piece is given when no alignment says otherwise. Measured on KsponSpeech, where the
# nominal layout scored 7.682 against 7.765 for the real word times.
FRAMES_PER_PIECE = 4.119

# Marks a colloquial transcript ends a word with, each of which costs the word a frame of its own.
TRAILING_PUNCTUATION = ".,?!~…\"')]》」”’。？！"


def split_trailing_punctuation(word: str) -> tuple:
    r"""
    Args:
        word (`str`):
            A word as the transcript spells it.

    Returns:
        `tuple`: The word without its trailing punctuation, and that punctuation. A word made of
        nothing but punctuation is returned whole, since there is no word left to carry it.
    """
    stripped = word.rstrip(TRAILING_PUNCTUATION)
    if not stripped:
        return word, ""
    return stripped, word[len(stripped) :]


class DuplexUtteranceDataset(Dataset):
    r"""
    Reads the cache [`DuplexCacheWriter`] writes: acoustic codes and word spans.

    Args:
        cache_dir (`str`):
            Directory holding `index.jsonl` and `codes.npy`.
        limit (`int`, *optional*):
            Cap on how many utterances are read, `0` for all of them.
    """

    def __init__(self, cache_dir: str, limit: int = 0):
        self.cache_dir = cache_dir
        self.codes = np.load(os.path.join(cache_dir, "codes.npy"), mmap_mode="r")
        self.entries = DuplexManifest(os.path.join(cache_dir, "index.jsonl"), limit)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict:
        entry = self.entries[index]
        start = entry["offset"]
        return {
            "codes": torch.from_numpy(
                np.asarray(self.codes[:, start : start + entry["frames"]], dtype=np.int64)
            ),
            "words": entry["text"].split(),
            "starts": entry["start"],
        }

    def example_lengths(self) -> list:
        r"""
        Returns:
            `list`: Frame count of each utterance, for a sampler that groups by length.
        """
        return self.entries.frames.tolist()


class DuplexManifest:
    r"""
    Constructs a read-only view of a cache manifest that parses one record at a time.

    The manifest is held as one byte array with line offsets and a frame count per record, so a forked
    data loader worker shares it instead of copying millions of Python objects as their reference counts
    change.

    Args:
        path (`str`):
            The `index.jsonl` file.
        limit (`int`, *optional*, defaults to 0):
            Cap on how many records are read, `0` for all of them.
    """

    def __init__(self, path: str, limit: int = 0):
        with open(path, "rb") as manifest:
            raw = manifest.read()
        ends = np.flatnonzero(np.frombuffer(raw, dtype=np.uint8) == ord("\n")) + 1
        if raw and not raw.endswith(b"\n"):
            ends = np.append(ends, len(raw))
        if limit:
            ends = ends[:limit]
            raw = raw[: int(ends[-1])] if len(ends) else b""
        self.data = np.frombuffer(raw, dtype=np.uint8)
        self.offsets = np.concatenate([[0], ends]).astype(np.int64)
        self.frames = np.array([self[index]["frames"] for index in range(len(self))], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return json.loads(self.data[self.offsets[index] : self.offsets[index + 1]].tobytes())

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


class DuplexCacheWriter:
    r"""
    Constructs a writer of the cache [`DuplexUtteranceDataset`] reads: `index.jsonl` with one record per
    utterance and `codes.npy` with every utterance's codes laid end to end along the frame axis.

    Args:
        output_dir (`str`):
            Directory the cache is written to. Created if missing.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._manifest = open(os.path.join(output_dir, "index.jsonl"), "w", encoding="utf-8")
        self._codes = []
        self.frames = 0
        self.count = 0

    def add(self, text: str, starts: list, ends: list, codes: np.ndarray, utterance_id: str | None = None):
        r"""
        Appends one utterance.

        Args:
            text (`str`):
                Transcript, words separated by spaces.
            starts (`list`):
                Onset of each word, in seconds.
            ends (`list`):
                End of each word, in seconds.
            codes (`numpy.ndarray` of shape `(codebooks, frames)`):
                Acoustic codes of the utterance.
            utterance_id (`str`, *optional*):
                Identifier written to the record, numbered in order when omitted.
        """
        record = {
            "utterance_id": utterance_id or f"utt-{self.count:08d}",
            "text": text,
            "start": list(starts),
            "end": list(ends),
            "frames": int(codes.shape[-1]),
            "offset": self.frames,
        }
        self._manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._codes.append(np.asarray(codes, dtype=np.int16))
        self.frames += int(codes.shape[-1])
        self.count += 1

    def close(self):
        r"""
        Writes `codes.npy` and closes the manifest.
        """
        self._manifest.close()
        if self._codes:
            np.save(os.path.join(self.output_dir, "codes.npy"), np.concatenate(self._codes, axis=-1))
        self._codes = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @staticmethod
    def merge(sources: list, output_dir: str, exclude: set | None = None) -> int:
        r"""
        Concatenates caches into one, shifting each record's offset by the frames before it.

        Args:
            sources (`list`):
                Cache directories, in the order their utterances appear in the result.
            output_dir (`str`):
                Directory the merged cache is written to.
            exclude (`set`, *optional*):
                Utterance texts left out of the index, such as the samples an earlier run trained on. Their
                codes stay in `codes.npy`, where nothing reads them.

        Returns:
            `int`: Number of utterances in the merged cache.
        """
        os.makedirs(output_dir, exist_ok=True)
        codes = [np.load(os.path.join(source, "codes.npy"), mmap_mode="r") for source in sources]
        np.save(os.path.join(output_dir, "codes.npy"), np.concatenate(codes, axis=1))
        shift = count = 0
        with open(os.path.join(output_dir, "index.jsonl"), "w", encoding="utf-8") as manifest:
            for source, block in zip(sources, codes):
                name = os.path.basename(os.path.normpath(source))
                with open(os.path.join(source, "index.jsonl"), encoding="utf-8") as lines:
                    for line in lines:
                        record = json.loads(line)
                        if exclude and record["text"] in exclude:
                            continue
                        record["offset"] += shift
                        record["utterance_id"] = f"{name}-{record['utterance_id']}"
                        manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                        count += 1
                shift += block.shape[1]
        return count


def nominal_layout(words: list, tokenizer, frames_per_piece: float = FRAMES_PER_PIECE) -> tuple:
    r"""
    Spaces the words out at the nominal rhythm, one fixed number of frames per text piece.

    Args:
        words (`list`):
            Words of the utterance, in order.
        tokenizer (`PreTrainedTokenizerBase`):
            Tokenizer of the text stream, which decides how many pieces a word costs.
        frames_per_piece (`float`, *optional*, defaults to 4.119):
            Frames a piece occupies. It belongs to the tokenizer: a vocabulary with coarser pieces covers the
            same speech in fewer of them.

    Returns:
        `tuple`: Onsets in seconds, ends in seconds, and the frame count the utterance needs.
    """
    starts, ends, pieces_before = [], [], 0
    for word in words:
        pieces = max(1, len(tokenizer.encode(" " + word, add_special_tokens=False)))
        starts.append(pieces_before * frames_per_piece / FRAMES_PER_SECOND)
        pieces_before += pieces
        ends.append(pieces_before * frames_per_piece / FRAMES_PER_SECOND)
    # One extra piece of room so the last word's pieces and the mark after it still fit.
    frames = int(np.ceil((pieces_before + 1) * frames_per_piece))
    return starts, ends, frames


def codec_silent_frame(processor, device: str = "cuda") -> np.ndarray:
    r"""
    Encodes two seconds of silence and returns one frame from the middle of it.

    Args:
        processor (`PolyplexProcessor`):
            Processor whose codec encodes the silence.
        device (`str`, *optional*, defaults to `"cuda"`):
            Device the codec runs on.

    Returns:
        `numpy.ndarray` of shape `(codebooks, 1)`: The codes of one silent frame, as `int16`.
    """
    processor.audio_tokenizer.to(device)
    rate = processor.audio_tokenizer.config.sampling_rate
    waveform = np.zeros(int(rate * 2.0), dtype=np.float32)
    encoded = processor(assistant_audio=waveform, sampling_rate=rate, return_tensors="pt")
    silence = encoded["assistant_audio_codes"][0].to(torch.int16).cpu().numpy()
    # The interior of a silent stretch is what a padded slot should look like, not its onset.
    return silence[:, silence.shape[-1] // 2 : silence.shape[-1] // 2 + 1]


def build_text_cache(
    dataset_dir: str,
    model_dir: str,
    output_dir: str,
    limit: int = 0,
    max_frames: int = 0,
    frames_per_piece: float = FRAMES_PER_PIECE,
    device: str = "cuda",
    origin: str | None = None,
    skip: int = 0,
) -> int:
    r"""
    Writes a duplex cache from a Phase 1 text dataset, with codec silence in the audio slots.

    Phase 1 needs no speech: the nominal rhythm scores the same as a real alignment to within measurement
    error, so a plain text corpus is enough. The audio slots hold the codes of a silent frame rather than
    nothing, since a backbone given an input it never saw in pretraining does not learn.

    Args:
        dataset_dir (`str`):
            Directory of a saved [`Phase1Dataset`].
        model_dir (`str`):
            Expanded checkpoint the processor is read from.
        output_dir (`str`):
            Directory the cache is written to.
        limit (`int`, *optional*, defaults to 0):
            Cap on utterances, `0` for all.
        max_frames (`int`, *optional*, defaults to 0):
            Utterances longer than this are dropped, `0` keeps all.
        frames_per_piece (`float`, *optional*, defaults to 4.119):
            Frames a text piece stands for.
        device (`str`, *optional*, defaults to `"cuda"`):
            Device the codec runs on.
        origin (`str`, *optional*):
            Keep only the examples of this origin, such as `"translation"`. `limit` counts kept examples.
        skip (`int`, *optional*, defaults to 0):
            Examples passed over before the first one is read, counted the way `limit` counts them. A cache
            built with `skip` equal to an earlier cache's `limit` holds none of that cache's examples.

    Returns:
        `int`: Number of utterances written.
    """
    from ..models.polyplex import PolyplexProcessor
    from .phase_data import Phase1Dataset

    processor = PolyplexProcessor.from_pretrained(model_dir)
    frame = codec_silent_frame(processor, device)

    examples = Phase1Dataset.from_pretrained(dataset_dir).examples
    if origin is not None:
        examples = [example for example in examples if example.origin == origin]
    examples = examples[skip:]
    examples = examples[:limit] if limit else examples
    with DuplexCacheWriter(output_dir) as writer:
        for position, example in enumerate(examples):
            words = example.target.split()
            if not words:
                continue
            starts, ends, frames = nominal_layout(words, processor.tokenizer, frames_per_piece)
            if max_frames and frames > max_frames:
                continue
            writer.add(example.target, starts, ends, np.repeat(frame, frames, axis=-1), f"text-{position:08d}")
        return writer.count


def text_stream(
    words: list, starts: list, frames: int, tokenizer, offset: int, prefix: torch.Tensor | None = None
) -> torch.Tensor:
    r"""
    Places the words on the frames they were spoken on, in the inner monologue layout.

    A word's pieces run from its onset frame, `EPAD` occupies the frame before the first piece unless
    the previous word's last piece is already there, and every other frame holds `PAD`. A word that starts
    before the previous word's pieces end overwrites them from its onset on, so the previous word is cut. Only the first piece of a word is announced: `EPAD` marks the end of
    a silence, not the start of a token.

    Trailing punctuation is placed after the word rather than counted as part of it. A short spoken
    form often lasts a single frame, so a mark attached to it pushes the word past the span it was
    spoken in and the whole word is lost, which costs 4.15% of the words of KsponSpeech against 1.70%
    once the mark is moved off. The mark is written on the frame following the word's last piece only
    when that frame would otherwise hold padding: a real word's pieces and the `EPAD` announcing them
    are never displaced, so the mark is dropped instead whenever the next word starts too soon.

    The prefix is written first so that a word beginning on the utterance's very first frame can still
    be announced: its `EPAD` lands on the prefix's closing frame, which holds padding anyway. Writing
    the prefix afterwards would erase that announcement, and dropping such a word instead would lose
    the first word of every utterance whose speech starts at zero.

    Args:
        words (`list`):
            The words of the utterance, in order.
        starts (`list`):
            Onset of each word, in seconds.
        frames (`int`):
            Length of the stream, in frames.
        tokenizer (`PreTrainedTokenizerBase`):
            Tokenizer of the text stream.
        offset (`int`):
            Frame the utterance begins on, i.e. the length of the prefix in front of it.
        prefix (`torch.LongTensor`, *optional*):
            The prefix's own text stream, written over the first `offset` frames.

    Returns:
        `torch.LongTensor` of shape `(frames,)`: The text stream.
    """
    stream = torch.full((frames,), PAD_TOKEN_ID, dtype=torch.long)
    if prefix is not None:
        stream[: prefix.shape[0]] = prefix
    # The word's own frames are counted from the utterance, the prefix sits before all of them.
    onsets = [offset + int(start * 12.5) for start in starts]
    # The frame after the last piece written.
    end = offset
    for index, (onset, word) in enumerate(zip(onsets, words)):
        pieces = tokenizer.encode(" " + word, add_special_tokens=False)
        if not pieces or onset < 1:
            continue

        spoken, marks = split_trailing_punctuation(word)
        trailing = []
        if marks:
            # The word keeps the ids it tokenizes to, so the mark is split off only where it is a
            # clean suffix of them and the two halves rejoin to the original sequence.
            head = tokenizer.encode(" " + spoken, add_special_tokens=False)
            if head and pieces[: len(head)] == head:
                pieces, trailing = head, pieces[len(head) :]

        if onset + len(pieces) > frames:
            continue
        # Moshi 3.4.4 leaves the `EPAD` out when its frame holds a piece of the previous word. The prefix's
        # frames are never a word's, so a word on the utterance's first frame is still announced.
        if onset - 1 < offset or stream[onset - 1] == PAD_TOKEN_ID:
            stream[onset - 1] = EPAD_TOKEN_ID
        # Equation 5 applied word by word: a word starting inside the previous one overwrites its pieces, and
        # whatever the previous word still holds past the new pieces is cleared, so it ends cut.
        stream[onset : onset + len(pieces)] = torch.tensor(pieces, dtype=torch.long)
        if end > onset + len(pieces):
            stream[onset + len(pieces) : end] = PAD_TOKEN_ID
        end = onset + len(pieces)

        # The next word's `EPAD` sits one frame ahead of it, and the mark may not reach that far. A
        # word whose pieces run past the next onset holds its frames too, so only padding is taken.
        after = onset + len(pieces)
        limit = onsets[index + 1] - 1 if index + 1 < len(onsets) else frames
        room = stream[after : after + len(trailing)]
        if trailing and after + len(trailing) <= min(limit, frames) and (room == PAD_TOKEN_ID).all():
            stream[after : after + len(trailing)] = torch.tensor(trailing, dtype=torch.long)
    return stream


def packed_words(words, starts, frames, tokenizer, offset, prefix=None):
    r"""
    Puts the words at the front of the stream instead of on the frames they were spoken on.

    A corpus that ships no word times leaves nothing else to do, and the English corpora that carry
    their own speech are in that position. A state is English whether or not its word sits on the frame the speaker
    reached it on, so a term that only asks what the model does on English input loses nothing to the
    missing alignment.

    Args:
        words (`list`):
            The words of the utterance, in order.
        starts (`list`):
            Ignored, the argument the aligned layout reads its onsets from.
        frames (`int`):
            Length of the stream, in frames.
        tokenizer (`PreTrainedTokenizerBase`):
            Tokenizer of the text stream.
        offset (`int`):
            Frame the utterance begins on, i.e. the length of the prefix in front of it.
        prefix (`torch.LongTensor`, *optional*):
            The prefix's own text stream, written over the first `offset` frames.

    Returns:
        `torch.LongTensor` of shape `(frames,)`: The text stream.
    """
    stream = torch.full((frames,), PAD_TOKEN_ID, dtype=torch.long)
    if prefix is not None:
        stream[: prefix.shape[0]] = prefix

    cursor = offset
    for word in words:
        pieces = tokenizer.encode(" " + word, add_special_tokens=False)
        if cursor + len(pieces) + 1 > frames:
            break
        # Every word is announced the way the aligned layout announces one, `EPAD` then its pieces.
        stream[cursor] = EPAD_TOKEN_ID
        stream[cursor + 1 : cursor + 1 + len(pieces)] = torch.tensor(pieces, dtype=torch.long)
        cursor += len(pieces) + 1
    return stream


def duplex_collate_fn(
    processor,
    model,
    persona: str = "You are a helpful assistant.",
    score_words_only: bool = True,
    drop_audio: bool = False,
    max_frames: int | None = None,
    audio_labels: bool = False,
    layout=None,
):
    r"""
    Builds the `collate_fn` that turns cached utterances into the batch the model reads.

    Args:
        processor (`PolyplexProcessor`):
            Processor of the expanded checkpoint. Supplies the persona prefix, the tokenizer and the
            silence a stream with no audio is filled with.
        model (`PolyplexForConditionalGeneration`):
            The model, for its opening frame and its delay pattern. Nothing is run on it here.
        persona (`str`, *optional*):
            Instruction the prefix carries.
        score_words_only (`bool`, *optional*, defaults to `True`):
            Whether `PAD` and `EPAD` frames are left out of the loss. A run with no acoustic stream has
            no timing to learn from, and scoring those frames there teaches it to speak over silence.
        drop_audio (`bool`, *optional*, defaults to `False`):
            Whether the assistant's speech is replaced by silence, which is the ablation that says what
            the acoustic stream is worth.
        max_frames (`int`, *optional*):
            Hard cap on the length of an item, in frames.
        audio_labels (`bool`, *optional*, defaults to `False`):
            Whether the batch also carries `audio_labels`, the assistant's codes as targets for the depth
            decoder. The opening frame and the persona prefix are masked out: the prefix is ours, not
            speech the model is asked to produce.
        layout (`Callable`, *optional*):
            Writes the text stream of an item, with the signature of [`text_stream`]. Defaults to
            [`text_stream`], the aligned layout; [`packed_words`] serves corpora with no word times.

    Returns:
        `Callable`: A `collate_fn` producing `input_ids`, `attention_mask`, both acoustic streams and
        `text_labels`. An example is a dict with `codes`, `words` and `starts`, and may carry
        `user_codes` as well, the speech the user side hears over the same frames. Without them the user
        side is silent.
    """
    # The processor builds the prefix on the codec's device; a collator hands the trainer host tensors.
    prefix = {
        key: value.cpu() for key, value in processor(text_prompt=persona, return_tensors="pt").items()
    }
    prefix_frames = prefix["input_ids"].shape[-1]
    reserved = model.config.audio_vocab_size
    codebooks = model.num_codebooks
    tokenizer = processor.tokenizer

    # The processor allocates the silence on the codec's device and a forked dataloader worker may not
    # touch CUDA. Silence is the same in every batch, so it is built once here, in the process that
    # owns the codec, and sliced afterwards.
    silence = processor.get_silence_audio_codes(max_frames or 4096)[0].cpu().to(torch.long)

    def silent(frames: int) -> torch.Tensor:
        """The opening `frames` of silence, holding the last frame over if more is asked for."""
        if frames <= silence.shape[-1]:
            return silence[:, :frames]
        held = silence[:, -1:].expand(-1, frames - silence.shape[-1])
        return torch.cat([silence, held], dim=-1)

    def delayed(codes: torch.Tensor) -> torch.Tensor:
        """Returns the wire format: every codebook after the first pushed one frame later."""
        _, mask = model.build_delay_pattern_mask(
            codes, bos_token_id=reserved, pad_token_id=reserved, max_length=codes.shape[-1]
        )
        return model.apply_delay_pattern_mask(codes, mask)

    def collate(examples: list) -> dict:
        streams, assistants, users, labels = [], [], [], []
        for example in examples:
            codes = example["codes"]
            if max_frames is not None:
                codes = codes[:, : max_frames - prefix_frames]
            body = codes.shape[-1]
            if drop_audio:
                codes = silent(body)
            # Speech on the user side is what listening is fitted on. Without it the user is silent,
            # which is what every assistant side example has always carried there.
            heard = example.get("user_codes")
            if heard is None:
                heard = silent(body)
            else:
                heard = heard[:, :body].to(torch.long)
                if heard.shape[-1] < body:
                    heard = torch.cat([heard, silent(body)[:, heard.shape[-1] :]], dim=-1)

            total = prefix_frames + body
            # Looked up per call, so a layout swapped on the module reaches a collator built before it.
            stream = (layout or text_stream)(
                example["words"], example["starts"], total, tokenizer, prefix_frames,
                prefix=prefix["input_ids"][0],
            )

            label = stream.clone()
            label[:prefix_frames] = -100  # The instruction is ours, not something to predict.
            if score_words_only:
                label[(label == PAD_TOKEN_ID) | (label == EPAD_TOKEN_ID)] = -100

            streams.append(stream)
            assistants.append(torch.cat([prefix["assistant_audio_codes"][0], codes], dim=-1))
            users.append(heard)
            labels.append(label)

        width = max(stream.shape[0] for stream in streams)
        batch = len(streams)
        input_ids = torch.full((batch, width), PAD_TOKEN_ID, dtype=torch.long)
        text_labels = torch.full((batch, width), -100, dtype=torch.long)
        attention_mask = torch.zeros((batch, width), dtype=torch.bool)
        assistant = torch.full((batch, codebooks, width), reserved, dtype=torch.long)
        user = torch.full((batch, codebooks, width), reserved, dtype=torch.long)

        for row, (stream, codes, heard, label) in enumerate(zip(streams, assistants, users, labels)):
            length = stream.shape[0]
            input_ids[row, :length] = stream
            text_labels[row, :length] = label
            attention_mask[row, :length] = True
            assistant[row, :, :length] = codes
            user[row, :, :prefix_frames] = prefix["user_audio_codes"][0]
            user[row, :, prefix_frames:length] = heard

        # The opening frame every absolute position downstream is measured from.
        opening_text = torch.full((batch, 1), model.config.vocab_size, dtype=torch.long)
        opening_codes = torch.full((batch, codebooks, 1), reserved, dtype=torch.long)
        batch_out = {
            "input_ids": torch.cat([opening_text, input_ids], dim=1),
            "attention_mask": torch.cat([torch.ones_like(opening_text, dtype=torch.bool), attention_mask], 1),
            "assistant_audio_codes": delayed(torch.cat([opening_codes, assistant], dim=-1)),
            "user_audio_codes": delayed(torch.cat([opening_codes, user], dim=-1)),
            "text_labels": torch.cat([torch.full((batch, 1), -100, dtype=torch.long), text_labels], 1),
        }
        if audio_labels:
            targets = batch_out["assistant_audio_codes"].clone()
            # Column 0 is the opening frame and columns 1..prefix_frames the prefix. The delay moves every
            # codebook after the first one frame later, so their prefix reaches one column further.
            targets[:, :, : prefix_frames + 1] = -100
            targets[:, 1:, prefix_frames + 1] = -100
            batch_out["audio_labels"] = targets
        return batch_out

    return collate


def dense_collate_fn(processor, model, silent_audio: bool = True):
    r"""
    Builds the `collate_fn` that turns packed text rows into the batch the model reads, with no PAD frames.

    Every document of a row opens on its own opening frame and counts its positions from zero, and the batch
    carries `position_ids` and no `attention_mask`. The backbone then builds one causal block per document,
    so a document never attends to the one packed before it. With `silent_audio`, both acoustic streams hold codec
    silence, laid out in the delay pattern per document; without it the batch carries no audio codes and the
    model reads the text embeddings alone, the dense input of SPEC S13.1a. A row shorter than the batch is filled
    with one unscored segment.

    Args:
        processor (`PolyplexProcessor`):
            Processor of the expanded checkpoint, for the silence the acoustic streams hold.
        model (`PolyplexForConditionalGeneration`):
            The model, for its opening frame and its delay pattern. Nothing is run on it here.
        silent_audio (`bool`, *optional*, defaults to `True`):
            Whether both acoustic streams hold codec silence.

    Returns:
        `Callable`: A `collate_fn` taking examples with a `segments` list of token tensors and producing
        `input_ids`, `position_ids`, `text_labels` and, with `silent_audio`, both acoustic streams.
    """
    reserved = model.config.audio_vocab_size
    opening = model.config.vocab_size
    codebooks = model.num_codebooks
    silence = processor.get_silence_audio_codes(4096)[0].cpu().to(torch.long)

    def silent(frames: int) -> torch.Tensor:
        if frames <= silence.shape[-1]:
            return silence[:, :frames]
        held = silence[:, -1:].expand(-1, frames - silence.shape[-1])
        return torch.cat([silence, held], dim=-1)

    def delayed(codes: torch.Tensor) -> torch.Tensor:
        _, mask = model.build_delay_pattern_mask(
            codes, bos_token_id=reserved, pad_token_id=reserved, max_length=codes.shape[-1]
        )
        return model.apply_delay_pattern_mask(codes, mask)

    def segment(tokens: torch.Tensor, scored: bool) -> tuple:
        tokens = torch.as_tensor(tokens, dtype=torch.long)
        ids = torch.cat([torch.tensor([opening]), tokens])
        labels = torch.cat([torch.tensor([-100]), tokens if scored else torch.full_like(tokens, -100)])
        codes = torch.cat([torch.full((codebooks, 1), reserved, dtype=torch.long), silent(tokens.shape[0])], dim=-1)
        return ids, labels, torch.arange(ids.shape[0]), delayed(codes[None])[0]

    def collate(examples: list) -> dict:
        rows = [[segment(tokens, True) for tokens in example["segments"]] for example in examples]
        width = max(sum(part[0].shape[0] for part in row) for row in rows)
        columns = {"input_ids": [], "text_labels": [], "position_ids": [], "codes": []}
        for row in rows:
            missing = width - sum(part[0].shape[0] for part in row)
            if missing:
                row.append(segment(torch.full((missing - 1,), PAD_TOKEN_ID), False))
            for key, index in (("input_ids", 0), ("text_labels", 1), ("position_ids", 2)):
                columns[key].append(torch.cat([part[index] for part in row]))
            columns["codes"].append(torch.cat([part[3] for part in row], dim=-1))
        batch = {
            "input_ids": torch.stack(columns["input_ids"]),
            "position_ids": torch.stack(columns["position_ids"]),
            "text_labels": torch.stack(columns["text_labels"]),
        }
        if silent_audio:
            codes = torch.stack(columns["codes"])
            batch["assistant_audio_codes"], batch["user_audio_codes"] = codes, codes.clone()
        return batch

    return collate


def main():
    r"""
    Command line: `text` writes a cache from a text dataset, `unseen` writes a text dataset of sentences
    no listed file holds, `dialogue` writes a cache of dialogues followed from one speaker, `dense` packs
    a document corpus into rows for the dense stage, `merge` concatenates caches.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Duplex caches.")
    commands = parser.add_subparsers(dest="command", required=True)
    text = commands.add_parser("text", help="Write a cache from a Phase 1 text dataset.")
    text.add_argument("--dataset", required=True)
    text.add_argument("--model", required=True)
    text.add_argument("--output", required=True)
    text.add_argument("--limit", type=int, default=0)
    text.add_argument("--max-frames", type=int, default=0)
    text.add_argument("--frames-per-piece", type=float, default=FRAMES_PER_PIECE)
    text.add_argument("--device", default="cuda")
    text.add_argument("--origin")
    text.add_argument("--skip", type=int, default=0)
    unseen = commands.add_parser("unseen", help="Write a text dataset of sentences no listed file holds.")
    unseen.add_argument("--conversation-root", required=True)
    unseen.add_argument("--translation", required=True)
    unseen.add_argument("--seen", nargs="+", required=True)
    unseen.add_argument("--output", required=True)
    unseen.add_argument("--seed", type=int, default=0)
    unseen.add_argument("--translation-share", type=float, default=0.3)
    dialogue = commands.add_parser("dialogue", help="Write a cache of dialogues followed from one speaker.")
    dialogue.add_argument("--corpus-root", required=True)
    dialogue.add_argument("--split", default="train")
    dialogue.add_argument("--model", required=True)
    dialogue.add_argument("--output", required=True)
    dialogue.add_argument("--max-frames", type=int, default=1024)
    dialogue.add_argument("--exclude", nargs="*", default=[])
    dialogue.add_argument("--limit", type=int, default=0)
    dialogue.add_argument("--device", default="cuda")
    dialogue.add_argument("--skip", type=int, default=0)
    dense = commands.add_parser("dense", help="Pack a JSON lines document corpus into dense rows.")
    dense.add_argument("--documents", required=True)
    dense.add_argument("--model", required=True)
    dense.add_argument("--output", required=True)
    dense.add_argument("--max-tokens", type=int, default=1024)
    merge = commands.add_parser("merge", help="Concatenate caches into one.")
    merge.add_argument("--output", required=True)
    merge.add_argument("--exclude", nargs="*", default=[])
    merge.add_argument("sources", nargs="+")
    arguments = parser.parse_args()

    if arguments.command == "text":
        count = build_text_cache(
            arguments.dataset, arguments.model, arguments.output, arguments.limit, arguments.max_frames,
            arguments.frames_per_piece, arguments.device, arguments.origin, arguments.skip,
        )
    elif arguments.command == "unseen":
        count = write_unseen_dataset(
            arguments.conversation_root, arguments.translation, arguments.seen, arguments.output, arguments.seed,
            arguments.translation_share,
        )
    elif arguments.command == "dialogue":
        from .multisession import iter_dialogues

        exclude = set()
        for path in arguments.exclude:
            exclude |= read_texts(path)
        count = build_dialogue_cache(
            iter_dialogues(arguments.corpus_root, arguments.split), arguments.model, arguments.output,
            arguments.max_frames, exclude, arguments.limit, device=arguments.device, skip=arguments.skip,
        )
    elif arguments.command == "dense":
        from ..models.polyplex import PolyplexProcessor
        from .phase_data import write_dense_cache

        tokenizer = PolyplexProcessor.from_pretrained(arguments.model).tokenizer
        count = write_dense_cache(arguments.documents, tokenizer, arguments.output, arguments.max_tokens)
        print(f"[dense] {count} rows -> {arguments.output}", flush=True)
        return
    else:
        exclude = set()
        for path in arguments.exclude:
            exclude |= read_texts(path)
        count = DuplexCacheWriter.merge(arguments.sources, arguments.output, exclude)
    print(f"[cache] {count} utterances -> {arguments.output}", flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "DuplexCacheWriter",
    "DuplexManifest",
    "DuplexUtteranceDataset",
    "build_text_cache",
    "codec_silent_frame",
    "dense_collate_fn",
    "duplex_collate_fn",
    "nominal_layout",
    "packed_words",
    "text_stream",
]
