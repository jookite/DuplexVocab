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
"""KsponSpeech single-turn TTS dataset."""

import json
import os
import re

import torch

from .base import AudioFileFormat, SingleTurnTTSDataset


KSPONSPEECH_URL = "https://dolab-data.duckdns.org/api/public/dl/T8VlWVUT?inline=true"


SPELLING_PAIR = re.compile(r"\(([^()]*)\)/\(([^()]*)\)")
SPEECH_EVENT = re.compile(r"[bnlou]/|\*|\+|/")


def normalize_transcription(text: str, transcription_form: str = "phonetic") -> str:
    r"""
    Resolves KsponSpeech spelling pairs and strips speech event markers.

    Args:
        text (`str`):
            Transcription as stored in the corpus.
        transcription_form (`str`, *optional*, defaults to `"phonetic"`):
            Which side of a `(A)/(B)` spelling pair to keep, `"phonetic"` keeps the spoken form
            and `"orthographic"` keeps the written form.

    Returns:
        `str`: Transcription usable as text conditioning.
    """
    keep = 2 if transcription_form == "phonetic" else 1
    text = SPELLING_PAIR.sub(lambda match: match.group(keep), text)
    text = SPEECH_EVENT.sub(" ", text)
    return " ".join(text.split())


class KsponSpeechDataset(SingleTurnTTSDataset):
    r"""
    Constructs a KsponSpeech dataset for text to audio token training.

    The corpus stores headerless 16 kHz little endian 16 bit PCM utterances and a
    transcript file whose lines follow `<relative_pcm_path> :: <transcription>`.

    Args:
        root (`str`, *optional*):
            Directory holding the `KsponSpeech_0*` shards, defaults to the extracted
            corpus cache. Pass `download=True` to fetch it when the directory is missing.
        transcript_file (`str`, *optional*):
            Path of the transcript file listing the utterances of a split, defaults to the
            transcript discovered below `root`.
        split (`str`, *optional*, defaults to `"train"`):
            Split whose transcript file is looked up when `transcript_file` is unset.
        download (`bool`, *optional*, defaults to `False`):
            Whether to download and extract the corpus archive when `root` is missing.
        transcription_form (`str`, *optional*, defaults to `"phonetic"`):
            Which side of a `(A)/(B)` spelling pair to keep, `"phonetic"` keeps the
            spoken form and `"orthographic"` keeps the written form.
        sampling_rate (`int`, *optional*, defaults to 24000):
            Sampling rate expected by the audio encoder.
        min_duration (`float`, *optional*, defaults to 0.5):
            Utterances shorter than this many seconds are skipped.
        max_duration (`float`, *optional*, defaults to 30.0):
            Utterances longer than this many seconds are skipped.
        target_loudness (`float`, *optional*, defaults to -23.0):
            Integrated loudness in LUFS every utterance is scaled to, the EBU R128 reference. `None`
            leaves the decoded amplitude untouched.
        peak_ceiling (`float`, *optional*, defaults to 0.99):
            Highest absolute sample allowed after the loudness gain. An utterance whose gain would push
            it past this is scaled back to it, which costs a little loudness accuracy and avoids clipping.
    """

    def __init__(
        self,
        root: str | None = None,
        transcript_file: str | None = None,
        split: str = "train",
        download: bool = False,
        transcription_form: str = "phonetic",
        sampling_rate: int = 24000,
        min_duration: float = 0.5,
        max_duration: float = 30.0,
        target_loudness: float | None = -23.0,
        peak_ceiling: float = 0.99,
    ):
        if transcription_form not in ("phonetic", "orthographic"):
            raise ValueError(
                f"`transcription_form` must be 'phonetic' or 'orthographic', got {transcription_form}."
            )
        root = self.resolve_corpus_root(
            KSPONSPEECH_URL, "ksponspeech", root=root, download=download
        )

        self.target_loudness = target_loudness
        self.peak_ceiling = peak_ceiling
        self._meters: dict = {}
        self.split = split
        self.transcript_file = transcript_file or self._find_transcript(root, split)
        self.transcription_form = transcription_form
        super().__init__(
            root=root,
            sampling_rate=sampling_rate,
            min_duration=min_duration,
            max_duration=max_duration,
            audio_file_format=AudioFileFormat(
                headerless_extensions=(".pcm", ".raw"), sampling_rate=16000, dtype="<i2"
            ),
        )

    @staticmethod
    def _find_transcript(root: str, split: str) -> str:
        r"""
        Locates the transcript file of a split below the corpus root.

        Args:
            root (`str`):
                Directory holding the extracted corpus.
            split (`str`):
                Split name searched for in the transcript file names.

        Returns:
            `str`: Path of the matching transcript file.

        Raises:
            FileNotFoundError: If no transcript file matches the split.
        """
        candidates = []
        for directory, _, file_names in os.walk(root):
            for file_name in file_names:
                if file_name.endswith(".txt") and "trn" not in file_name and "script" not in file_name:
                    continue
                if split in file_name:
                    candidates.append(os.path.join(directory, file_name))
        if not candidates:
            raise FileNotFoundError(
                f"No transcript file for split '{split}' below {root}, pass `transcript_file` explicitly."
            )
        return sorted(candidates, key=len)[0]

    def _build_index(self) -> list:
        r"""
        Reads the transcript file and resolves audio paths against `self.root`.

        Returns:
            `list`: Entries of `(audio_path, raw_transcription)`.
        """
        entries = []
        with open(self.transcript_file, encoding="utf-8") as transcripts:
            for line in transcripts:
                line = line.strip()
                if not line:
                    continue
                relative_path, _, transcription = line.partition("::")
                if not transcription:
                    continue
                relative_path = relative_path.strip().replace("\\", "/")
                entries.append((os.path.join(self.root, relative_path), transcription.strip()))
        return entries

    def normalize_text(self, text: str) -> str:
        r"""
        Resolves KsponSpeech spelling pairs and strips speech event markers.

        Args:
            text (`str`):
                Transcription as stored in the corpus.

        Returns:
            `str`: Transcription usable as text conditioning.
        """
        return normalize_transcription(text, self.transcription_form)

    def _meter(self, sampling_rate: int):
        r"""
        Args:
            sampling_rate (`int`):
                Rate the meter is built for.

        Returns:
            `pyloudnorm.Meter`: A BS.1770 meter, built once per rate and per worker.

        Raises:
            ImportError: If `pyloudnorm` is missing while loudness normalization is on.
        """
        if sampling_rate not in self._meters:
            try:
                import pyloudnorm
            except ImportError as error:
                raise ImportError(
                    "Loudness normalization needs `pyloudnorm`, install the `preprocess` extra or pass "
                    "`target_loudness=None`."
                ) from error
            self._meters[sampling_rate] = pyloudnorm.Meter(sampling_rate)
        return self._meters[sampling_rate]

    def normalize_loudness(self, waveform: torch.FloatTensor) -> torch.FloatTensor:
        r"""
        Scales one waveform to `self.target_loudness`.

        The corpus was recorded across many devices and rooms, so utterance amplitude varies by more
        than an order of magnitude. Left alone that variance reaches the acoustic codes, and the adapter
        would have to spend capacity on it rather than on Korean phonetics.

        The gain is a single scalar, so the waveform is not otherwise reshaped. Two cases are left as
        they are: an utterance shorter than the 400 ms BS.1770 block, which has no measurable integrated
        loudness, and a digitally silent one, whose loudness is `-inf`.

        Args:
            waveform (`torch.FloatTensor` of shape `(1, num_samples)`):
                Decoded mono waveform.

        Returns:
            `torch.FloatTensor` of shape `(1, num_samples)`: The waveform at the target loudness.
        """
        if self.target_loudness is None:
            return waveform

        samples = waveform.squeeze(0).numpy()
        if samples.shape[-1] < int(0.4 * self.sampling_rate):
            return waveform

        loudness = self._meter(self.sampling_rate).integrated_loudness(samples)
        if not float(loudness) > float("-inf"):
            return waveform

        gain = 10.0 ** ((self.target_loudness - loudness) / 20.0)
        peak = waveform.abs().max().item()
        if peak * gain > self.peak_ceiling:
            gain = self.peak_ceiling / peak
        return waveform * gain

    def load_audio(self, audio_path: str) -> torch.FloatTensor:
        r"""
        Decodes one utterance and brings it to the target loudness.

        Args:
            audio_path (`str`):
                Path of the audio file.

        Returns:
            `torch.FloatTensor` of shape `(1, num_samples)`: The decoded, normalized waveform.
        """
        return self.normalize_loudness(super().load_audio(audio_path))


SOURCE_RATE = 16000


class KsponSpeechArchive:
    r"""
    Constructs a reader of the KsponSpeech distribution archives, which pulls utterances out without
    extracting them.

    The release ships one zip per shard (`KsponSpeech_01.zip`, ...) holding headerless 16 kHz PCM files,
    and `.trn` transcripts whose lines read `<relative_pcm_path> :: <transcription>`.

    Args:
        archives_dir (`str`):
            Directory holding the shard zips.
    """

    def __init__(self, archives_dir: str):
        self.archives_dir = archives_dir
        self._open = {}

    @staticmethod
    def read_transcript(path: str) -> dict:
        r"""
        Groups the utterances of a transcript file by the shard that holds their audio.

        Args:
            path (`str`):
                Path of a `.trn` transcript.

        Returns:
            `dict`: Shard name mapped to a list of `(member_path, raw_transcription)`.
        """
        shards = {}
        with open(path, encoding="utf-8") as transcripts:
            for line in transcripts:
                relative_path, _, transcription = line.strip().partition("::")
                if not transcription:
                    continue
                relative_path = relative_path.strip().replace("\\", "/")
                shards.setdefault(relative_path.split("/")[0], []).append(
                    (relative_path, transcription.strip())
                )
        return shards

    @staticmethod
    def decode_pcm(payload: bytes) -> torch.Tensor:
        r"""
        Args:
            payload (`bytes`):
                Headerless little endian 16 bit mono samples.

        Returns:
            `torch.Tensor` of shape `(num_samples,)`: The waveform scaled to `[-1, 1)`.
        """
        import numpy as np

        usable = len(payload) - (len(payload) % 2)
        samples = np.frombuffer(payload, dtype="<i2", count=usable // 2)
        return torch.from_numpy(samples.astype(np.float32) / 32768.0)

    def read(self, member: str) -> torch.Tensor:
        r"""
        Reads one utterance out of the shard zip its path names.

        Args:
            member (`str`):
                Path the transcript names. The eval archive drops the shard component, so both forms
                are tried.

        Returns:
            `torch.Tensor` of shape `(num_samples,)`: The waveform at 16 kHz, scaled to `[-1, 1)`.
        """
        import zipfile

        shard = member.split("/")[0]
        if shard not in self._open:
            self._open[shard] = zipfile.ZipFile(os.path.join(self.archives_dir, shard + ".zip"))
        archive = self._open[shard]
        try:
            payload = archive.read(member)
        except KeyError:
            payload = archive.read(member.split("/", 1)[1])
        return self.decode_pcm(payload)


def subsample(entries: list, max_hours: float | None, mean_duration: float = 5.55) -> list:
    r"""
    Thins an utterance list down to roughly a target amount of audio, at a fixed stride so that a capped
    run still spans the speakers and sessions the shard is ordered by.

    Args:
        entries (`list`):
            Utterances of one shard.
        max_hours (`float` or `None`):
            Target amount of audio, `None` keeps everything.
        mean_duration (`float`, *optional*, defaults to 5.55):
            Assumed seconds per utterance, used to turn hours into a count before decoding.

    Returns:
        `list`: The kept utterances, in their original order.
    """
    if max_hours is None:
        return entries
    target = int(max_hours * 3600 / mean_duration)
    if target >= len(entries):
        return entries
    stride = len(entries) / target
    return [entries[int(index * stride)] for index in range(target)]


def read_alignment_index(alignment_dir: str, subsets: tuple, minimum: float, maximum: float, limit: int) -> list:
    r"""
    Reads the word alignment index of KsponSpeech, keeping the utterances a run can use.

    Args:
        alignment_dir (`str`):
            Directory holding `alignment.jsonl` (from the CTC pass) and `audio_paths.json`.
        subsets (`tuple`):
            Splits to keep, out of `"train"`, `"valid"` and `"test"`.
        minimum (`float`):
            Shortest utterance kept, in seconds.
        maximum (`float`):
            Longest utterance kept, in seconds.
        limit (`int`):
            Cap on how many are kept, `0` for all of them.

    Returns:
        `list`: Entries carrying the transcript, the word spans and the archive member.
    """
    with open(os.path.join(alignment_dir, "audio_paths.json"), encoding="utf-8") as handle:
        paths = json.load(handle)
    kept = []
    with open(os.path.join(alignment_dir, "alignment.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item["subset"] not in subsets or not minimum <= item["duration"] <= maximum:
                continue
            member = paths.get(item["utterance_id"])
            if member is None:
                continue
            item["member"] = member
            kept.append(item)
            if limit and len(kept) >= limit:
                break
    return kept


def write_duplex_cache(
    alignment_dir: str,
    archives_dir: str,
    model_dir: str,
    output_dir: str,
    subsets: tuple = ("train",),
    limit: int = 0,
    min_duration: float = 2.0,
    max_duration: float = 12.0,
    device: str = "cuda",
) -> int:
    r"""
    Encodes KsponSpeech into the acoustic codes and word spans a duplex batch is assembled from.

    The processor encodes one utterance per call, which is what deployment does. The frame layout is not
    cached, since the text stream depends on the persona prefix a run chooses.

    Args:
        alignment_dir (`str`):
            Directory of the alignment index, see [`read_alignment_index`].
        archives_dir (`str`):
            Directory of the release archives.
        model_dir (`str`):
            Expanded checkpoint the processor is read from.
        output_dir (`str`):
            Directory the cache is written to.
        subsets (`tuple`, *optional*, defaults to `("train",)`):
            Splits to encode.
        limit (`int`, *optional*, defaults to 0):
            Cap on utterances, `0` for all.
        min_duration (`float`, *optional*, defaults to 2.0):
            Shortest utterance kept, in seconds.
        max_duration (`float`, *optional*, defaults to 12.0):
            Longest utterance kept, in seconds.
        device (`str`, *optional*, defaults to `"cuda"`):
            Device the codec runs on.

    Returns:
        `int`: Number of utterances written.
    """
    import torchaudio

    from ..models.polyplex import PolyplexProcessor
    from .duplex_data import DuplexCacheWriter

    processor = PolyplexProcessor.from_pretrained(model_dir)
    processor.audio_tokenizer.to(device)
    rate = processor.audio_tokenizer.config.sampling_rate
    resampler = torchaudio.transforms.Resample(SOURCE_RATE, rate)
    archive = KsponSpeechArchive(archives_dir)

    entries = read_alignment_index(alignment_dir, tuple(subsets), min_duration, max_duration, limit)
    with DuplexCacheWriter(output_dir) as writer:
        for item in entries:
            waveform = resampler(archive.read(item["member"])).numpy()
            encoded = processor(assistant_audio=waveform, sampling_rate=rate, return_tensors="pt")
            codes = encoded["assistant_audio_codes"][0].to(torch.int16).cpu().numpy()
            writer.add(item["text"], item["start"], item["end"], codes, item["utterance_id"])
        return writer.count


def import_code_cache(source: str, output_prefix: str, min_duration: float = 2.0, max_duration: float = 12.0) -> dict:
    r"""
    Builds the duplex caches Phase 2 reads out of an existing per utterance KsponSpeech code cache.

    The source holds one `<key>.npz` per utterance with the Mimi codes and the word onsets and ends, and a
    `<key>.json` beside it with the transcript and the split. It was encoded with every clip brought to the
    same loudness first, so every split is taken from this one source and training and evaluation see the
    same preprocessing. Utterances in which another speaker is heard are dropped, since Phase 2 teaches the
    model to produce one voice. Utterances whose word count disagrees with their onsets are dropped too,
    since the layout places word `i` at onset `i`.

    Args:
        source (`str`):
            Directory holding the `<key>.npz` and `<key>.json` pairs.
        output_prefix (`str`):
            Prefix of the split directories, written as `<prefix>_train`, `<prefix>_valid` and `<prefix>_test`.
        min_duration (`float`, *optional*, defaults to 2.0):
            Shortest utterance kept, in seconds.
        max_duration (`float`, *optional*, defaults to 12.0):
            Longest utterance kept, in seconds.

    Returns:
        `dict`: Split name mapped to the number of utterances written.
    """
    import numpy as np

    from .duplex_data import DuplexCacheWriter

    splits = ("train", "valid", "test")
    kept = {split: [] for split in splits}
    with os.scandir(source) as listing:
        for item in listing:
            if not item.name.endswith(".json"):
                continue
            try:
                with open(item.path, encoding="utf-8") as handle:
                    meta = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            if meta.get("subset") not in kept or meta.get("has_other_speaker"):
                continue
            if not min_duration <= float(meta.get("duration", 0.0)) <= max_duration:
                continue
            meta["key"] = item.name[: -len(".json")]
            kept[meta["subset"]].append(meta)

    counts = {}
    for split in splits:
        entries = sorted(kept[split], key=lambda meta: meta.get("utterance_id", meta["key"]))
        with DuplexCacheWriter(f"{output_prefix}_{split}") as writer:
            for meta in entries:
                arrays = np.load(os.path.join(source, meta["key"] + ".npz"))
                starts = arrays["align_start"].astype(float).tolist()
                if len(meta["text"].split()) != len(starts):
                    continue
                ends = arrays["align_end"].astype(float).tolist()
                writer.add(meta["text"], starts, ends, arrays["codes"], meta.get("utterance_id", meta["key"]))
            counts[split] = writer.count
    return counts


def encode_mimi_batch(mimi, waveforms: list, num_codebooks: int, device: str) -> list:
    r"""
    Encodes a batch of waveforms, keeping only the frames that are unaffected by padding.

    Mimi is causal, so right padding leaves every complete frame untouched while the trailing partial frame
    changes. Each clip is therefore cut to its whole frames.

    Args:
        mimi (`MimiModel`):
            The frozen codec.
        waveforms (`list` of `torch.Tensor`):
            Waveforms at the codec sampling rate.
        num_codebooks (`int`):
            Number of codebooks kept per frame.
        device (`str`):
            Device the codec runs on.

    Returns:
        `list` of `numpy.ndarray`: Per clip codes of shape `(num_codebooks, num_frames)`.
    """
    import numpy as np

    samples_per_frame = int(mimi.config.sampling_rate / mimi.config.frame_rate)
    longest = max(waveform.shape[-1] for waveform in waveforms)
    padded = torch.zeros(len(waveforms), 1, longest)
    for position, waveform in enumerate(waveforms):
        padded[position, 0, : waveform.shape[-1]] = waveform
    with torch.no_grad():
        codes = mimi.encode(padded.to(device), num_quantizers=num_codebooks).audio_codes
    codes = codes.cpu().numpy().astype(np.int16)
    return [codes[position, :, : waveform.shape[-1] // samples_per_frame] for position, waveform in enumerate(waveforms)]


def write_mimi_cache(
    archive_path: str,
    transcript_path: str,
    output_dir: str,
    shards: list | None = None,
    max_hours: float | None = None,
    num_codebooks: int = 8,
    max_batch_seconds: float = 48.0,
    sort_window: int = 256,
    min_duration: float = 0.5,
    max_duration: float = 30.0,
    transcription_form: str = "phonetic",
    device: str = "cuda",
    force: bool = False,
):
    r"""
    Encodes KsponSpeech into the per shard Mimi code cache [`MimiCacheMixin`] reads, streaming the
    distribution archive without extracting it. Used by the code rotation comparison arm of Phase 2.

    Each shard becomes one `<shard>.npy` with every utterance's codes end to end and one `<shard>.jsonl`
    naming the slice each utterance occupies.

    Args:
        archive_path (`str`):
            Path of the KsponSpeech distribution zip.
        transcript_path (`str`):
            Path of the `.trn` transcript to encode.
        output_dir (`str`):
            Directory the codes and manifests go to.
        shards (`list`, *optional*):
            Shards to process, all of them when omitted.
        max_hours (`float`, *optional*):
            Audio kept per shard, all of it when omitted.
        num_codebooks (`int`, *optional*, defaults to 8):
            Number of codebooks kept per frame.
        max_batch_seconds (`float`, *optional*, defaults to 48.0):
            Padded audio per batch, which bounds codec memory.
        sort_window (`int`, *optional*, defaults to 256):
            Utterances buffered and sorted by length before batching.
        min_duration (`float`, *optional*, defaults to 0.5):
            Shortest utterance kept, in seconds.
        max_duration (`float`, *optional*, defaults to 30.0):
            Longest utterance kept, in seconds.
        transcription_form (`str`, *optional*, defaults to `"phonetic"`):
            Which spelling of the dual transcription is kept, see [`normalize_transcription`].
        device (`str`, *optional*, defaults to `"cuda"`):
            Device the codec runs on.
        force (`bool`, *optional*, defaults to `False`):
            Re-encode shards that are already cached.
    """
    import zipfile

    import numpy as np
    import torchaudio
    from transformers import MimiModel

    os.makedirs(output_dir, exist_ok=True)
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()
    resampler = torchaudio.transforms.Resample(SOURCE_RATE, mimi.config.sampling_rate)
    budget = int(max_batch_seconds * mimi.config.sampling_rate)
    transcript = KsponSpeechArchive.read_transcript(transcript_path)

    with zipfile.ZipFile(archive_path) as archive:
        for shard in shards or sorted(transcript):
            codes_path = os.path.join(output_dir, f"{shard}.npy")
            manifest_path = os.path.join(output_dir, f"{shard}.jsonl")
            if shard not in transcript or (os.path.exists(codes_path) and os.path.exists(manifest_path) and not force):
                continue
            chunks, manifest, offset = [], [], 0

            def drain(window):
                nonlocal offset
                # Sorting by length keeps the padded batch close to the real audio the budget is sized against.
                window.sort(key=lambda item: item[0].shape[-1])
                batches, current = [], []
                for waveform, meta in window:
                    longest = max([waveform.shape[-1]] + [w.shape[-1] for w, _ in current])
                    if current and (len(current) + 1) * longest > budget:
                        batches.append(current)
                        current = []
                    current.append((waveform, meta))
                if current:
                    batches.append(current)
                for batch in batches:
                    encoded = encode_mimi_batch(mimi, [w for w, _ in batch], num_codebooks, device)
                    for codes, (_, meta) in zip(encoded, batch):
                        if codes.shape[-1] == 0:
                            continue
                        chunks.append(codes)
                        manifest.append({**meta, "offset": offset, "frames": int(codes.shape[-1])})
                        offset += int(codes.shape[-1])

            with archive.open(f"{shard}.zip") as stream, zipfile.ZipFile(stream) as shard_archive:
                window = []
                for member, raw_text in subsample(transcript[shard], max_hours):
                    try:
                        waveform = KsponSpeechArchive.decode_pcm(shard_archive.read(member))
                    except KeyError:
                        continue
                    duration = waveform.shape[-1] / SOURCE_RATE
                    text = normalize_transcription(raw_text, transcription_form)
                    if not text or not min_duration <= duration <= max_duration:
                        continue
                    meta = {"id": os.path.splitext(os.path.basename(member))[0], "text": text, "duration": round(duration, 3)}
                    window.append((resampler(waveform), meta))
                    if len(window) >= sort_window:
                        drain(window)
                        window = []
                if window:
                    drain(window)

            if chunks:
                np.save(codes_path, np.concatenate(chunks, axis=1))
                with open(manifest_path, "w", encoding="utf-8") as sink:
                    for row in manifest:
                        sink.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    r"""
    Command line: `mimi` writes the per shard Mimi code cache, `duplex` writes a duplex cache from the
    aligned release, and `import` builds the Phase 2 duplex caches from an existing per utterance cache.
    """
    import argparse

    parser = argparse.ArgumentParser(description="KsponSpeech caches.")
    commands = parser.add_subparsers(dest="command", required=True)

    mimi = commands.add_parser("mimi", help="Per shard Mimi code cache, for the rotation comparison arm.")
    mimi.add_argument("--archive", required=True)
    mimi.add_argument("--transcript", required=True)
    mimi.add_argument("--output", required=True)
    mimi.add_argument("--shards", nargs="*", default=None)
    mimi.add_argument("--max-hours", type=float, default=None)
    mimi.add_argument("--num-codebooks", type=int, default=8)
    mimi.add_argument("--max-batch-seconds", type=float, default=48.0)
    mimi.add_argument("--sort-window", type=int, default=256)
    mimi.add_argument("--min-duration", type=float, default=0.5)
    mimi.add_argument("--max-duration", type=float, default=30.0)
    mimi.add_argument("--transcription-form", default="phonetic", choices=("phonetic", "orthographic"))
    mimi.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    mimi.add_argument("--force", action="store_true")

    duplex = commands.add_parser("duplex", help="Duplex cache from the aligned release.")
    duplex.add_argument("--alignment", required=True)
    duplex.add_argument("--archives", required=True)
    duplex.add_argument("--model", required=True)
    duplex.add_argument("--output", required=True)
    duplex.add_argument("--subsets", nargs="+", default=["train"])
    duplex.add_argument("--limit", type=int, default=0)
    duplex.add_argument("--min-duration", type=float, default=2.0)
    duplex.add_argument("--max-duration", type=float, default=12.0)
    duplex.add_argument("--device", default="cuda")

    imported = commands.add_parser("import", help="Phase 2 duplex caches from a per utterance code cache.")
    imported.add_argument("--source", required=True)
    imported.add_argument("--output", required=True)
    imported.add_argument("--min-duration", type=float, default=2.0)
    imported.add_argument("--max-duration", type=float, default=12.0)

    arguments = parser.parse_args()
    if arguments.command == "mimi":
        write_mimi_cache(
            arguments.archive, arguments.transcript, arguments.output, arguments.shards, arguments.max_hours,
            arguments.num_codebooks, arguments.max_batch_seconds, arguments.sort_window, arguments.min_duration,
            arguments.max_duration, arguments.transcription_form, arguments.device, arguments.force,
        )
    elif arguments.command == "duplex":
        count = write_duplex_cache(
            arguments.alignment, arguments.archives, arguments.model, arguments.output, tuple(arguments.subsets),
            arguments.limit, arguments.min_duration, arguments.max_duration, arguments.device,
        )
        print(f"[kspon] {count} utterances -> {arguments.output}", flush=True)
    else:
        counts = import_code_cache(arguments.source, arguments.output, arguments.min_duration, arguments.max_duration)
        print(f"[kspon] {counts}", flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "KSPONSPEECH_URL",
    "KsponSpeechArchive",
    "KsponSpeechDataset",
    "encode_mimi_batch",
    "import_code_cache",
    "normalize_transcription",
    "read_alignment_index",
    "subsample",
    "write_duplex_cache",
    "write_mimi_cache",
]
