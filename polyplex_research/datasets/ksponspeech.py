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


if __name__ == "__main__":
    main()


__all__ = [
    "KSPONSPEECH_URL",
    "KsponSpeechArchive",
    "KsponSpeechDataset",
    "normalize_transcription",
]
