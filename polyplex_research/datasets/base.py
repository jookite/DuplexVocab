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
"""Base classes and mixins for single-turn TTS datasets."""

import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


CHUNK_SIZE = 1 << 22

EXPORT_CONFIG_NAME = "dataset_config.json"
EXPORT_DATA_NAME = "data.jsonl"
EXPORT_ARRAY_NAME = "codes.npy"


@dataclass
class SingleTurnTTSExample:
    r"""
    Container for one single-turn utterance.

    Args:
        text (`str`):
            Normalized transcription of the utterance.
        audio (`torch.FloatTensor` of shape `(1, num_samples)`, *optional*):
            Mono waveform resampled to the audio encoder sampling rate. Unset when the utterance was
            read from a code cache, in which case `codes` carries the audio instead.
        sampling_rate (`int`, *optional*, defaults to 24000):
            Sampling rate of `audio`.
        audio_path (`str`, *optional*):
            Path the waveform was decoded from.
        codes (`torch.LongTensor` of shape `(num_codebooks, num_frames)`, *optional*):
            Pre-encoded acoustic codes, which let a batch skip the codec entirely.
    """

    text: str
    audio: torch.FloatTensor | None = None
    sampling_rate: int = 24000
    audio_path: str = ""
    codes: torch.LongTensor | None = None


@dataclass
class AudioFileFormat:
    r"""
    Decoding hints for corpora that ship headerless audio.

    Args:
        headerless_extensions (`tuple`, *optional*, defaults to `(".pcm", ".raw")`):
            File suffixes that carry no container header and must be read as raw samples.
        sampling_rate (`int`, *optional*, defaults to 16000):
            Sampling rate of the headerless samples.
        dtype (`str`, *optional*, defaults to `"<i2"`):
            Numpy style dtype string describing one raw sample.
    """

    headerless_extensions: tuple = (".pcm", ".raw")
    sampling_rate: int = 16000
    dtype: str = "<i2"


class ArchiveDownloadMixin:
    r"""
    Fetches and unpacks the archive a corpus is published as.

    Every method is a `staticmethod` or a `classmethod` because none of them depend on dataset state,
    which lets a module that is not a dataset call them off the class directly. Mixing the class into a
    dataset is what gives a subclass [`resolve_corpus_root`], the one step a corpus actually varies in.
    """

    @staticmethod
    def get_cache_dir() -> str:
        r"""
        Resolves the directory that holds downloaded corpora.

        Returns:
            `str`: `$HF_HOME/datasets` when set, otherwise `.cache/datasets` of the repository.
        """
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            return os.path.join(hf_home, "datasets")
        repository_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(repository_root, ".cache", "datasets")

    @staticmethod
    def download_file(url: str, destination: str, force: bool = False) -> str:
        r"""
        Downloads a URL to disk, resuming a partial transfer when the server supports it.

        Args:
            url (`str`):
                Address of the archive.
            destination (`str`):
                Path the archive is written to.
            force (`bool`, *optional*, defaults to `False`):
                Whether to discard an existing file and download again.

        Returns:
            `str`: Path of the downloaded archive.

        Raises:
            RuntimeError: If the transfer ends before the announced content length.
        """
        if os.path.exists(destination) and not force:
            return destination

        os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
        partial_path = destination + ".part"
        downloaded = os.path.getsize(partial_path) if os.path.exists(partial_path) else 0

        request = urllib.request.Request(url)
        if downloaded:
            request.add_header("Range", f"bytes={downloaded}-")

        with urllib.request.urlopen(request) as response:
            resumed = response.status == 206
            if not resumed:
                downloaded = 0
            expected = response.headers.get("Content-Length")
            expected = downloaded + int(expected) if expected is not None else None

            with open(partial_path, "ab" if resumed else "wb") as archive:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    archive.write(chunk)
                    downloaded += len(chunk)

        if expected is not None and downloaded != expected:
            raise RuntimeError(
                f"Incomplete download of {url}, got {downloaded} bytes out of {expected}."
            )

        os.replace(partial_path, destination)
        return destination

    @staticmethod
    def decode_member_name(info: zipfile.ZipInfo) -> str:
        r"""
        Decodes a zip member name that was stored without the unicode flag.

        Args:
            info (`zipfile.ZipInfo`):
                Member metadata of the archive.

        Returns:
            `str`: The member name, decoded as CP949 when it is not already unicode.
        """
        if info.flag_bits & 0x800:
            return info.filename
        try:
            return info.filename.encode("cp437").decode("cp949")
        except (UnicodeDecodeError, UnicodeEncodeError):
            return info.filename

    @classmethod
    def extract_archive(cls, archive_path: str, destination: str, force: bool = False) -> str:
        r"""
        Extracts a zip or tar archive.

        Args:
            archive_path (`str`):
                Path of the archive.
            destination (`str`):
                Directory the archive is extracted into.
            force (`bool`, *optional*, defaults to `False`):
                Whether to extract again over an already populated directory.

        Returns:
            `str`: Path of the directory holding the extracted files.

        Raises:
            ValueError: If the archive format is neither zip nor tar.
        """
        marker = os.path.join(destination, ".extracted")
        if os.path.exists(marker) and not force:
            return destination

        staging = destination + ".staging"
        shutil.rmtree(staging, ignore_errors=True)
        os.makedirs(staging, exist_ok=True)

        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as archive:
                for info in archive.infolist():
                    member_path = os.path.join(staging, cls.decode_member_name(info))
                    if info.is_dir():
                        os.makedirs(member_path, exist_ok=True)
                        continue
                    os.makedirs(os.path.dirname(member_path), exist_ok=True)
                    with archive.open(info) as source, open(member_path, "wb") as sink:
                        shutil.copyfileobj(source, sink, CHUNK_SIZE)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path) as archive:
                archive.extractall(staging, filter="data")
        else:
            shutil.rmtree(staging, ignore_errors=True)
            raise ValueError(f"Unsupported archive format for {archive_path}.")

        os.makedirs(destination, exist_ok=True)
        for entry in os.listdir(staging):
            target = os.path.join(destination, entry)
            shutil.rmtree(target, ignore_errors=True)
            if os.path.isfile(target):
                os.remove(target)
            shutil.move(os.path.join(staging, entry), target)
        shutil.rmtree(staging, ignore_errors=True)

        open(marker, "w").close()
        return destination

    @classmethod
    def extract_nested_zips(cls, directory: str, remove_archives: bool = False) -> str:
        r"""
        Recursively extracts zip archives that are nested inside an extracted directory.

        Args:
            directory (`str`):
                Directory that may contain further zip archives.
            remove_archives (`bool`, *optional*, defaults to `False`):
                Whether to delete a nested archive once it has been extracted.

        Returns:
            `str`: The directory that was processed.
        """
        while True:
            pending = [
                os.path.join(parent, file_name)
                for parent, _, file_names in os.walk(directory)
                for file_name in file_names
                if file_name.endswith(".zip")
            ]
            if not pending:
                return directory

            for archive_path in pending:
                target = os.path.splitext(archive_path)[0]
                with zipfile.ZipFile(archive_path) as archive:
                    for info in archive.infolist():
                        if info.is_dir():
                            continue
                        member_path = os.path.join(target, cls.decode_member_name(info))
                        os.makedirs(os.path.dirname(member_path), exist_ok=True)
                        with archive.open(info) as source, open(member_path, "wb") as sink:
                            shutil.copyfileobj(source, sink, CHUNK_SIZE)
                if remove_archives:
                    os.remove(archive_path)
                else:
                    os.rename(archive_path, archive_path + ".done")

    @staticmethod
    def find_local_archive(cache_dir: str, name: str) -> str | None:
        r"""
        Looks for an archive that was placed in the cache by hand.

        A corpus too large or too slow to fetch over the network is often copied in manually, under
        whatever name it was published with. Any file in the archives directory whose stem matches the
        corpus counts, ignoring case, so `KSponSpeech.zip` is picked up as readily as the name
        [`download_and_extract`] would have written itself.

        Args:
            cache_dir (`str`):
                Directory holding downloaded corpora.
            name (`str`):
                Corpus name.

        Returns:
            `str` or `None`: Path of the archive, when one is present.
        """
        archives_dir = os.path.join(cache_dir, "archives")
        if not os.path.isdir(archives_dir):
            return None
        wanted = name.lower()
        for file_name in sorted(os.listdir(archives_dir)):
            stem = os.path.splitext(file_name)[0].lower()
            if stem == wanted or file_name.lower() == f"{wanted}.archive":
                candidate = os.path.join(archives_dir, file_name)
                if os.path.isfile(candidate):
                    return candidate
        return None

    @classmethod
    def download_and_extract(
        cls,
        url: str,
        name: str,
        cache_dir: str | None = None,
        force: bool = False,
        nested: bool = False,
    ) -> str:
        r"""
        Downloads an archive and extracts it below the corpus cache directory.

        Args:
            url (`str`):
                Address of the archive.
            name (`str`):
                Corpus name used as the cache subdirectory.
            cache_dir (`str`, *optional*):
                Directory holding downloaded corpora, defaults to [`get_cache_dir`].
            force (`bool`, *optional*, defaults to `False`):
                Whether to download and extract again.
            nested (`bool`, *optional*, defaults to `False`):
                Whether to also extract zip archives found inside the extracted directory.

        Returns:
            `str`: Path of the extracted corpus directory.
        """
        cache_dir = cache_dir or cls.get_cache_dir()
        archive_path = cls.find_local_archive(cache_dir, name) or os.path.join(
            cache_dir, "archives", f"{name}.archive"
        )
        extraction_dir = os.path.join(cache_dir, name)

        cls.download_file(url, archive_path, force=force)
        cls.extract_archive(archive_path, extraction_dir, force=force)
        if nested:
            cls.extract_nested_zips(extraction_dir)
        return extraction_dir

    @classmethod
    def resolve_corpus_root(
        cls,
        url: str,
        name: str,
        root: str | None = None,
        download: bool = False,
        nested: bool = False,
    ) -> str:
        r"""
        Settles on the directory a corpus is read from, fetching it first when asked to.

        An explicit `root` wins, an absent one falls back to the corpus cache, and the archive is only
        pulled when the directory is still missing. This is the whole of what the corpus subclasses used
        to repeat around their own constants.

        Args:
            url (`str`):
                Address of the corpus archive.
            name (`str`):
                Corpus name used as the cache subdirectory.
            root (`str`, *optional*):
                Directory of an already extracted corpus.
            download (`bool`, *optional*, defaults to `False`):
                Whether a missing corpus may be fetched.
            nested (`bool`, *optional*, defaults to `False`):
                Whether the archive holds further zip archives.

        Returns:
            `str`: Directory holding the extracted corpus.

        Raises:
            FileNotFoundError: If the corpus is missing and `download` is `False`.
        """
        if root is None:
            root = os.path.join(cls.get_cache_dir(), name)
        if download and not os.path.isdir(root):
            root = cls.download_and_extract(
                url, os.path.basename(root), cache_dir=os.path.dirname(root), nested=nested
            )
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"Corpus root {root} does not exist, pass `download=True` to fetch it."
            )
        return root


class ReplicableMixin:
    r"""
    Turns a dataset that was built from raw corpora into one that can be rebuilt from a folder.

    A phase dataset is expensive and awkward to construct: it crawls archives that have to be fetched or
    copied in by hand, mixes several corpora under a seed, and on the audio side depends on a codec pass.
    None of that should have to run on the training machine, and none of it reproduces byte for byte
    across machines. This mixin separates the two, so a dataset is built once and replicated everywhere
    else:

    1. `__init__` builds the dataset from the raw corpora, on whichever machine holds them.
    2. [`save_pretrained`] writes what was built to a folder: the records as JSON lines, the acoustic
       codes as one array, and the settings needed to reconstruct the instance.
    3. [`push_to_hub`] uploads that folder as a Hub dataset repository.
    4. [`from_pretrained`] rebuilds the dataset from either, touching no corpus and no codec.

    A subclass implements the export surface, [`export_config`], [`export_records`], [`export_arrays`]
    and [`from_export`], and inherits the three replication methods.
    """

    export_config_name = EXPORT_CONFIG_NAME
    export_data_name = EXPORT_DATA_NAME
    export_array_name = EXPORT_ARRAY_NAME

    # -----------------------------------------------------------------
    # Subclass surface
    # -----------------------------------------------------------------

    def export_config(self) -> dict:
        r"""
        Returns:
            `dict`: The JSON serializable settings [`from_export`] needs to rebuild the instance.

        Raises:
            NotImplementedError: Always, subclasses describe their own settings.
        """
        raise NotImplementedError

    def export_records(self):
        r"""
        Yields:
            `dict`: One JSON serializable record per example, in dataset order.

        Raises:
            NotImplementedError: Always, subclasses describe their own records.
        """
        raise NotImplementedError

    def export_arrays(self):
        r"""
        Returns:
            `np.ndarray` or `None`: Acoustic codes the records index into, `None` for a text dataset.
        """
        return None

    @classmethod
    def from_export(cls, config: dict, records: list, arrays):
        r"""
        Args:
            config (`dict`):
                Settings written by [`export_config`].
            records (`list` of `dict`):
                Records written by [`export_records`], in dataset order.
            arrays (`np.ndarray` or `None`):
                Codes written by [`export_arrays`], memory mapped.

        Returns:
            `ReplicableMixin`: The rebuilt dataset.

        Raises:
            NotImplementedError: Always, subclasses rebuild themselves.
        """
        raise NotImplementedError

    # -----------------------------------------------------------------
    # Replication
    # -----------------------------------------------------------------

    def save_pretrained(self, save_directory: str) -> str:
        r"""
        Writes the built dataset to a folder.

        Args:
            save_directory (`str`):
                Directory the files are written to, created when missing.

        Returns:
            `str`: The directory that was written.
        """
        save_directory = os.path.expanduser(save_directory)
        os.makedirs(save_directory, exist_ok=True)

        count = 0
        with open(os.path.join(save_directory, self.export_data_name), "w", encoding="utf-8") as sink:
            for record in self.export_records():
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        arrays = self.export_arrays()
        if arrays is not None:
            np.save(os.path.join(save_directory, self.export_array_name), arrays)

        config = dict(self.export_config())
        config["dataset_class"] = type(self).__name__
        config["num_records"] = count
        config["has_arrays"] = arrays is not None
        with open(os.path.join(save_directory, self.export_config_name), "w", encoding="utf-8") as sink:
            json.dump(config, sink, ensure_ascii=False, indent=2)

        return save_directory

    def push_to_hub(
        self,
        repo_id: str,
        save_directory: str | None = None,
        private: bool = True,
        token: str | None = None,
        commit_message: str | None = None,
    ) -> str:
        r"""
        Uploads the built dataset as a Hub dataset repository.

        Args:
            repo_id (`str`):
                Target repository, as `owner/name`.
            save_directory (`str`, *optional*):
                Folder to write before uploading. A temporary one is used and removed when omitted.
            private (`bool`, *optional*, defaults to `True`):
                Visibility of a repository that has to be created. The corpora behind these datasets carry
                their own licences, so a new repository is closed by default.
            token (`str`, *optional*):
                Hub token, defaults to the locally stored login.
            commit_message (`str`, *optional*):
                Message of the upload commit.

        Returns:
            `str`: URL of the commit.
        """
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

        temporary = save_directory is None
        save_directory = save_directory or tempfile.mkdtemp(prefix="polyplex-dataset-")
        try:
            self.save_pretrained(save_directory)
            return str(
                api.upload_folder(
                    repo_id=repo_id,
                    folder_path=save_directory,
                    repo_type="dataset",
                    commit_message=commit_message or f"Upload {type(self).__name__}",
                )
            )
        finally:
            if temporary:
                shutil.rmtree(save_directory, ignore_errors=True)

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str,
        token: str | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
    ):
        r"""
        Rebuilds a dataset from a folder written by [`save_pretrained`] or from a Hub repository.

        Args:
            path_or_repo_id (`str`):
                Local directory, or the `owner/name` of a Hub dataset repository.
            token (`str`, *optional*):
                Hub token, defaults to the locally stored login.
            revision (`str`, *optional*):
                Branch, tag or commit to download.
            cache_dir (`str`, *optional*):
                Directory the download is cached in.

        Returns:
            `ReplicableMixin`: The rebuilt dataset.

        Raises:
            ValueError: If the export was written by a different dataset class.
        """
        directory = os.path.expanduser(path_or_repo_id)
        if not os.path.isdir(directory):
            from huggingface_hub import snapshot_download

            directory = snapshot_download(
                path_or_repo_id,
                repo_type="dataset",
                revision=revision,
                cache_dir=cache_dir,
                token=token,
            )

        with open(os.path.join(directory, cls.export_config_name), encoding="utf-8") as handle:
            config = json.load(handle)

        stored_class = config.pop("dataset_class", None)
        if stored_class is not None and stored_class != cls.__name__:
            raise ValueError(
                f"{directory} was written by {stored_class}, load it with that class rather than "
                f"{cls.__name__}."
            )

        records = []
        with open(os.path.join(directory, cls.export_data_name), encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        arrays = None
        array_path = os.path.join(directory, cls.export_array_name)
        if config.get("has_arrays") and os.path.isfile(array_path):
            arrays = np.load(array_path, mmap_mode="r")

        return cls.from_export(config, records, arrays)


class MimiCacheMixin:
    r"""
    Reads utterances out of a directory of cached Mimi codes.

    Each shard of the cache is one `.npy` holding every utterance's codes end to end and one `.jsonl`
    naming the slice of that array each utterance occupies. The arrays are memory mapped, so a shard
    costs a file handle rather than its size, and a batch reads only the frames it needs.

    A dataset mixes this in to gain a second way of being indexed, next to walking audio files: call
    [`index_code_cache`] from the constructor and serve `self.entries` through [`code_example`].
    """

    def index_code_cache(
        self,
        cache_dir: str,
        shards: list | None = None,
        min_duration: float = 0.5,
        max_duration: float = 30.0,
        num_codebooks: int = 8,
    ) -> list:
        r"""
        Reads the shard manifests and builds the utterance index.

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

        Returns:
            `list`: Entries of `(shard, manifest_row)`.

        Raises:
            FileNotFoundError: If the cache holds no shard, or a named shard is incomplete.
        """
        self.cache_dir = os.path.expanduser(cache_dir)
        self.num_codebooks = num_codebooks
        self._arrays: dict = {}

        if shards is None:
            shards = sorted(
                os.path.splitext(name)[0]
                for name in os.listdir(self.cache_dir)
                if name.endswith(".jsonl")
            )
        if not shards:
            raise FileNotFoundError(f"No cached shards below {self.cache_dir}.")

        entries = []
        for shard in shards:
            manifest_path = os.path.join(self.cache_dir, f"{shard}.jsonl")
            codes_path = os.path.join(self.cache_dir, f"{shard}.npy")
            if not (os.path.isfile(manifest_path) and os.path.isfile(codes_path)):
                raise FileNotFoundError(f"Shard {shard} is missing its codes or its manifest.")
            with open(manifest_path, encoding="utf-8") as manifest:
                for line in manifest:
                    row = json.loads(line)
                    if not min_duration <= row["duration"] <= max_duration:
                        continue
                    if not row["text"]:
                        continue
                    entries.append((shard, row))
        return entries

    def shard_codes(self, shard: str) -> np.ndarray:
        r"""
        Args:
            shard (`str`):
                Shard name.

        Returns:
            `np.ndarray` of shape `(num_codebooks, total_frames)`: The memory mapped codes of the shard.
        """
        # Opened lazily and per worker: a memory map cannot be shared across a fork boundary safely,
        # and building it in __init__ would map every shard even for a run that touches one.
        if shard not in self._arrays:
            self._arrays[shard] = np.load(
                os.path.join(self.cache_dir, f"{shard}.npy"), mmap_mode="r"
            )
        return self._arrays[shard]

    def code_example(self, shard: str, row: dict) -> SingleTurnTTSExample:
        r"""
        Args:
            shard (`str`):
                Shard the utterance lives in.
            row (`dict`):
                Manifest row naming its slice of the shard.

        Returns:
            `SingleTurnTTSExample`: The utterance, carrying codes instead of a waveform.
        """
        codes = self.shard_codes(shard)
        window = codes[: self.num_codebooks, row["offset"] : row["offset"] + row["frames"]]
        return SingleTurnTTSExample(
            text=row["text"],
            codes=torch.from_numpy(np.asarray(window, dtype=np.int64)),
            audio_path=row["id"],
        )


class SingleTurnTTSDataset(ArchiveDownloadMixin, Dataset):
    r"""
    Base dataset that yields text and target speech pairs for text to audio token training.

    Subclasses only implement corpus specific indexing and transcription cleanup. The
    sequence layout for duplex spoken language models is built by
    [`SingleTurnTTSCollator`], so this class never emits ASR style targets.

    Args:
        root (`str`):
            Root directory of the extracted corpus.
        sampling_rate (`int`, *optional*, defaults to 24000):
            Sampling rate expected by the audio encoder, waveforms are resampled to it.
        min_duration (`float`, *optional*, defaults to 0.5):
            Utterances shorter than this many seconds are skipped.
        max_duration (`float`, *optional*, defaults to 30.0):
            Utterances longer than this many seconds are skipped.
        audio_file_format (`AudioFileFormat`, *optional*):
            Decoding hints used for headerless audio files.
    """

    def __init__(
        self,
        root: str,
        sampling_rate: int = 24000,
        min_duration: float = 0.5,
        max_duration: float = 30.0,
        audio_file_format: AudioFileFormat | None = None,
    ):
        self.root = root
        self.sampling_rate = sampling_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.audio_file_format = audio_file_format or AudioFileFormat()
        self._resamplers: dict = {}
        self.entries = self._build_index()

    def _build_index(self) -> list:
        r"""
        Collects the utterance index of the corpus.

        Returns:
            `list`: Entries of `(audio_path, raw_transcription)`.

        Raises:
            NotImplementedError: Always, subclasses must provide the corpus layout.
        """
        raise NotImplementedError

    def normalize_text(self, text: str) -> str:
        r"""
        Cleans a raw transcription.

        Args:
            text (`str`):
                Transcription as stored in the corpus.

        Returns:
            `str`: Transcription usable as text conditioning.
        """
        return " ".join(text.split())

    def load_audio(self, audio_path: str) -> torch.FloatTensor:
        r"""
        Decodes one audio file into a mono waveform at `self.sampling_rate`.

        Args:
            audio_path (`str`):
                Path of the audio file.

        Returns:
            `torch.FloatTensor` of shape `(1, num_samples)`: The decoded waveform.
        """
        extension = os.path.splitext(audio_path)[-1].lower()
        if extension in self.audio_file_format.headerless_extensions:
            with open(audio_path, "rb") as raw_file:
                payload = raw_file.read()
            # KsponSpeech ships a few utterances whose byte count is not a whole number
            # of 16 bit samples, the trailing byte belongs to no sample.
            usable = len(payload) - (len(payload) % 2)
            raw = torch.frombuffer(bytearray(payload[:usable]), dtype=torch.int16).clone()
            waveform = raw.to(torch.float32).unsqueeze(0) / 32768.0
            source_rate = self.audio_file_format.sampling_rate
        else:
            waveform, source_rate = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if source_rate != self.sampling_rate:
            if source_rate not in self._resamplers:
                self._resamplers[source_rate] = torchaudio.transforms.Resample(
                    source_rate, self.sampling_rate
                )
            waveform = self._resamplers[source_rate](waveform)

        return waveform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> SingleTurnTTSExample:
        audio_path, raw_text = self.entries[index]
        waveform = self.load_audio(audio_path)
        duration = waveform.shape[-1] / self.sampling_rate
        if not self.min_duration <= duration <= self.max_duration:
            return self[(index + 1) % len(self)]

        return SingleTurnTTSExample(
            text=self.normalize_text(raw_text),
            audio=waveform,
            sampling_rate=self.sampling_rate,
            audio_path=audio_path,
        )


class MimiCodeDataset(MimiCacheMixin, Dataset):
    r"""
    Constructs a single-turn TTS dataset from cached acoustic codes.

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
        self.entries = self.index_code_cache(
            cache_dir,
            shards=shards,
            min_duration=min_duration,
            max_duration=max_duration,
            num_codebooks=num_codebooks,
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> SingleTurnTTSExample:
        shard, row = self.entries[index]
        return self.code_example(shard, row)
