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
"""Korean-English translation dataset for text injection."""

import json
import os
from dataclasses import dataclass

from torch.utils.data import Dataset

from .base import ArchiveDownloadMixin


COLLOQUIAL_KOEN_URL = "https://dolab-data.duckdns.org/api/public/dl/wxvXxLQa?inline=true"


@dataclass
class TranslationExample:
    r"""
    Container for one translation pair.

    Args:
        source (`str`):
            Sentence in the source language.
        target (`str`):
            Sentence in the target language.
        source_language (`str`):
            Language code of `source`.
        target_language (`str`):
            Language code of `target`.
        domain (`str`, *optional*, defaults to `""`):
            Domain label attached to the pair by the corpus.
        subdomain (`str`, *optional*, defaults to `""`):
            Finer grained label attached to the pair by the corpus.
    """

    source: str
    target: str
    source_language: str
    target_language: str
    domain: str = ""
    subdomain: str = ""


class ColloquialKoEnDataset(ArchiveDownloadMixin, Dataset):
    r"""
    Constructs the AI Hub colloquial Korean-English corpus for Phase 1 text injection.

    Each split ships JSON files holding a `data` array whose records carry the human
    written `ko` and `en` sides next to a machine translated `mt` field. Only the human
    sides are used, so the decoding target is never machine translation output.

    Args:
        root (`str`, *optional*):
            Directory holding the extracted corpus, defaults to the corpus cache.
        split (`str`, *optional*, defaults to `"train"`):
            Split to load, either `"train"` or `"valid"`.
        direction (`str`, *optional*, defaults to `"en2ko"`):
            Translation direction, `"en2ko"` teaches decoding into Korean and `"ko2en"`
            teaches decoding into English.
        download (`bool`, *optional*, defaults to `False`):
            Whether to download and extract the corpus when `root` is missing.
        max_examples (`int`, *optional*):
            Upper bound on the number of loaded pairs, the corpus is meant to be used in
            small verified subsets.
    """

    def __init__(
        self,
        root: str | None = None,
        split: str = "train",
        direction: str = "en2ko",
        download: bool = False,
        max_examples: int | None = None,
    ):
        if split not in ("train", "valid"):
            raise ValueError(f"`split` must be 'train' or 'valid', got {split}.")
        if direction not in ("en2ko", "ko2en"):
            raise ValueError(f"`direction` must be 'en2ko' or 'ko2en', got {direction}.")

        self.root = self.resolve_corpus_root(
            COLLOQUIAL_KOEN_URL, "colloquial_koen", root=root, download=download, nested=True
        )
        self.split = split
        self.direction = direction
        self.max_examples = max_examples
        self.examples = self._load()

    def _find_json_files(self) -> list:
        r"""
        Collects the label JSON files of the requested split.

        Returns:
            `list`: Paths of the JSON files, sorted for a reproducible order.

        Raises:
            FileNotFoundError: If the split has no JSON file below `self.root`.
        """
        matches = []
        for parent, _, file_names in os.walk(self.root):
            for file_name in file_names:
                if file_name.endswith(".json") and self.split in file_name:
                    matches.append(os.path.join(parent, file_name))
        if not matches:
            raise FileNotFoundError(f"No '{self.split}' JSON file below {self.root}.")
        return sorted(matches)

    def _load(self) -> list:
        r"""
        Parses the split into deduplicated translation pairs.

        Returns:
            `list` of `TranslationExample`: The loaded pairs.
        """
        source_key, target_key = ("en", "ko") if self.direction == "en2ko" else ("ko", "en")
        examples = []
        seen = set()

        for json_path in self._find_json_files():
            with open(json_path, encoding="utf-8-sig") as handle:
                records = json.load(handle)
            records = records.get("data", records) if isinstance(records, dict) else records

            # The corpus files are ordered by domain, so the capped run is drawn at a fixed stride
            # across the whole file rather than from its front, which is one domain.
            if self.max_examples is not None and len(records) > self.max_examples:
                stride = len(records) / self.max_examples
                records = [records[int(index * stride)] for index in range(self.max_examples)]

            for record in records:
                source = (record.get(source_key) or "").strip()
                target = (record.get(target_key) or "").strip()
                if not source or not target:
                    continue
                key = (source, target)
                if key in seen:
                    continue
                seen.add(key)
                examples.append(
                    TranslationExample(
                        source=source,
                        target=target,
                        source_language=source_key,
                        target_language=target_key,
                        domain=record.get("domain", "") or "",
                        subdomain=record.get("subdomain", "") or "",
                    )
                )
                if self.max_examples is not None and len(examples) >= self.max_examples:
                    return examples

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TranslationExample:
        return self.examples[index]
