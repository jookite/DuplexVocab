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
"""Korean instruction corpora used as standard register text."""

import json
import os
import re

from .base import ArchiveDownloadMixin


KOREAN_INSTRUCTION_URLS = {
    "koalpaca": "https://huggingface.co/datasets/heegyu/open-korean-instructions/resolve/main/koalpaca.json",
    "korquad_chat": "https://huggingface.co/datasets/heegyu/open-korean-instructions/resolve/main/korquad-chat.json",
}

SPEAKER_TAG = re.compile(r"<(?:usr|bot|sys)>")
HANGUL = re.compile(r"[가-힣]")


def iter_instruction_texts(
    subsets: tuple = ("koalpaca",),
    cache_dir: str | None = None,
    download: bool = False,
    max_lines: int | None = None,
):
    r"""
    Yields plain sentences from Korean instruction corpora.

    The corpora store one JSON record per line whose `text` field interleaves speaker
    tags. The tags are stripped so that only the standard register wording reaches the
    vocabulary trainer, and a line carrying no Hangul is skipped.

    Args:
        subsets (`tuple`, *optional*, defaults to `("koalpaca",)`):
            Keys of [`KOREAN_INSTRUCTION_URLS`] to read.
        cache_dir (`str`, *optional*):
            Directory holding the downloaded files, defaults to the corpus cache.
        download (`bool`, *optional*, defaults to `False`):
            Whether to fetch a subset that is not cached yet.
        max_lines (`int`, *optional*):
            Upper bound on the number of yielded sentences.

    Yields:
        `str`: One sentence of the instruction corpora.

    Raises:
        KeyError: If a requested subset has no known URL.
        FileNotFoundError: If a subset is missing and `download` is `False`.
    """
    cache_dir = cache_dir or os.path.join(ArchiveDownloadMixin.get_cache_dir(), "korean_instructions")
    produced = 0

    for subset in subsets:
        if subset not in KOREAN_INSTRUCTION_URLS:
            raise KeyError(f"Unknown instruction subset '{subset}'.")

        path = os.path.join(cache_dir, f"{subset}.jsonl")
        if not os.path.exists(path):
            if not download:
                raise FileNotFoundError(
                    f"Instruction subset '{subset}' is missing at {path}, pass `download=True`."
                )
            ArchiveDownloadMixin.download_file(KOREAN_INSTRUCTION_URLS[subset], path)

        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    text = json.loads(line).get("text", "")
                except json.JSONDecodeError:
                    continue

                for sentence in SPEAKER_TAG.sub("\n", text).splitlines():
                    sentence = " ".join(sentence.split())
                    if not HANGUL.search(sentence):
                        continue
                    yield sentence
                    produced += 1
                    if max_lines is not None and produced >= max_lines:
                        return
