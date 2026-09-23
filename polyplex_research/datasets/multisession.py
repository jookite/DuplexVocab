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
"""AI Hub multi-session daily dialogue, read as assistant side turns."""

import glob
import json
import os
import re
import zlib
from dataclasses import dataclass

from .base import ArchiveDownloadMixin


MULTISESSION_DIRECTORY = "multisession_daily"

MULTISESSION_SPLITS = {"train": "Training", "valid": "Validation", "test": "Validation"}

HELD_OUT_BUCKETS = 2

UNREADABLE_TEXT = re.compile(r"[\u3131-\u318e]|[\U0001F300-\U0001FAFF]|~~")


@dataclass
class MultisessionTurn:
    r"""
    One turn of the corpus together with what a leak free split needs.

    Args:
        text (`str`):
            The utterance as the corpus stores it.
        conversation (`str`):
            Identifier shared by every turn of the same multi session dialogue, which is the unit a
            split has to respect.
        topic (`str`):
            Coarse topic label the corpus carries, used to report how narrow a share is.
    """

    text: str
    conversation: str
    topic: str = ""


@dataclass
class DialogueTurn:
    r"""
    One turn of a dialogue read from one speaker's side.

    Args:
        text (`str`):
            The utterance, with whitespace collapsed.
        own (`bool`):
            Whether the followed speaker says it. The other side's turns are only heard.
    """

    text: str
    own: bool


def _wizard_speaker(record: dict) -> str | None:
    r"""
    Finds which speaker of a record is the chatbot side.

    The corpus names the two roles in `personaInfo` rather than in the turns, so the turns can only be
    attributed after reading which `speakerType` the wizard block claims.

    Args:
        record (`dict`):
            One parsed label file.

    Returns:
        `str` or `None`: The `speaker` value of the wizard, or `None` when the record names no wizard.
    """
    for key, block in record.get("personaInfo", {}).items():
        if isinstance(block, dict) and "wizard" in key.lower():
            return block.get("speakerType")
    return None


def _quotes_document(turn: dict) -> bool:
    r"""
    Whether a turn reads from a retrieved document.

    Args:
        turn (`dict`):
            One entry of a session's `dialog`.

    Returns:
        `bool`: `True` when `selectedContents` names any passage.
    """
    selected = turn.get("selectedContents") or []
    return any(any(entry) for entry in selected if isinstance(entry, list))


def _in_split(conversation: str, split: str) -> bool:
    r"""
    Decides whether a conversation belongs to the requested half of the validation directory.

    The corpus ships a training and a validation directory and the phase needs three splits, so the
    validation directory is halved. The bucket is a checksum of the identifier rather than a shuffle,
    so the same conversation lands in the same split whatever order the files are read in.

    Args:
        conversation (`str`):
            Identifier of the dialogue.
        split (`str`):
            One of `"train"`, `"valid"` or `"test"`.

    Returns:
        `bool`: Whether the conversation is part of the split.
    """
    if split == "train":
        return True
    bucket = zlib.crc32(conversation.encode("utf-8")) % HELD_OUT_BUCKETS
    return bucket == (0 if split == "valid" else 1)


def resolve_multisession_root(root: str | None = None) -> str:
    r"""
    Args:
        root (`str`, *optional*):
            Directory holding the extracted corpus, defaults to the corpus cache.

    Returns:
        `str`: The resolved directory.

    Raises:
        FileNotFoundError: If the directory is missing. The corpus is application gated, so it cannot
            be fetched and has to be placed by hand.
    """
    root = root or os.path.join(ArchiveDownloadMixin.get_cache_dir(), MULTISESSION_DIRECTORY)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"Multi session corpus root {root} does not exist. The corpus is distributed by "
            "application rather than by download, so extract it there first."
        )
    return root


def iter_wizard_turns(
    root: str | None = None,
    split: str = "train",
    drop_quoted: bool = True,
    max_turns: int | None = None,
):
    r"""
    Yields the chatbot side turns of the corpus.

    The wizard answers with a persona that states it is a chatbot, which is the register the text
    stream carries at deployment, and the apprentice side is left out because it is the user. A turn
    that quotes a retrieved document is dropped by default: those run to a median of 52 pieces against
    22 for the rest, which is past the length a single frame aligned turn can hold.

    Args:
        root (`str`, *optional*):
            Directory holding the extracted corpus, defaults to the corpus cache.
        split (`str`, *optional*, defaults to `"train"`):
            `"train"` reads the training directory, `"valid"` and `"test"` split the validation
            directory by conversation so that no dialogue appears in both.
        drop_quoted (`bool`, *optional*, defaults to `True`):
            Whether to skip turns that read from a retrieved document.
        max_turns (`int`, *optional*):
            Upper bound on the number of yielded turns.

    Yields:
        `MultisessionTurn`: One wizard turn.

    Raises:
        KeyError: If the split is not one the corpus ships.
        FileNotFoundError: If the corpus root is missing.
    """
    if split not in MULTISESSION_SPLITS:
        raise KeyError(f"Unknown split '{split}', expected one of {sorted(MULTISESSION_SPLITS)}.")

    directory = os.path.join(resolve_multisession_root(root), MULTISESSION_SPLITS[split])
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Split directory {directory} does not exist.")

    produced = 0
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        try:
            with open(path, encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue

        conversation = record.get("multisessionInfo", {}).get("multisessionID") or os.path.basename(path)
        if not _in_split(conversation, split):
            continue

        wizard = _wizard_speaker(record)
        if wizard is None:
            continue
        topic = record.get("topicInfo", {}).get("largeCategory", "")

        for session in record.get("sessionInfo", []):
            for turn in session.get("dialog", []):
                if turn.get("speaker") != wizard:
                    continue
                if drop_quoted and _quotes_document(turn):
                    continue
                text = " ".join((turn.get("utterance") or "").split())
                if not text:
                    continue
                yield MultisessionTurn(text=text, conversation=conversation, topic=topic)
                produced += 1
                if max_turns is not None and produced >= max_turns:
                    return


def iter_conversation_turns(
    root: str,
    split: str = "train",
    min_words: int = 2,
    max_words: int = 25,
):
    r"""
    Yields every speaker's turns of a conversation corpus between two people.

    Neither side plays a chatbot, so both are kept. A turn holding bare jamo (laughter, crying), an emoji or
    a tilde run is dropped, because none of them is said aloud.

    Args:
        root (`str`):
            Directory holding one subdirectory per split, each with the label JSON files.
        split (`str`, *optional*, defaults to `"train"`):
            Key of `MULTISESSION_SPLITS` naming the subdirectory to read.
        min_words (`int`, *optional*, defaults to 2):
            Fewest space separated words a kept turn has.
        max_words (`int`, *optional*, defaults to 25):
            Most space separated words a kept turn has.

    Yields:
        `MultisessionTurn`: One kept turn, with whitespace collapsed.

    Raises:
        KeyError: If the split is not one the corpus ships.
    """
    if split not in MULTISESSION_SPLITS:
        raise KeyError(f"Unknown split '{split}', expected one of {sorted(MULTISESSION_SPLITS)}.")

    directory = os.path.join(root, MULTISESSION_SPLITS[split])
    for path in sorted(glob.glob(os.path.join(directory, "**", "*.json"), recursive=True)):
        try:
            with open(path, encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue

        conversation = record.get("multisessionInfo", {}).get("multisessionID") or os.path.basename(path)
        topic = record.get("topicInfo", {}).get("topicType", "")
        for session in record.get("sessionInfo", []):
            for turn in session.get("dialog", []):
                text = " ".join((turn.get("utterance") or "").split())
                if not text or UNREADABLE_TEXT.search(text):
                    continue
                if not min_words <= len(text.split()) <= max_words:
                    continue
                yield MultisessionTurn(text=text, conversation=conversation, topic=topic)


def iter_dialogues(
    root: str,
    split: str = "train",
    min_words: int = 2,
    max_words: int = 25,
    drop_quoted: bool = True,
):
    r"""
    Yields the sessions of a multi session corpus as dialogues followed from one speaker's side.

    A record that names a wizard is followed from the wizard, the chatbot side. A record between two people
    is followed from one of its two speakers per session, chosen by a checksum of the session identifier, so
    both sides are used across the corpus and the choice does not depend on reading order. An own turn the
    text stream cannot carry (bare jamo, an emoji, a tilde run, a quoted document, a word count outside the
    bounds) ends the dialogue there, and the turns after it start a new one. The other side's turns are kept
    as they are, since only their length is used.

    Args:
        root (`str`):
            Directory holding one subdirectory per split, each with the label JSON files.
        split (`str`, *optional*, defaults to `"train"`):
            Key of `MULTISESSION_SPLITS`. `"valid"` and `"test"` halve the validation directory by
            conversation.
        min_words (`int`, *optional*, defaults to 2):
            Fewest space separated words a kept own turn has.
        max_words (`int`, *optional*, defaults to 25):
            Most space separated words a kept own turn has.
        drop_quoted (`bool`, *optional*, defaults to `True`):
            Whether an own turn that reads from a retrieved document ends the dialogue.

    Yields:
        `list[DialogueTurn]`: The turns of one dialogue in order, with at least one own turn.

    Raises:
        KeyError: If the split is not one the corpus ships.
    """
    if split not in MULTISESSION_SPLITS:
        raise KeyError(f"Unknown split '{split}', expected one of {sorted(MULTISESSION_SPLITS)}.")

    directory = os.path.join(root, MULTISESSION_SPLITS[split])
    for path in sorted(glob.glob(os.path.join(directory, "**", "*.json"), recursive=True)):
        try:
            with open(path, encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue

        conversation = record.get("multisessionInfo", {}).get("multisessionID") or os.path.basename(path)
        if not _in_split(conversation, split):
            continue
        wizard = _wizard_speaker(record)
        for index, session in enumerate(record.get("sessionInfo", [])):
            dialog = session.get("dialog", [])
            followed = wizard
            if followed is None:
                speakers = sorted({turn.get("speaker") for turn in dialog if turn.get("speaker")})
                if not speakers:
                    continue
                identifier = str(session.get("sessionID") or f"{conversation}-{index}")
                followed = speakers[zlib.crc32(identifier.encode("utf-8")) % len(speakers)]

            turns = []
            for turn in dialog:
                text = " ".join((turn.get("utterance") or "").split())
                if not text:
                    continue
                if turn.get("speaker") != followed:
                    turns.append(DialogueTurn(text=text, own=False))
                    continue
                usable = (
                    not UNREADABLE_TEXT.search(text)
                    and min_words <= len(text.split()) <= max_words
                    and not (drop_quoted and _quotes_document(turn))
                )
                if usable:
                    turns.append(DialogueTurn(text=text, own=True))
                    continue
                if any(t.own for t in turns):
                    yield turns
                turns = []
            if any(t.own for t in turns):
                yield turns
