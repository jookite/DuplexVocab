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
"""Web text and Wikipedia, read as documents for dense adaptation and for choosing the tokens of a language."""

import glob
import hashlib
import json
import os
import re
import zlib


WEBTEXT_REPO = "HAERAE-HUB/KOREAN-WEBTEXT"
FINEWEB_EDU_REPO = "HuggingFaceFW/fineweb-edu"
FINEWEB2_REPO = "HuggingFaceFW/fineweb-2"
FINEWEB2_HQ_REPO = "epfml/FineWeb2-HQ"
WEBTEXT_SCORES_REPO = "eliceai/korean-webtext-edu"
WIKIPEDIA_REPO = "wikimedia/wikipedia"
WIKIPEDIA_DUMP = "20231101"

# One document in this many goes to the held out split, chosen by a checksum of its text.
HELD_OUT_EVERY = 200

# Sentence-final marks of the scripts Wikipedia is read in: Latin, CJK full width, Devanagari danda, Arabic and
# Greek question marks, Ethiopic full stop and Armenian full stop.
SENTENCE_END = re.compile(r"[.?!。．！？।॥؟\u037e።։][\"'”’)\]」』》）]*$")


def held_out(text: str) -> bool:
    r"""
    Args:
        text (`str`):
            A document.

    Returns:
        `bool`: Whether the document belongs to the held out split. The choice depends on the text only, so
        it does not change with reading order.
    """
    return zlib.crc32(text[:512].encode("utf-8")) % HELD_OUT_EVERY == 0


def _in_split(text: str, split: str) -> bool:
    if split not in ("train", "valid"):
        raise KeyError(f"Unknown split '{split}', expected 'train' or 'valid'.")
    return held_out(text) == (split == "valid")


def download_corpus(repo_id: str, allow_patterns: list) -> str:
    r"""
    Fetches the files of a Hub dataset repository into the Hub cache under `HF_HOME`.

    Args:
        repo_id (`str`):
            Dataset repository.
        allow_patterns (`list`):
            Glob patterns of the files to fetch.

    Returns:
        `str`: Local directory of the snapshot.
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id, repo_type="dataset", allow_patterns=allow_patterns)


def clean_wikipedia_text(text: str) -> str:
    r"""
    Keeps the prose of a Wikipedia article.

    A line is kept when it ends a sentence. Section titles, list items, infobox residue and the category
    lines at the end of an article do not, so they are dropped.

    Args:
        text (`str`):
            Article text as the dump stores it, one paragraph or item per line.

    Returns:
        `str`: The kept lines joined by line breaks, empty when none is kept.
    """
    lines = (line.strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line and SENTENCE_END.search(line))


def text_key(text: str) -> bytes:
    r"""
    Args:
        text (`str`):
            A document.

    Returns:
        `bytes`: A digest of the document with whitespace collapsed, equal for two copies of a document that
        differ only in line breaks.
    """
    return hashlib.blake2b(" ".join(text.split()).encode("utf-8"), digest_size=12).digest()


class KoreanWebTextCorpus:
    r"""
    Constructs a reader of HAERAE KOREAN-WEBTEXT, filtered by the educational score `eliceai/korean-webtext-edu`
    gives each of its documents.

    The score repository holds the same documents with line breaks removed, sharded differently, so a
    document's score is found by its text (see [`text_key`]) rather than by its row. A document the score
    repository does not hold is dropped.

    Args:
        text_dir (`str`, *optional*):
            Snapshot of `HAERAE-HUB/KOREAN-WEBTEXT`. Fetched when omitted.
        score_dir (`str`, *optional*):
            Snapshot of `eliceai/korean-webtext-edu`. Fetched when omitted.
        min_score (`float`, *optional*, defaults to 1.0):
            Lowest educational score a kept document has. A score of 0 marks pages with no informational
            content.
        split (`str`, *optional*, defaults to `"train"`):
            `"train"` or `"valid"`, see [`held_out`].
    """

    def __init__(self, text_dir: str | None = None, score_dir: str | None = None, min_score: float = 1.0, split: str = "train"):
        self.text_dir = text_dir or download_corpus(WEBTEXT_REPO, ["data/*.parquet", "README.md"])
        self.score_dir = score_dir or download_corpus(WEBTEXT_SCORES_REPO, ["data-*-of-00018.arrow", "*.json"])
        self.min_score = min_score
        self.split = split

    def scores(self) -> dict:
        r"""
        Returns:
            `dict`: The score of every document of the score repository, keyed by [`text_key`].
        """
        import pyarrow as pa

        scores = {}
        for path in sorted(glob.glob(os.path.join(self.score_dir, "data-*-of-00018.arrow"))):
            with pa.memory_map(path) as source:
                table = pa.ipc.open_stream(source).read_all()
            for text, score in zip(table.column("text").to_pylist(), table.column("score").to_pylist()):
                if text and score is not None:
                    scores[text_key(text)] = score
        return scores

    def __iter__(self):
        import pyarrow.parquet as pq

        scores = self.scores()
        for path in sorted(glob.glob(os.path.join(self.text_dir, "data", "train-*.parquet"))):
            for text in pq.read_table(path, columns=["text"]).column("text").to_pylist():
                if not text or not _in_split(text, self.split):
                    continue
                score = scores.get(text_key(text))
                if score is not None and score >= self.min_score:
                    yield text


class WikipediaCorpus:
    r"""
    Constructs a reader of one language's Wikipedia dump in `wikimedia/wikipedia`, keeping the prose of each
    article (see [`clean_wikipedia_text`]).

    Args:
        language (`str`):
            Wikipedia language code of the dump, such as `"ko"` or `"ja"`.
        root (`str`, *optional*):
            Snapshot of `wikimedia/wikipedia` holding the dump's files. Fetched when omitted.
        dump (`str`, *optional*, defaults to `"20231101"`):
            Date of the dump.
        min_characters (`int`, *optional*, defaults to 200):
            Shortest cleaned article kept. Stubs are mostly one definition sentence.
        split (`str`, *optional*, defaults to `"train"`):
            `"train"` or `"valid"`, see [`held_out`].
    """

    def __init__(
        self,
        language: str,
        root: str | None = None,
        dump: str = WIKIPEDIA_DUMP,
        min_characters: int = 200,
        split: str = "train",
    ):
        self.language = language
        self.config = f"{dump}.{language}"
        self.root = root or download_corpus(WIKIPEDIA_REPO, [f"{self.config}/*"])
        self.min_characters = min_characters
        self.split = split

    def __iter__(self):
        import pyarrow.parquet as pq

        for path in sorted(glob.glob(os.path.join(self.root, self.config, "*.parquet"))):
            for text in pq.read_table(path, columns=["text"]).column("text").to_pylist():
                cleaned = clean_wikipedia_text(text or "")
                if len(cleaned) >= self.min_characters and _in_split(cleaned, self.split):
                    yield cleaned


class KoreanWikipediaCorpus(WikipediaCorpus):
    r"""
    Constructs a reader of the Korean Wikipedia dump (`20231101.ko`), the one the dense stage of the first design
    read.

    Args:
        root (`str`, *optional*):
            Snapshot of `wikimedia/wikipedia` holding the `20231101.ko` files. Fetched when omitted.
        min_characters (`int`, *optional*, defaults to 200):
            Shortest cleaned article kept.
        split (`str`, *optional*, defaults to `"train"`):
            `"train"` or `"valid"`, see [`held_out`].
    """

    def __init__(self, root: str | None = None, min_characters: int = 200, split: str = "train"):
        super().__init__("ko", root=root, min_characters=min_characters, split=split)


class FineWebCorpus:
    r"""
    Constructs a reader of the `text` column of one subset of a FineWeb repository, such as a language of
    `HuggingFaceFW/fineweb-2` (`data/kor_Hang/train`) or a sample of `HuggingFaceFW/fineweb-edu` (`sample/10BT`).

    Args:
        repo_id (`str`):
            Dataset repository.
        subset (`str`):
            Directory of the repository whose parquet files are read.
        root (`str`, *optional*):
            Snapshot of the repository holding the subset. Fetched when omitted.
        split (`str`, *optional*, defaults to `"train"`):
            `"train"` or `"valid"`, see [`held_out`].
        max_files (`int`, *optional*):
            Read only the first this many parquet files of the subset, in name order, and fetch only those.
        stream (`bool`, *optional*, defaults to `False`):
            Read the parquet files on the hub row group by row group instead of downloading them, so a reader that
            stops early fetches only the leading row groups of the first files.
    """

    def __init__(
        self,
        repo_id: str,
        subset: str,
        root: str | None = None,
        split: str = "train",
        max_files: int | None = None,
        stream: bool = False,
    ):
        self.repo_id = repo_id
        self.subset = subset
        self.split = split
        self.max_files = max_files
        self.files = None
        if root is None and stream:
            self.files = self._subset_files()
        elif root is None:
            patterns = [f"{subset}/*.parquet"] if max_files is None else self._subset_files()
            root = download_corpus(repo_id, patterns)
        self.root = root

    def _subset_files(self) -> list:
        from huggingface_hub import HfApi

        # Only the subset's folder is listed: a multilingual repository holds tens of thousands of files.
        tree = HfApi().list_repo_tree(self.repo_id, path_in_repo=self.subset, repo_type="dataset")
        files = [entry.path for entry in tree]
        return sorted(
            name for name in files if os.path.dirname(name) == self.subset and name.endswith(".parquet")
        )[: self.max_files]

    def __iter__(self):
        import pyarrow.parquet as pq

        if self.files is not None:
            import huggingface_hub

            hub = huggingface_hub.HfFileSystem()
            # Large reads ahead: each read of the Hub file is one request.
            sources = (
                hub.open(f"datasets/{self.repo_id}/{name}", "rb", block_size=64 << 20, cache_type="readahead")
                for name in self.files
            )
        else:
            paths = sorted(glob.glob(os.path.join(self.root, self.subset, "*.parquet")))
            sources = iter(paths[: self.max_files])
        for source in sources:
            # One batch per row group, so the first documents arrive without reading the 65,536 rows pyarrow batches by default.
            for batch in pq.ParquetFile(source).iter_batches(columns=["text"], batch_size=1000):
                for text in batch.column("text").to_pylist():
                    if text and _in_split(text, self.split):
                        yield text


class FineWebEduCorpus(FineWebCorpus):
    r"""
    Constructs a reader of one sample of `HuggingFaceFW/fineweb-edu`, English web pages kept by an educational quality
    classifier.

    Args:
        subset (`str`, *optional*, defaults to `"sample/10BT"`):
            Directory of the sample.
        **kwargs:
            `root`, `split`, `max_files` and `stream`, see [`FineWebCorpus`].
    """

    def __init__(self, subset: str = "sample/10BT", **kwargs):
        super().__init__(FINEWEB_EDU_REPO, subset, **kwargs)


class FineWeb2Corpus(FineWebCorpus):
    r"""
    Constructs a reader of one language of `HuggingFaceFW/fineweb-2`, unfiltered beyond FineWeb-2's own pipeline.

    Args:
        code (`str`):
            FineWeb-2 language code, such as `"kor_Hang"`.
        **kwargs:
            `root`, `split`, `max_files` and `stream`, see [`FineWebCorpus`].
    """

    def __init__(self, code: str, **kwargs):
        super().__init__(FINEWEB2_REPO, f"data/{code}/train", **kwargs)


class FineWeb2HQCorpus(FineWebCorpus):
    r"""
    Constructs a reader of one language of `epfml/FineWeb2-HQ`, the 10% of FineWeb-2's documents an XLM-RoBERTa
    quality classifier scores highest. Only the `text` column is read; the files also carry document embeddings.

    Args:
        code (`str`):
            FineWeb-2 language code, one of [`FINEWEB2_HQ_CODES`].
        **kwargs:
            `root`, `split`, `max_files` and `stream`, see [`FineWebCorpus`].

    Raises:
        ValueError: If the language has no FineWeb2-HQ subset.
    """

    def __init__(self, code: str, **kwargs):
        if code not in FINEWEB2_HQ_CODES:
            raise ValueError(f"FineWeb2-HQ holds no {code!r}.")
        super().__init__(FINEWEB2_HQ_REPO, code, **kwargs)


# FineWeb-2 codes of the languages Qwen3 lists, the ones the Polyplex vocabulary was built from (`build_vocab.ipynb`,
# `QWEN3_LANGUAGES`), in that order. English reads FineWeb-Edu, and a variety sharing its parent's subset appears once.
QWEN3_FINEWEB2 = (
    "fra_Latn", "por_Latn", "deu_Latn", "ron_Latn", "swe_Latn", "dan_Latn", "bul_Cyrl", "rus_Cyrl", "ces_Latn",
    "ell_Grek", "ukr_Cyrl", "spa_Latn", "nld_Latn", "slk_Latn", "hrv_Latn", "pol_Latn", "lit_Latn", "nob_Latn",
    "nno_Latn", "fas_Arab", "slv_Latn", "guj_Gujr", "lvs_Latn", "ita_Latn", "oci_Latn", "npi_Deva", "mar_Deva",
    "bel_Cyrl", "srp_Cyrl", "ltz_Latn", "vec_Latn", "asm_Beng", "cym_Latn", "szl_Latn", "ast_Latn", "hne_Deva",
    "awa_Deva", "mai_Deva", "bho_Deva", "snd_Arab", "gle_Latn", "fao_Latn", "hin_Deva", "pan_Guru", "ben_Beng",
    "ory_Orya", "tgk_Cyrl", "ydd_Hebr", "lmo_Latn", "lij_Latn", "scn_Latn", "fur_Latn", "srd_Latn", "glg_Latn",
    "cat_Latn", "isl_Latn", "als_Latn", "lim_Latn", "afr_Latn", "mkd_Cyrl", "sin_Sinh", "urd_Arab", "mag_Deva",
    "bos_Latn", "hye_Armn", "cmn_Hani", "yue_Hani", "mya_Mymr", "arb_Arab", "ars_Arab", "apc_Arab", "arz_Arab",
    "ary_Arab", "acm_Arab", "aeb_Arab", "heb_Hebr", "mlt_Latn", "ind_Latn", "zsm_Latn", "fil_Latn", "ceb_Latn",
    "jav_Latn", "sun_Latn", "min_Latn", "ban_Latn", "bjn_Latn", "pag_Latn", "ilo_Latn", "war_Latn", "tam_Taml",
    "tel_Telu", "kan_Knda", "mal_Mlym", "tur_Latn", "azj_Latn", "uzn_Latn", "kaz_Cyrl", "bak_Cyrl", "tat_Cyrl",
    "tha_Thai", "lao_Laoo", "fin_Latn", "ekk_Latn", "hun_Latn", "vie_Latn", "khm_Khmr", "jpn_Jpan", "kor_Hang",
    "kat_Geor", "eus_Latn", "hat_Latn", "pap_Latn", "kea_Latn", "tpi_Latn", "swh_Latn",
)
# The languages `epfml/FineWeb2-HQ` holds.
FINEWEB2_HQ_CODES = (
    "arb_Arab", "ces_Latn", "cmn_Hani", "dan_Latn", "deu_Latn", "ell_Grek", "fas_Arab", "fra_Latn", "hun_Latn",
    "ind_Latn", "ita_Latn", "jpn_Jpan", "nld_Latn", "pol_Latn", "por_Latn", "rus_Cyrl", "spa_Latn", "swe_Latn",
    "tur_Latn", "vie_Latn",
)
# The languages the dense stage is evaluated on (SPEC S13.4), mapped to their FineWeb-2 code; English reads FineWeb-Edu.
DENSE_EVALUATION = {
    "en": None,
    "ko": "kor_Hang",
    "zh": "cmn_Hani",
    "ru": "rus_Cyrl",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "ja": "jpn_Jpan",
}


def language_key(code: str) -> str:
    r"""
    Args:
        code (`str`):
            A FineWeb-2 language code, such as `"kor_Hang"`.

    Returns:
        `str`: The key the dense run uses for that language: the key of [`DENSE_EVALUATION`] with that code, or the
        code itself.
    """
    for language, evaluated in DENSE_EVALUATION.items():
        if evaluated == code:
            return language
    return code


def language_code(language: str) -> str | None:
    r"""
    Args:
        language (`str`):
            A key of [`DENSE_EVALUATION`] or a FineWeb-2 language code.

    Returns:
        `str | None`: Its FineWeb-2 code, `None` for English.
    """
    return DENSE_EVALUATION.get(language, language)


def fineweb2_bytes(codes: tuple = QWEN3_FINEWEB2, cache: str | None = None) -> dict:
    r"""
    Bytes of the parquet files of each FineWeb-2 language, from the hub's file listing, kept in `cache` so the listing
    is read once.

    Args:
        codes (`tuple[str]`, *optional*, defaults to [`QWEN3_FINEWEB2`]):
            FineWeb-2 language codes.
        cache (`str`, *optional*):
            JSON file holding the sizes already listed. `$HF_HOME/fineweb2_bytes.json` when omitted.

    Returns:
        `dict`: Code mapped to bytes.
    """
    from huggingface_hub import HfApi

    cache = cache or os.path.join(os.environ.get("HF_HOME", ".cache/huggingface"), "fineweb2_bytes.json")
    sizes = json.load(open(cache)) if os.path.exists(cache) else {}
    missing = [code for code in codes if code not in sizes]
    if missing:
        api = HfApi()
        for code in missing:
            entries = api.list_repo_tree(FINEWEB2_REPO, path_in_repo=f"data/{code}/train", repo_type="dataset", recursive=False)
            sizes[code] = sum(entry.size for entry in entries if entry.path.endswith(".parquet"))
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        with open(cache, "w") as handle:
            json.dump(sizes, handle, indent=1)
    return {code: sizes[code] for code in codes}


class MultilingualEduCorpus:
    r"""
    Constructs a reader of one language's documents for the dense stage (SPEC S13.4), from the quality filtered source
    that language has: FineWeb-Edu for English ([`FineWebEduCorpus`]), the educational score of KOREAN-WEBTEXT for
    Korean ([`KoreanWebTextCorpus`]), FineWeb2-HQ for its 20 languages ([`FineWeb2HQCorpus`]), and FineWeb-2 itself
    for the rest ([`FineWeb2Corpus`]). With `filtered=False` every language but English reads FineWeb-2, the source the
    dense evaluation reads.

    Args:
        language (`str`):
            A key of [`DENSE_EVALUATION`] or a FineWeb-2 language code.
        split (`str`, *optional*, defaults to `"train"`):
            `"train"` or `"valid"`, see [`held_out`].
        filtered (`bool`, *optional*, defaults to `True`):
            Whether a language with a quality filtered source reads it.
        max_files (`int`, *optional*):
            Parquet files of a FineWeb source read, see [`FineWebCorpus`].
        stream (`bool`, *optional*, defaults to `True`):
            Whether a FineWeb source is streamed from the hub, see [`FineWebCorpus`].
        min_score (`float`, *optional*, defaults to 1.0):
            Lowest educational score of a kept Korean document.
    """

    def __init__(
        self,
        language: str,
        split: str = "train",
        filtered: bool = True,
        max_files: int | None = None,
        stream: bool = True,
        min_score: float = 1.0,
    ):
        self.language = language
        self.source = self.source_of(language, filtered)
        code = language_code(language)
        options = {"split": split, "max_files": max_files, "stream": stream}
        if self.source == "fineweb-edu":
            self.corpus = FineWebEduCorpus(**options)
        elif self.source == "korean-webtext-edu":
            self.corpus = KoreanWebTextCorpus(min_score=min_score, split=split)
        elif self.source == "fineweb2-hq":
            self.corpus = FineWeb2HQCorpus(code, **options)
        else:
            self.corpus = FineWeb2Corpus(code, **options)

    @staticmethod
    def source_of(language: str, filtered: bool = True) -> str:
        r"""
        Args:
            language (`str`):
                A key of [`DENSE_EVALUATION`] or a FineWeb-2 language code.
            filtered (`bool`, *optional*, defaults to `True`):
                Whether a quality filtered source is taken when there is one.

        Returns:
            `str`: `"fineweb-edu"`, `"korean-webtext-edu"`, `"fineweb2-hq"` or `"fineweb-2"`.
        """
        code = language_code(language)
        if code is None:
            return "fineweb-edu"
        if filtered and code == "kor_Hang":
            return "korean-webtext-edu"
        if filtered and code in FINEWEB2_HQ_CODES:
            return "fineweb2-hq"
        return "fineweb-2"

    @classmethod
    def documents_key(cls, language: str, filtered: bool = True) -> str:
        r"""
        Args:
            language (`str`):
                A key of [`DENSE_EVALUATION`] or a FineWeb-2 language code.
            filtered (`bool`, *optional*, defaults to `True`):
                Whether a quality filtered source is taken when there is one.

        Returns:
            `str`: The key a run counts the documents it read from this language's source under: the language for
            FineWeb-Edu and FineWeb-2, `"<language>@<source>"` for a filtered source.
        """
        source = cls.source_of(language, filtered)
        return language if source in ("fineweb-edu", "fineweb-2") else f"{language}@{source}"

    @property
    def request(self) -> list:
        r"""
        Returns:
            `list`: What identifies the documents this reader yields, for a cache of them.
        """
        if isinstance(self.corpus, FineWebCorpus):
            return [self.corpus.repo_id, self.corpus.subset]
        return [WEBTEXT_REPO, f"edu>={self.corpus.min_score}"]

    def __iter__(self):
        return iter(self.corpus)


def dense_repositories() -> list:
    r"""
    Returns:
        `list[str]`: The repositories [`MultilingualEduCorpus`] reads, whose downloads a run frees once its documents
        are read.
    """
    return [FINEWEB_EDU_REPO, FINEWEB2_REPO, FINEWEB2_HQ_REPO, WEBTEXT_REPO, WEBTEXT_SCORES_REPO]


def write_documents(corpora: list, output_path: str) -> int:
    r"""
    Writes the documents of several corpora, one after another, as JSON lines with a `text` field.

    Args:
        corpora (`list`):
            Iterables of document strings, such as [`KoreanWebTextCorpus`] and [`KoreanWikipediaCorpus`].
        output_path (`str`):
            File the documents are written to.

    Returns:
        `int`: Number of documents written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as sink:
        for corpus in corpora:
            for text in corpus:
                sink.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                count += 1
    return count


def main():
    r"""
    Command line: writes the documents of the dense stage of Phase 1 for one split.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Documents of the dense stage of Phase 1.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--min-score", type=float, default=1.0)
    parser.add_argument("--corpora", nargs="+", default=["webtext", "wikipedia"])
    arguments = parser.parse_args()

    readers = {
        "webtext": lambda: KoreanWebTextCorpus(min_score=arguments.min_score, split=arguments.split),
        "wikipedia": lambda: KoreanWikipediaCorpus(split=arguments.split),
    }
    count = write_documents([readers[name]() for name in arguments.corpora], arguments.output)
    print(f"[documents] {count} -> {arguments.output}", flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "DENSE_EVALUATION",
    "FINEWEB2_HQ_CODES",
    "FineWeb2Corpus",
    "FineWeb2HQCorpus",
    "FineWebCorpus",
    "FineWebEduCorpus",
    "MultilingualEduCorpus",
    "QWEN3_FINEWEB2",
    "dense_repositories",
    "fineweb2_bytes",
    "language_code",
    "language_key",
    "KoreanWebTextCorpus",
    "KoreanWikipediaCorpus",
    "WikipediaCorpus",
    "clean_wikipedia_text",
    "download_corpus",
    "held_out",
    "text_key",
    "write_documents",
]
