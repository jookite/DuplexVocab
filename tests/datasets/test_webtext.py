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
"""Reading the Korean document corpora of the dense stage."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from polyplex_research.datasets.webtext import (
    KoreanWebTextCorpus,
    KoreanWikipediaCorpus,
    clean_wikipedia_text,
    held_out,
    write_documents,
)


def test_wikipedia_keeps_only_lines_that_end_a_sentence():
    article = "한국어는 한국의 공용어이다.\n\n역사\n- 목록 항목\n세종이 훈민정음을 만들었다고 한다.\n분류: 언어\n외부 링크"
    assert clean_wikipedia_text(article) == "한국어는 한국의 공용어이다.\n세종이 훈민정음을 만들었다고 한다."


def _write_shard(directory, index, texts, scores):
    (directory / "text" / "data").mkdir(parents=True, exist_ok=True)
    (directory / "score").mkdir(exist_ok=True)
    pq.write_table(pa.table({"text": texts}), directory / "text" / "data" / f"train-{index:05d}-of-00018.parquet")
    table = pa.table({"text": [t.replace("\n", " ") for t in texts], "score": scores})
    with pa.OSFile(str(directory / "score" / f"data-{index:05d}-of-00018.arrow"), "wb") as sink:
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)


def test_web_text_finds_each_score_by_text_across_shards_and_drops_low_or_missing_ones(tmp_path):
    texts = [f"문서 {i}번입니다.\n둘째 줄입니다." for i in range(400)]
    scores = [0.0 if i % 4 == 0 else 2.0 for i in range(400)]
    _write_shard(tmp_path, 0, texts[:300], scores[:250] + scores[300:350])
    # The score repository holds the documents in other shards and one of them not at all.
    table = pa.table({"text": [t.replace("\n", " ") for t in texts[1:]], "score": scores[1:]})
    with pa.OSFile(str(tmp_path / "score" / "data-00000-of-00018.arrow"), "wb") as sink:
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
    corpus = KoreanWebTextCorpus(str(tmp_path / "text"), str(tmp_path / "score"), min_score=1.0, split="train")
    kept = list(corpus)
    expected = [t for i, (t, s) in enumerate(zip(texts[:300], scores)) if i > 0 and s >= 1.0 and not held_out(t)]
    assert kept == expected
    valid = list(KoreanWebTextCorpus(str(tmp_path / "text"), str(tmp_path / "score"), split="valid"))
    assert not set(valid) & set(kept)


def test_wikipedia_drops_stubs_and_documents_are_written_as_json_lines(tmp_path):
    (tmp_path / "20231101.ko").mkdir()
    long_article = "긴 문서의 문장입니다. " * 30
    pq.write_table(pa.table({"text": [long_article, "짧은 문서이다."]}), tmp_path / "20231101.ko" / "train-00000-of-00001.parquet")
    corpus = KoreanWikipediaCorpus(str(tmp_path), split="train")
    output = tmp_path / "docs.jsonl"
    count = write_documents([corpus, ["추가 문서."]], str(output))
    lines = [json.loads(line)["text"] for line in output.read_text(encoding="utf-8").splitlines()]
    assert lines[-1] == "추가 문서." and "짧은 문서이다." not in lines
    assert count == len(lines) == (0 if held_out(long_article.strip()) else 1) + 1


def test_wikipedia_reads_the_dump_of_the_language_it_is_given(tmp_path):
    from polyplex_research.datasets.webtext import WikipediaCorpus

    (tmp_path / "20231101.ja").mkdir()
    (tmp_path / "20231101.ko").mkdir()
    article = "日本語は日本の言語である。" * 20
    pq.write_table(pa.table({"text": [article, "短い。"]}), tmp_path / "20231101.ja" / "train-00000-of-00001.parquet")
    pq.write_table(pa.table({"text": ["한국어 문서이다." * 40]}), tmp_path / "20231101.ko" / "train-00000-of-00001.parquet")
    documents = list(WikipediaCorpus("ja", root=str(tmp_path), split="train")) + list(
        WikipediaCorpus("ja", root=str(tmp_path), split="valid")
    )
    assert documents == [article]


def test_the_korean_wikipedia_is_the_wikipedia_of_ko(tmp_path):
    from polyplex_research.datasets.webtext import WikipediaCorpus

    assert issubclass(KoreanWikipediaCorpus, WikipediaCorpus)
    (tmp_path / "20231101.ko").mkdir()
    assert KoreanWikipediaCorpus(root=str(tmp_path)).language == "ko"


def test_fineweb_reads_the_text_column_of_a_subset(tmp_path):
    from polyplex_research.datasets.webtext import FineWebCorpus

    subset = tmp_path / "data" / "kor_Hang" / "train"
    subset.mkdir(parents=True)
    texts = [f"문서 {index}번의 내용이다." for index in range(400)]
    pq.write_table(pa.table({"text": texts, "id": list(range(400))}), subset / "000_00000.parquet")
    train = list(FineWebCorpus("HuggingFaceFW/fineweb-2", "data/kor_Hang/train", root=str(tmp_path)))
    valid = list(FineWebCorpus("HuggingFaceFW/fineweb-2", "data/kor_Hang/train", root=str(tmp_path), split="valid"))
    assert sorted(train + valid) == sorted(texts)
    assert all(held_out(text) for text in valid) and not any(held_out(text) for text in train)


def test_fineweb_reads_only_the_first_files_it_is_limited_to(tmp_path):
    from polyplex_research.datasets.webtext import FineWebCorpus

    subset = tmp_path / "sample" / "10BT"
    subset.mkdir(parents=True)
    for index in range(3):
        texts = [f"Document {index}-{line} of the sample." for line in range(50)]
        pq.write_table(pa.table({"text": texts}), subset / f"{index:03d}_00000.parquet")
    both = FineWebCorpus("HuggingFaceFW/fineweb-edu", "sample/10BT", root=str(tmp_path), max_files=2, split="train")
    every = list(FineWebCorpus("HuggingFaceFW/fineweb-edu", "sample/10BT", root=str(tmp_path), split="train"))
    assert {text.split()[1].split("-")[0] for text in both} == {"0", "1"}
    assert len(every) > len(list(both))


def test_a_streamed_subset_reads_the_hub_files_lazily_and_opens_only_what_is_read(tmp_path, monkeypatch):
    import fsspec
    import huggingface_hub

    from polyplex_research.datasets.webtext import FineWebCorpus

    subset = tmp_path / "datasets" / "HuggingFaceFW" / "fineweb-2" / "data" / "kor_Hang" / "train"
    subset.mkdir(parents=True)
    for index in range(2):
        texts = [f"문서 {index}-{line}번의 내용이다." for line in range(300)]
        pq.write_table(pa.table({"text": texts}), subset / f"{index:03d}_00000.parquet", row_group_size=20)
    opened = []
    local = fsspec.filesystem("file")

    class LocalHub:
        def open(self, path, mode="rb", **options):
            opened.append(path)
            return local.open(str(tmp_path / path), mode)

    names = sorted(f"data/kor_Hang/train/{path.name}" for path in subset.iterdir()) + ["data/kor_Hang/train/README.md"]
    monkeypatch.setattr(huggingface_hub, "HfFileSystem", LocalHub)
    listed = []

    def list_repo_tree(self, repo_id, path_in_repo=None, repo_type=None):
        # The subset's own folder is listed; the whole repository holds tens of thousands of files.
        listed.append(path_in_repo)
        return [type("Entry", (), {"path": name})() for name in names if name.startswith(path_in_repo + "/")]

    def list_repo_files(self, repo_id, repo_type=None):
        raise AssertionError("the whole repository is listed")

    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_tree", list_repo_tree)
    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_files", list_repo_files)
    first = []
    for text in FineWebCorpus("HuggingFaceFW/fineweb-2", "data/kor_Hang/train", stream=True):
        first.append(text)
        if len(first) == 5:
            break
    assert opened == ["datasets/HuggingFaceFW/fineweb-2/data/kor_Hang/train/000_00000.parquet"]
    assert listed == ["data/kor_Hang/train"]
    every = list(FineWebCorpus("HuggingFaceFW/fineweb-2", "data/kor_Hang/train", stream=True))
    root = str(tmp_path / "datasets" / "HuggingFaceFW" / "fineweb-2")
    assert every == list(FineWebCorpus("HuggingFaceFW/fineweb-2", "data/kor_Hang/train", root=root))
    assert first == every[:5]


def test_a_downloaded_subset_with_a_file_limit_lists_only_its_folder(monkeypatch, tmp_path):
    import huggingface_hub

    import polyplex_research.datasets.webtext as webtext

    names = ["data/kor_Hang/train/001_00000.parquet", "data/kor_Hang/train/000_00000.parquet", "data/kor_Hang/train/README.md"]
    listed, downloaded = [], []

    def list_repo_tree(self, repo_id, path_in_repo=None, repo_type=None):
        listed.append(path_in_repo)
        return [type("Entry", (), {"path": name})() for name in names]

    def list_repo_files(self, repo_id, repo_type=None):
        raise AssertionError("the whole repository is listed")

    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_tree", list_repo_tree)
    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_files", list_repo_files)
    monkeypatch.setattr(webtext, "download_corpus", lambda repo_id, patterns: downloaded.append(patterns) or str(tmp_path))
    webtext.FineWebCorpus("HuggingFaceFW/fineweb-2", "data/kor_Hang/train", max_files=1)
    assert listed == ["data/kor_Hang/train"]
    assert downloaded == [["data/kor_Hang/train/000_00000.parquet"]]


@pytest.mark.parametrize("line", ["日本語は言語である。", "यह भाषा है।", "هل هذا؟", "Τι είναι;", "한국어다."])
def test_sentence_ends_of_other_scripts_are_kept(line):
    assert clean_wikipedia_text(line) == line


def test_a_line_ending_in_a_semicolon_is_not_a_sentence():
    assert clean_wikipedia_text("a list item;") == ""


def test_a_dense_language_key_names_its_fineweb2_code():
    from polyplex_research.datasets.webtext import language_code, language_key

    assert language_key("kor_Hang") == "ko" and language_key("fra_Latn") == "fra_Latn"
    assert language_code("ko") == "kor_Hang" and language_code("fra_Latn") == "fra_Latn"
    assert language_code("en") is None


def test_each_fineweb_class_reads_its_own_repository(monkeypatch):
    import polyplex_research.datasets.webtext as webtext

    fetched = []
    monkeypatch.setattr(webtext, "download_corpus", lambda repo_id, patterns: fetched.append((repo_id, patterns)) or "root")
    assert webtext.FineWebEduCorpus().repo_id == "HuggingFaceFW/fineweb-edu"
    assert webtext.FineWeb2Corpus("kor_Hang").subset == "data/kor_Hang/train"
    hq = webtext.FineWeb2HQCorpus("jpn_Jpan")
    assert (hq.repo_id, hq.subset) == ("epfml/FineWeb2-HQ", "jpn_Jpan")
    assert fetched[-1] == ("epfml/FineWeb2-HQ", ["jpn_Jpan/*.parquet"])
    with pytest.raises(ValueError, match="hin_Deva"):
        webtext.FineWeb2HQCorpus("hin_Deva")


def test_fineweb2_hq_is_read_by_its_text_alone_and_split_as_the_rest(tmp_path):
    from polyplex_research.datasets.webtext import FineWeb2HQCorpus

    (tmp_path / "jpn_Jpan").mkdir()
    texts = [f"日本語の文書{index}です。" for index in range(400)]
    table = pa.table({"text": texts, "embeddings": [[[0.5] * 4] for _ in texts], "quality_score": [1.0] * len(texts)})
    pq.write_table(table, tmp_path / "jpn_Jpan" / "000_00000.parquet")
    train = list(FineWeb2HQCorpus("jpn_Jpan", root=str(tmp_path)))
    valid = list(FineWeb2HQCorpus("jpn_Jpan", root=str(tmp_path), split="valid"))
    assert sorted(train + valid) == sorted(texts)
    assert all(held_out(text) for text in valid) and not any(held_out(text) for text in train)


def test_the_multilingual_corpus_takes_the_filtered_source_a_language_has(monkeypatch):
    import polyplex_research.datasets.webtext as webtext

    source = webtext.MultilingualEduCorpus.source_of
    assert source("en") == "fineweb-edu" and source("en", filtered=False) == "fineweb-edu"
    assert source("ko") == "korean-webtext-edu" and source("ko", filtered=False) == "fineweb-2"
    assert source("ja") == "fineweb2-hq" and source("fra_Latn") == "fineweb2-hq"
    assert source("hi") == "fineweb-2" and source("pol_Latn", filtered=False) == "fineweb-2"
    key = webtext.MultilingualEduCorpus.documents_key
    assert key("en") == "en" and key("hi") == "hi" and key("ja") == "ja@fineweb2-hq"
    assert key("ja", filtered=False) == "ja" and key("ko") == "ko@korean-webtext-edu"
    monkeypatch.setattr(webtext, "download_corpus", lambda repo_id, patterns: "root")
    corpus = webtext.MultilingualEduCorpus("ja", stream=False)
    assert isinstance(corpus.corpus, webtext.FineWeb2HQCorpus) and corpus.request == ["epfml/FineWeb2-HQ", "jpn_Jpan"]
    unfiltered = webtext.MultilingualEduCorpus("ko", filtered=False, stream=False)
    assert unfiltered.request == ["HuggingFaceFW/fineweb-2", "data/kor_Hang/train"]
    korean = webtext.MultilingualEduCorpus("ko", stream=False)
    assert isinstance(korean.corpus, webtext.KoreanWebTextCorpus)


def test_fineweb2_bytes_sum_the_listed_parquet_sizes_once(tmp_path, monkeypatch):
    import huggingface_hub

    from polyplex_research.datasets.webtext import fineweb2_bytes

    calls = []

    class Entry:
        def __init__(self, path, size):
            self.path, self.size = path, size

    def tree(self, repo_id, path_in_repo=None, repo_type=None, recursive=False):
        calls.append(path_in_repo)
        return [Entry(f"{path_in_repo}/000.parquet", 10), Entry(f"{path_in_repo}/001.parquet", 5),
                Entry(f"{path_in_repo}/notes.md", 99)]

    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_tree", tree)
    cache = tmp_path / "sizes.json"
    assert fineweb2_bytes(("kor_Hang", "fra_Latn"), str(cache)) == {"kor_Hang": 15, "fra_Latn": 15}
    assert fineweb2_bytes(("kor_Hang", "fra_Latn"), str(cache)) == {"kor_Hang": 15, "fra_Latn": 15}
    assert calls == ["data/kor_Hang/train", "data/fra_Latn/train"]
